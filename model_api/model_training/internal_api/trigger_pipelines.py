"""
    Actual Pipelines that are called by internal_api's `internal_main.py` functions
"""
import logging
import pathlib
import sys
import pickle
import random
import time
import torch
import os

PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")
sys.path.append(PATH_TO_PARENT + "../../")

from trigger_ner.utilities.config import Config
from trigger_ner.utilities.reader import Reader
from trigger_ner.utilities.utils import batching_list_instances
from trigger_ner.utilities.duplicates import remove_duplicates
from trigger_ner.model.soft_inferencer_naive import SoftSequenceNaive, SoftSequenceNaiveTrainer
from trigger_ner.model.soft_matcher import SoftMatcher, SoftMatcherTrainer
from trigger_ner.model.soft_inferencer import SoftSequence, SoftSequenceTrainer

from fast_api.fast_api_util_functions import update_model_training, send_model_metadata


def standard_ner_pipeline(payload):
    start_time = time.time()
    build_data = payload["build_data"]
    conf = Config(payload)

    if conf.is_lean_life:
        update_model_training(
            conf.project_id, conf.experiment_name, -1, conf.num_epochs, -1, -1,
            "starting training pipeline"
        )

    if build_data:
        reader = Reader(conf.digit2zero)
        train_data = reader.build_data(payload["labeled_data"], "labeled")

        # TODO: split data 80/20 if not present
        if "dev_data" not in payload or payload["dev_data"] is None:
            dev_data = None
        else:
            dev_data = reader.build_data(payload["dev_data"], "dev")

        if "eval_data" not in payload or payload["eval_data"] is None:
            eval_data = None
        else:
            eval_data = reader.build_data(payload["eval_data"], "eval")

        # vocab
        conf.build_label_idx(train_data)
        conf.build_word_idx(train_data, dev_data)
        conf.build_emb_table()
        conf.dump_label_word_emb_data()

        conf.map_insts_ids(train_data, "labeled")
        if dev_data:
            conf.map_insts_ids(dev_data, "dev")
        if eval_data:
            conf.map_insts_ids(eval_data, "eval")

        # TODO: ask dongho about implementation
        # if conf.context_emb == ContextEmb.bert:
        #     print('Loading the BERT vectors for all datasets.')
        #     conf.context_emb_size = load_bert_vec(conf.trigger_file + "." + conf.context_emb.name + ".vec", dataset)

        if conf.is_lean_life:
            time_spent = time.time() - start_time
            update_model_training(
                conf.project_id, conf.experiment_name, -1, conf.num_epochs, time_spent, -1,
                "built training data"
            )
    else:
        # load_data
        with open(conf.generate_training_data_path("labeled"), 'rb') as f:
            train_data = pickle.load(f)

        with open(conf.generate_training_data_path("dev"), 'rb') as f:
            dev_data = pickle.load(f)

        with open(conf.generate_training_data_path("eval"), 'rb') as f:
            eval_data = pickle.load(f)

        # load vocab
        conf.read_label_word_emd_data()

    # dataset division
    numbers = int(len(train_data) * conf.percentage / 100)
    initial_trains = train_data[:numbers]
    random.seed(conf.seed)
    random.shuffle(initial_trains)

    encoder = SoftSequenceNaive(conf)
    trainer = SoftSequenceNaiveTrainer(encoder, conf, dev_data, eval_data)

    if conf.is_lean_life:
        time_spent = time.time() - start_time
        update_model_training(
            conf.project_id, conf.experiment_name, -1, conf.num_epochs, time_spent, -1, "starting training"
        )

    _, best_train_loss = trainer.train_model(conf.num_epochs, initial_trains)
    model_save_path = conf.generate_model_path("naive")

    if conf.is_lean_life:
        file_size = os.path.getsize(model_save_path)
        send_model_metadata(conf.project_id, conf.experiment_name, model_save_path, best_train_loss, file_size)

    return model_save_path


def evaluate_standard_ner_pipeline(payload):
    conf = Config(payload)

    # load vocab
    conf.read_label_word_emd_data()

    reader = Reader(conf.digit2zero)
    eval_data = [{'text': tup[0], 'label': tup[1]} for tup in payload["eval_data"]]
    eval_data = reader.build_data(eval_data, "eval")
    conf.map_insts_ids(eval_data, "eval")

    encoder = SoftSequenceNaive(conf)
    encoder.load_state_dict(
        torch.load(conf.generate_model_path("naive"))
    )
    trainer = SoftSequenceNaiveTrainer(encoder, conf)

    encoder.eval()
    test_batches = batching_list_instances(conf, eval_data)
    test_metrics = trainer.evaluate_model(test_batches, "eval", eval_data)
    return test_metrics


def predict_standard_ner_pipeline(payload):
    conf = Config(payload)

    # load vocab
    conf.read_label_word_emd_data()

    reader = Reader(conf.digit2zero)
    pred_data = [{'text': row} for row in payload["prediction_data"]]
    pred_data = reader.build_data(pred_data, "pred")
    conf.map_insts_ids(pred_data, "pred")

    encoder = SoftSequenceNaive(conf)
    encoder.load_state_dict(
        torch.load(conf.generate_model_path("naive"))
    )
    trainer = SoftSequenceNaiveTrainer(encoder, conf)

    encoder.eval()
    pred_batches = batching_list_instances(conf, pred_data)
    trainer.predict_model(pred_batches, pred_data)

    return list(map(lambda x: " ".join(x.prediction), pred_data))


def trigger_soft_match_pipeline(payload):
    start_time = time.time()
    build_data = payload["build_data"]
    conf = Config(payload)

    if conf.is_lean_life:
        update_model_training(
            conf.project_id, conf.experiment_name, -1, conf.num_epochs, -1, -1,
            "starting training pipeline"
        )

    conf.optimizer = conf.trig_optimizer
    reader = Reader(conf.digit2zero)

    if build_data:
        train_data, max_length, label_length = reader.build_trigger_data(payload["labeled_data"])
        reader.merge_labels(train_data)

        # TODO: split data 80/20 if not present
        if "dev_data" not in payload or payload["dev_data"] is None:
            dev_data = None
        else:
            dev_data = reader.build_data(payload["dev_data"], "dev")

        if "eval_data" not in payload or payload["eval_data"] is None:
            eval_data = None
        else:
            eval_data = reader.build_data(payload["eval_data"], "eval")

        # vocab
        conf.build_label_idx(train_data)
        conf.build_word_idx(train_data, dev_data)
        conf.build_emb_table()
        conf.dump_label_word_emb_data(label_length)

        conf.map_insts_ids(train_data, "trigger")
        if dev_data:
            conf.map_insts_ids(dev_data, "dev")
        if eval_data:
            conf.map_insts_ids(eval_data, "eval")

        # TODO: ask dongho about implementation
        # if conf.context_emb == ContextEmb.bert:
        #     print('Loading the BERT vectors for all datasets.')
        #     conf.context_emb_size = load_bert_vec(conf.trigger_file + "." + conf.context_emb.name + ".vec", dataset)

        if conf.is_lean_life:
            time_spent = time.time() - start_time
            update_model_training(
                conf.project_id, conf.experiment_name, -1, conf.num_epochs, time_spent, -1,
                "built training data"
            )
    else:
        # load_data
        with open(conf.generate_training_data_path("trigger"), 'rb') as f:
            train_data = pickle.load(f)

        with open(conf.generate_training_data_path("dev"), 'rb') as f:
            dev_data = pickle.load(f)

        with open(conf.generate_training_data_path("eval"), 'rb') as f:
            eval_data = pickle.load(f)

        # load vocab
        label_length = conf.read_label_word_emd_data()

    dataset = reader.trigger_percentage(train_data, conf.percentage)
    encoder = SoftMatcher(conf, label_length)
    trainer = SoftMatcherTrainer(encoder, conf, dev_data, eval_data)

    # matching module training
    random.shuffle(dataset)
    if conf.is_lean_life:
        time_spent = time.time() - start_time
        update_model_training(
            conf.project_id, conf.experiment_name, -1, conf.num_epochs, time_spent, -1, "starting pre-training"
        )
    trainer.train_model(conf.num_epochs_soft, dataset)
    if conf.is_lean_life:
        time_spent = time.time() - start_time
        update_model_training(
            conf.project_id, conf.experiment_name, -1, conf.num_epochs, time_spent, -1, "completed pre-training"
        )
    logits, predicted, triggers = trainer.get_triggervec(dataset)
    triggers_remove = remove_duplicates(logits, predicted, triggers, dataset)

    # write trigger data to file
    path = conf.generate_training_data_path("soft_triggers")
    with open(path, 'wb') as f:
        pickle.dump(triggers_remove, f)

    # sequence labeling module training
    random.shuffle(dataset)
    inference = SoftSequence(conf, encoder)
    sequence_trainer = SoftSequenceTrainer(inference, conf, dev_data, eval_data, triggers_remove)

    if conf.is_lean_life:
        time_spent = time.time() - start_time
        update_model_training(
            conf.project_id, conf.experiment_name, -1, conf.num_epochs, time_spent, -1, "starting training"
        )

    _, best_train_loss = sequence_trainer.train_model(conf.num_epochs, dataset, True)
    model_save_path = conf.generate_model_path("trigger")

    if conf.is_lean_life:
        file_size = os.path.getsize(model_save_path)
        send_model_metadata(conf.project_id, conf.experiment_name, model_save_path, best_train_loss, file_size)

    return model_save_path


def evaluate_trigger_ner_pipeline(payload):
    conf = Config(payload)

    # load vocab
    label_length = conf.read_label_word_emd_data()

    reader = Reader(conf.digit2zero)
    eval_data = [{'text': tup[0], 'label': tup[1]} for tup in payload["eval_data"]]
    eval_data = reader.build_data(eval_data, "eval")
    conf.map_insts_ids(eval_data, "eval")

    # load trigger data
    path = conf.generate_training_data_path("soft_triggers")
    with open(path, 'rb') as f:
        triggers = pickle.load(f)

    encoder = SoftMatcher(conf, label_length)
    encoder.load_state_dict(
        torch.load(conf.generate_model_path("trigger_soft"))
    )

    inference = SoftSequence(conf, encoder)
    inference.load_state_dict(
        torch.load(conf.generate_model_path("trigger"))
    )
    sequence_trainer = SoftSequenceTrainer(inference, conf, None, None, triggers)

    encoder.eval()
    inference.eval()
    test_batches = batching_list_instances(conf, eval_data)
    test_metrics = sequence_trainer.evaluate_model(test_batches, "eval", eval_data, triggers)

    return test_metrics


def predict_trigger_ner_pipeline(payload):
    conf = Config(payload)

    # load vocab
    label_length = conf.read_label_word_emd_data()

    reader = Reader(conf.digit2zero)
    pred_data = [{'text': tup[0], 'label': tup[1]} for tup in payload["prediction_data"]]
    pred_data = reader.build_data(pred_data, "pred")
    conf.map_insts_ids(pred_data, "pred")

    # load trigger data
    path = conf.generate_training_data_path("soft_triggers")
    with open(path, 'rb') as f:
        triggers = pickle.load(f)

    encoder = SoftMatcher(conf, label_length)
    encoder.load_state_dict(
        torch.load(conf.generate_model_path("trigger_soft"))
    )

    inference = SoftSequence(conf, encoder)
    inference.load_state_dict(
        torch.load(conf.generate_model_path("trigger"))
    )
    sequence_trainer = SoftSequenceTrainer(inference, conf, None, None, triggers)

    encoder.eval()
    inference.eval()

    pred_batches = batching_list_instances(conf, pred_data)
    sequence_trainer.predict_model(pred_batches, pred_data, triggers)

    return list(map(lambda x: " ".join(x.prediction), pred_data))
