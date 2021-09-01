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
from trigger_ner.model.soft_inferencer_naive import SoftSequenceNaive, SoftSequenceNaiveTrainer

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

        # vocab
        conf.build_label_idx(train_data)
        conf.build_word_idx(train_data, dev_data)
        conf.build_emb_table()
        conf.dump_label_word_emb_data()

        conf.map_insts_ids(train_data, "labeled")
        if dev_data:
            conf.map_insts_ids(dev_data, "dev")

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

        # load vocab
        conf.read_label_word_emd_data()

    # dataset division
    # numbers = int(len(dataset) * conf.percentage / 100)
    # initial_trains = dataset[:numbers]
    initial_trains = train_data
    random.shuffle(initial_trains)

    encoder = SoftSequenceNaive(conf)
    trainer = SoftSequenceNaiveTrainer(encoder, conf, dev_data, None)

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
