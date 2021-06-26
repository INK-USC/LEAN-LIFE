import time
import random
import pickle
import csv
import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.model_selection import train_test_split

import pathlib
import sys

PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")
sys.path.append(PATH_TO_PARENT + "../../")

from fast_api.fast_api_util_functions import update_model_training, send_model_metadata
from defaults import TRAINING_DEFAULTS, BILSTM_DEFAULTS

from next_framework.training.next_constants import TACRED_ENTITY_TYPES, SPACY_NERS
from next_framework.training.next_util_functions import generate_save_string, build_custom_vocab, \
    set_re_dataset_ner_label_space, set_ner_label_space, build_vocab
from next_framework.training.next_training_util_functions import build_labeled_dataset, predictions_clf, \
    build_prediction_dataset
from next_framework.training.standard_training_util_functions import build_datasets_from_text, evaluate_clf
from next_framework.models import BiLSTM_Att_Clf


def _check_or_load_defaults(payload, default, key):
    if key in payload and payload[key] != None:
        return payload[key]
    else:
        return default[key]


def standard_pipeline(payload):
    start_time = time.time()
    # Step 1.
    build_data = payload["build_data"]
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    label_map = payload["label_map"]
    task = payload["task"]

    clip_gradient_size = BILSTM_DEFAULTS["clip_gradient_size"]
    n_layer_x_n_directions = BILSTM_DEFAULTS["layer_x_directions"]

    match_batch_size = int(_check_or_load_defaults(payload, BILSTM_DEFAULTS, "match_batch_size"))
    eval_batch_size = int(_check_or_load_defaults(payload, BILSTM_DEFAULTS, "eval_batch_size"))
    learning_rate = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "learning_rate")
    epochs = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "epochs")
    embeddings = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "embeddings")
    random_state = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "random_state")
    emb_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "emb_dim")
    hidden_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "hidden_dim")
    none_label_key = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "none_label_key")

    load_model = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "load_model")
    start_epoch = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "start_epoch")

    save_string = generate_save_string(dataset_name, embeddings)
    vocab_path = PATH_TO_PARENT + "../next_framework/data/vocabs/vocab_{}.p".format(save_string)

    number_of_classes = len(label_map)
    if none_label_key != None:
        none_label_id = label_map[none_label_key]
    else:
        none_label_id = -1

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    relation_ner_types = None
    if task == "re":
        ner_labels = payload["ner_labels"]
        set_re_dataset_ner_label_space(dataset_name, ner_labels)

        if "relation_ner_types" in payload and payload["relation_ner_types"] is not None:
            relation_ner_types = payload["relation_ner_types"]
        else:
            # FILL IN MORE DATASETS HERE
            # if using the api, you can just provide the mapping between relation to ner types (above)
            # however this is not necessary to send
            if dataset_name == "tacred":
                relation_ner_types = TACRED_ENTITY_TYPES
    else:
        temp = SPACY_NERS[:]
        temp.append("<PAD>")
        set_ner_label_space(temp)

    torch.manual_seed(random_state)
    random.seed(random_state)

    if build_data:
        if "dev_data" not in payload or payload["dev_data"] is None:
            labeled_data, dev_data = train_test_split(payload["labeled_data"], train_size=0.8,
                                                      random_state=random_state)
            payload["labeled_data"] = labeled_data
            payload["dev_data"] = dev_data
        vocab = build_vocab(list(map(lambda x: x['text'], payload["labeled_data"])), embeddings, save_string)
    else:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

    pad_idx = vocab["<pad>"]

    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    # you can either send us tokens
    # load_spacy_to_custom_dataset_ner_mapping <- remember to say how to update this
    custom_vocab = {}
    if "custom_vocab_tokens" in payload and payload["custom_vocab_tokens"] is not None:
        tokens = payload["custom_vocab_tokens"]
        custom_vocab = build_custom_vocab("", len(vocab), tokens)
    # or if this is an "re" task and you have sent ner_labels, we will create a custom
    # vocab for you of [SUBJ-ner_label, OBJ-ner_label] for each label in ner_labels
    # we do this due to parsing. Look at tokenize() in:
    # model_api/model_training/next_framework/training/util_functions.py
    else:
        if task == "re" and len(ner_labels) > 0:
            custom_vocab = build_custom_vocab("", len(vocab), ner_labels, "re")
        else:
            # finally, if the task is not "re", and you just want to set a custom vocab
            # we allow you to do so
            custom_vocab = build_custom_vocab(dataset_name, len(vocab))

    custom_vocab_length = len(custom_vocab)

    if payload["leanlife"]:
        update_model_training(payload["project_id"], experiment_name, -1, epochs, -1, -1, "starting training pipeline")

    # Step 2. Build data
    if build_data:
        build_datasets_from_text(payload["labeled_data"], vocab_path, payload["dev_data"], custom_vocab, save_string,
                                 label_map)

        if "eval_data" in payload and payload["eval_data"] is not None:
            eval_data = payload["eval_data"]
            build_labeled_dataset([tup[0] for tup in eval_data],
                                  [tup[1] for tup in eval_data],
                                  vocab, save_string, "test", label_map, custom_vocab)

        if payload["leanlife"]:
            time_spent = time.time() - start_time
            update_model_training(payload["project_id"], experiment_name, -1, epochs, time_spent, -1,
                                  "built training data")

    train_data_path = PATH_TO_PARENT + "../next_framework/data/training_data/labeled_data_{}.p".format(save_string)
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)

    dev_data_path = PATH_TO_PARENT + "../next_framework/data/training_data/dev_data_{}.p".format(save_string)

    if "eval_data" in payload and payload["eval_data"] is not None:
        eval_data_path = PATH_TO_PARENT + "../next_framework/data/training_data/test_data_{}.p".format(save_string)
    else:
        eval_data_path = ""

    # Step 3. Load model
    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, emb_dim, hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)

    loss_per_epoch = []
    dev_epoch_f1_scores = []
    best_dev_f1_score = -1
    if len(eval_data_path):
        test_epoch_f1_scores = []
        best_test_f1_score = -1
    best_loss = 1e30

    if load_model:
        clf.load_state_dict(
            torch.load(PATH_TO_PARENT + "../next_framework/data/saved_models/Clf_{}.p".format(experiment_name))
        )
        logging.info("loaded model")

        with open(PATH_TO_PARENT + "../next_framework/data/result_data/train_loss_per_epoch_Clf_{}.csv".format(
                experiment_name)) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                loss_per_epoch.append(tuple(map(float, row)))
                if float(row[0]) < best_loss:
                    best_loss = float(row[0])

        with open(PATH_TO_PARENT + "../next_framework/data/result_data/dev_f1_per_epoch_Clf_{}.csv".format(
                experiment_name)) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                dev_epoch_f1_scores.append(row)
                if float(row[-1]) > best_dev_f1_score:
                    best_dev_f1_score = float(row[-1])

        if len(eval_data_path):
            with open(PATH_TO_PARENT + "../next_framework/data/result_data/test_f1_per_epoch_Clf_{}.csv".format(
                    experiment_name)) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    test_epoch_f1_scores.append(row)
                    if float(row[-1]) > best_test_f1_score:
                        best_test_f1_score = float(row[-1])

        logging.info("loaded past results")

    clf.to(device)
    # define optimizer
    optimizer = SGD(clf.parameters(), lr=learning_rate)

    del vocab

    # define loss functions
    train_loss_function = nn.CrossEntropyLoss()

    # Step 4 Training
    h0 = torch.empty(n_layer_x_n_directions, match_batch_size, hidden_dim).to(device)
    c0 = torch.empty(n_layer_x_n_directions, match_batch_size, hidden_dim).to(device)
    nn.init.xavier_normal_(h0)
    nn.init.xavier_normal_(c0)

    for epoch in range(start_epoch, start_epoch + epochs):
        logging.info('\n Epoch {:} / {:}'.format(epoch + 1, start_epoch + epochs))

        train_total_loss, strict_total_loss, soft_total_loss = 0, 0, 0
        batch_count = 0
        clf.train()
        for step, batch in enumerate(tqdm(train_data.as_batches(batch_size=match_batch_size, seed=epoch))):
            train_tokens, train_lengths, train_labels = batch

            train_tokens = train_tokens.to(device)
            train_labels = train_labels.to(device)

            train_predictions = clf.forward(train_tokens, train_lengths, h0, c0)

            train_loss = train_loss_function(train_predictions, train_labels)
            train_total_loss += train_loss.item()
            batch_count += 1

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), clip_gradient_size)

            optimizer.step()

        # compute the training loss of the epoch
        train_avg_loss = train_total_loss / batch_count
        train_avg_strict_loss = strict_total_loss / batch_count
        train_avg_soft_loss = soft_total_loss / batch_count

        logging.info("Train Losses")
        loss_tuples = ("%.5f" % train_avg_loss, "%.5f" % train_avg_strict_loss, "%.5f" % train_avg_soft_loss)
        logging.info(
            "Avg Train Total Loss: {}, Avg Train Strict Loss: {}, Avg Train Soft Loss: {}".format(*loss_tuples)
        )

        loss_per_epoch.append((train_avg_loss, train_avg_strict_loss, train_avg_soft_loss))

        train_results = evaluate_clf(train_data_path, clf, train_loss_function, batch_size=eval_batch_size,
                                     none_label_id=none_label_id, hidden_dim=hidden_dim)
        avg_loss, avg_train_f1_score, total_train_class_probs = train_results

        logging.info("Train Results")
        train_tuple = ("%.5f" % avg_loss, "%.5f" % avg_train_f1_score)
        logging.info("Avg Train Loss: {}, Avg Train F1 Score: {}".format(*train_tuple))

        dev_results = evaluate_clf(dev_data_path, clf, train_loss_function, batch_size=eval_batch_size,
                                   none_label_id=none_label_id, hidden_dim=hidden_dim)

        avg_loss, avg_dev_f1_score, total_dev_class_probs = dev_results

        logging.info("Dev Results")
        dev_tuple = ("%.5f" % avg_loss, "%.5f" % avg_dev_f1_score)
        logging.info("Avg Dev Loss: {}, Avg Dev F1 Score: {}".format(*dev_tuple))

        dev_epoch_f1_scores.append((avg_loss, avg_dev_f1_score))

        if avg_dev_f1_score > best_dev_f1_score:
            best_dev_f1_score = avg_dev_f1_score
            logging.info("Updated Dev F1 Score")

        # Step 5 evaluate
        if len(eval_data_path):
            test_results = evaluate_clf(eval_data_path, clf, train_loss_function, batch_size=eval_batch_size,
                                        none_label_id=none_label_id, hidden_dim=hidden_dim)

            avg_loss, avg_test_f1_score, total_test_class_probs = test_results

            logging.info("Test Results")
            test_tuple = ("%.5f" % avg_loss, "%.5f" % avg_test_f1_score)
            logging.info("Avg Test Loss: {}, Avg Test F1 Score: {}".format(*test_tuple))

            test_epoch_f1_scores.append((avg_loss, avg_test_f1_score))

            if best_test_f1_score < avg_test_f1_score:
                logging.info("Saving Model")
                dir_name = PATH_TO_PARENT + "../next_framework/data/saved_models/"
                torch.save(clf.state_dict(), "{}Clf_{}.p".format(dir_name, experiment_name))
                with open(PATH_TO_PARENT + "../next_framework/data/result_data/test_predictions_Clf_{}.p".format(
                        experiment_name), "wb") as f:
                    pickle.dump(total_test_class_probs, f)
                with open(PATH_TO_PARENT + "../next_framework/data/result_data/dev_predictions_Clf_{}.p".format(
                        experiment_name), "wb") as f:
                    pickle.dump(total_dev_class_probs, f)

                best_test_f1_score = avg_test_f1_score

            logging.info("Best Test F1: {}".format("%.5f" % best_test_f1_score))
            logging.info(test_epoch_f1_scores[-3:])
        else:
            logging.info("Saving Model")
            dir_name = PATH_TO_PARENT + "../next_framework/data/saved_models/"
            torch.save(clf.state_dict(), "{}Clf_{}.p".format(dir_name, experiment_name))

        if payload["leanlife"]:
            time_spent = time.time() - start_time
            update_model_training(payload["project_id"], experiment_name, epoch + 1, epochs, time_spent, train_avg_loss,
                                  "training")

    # Step 6 save metadata
    with open(PATH_TO_PARENT + "../next_framework/data/result_data/train_loss_per_epoch_Clf_{}.csv".format(
            experiment_name), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['train_avg_loss', 'train_avg_strict_loss', 'train_avg_soft_loss'])
        for row in loss_per_epoch:
            writer.writerow(row)

    with open(PATH_TO_PARENT + "../next_framework/data/result_data/dev_f1_per_epoch_Clf_{}.csv".format(
            experiment_name), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['avg_loss', 'f1_score'])
        for row in dev_epoch_f1_scores:
            writer.writerow(row)

    if len(eval_data_path):
        with open(PATH_TO_PARENT + "../next_framework/data/result_data/test_f1_per_epoch_Clf_{}.csv".format(
                experiment_name), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['avg_loss', 'f1_score'])
            for row in test_epoch_f1_scores:
                writer.writerow(row)

    logging.info(experiment_name)
    save_path = PATH_TO_PARENT + "../next_framework/data/saved_models/Clf_{}.p".format(experiment_name)
    logging.info(save_path)

    if payload["leanlife"]:
        best_train_loss = min([row[0] for row in loss_per_epoch])
        file_size = os.path.getsize(save_path)
        send_model_metadata(payload["project_id"], experiment_name, save_path, best_train_loss, file_size)

    return save_path


def evaluate_standard_clf_pipeline(payload):
    # ready parameters
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    label_map = payload["label_map"]
    task = payload["task"]

    eval_batch_size = int(_check_or_load_defaults(payload, BILSTM_DEFAULTS, "eval_batch_size"))
    embeddings = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "embeddings")
    emb_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "emb_dim")
    hidden_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "hidden_dim")
    none_label_key = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "none_label_key")

    save_string = generate_save_string(dataset_name, embeddings)
    vocab_path = PATH_TO_PARENT + "../next_framework/data/vocabs/vocab_{}.p".format(save_string)

    number_of_classes = len(label_map)
    if none_label_key != None:
        none_label_id = label_map[none_label_key]
    else:
        none_label_id = -1

    if task == "re":
        ner_labels = payload["ner_labels"]
        set_re_dataset_ner_label_space(dataset_name, ner_labels)
    else:
        temp = SPACY_NERS[:]
        temp.append("<PAD>")
        set_ner_label_space(temp)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    pad_idx = vocab["<pad>"]

    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    # you can either send us tokens
    # load_spacy_to_custom_dataset_ner_mapping <- remember to say how to update this
    custom_vocab = {}
    if "custom_vocab_tokens" in payload and payload["custom_vocab_tokens"] is not None:
        tokens = payload["custom_vocab_tokens"]
        custom_vocab = build_custom_vocab("", len(vocab), tokens)
    # or if this is an "re" task and you have sent ner_labels, we will create a custom
    # vocab for you of [SUBJ-ner_label, OBJ-ner_label] for each label in ner_labels
    # we do this due to parsing. Look at tokenize() in:
    # model_api/model_training/next_framework/training/util_functions.py
    else:
        if task == "re" and len(ner_labels) > 0:
            custom_vocab = build_custom_vocab("", len(vocab), ner_labels, "re")
        else:
            # finally, if the task is not "re", and you just want to set a custom vocab
            # we allow you to do so
            custom_vocab = build_custom_vocab(dataset_name, len(vocab))

    custom_vocab_length = len(custom_vocab)

    # prepare eval data
    eval_data = payload["eval_data"]
    build_labeled_dataset([tup[0] for tup in eval_data],
                          [tup[1] for tup in eval_data],
                          vocab, save_string, "test", label_map, custom_vocab)
    eval_data_path = PATH_TO_PARENT + "../next_framework/data/training_data/test_data_{}.p".format(save_string)

    # load model
    train_loss_function = nn.CrossEntropyLoss()

    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, emb_dim, hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)
    clf.load_state_dict(
        torch.load(PATH_TO_PARENT + "../next_framework/data/saved_models/Clf_{}.p".format(experiment_name))
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    clf = clf.to(device)

    logging.info("loaded model")

    # evaluate
    test_results = evaluate_clf(eval_data_path, clf, train_loss_function, batch_size=eval_batch_size,
                                none_label_id=none_label_id, hidden_dim=hidden_dim)

    avg_loss, avg_test_f1_score, total_test_class_probs = test_results

    logging.info("Test Results")
    test_tuple = ("%.5f" % avg_loss, "%.5f" % avg_test_f1_score)
    logging.info("Avg Test Loss: {}, Avg Test F1 Score: {}".format(*test_tuple))

    return test_tuple[0], avg_test_f1_score


def predict_standard_clf_pipeline(payload):
    # ready parameters
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    label_map = payload["label_map"]
    task = payload["task"]
    prediction_data = payload["prediction_data"]

    pred_batch_size = int(_check_or_load_defaults(payload, BILSTM_DEFAULTS, "eval_batch_size"))
    embeddings = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "embeddings")
    emb_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "emb_dim")
    hidden_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "hidden_dim")

    save_string = generate_save_string(dataset_name, embeddings)
    vocab_path = PATH_TO_PARENT + "../next_framework/data/vocabs/vocab_{}.p".format(save_string)

    number_of_classes = len(label_map)

    if task == "re":
        ner_labels = payload["ner_labels"]
        set_re_dataset_ner_label_space(dataset_name, ner_labels)
    else:
        temp = SPACY_NERS[:]
        temp.append("<PAD>")
        set_ner_label_space(temp)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    pad_idx = vocab["<pad>"]

    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    # you can either send us tokens
    # load_spacy_to_custom_dataset_ner_mapping <- remember to say how to update this
    custom_vocab = {}
    if "custom_vocab_tokens" in payload and payload["custom_vocab_tokens"] is not None:
        tokens = payload["custom_vocab_tokens"]
        custom_vocab = build_custom_vocab("", len(vocab), tokens)
    # or if this is an "re" task and you have sent ner_labels, we will create a custom
    # vocab for you of [SUBJ-ner_label, OBJ-ner_label] for each label in ner_labels
    # we do this due to parsing. Look at tokenize() in:
    # model_api/model_training/next_framework/training/util_functions.py
    else:
        if task == "re" and len(ner_labels) > 0:
            custom_vocab = build_custom_vocab("", len(vocab), ner_labels, "re")
        else:
            # finally, if the task is not "re", and you just want to set a custom vocab
            # we allow you to do so
            custom_vocab = build_custom_vocab(dataset_name, len(vocab))

    custom_vocab_length = len(custom_vocab)

    # prepare eval data
    build_prediction_dataset(prediction_data, vocab, save_string, custom_vocab)
    logging.info("built prediction dataset")

    pred_path = PATH_TO_PARENT + "../next_framework/data/training_data/prediction_data_{}.p".format(save_string)

    # load model
    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, emb_dim, hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)
    clf.load_state_dict(
        torch.load(PATH_TO_PARENT + "../next_framework/data/saved_models/Clf_{}.p".format(experiment_name))
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    clf = clf.to(device)

    logging.info("loaded model")

    # evaluate
    total_class_probs, class_predictions = predictions_clf(pred_path, clf, label_map, batch_size=pred_batch_size)

    return total_class_probs.tolist(), class_predictions.tolist()
