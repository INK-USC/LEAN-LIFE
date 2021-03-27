"""
    Actual Pipelines that are called by internal_api's `internal_main.py` functions
"""
import argparse
import csv
import logging
import os
import pathlib
import pickle
import random
import sys
import time
import dill
from tqdm import tqdm
import numpy as np
from transformers import AdamW
import torch
import torch.nn as nn
from torch.optim import SGD
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")
sys.path.append(PATH_TO_PARENT + "../../")
from fast_api.fast_api_util_functions import update_model_training, send_model_metadata
from internal_api.defaults import FIND_MODULE_DEFAULTS, TRAINING_DEFAULTS, BILSTM_DEFAULTS
from next_framework.training.find_training_util_functions import build_pre_train_find_datasets, \
                                                                 evaluate_find_module
from next_framework.training.next_training_util_functions import batch_type_restrict_re, build_phrase_input, build_mask_mat_for_batch,\
                                                                 build_datasets_from_text, evaluate_next_clf, apply_strict_matching
from next_framework.training.next_util_functions import similarity_loss_function, generate_save_string, build_custom_vocab,\
                                                        set_re_dataset_ner_label_space
from next_framework.training.next_util_classes import BaseVariableLengthDataset
from next_framework.training.next_constants import TACRED_ENTITY_TYPES
from next_framework.models import Find_Module, BiLSTM_Att_Clf



def _check_or_load_defaults(payload, default, key):
    if key in payload and payload[key] != None:
        return payload[key]
    else:
        return default[key]

def pre_train_find_module_pipeline(payload):
    """
        Pipeline that takes in a payload of parameters and optionally data, and will preform the following steps:
            1. Read parameters sent in and fill with default values where appropriate
            2. Builds datasets that will be needed for training. The datasets will be cached, future experiments
               running off the same data will not need data to be passed in.
            3. Loads cached data and sets up model, loading a previously checkpointed model if needed
            4. Trains a Find_Module model instance
            5. Evaluates the model and will halt early if an F1 score of 90 percent is reached. Will also checkpoint
               a model each time it improves over its current best f1 score.
            6. Save performance data across epochs to a csv data.
    """
    # Step 1.
    start_time = time.time()

    pos_weight = FIND_MODULE_DEFAULTS["pos_weight"]
    clip_gradient_size = FIND_MODULE_DEFAULTS["clip_gradients"]
    lower_bound = FIND_MODULE_DEFAULTS["lower_bound"]

    build_data = payload["pre_train_build_data"]
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    dataset_size = payload["dataset_size"]
    task = payload["task"]

    train_batch_size = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_batch_size")
    eval_batch_size = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_eval_batch_size")
    learning_rate = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_learning_rate")
    epochs = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_epochs")
    embeddings = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "embeddings")
    gamma = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_gamma")
    emb_dim = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_emb_dim")
    hidden_dim = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_hidden_dim")
    training_size = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_training_size")
    random_state = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_random_state")

    load_model = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "pre_train_load_model")
    start_epoch = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "pre_train_start_epoch")

    sample_rate = training_size / dataset_size

    if sample_rate > 1:
        sample_rate = -1.0 # no sampling
    
    save_string = generate_save_string(dataset_name, embeddings, sample=sample_rate)

    torch.manual_seed(random_state)
    random.seed(random_state)

    if payload["leanlife"]:
        update_model_training(experiment_name, -1, epochs, -1, -1, "starting pre_training pipeline")

    # Step 2.
    if build_data:
        text_data = payload["training_data"]
        explanation_data = payload["explanation_data"]
        build_pre_train_find_datasets(text_data, explanation_data, save_string, embeddings, random_state, dataset_name, sample_rate)
        if payload["leanlife"]:
            time_spent = time.time() - start_time
            update_model_training(experiment_name, -1, epochs, time_spent, -1, "built pre_training data")
    
    # Step 3.
    with open(PATH_TO_PARENT + "../next_framework/data/pre_train_data/train_data_{}.p".format(save_string), "rb") as f:
        train_dataset = pickle.load(f)
    
    primary_eval_path = PATH_TO_PARENT + "../next_framework/data/pre_train_data/rq_data_{}.p".format(save_string)
    
    # optional secondary eval, can set this to the empty string
    secondary_eval_path = PATH_TO_PARENT + "../next_framework/data/pre_train_data/dev_data_{}.p".format(save_string)
    
    with open(PATH_TO_PARENT + "../next_framework/data/vocabs/vocab_{}.p".format(save_string), "rb") as f:
        vocab = pickle.load(f)
    
    with open(PATH_TO_PARENT + "../next_framework/data/pre_train_data/sim_data_{}.p".format(save_string), "rb") as f:
        sim_data = pickle.load(f)
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    # you can either send us tokens
    if "custom_vocab_tokens" in payload:
        tokens = payload["custom_vocab_tokens"]
        custom_vocab = build_custom_vocab("", len(vocab), tokens)
    # or if this is an "re" task and you have sent ner_labels, we will create a custom
    # vocab for you of [SUBJ-ner_label, OBJ-ner_label] for each label in ner_labels
    # we do this due to parsing. Look at tokenize() in:
    # model_api/model_training/next_framework/training/util_functions.py
    else:
        if task == "re" and len(payload["ner_labels"]) == 0:
            ner_labels = payload["ner_labels"]
            custom_vocab = build_custom_vocab("", len(vocab), ner_labels, "re")
        else:
            # finally, if the task is not "re", and you just want to set a custom vocab
            # we allow you to do so
            custom_vocab = build_custom_vocab(dataset_name, len(vocab))

    custom_vocab_length = len(custom_vocab)

    model = Find_Module.Find_Module(emb_weight=vocab.vectors, padding_idx=pad_idx, emb_dim=emb_dim,
                                    hidden_dim=hidden_dim, cuda=torch.cuda.is_available(),
                                    custom_token_count=custom_vocab_length)
    del vocab

    # prepping variables for storing training progress
    epochs = epochs
    epoch_string = str(epochs)
    epoch_losses = []
    dev_2_epoch_losses = []
    best_f1_score = -1
    best_dev_2_f1_score = -1
    best_dev_loss = float('inf') 
    
    if load_model:
        model.load_state_dict(torch.load(PATH_TO_PARENT+"../next_framework/data/saved_models/Find-Module-pt_{}.p".format(experiment_name)))
        logging.info("loaded model")

        with open(PATH_TO_PARENT+"../next_framework/data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                epoch_losses.append(row)
                if float(row[-1]) > best_f1_score:
                    best_f1_score = float(row[-1])
                if float(row[3]) < best_dev_loss:
                    best_dev_loss = float(row[3])
        
        with open(PATH_TO_PARENT+"../next_framework/data/result_data/dev_2_loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                dev_2_epoch_losses.append(row)
                if float(row[-1]) > best_dev_2_f1_score:
                    best_dev_2_f1_score = float(row[-1])
        
        logging.info("loaded past results")
    
    model = model.to(device)

    # Get L_sim Data ready
    real_query_tokens, _ = BaseVariableLengthDataset.variable_length_batch_as_tensors(sim_data["queries"], pad_idx)
    real_query_tokens = real_query_tokens.to(device)
    query_labels = sim_data["labels"]
    
    queries_by_label = {}
    for i, label in enumerate(query_labels):
        if label in queries_by_label:
            queries_by_label[label][i] = 1
        else:
            queries_by_label[label] = [0] * len(query_labels)
            queries_by_label[label][i] = 1
    
    query_index_matrix = []
    for i, label in enumerate(query_labels):
        query_index_matrix.append(queries_by_label[label][:])
    
    query_index_matrix = torch.tensor(query_index_matrix)
    neg_query_index_matrix = 1 - query_index_matrix
    for i, row in enumerate(neg_query_index_matrix):
        neg_query_index_matrix[i][i] = 1

    query_index_matrix = query_index_matrix.to(device)
    neg_query_index_matrix = neg_query_index_matrix.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
         
    # define loss functions
    find_loss_function  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    sim_loss_function = similarity_loss_function
    
    if payload["leanlife"]:
        time_spent = time.time() - start_time
        update_model_training(experiment_name, -1, epochs, time_spent, -1, "starting pre-training")

    # Step 4.
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch+epochs):
        logging.info('\n Epoch {:} / {:}'.format(epoch + 1, start_epoch+epochs))

        total_loss, find_total_loss, sim_total_loss = 0, 0, 0
        batch_count = 0
        model.train()
        # iterate over batches
        for step, batch in enumerate(tqdm(train_dataset.as_batches(batch_size=train_batch_size, seed=epoch))):
            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            tokens, queries, labels = batch

            # clear previously calculated gradients 
            model.zero_grad()        

            # get model predictions for the current batch
            token_scores = model.find_forward(tokens, queries, lower_bound)
            pos_scores, neg_scores = model.sim_forward(real_query_tokens, query_index_matrix, neg_query_index_matrix)
            
            # compute the loss between actual and predicted values
            find_loss = find_loss_function(token_scores, labels)
            sim_loss = sim_loss_function(pos_scores, neg_scores)
            string_loss = find_loss + gamma * sim_loss

            # add on to the total loss
            find_total_loss = find_total_loss  + find_loss.item()
            sim_total_loss = sim_total_loss + sim_loss.item()
            total_loss = total_loss + string_loss.item()
            batch_count += 1

            # backward pass to calculate the gradients
            string_loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_size)

            # update parameters
            optimizer.step()

        # compute the training loss of the epoch
        train_avg_loss = total_loss / batch_count
        train_avg_find_loss = find_total_loss / batch_count
        train_avg_sim_loss = sim_total_loss / batch_count

        # Step 5.
        logging.info("Starting Primary Evaluation")
        eval_results = evaluate_find_module(primary_eval_path, real_query_tokens, query_index_matrix, neg_query_index_matrix, lower_bound,
                                            model, find_loss_function, sim_loss_function, eval_batch_size, gamma)
        dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score, total_og_scores, total_new_scores = eval_results
        logging.info("Finished Primary Evaluation")
        
        if dev_f1_score > best_f1_score or (dev_f1_score == best_f1_score and dev_avg_loss < best_dev_loss):
            logging.info("Saving Model")
            dir_name = PATH_TO_PARENT + "../next_framework/data/saved_models/"
            torch.save(model.state_dict(), "{}Find-Module-pt_{}.p".format(dir_name, experiment_name))
            with open(PATH_TO_PARENT + "../next_framework/data/result_data/best_dev_total_og_scores_{}.p".format(experiment_name), "wb") as f:
                pickle.dump(total_og_scores, f)
            with open(PATH_TO_PARENT + "../next_framework/data/result_data/best_dev_total_new_scores_{}.p".format(experiment_name), "wb") as f:
                pickle.dump(total_new_scores, f)
            best_f1_score = dev_f1_score
            best_dev_loss = dev_avg_loss

        epoch_losses.append((train_avg_loss, train_avg_find_loss, train_avg_sim_loss,
                             dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score))
        logging.info("Best Primary F1: {}".format(str(best_f1_score)))
        logging.info(epoch_losses[-3:])
        
        if len(secondary_eval_path) > 0:
            logging.info("Starting Secondary Evaluation")
            eval_results = evaluate_find_module(secondary_eval_path, real_query_tokens, query_index_matrix, neg_query_index_matrix, lower_bound,
                                                model, find_loss_function, sim_loss_function, eval_batch_size, gamma)
            dev_2_avg_loss, dev_2_avg_find_loss, dev_2_avg_sim_loss, dev_2_f1_score, total_og_scores, total_new_scores = eval_results
            logging.info("Finished Secondary Evaluation")

            if dev_2_f1_score > best_dev_2_f1_score:
                best_dev_2_f1_score = dev_2_f1_score
                with open(PATH_TO_PARENT + "../next_framework/data/result_data/best_dev_2_total_og_scores_{}.p".format(experiment_name), "wb") as f:
                    pickle.dump(total_og_scores, f)
                with open(PATH_TO_PARENT + "../next_framework/data/result_data/best_dev_2_total_new_scores_{}.p".format(experiment_name), "wb") as f:
                    pickle.dump(total_new_scores, f)
            
            dev_2_epoch_losses.append((dev_2_avg_loss, dev_2_avg_find_loss, dev_2_avg_sim_loss,
                                       dev_2_f1_score))
            logging.info("Best Secondary F1: {}".format(str(best_dev_2_f1_score)))
            logging.info(dev_2_epoch_losses[-3:])
        
        if payload["leanlife"]:
            time_spent = time.time() - start_time
            update_model_training(experiment_name, epoch+1, epochs, time_spent, train_avg_loss, "pre-training")

        if best_f1_score > 0.9:
            break
    
    # Step 6.
    with open(PATH_TO_PARENT + "../next_framework/data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_loss','train_find_loss', 'train_sim_loss', 'dev_loss', 'dev_find_loss', 'dev_sim_loss', 'dev_f1_score'])
        for row in epoch_losses:
            writer.writerow(row)
    
    if len(secondary_eval_path) > 0:
        with open(PATH_TO_PARENT + "../next_framework/data/result_data/dev_2_loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name), "w") as f:
            writer=csv.writer(f)
            writer.writerow(["dev_2_avg_loss", "dev_2_avg_find_loss", "dev_2_avg_sim_loss", "dev_2_f1_score"])
            for row in dev_2_epoch_losses:
                writer.writerow(row)
    
    save_path = "../model_training/next_framework/data/saved_models/Find-Module-pt_{}".format(experiment_name)

    # Currently not possible
    if payload["leanlife"] and payload["find"]:
        best_train_loss = min([row[0] for row in epoch_losses])
        file_size = os.path.getsize(save_path)
        send_model_metadata(experiment_name, save_path, best_train_loss, file_size)
    
    return save_path
    
def train_next_bilstm_pipeline(payload):
    """
        Pipeline that takes in a payload of parameters and optionally data, and will preform the following steps:
            0. If needed will kick off a pre_training job for the Find_Module needed for the NExT Framework Training Algo
            1. Read parameters sent in and fill with default values where appropriate
            2. Builds datasets that will be needed for training. The datasets will be cached, future experiments
               running off the same data will not need data to be passed in.
            3. Loads cached data and sets up model, loading a previously checkpointed model if needed
            4. If needed will compute soft-match scores
            5. Trains a BiLSTM+Att classifier instance using the NExT Framework's Training Algorithm
            6. Optionally evaluates a model if a evaluation data is sent and checkpoints when prior f1 
               performance. Otherwise just saves after every epoch.
            7. Save performance data across epochs to a csv data.
    """
    # Step 0.
    if payload["stage"] == "both":
        _ = pre_train_find_module_pipeline(payload)

    start_time = time.time()
    
    # Step 1.
    build_data = payload["build_data"]
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    dataset_size = payload["dataset_size"]
    label_map = payload["label_map"]
    task = payload["task"]
    
    clip_gradient_size = BILSTM_DEFAULTS["clip_gradients"]
    lower_bound = FIND_MODULE_DEFAULTS["lower_bound"]
    n_layer_x_n_directions = BILSTM_DEFAULTS["layer_x_directions"]

    match_batch_size = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "match_batch_size")
    unlabeled_batch_size = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "unlabeled_batch_size")
    eval_batch_size = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "eval_batch_size")
    learning_rate = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "learning_rate")
    epochs = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "epochs")
    embeddings = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "embeddings")
    gamma = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "gamma")
    emb_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "emb_dim")
    hidden_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "hidden_dim")
    random_state = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "random_state")
    none_label_key = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "none_label_key")

    full_batch_size = match_batch_size + unlabeled_batch_size

    load_model = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "load_model")
    start_epoch = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "start_epoch")

    pre_training_size = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_training_size")
    find_module_hidden_dim = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_hidden_dim")
    
    find_module_path =  PATH_TO_PARENT + "../next_framework/data/saved_models/Find-Module-pt_{}.p".format(experiment_name)
    find_module_sample_rate = pre_training_size / dataset_size
    if find_module_sample_rate > 1:
        find_module_sample_rate = -1.0 # no sampling
    
    save_string = generate_save_string(dataset_name, embeddings, find_module_sample_rate)
    vocab_path = PATH_TO_PARENT + "../next_framework/data/vocabs/vocab_{}.p".format(save_string)

    number_of_classes = len(label_map)
    if none_label_key != None:
        none_label_id = label_map[none_label_key]
    else:
        none_label_id = -1

    relation_ner_types = None

    if task == "re":
        ner_labels = payload["ner_labels"]
        set_re_dataset_ner_label_space(dataset, ner_labels)

        if "relation_ner_types" in payload:
            relation_ner_types = payload["relation_ner_types"]
        else:
            # FILL IN MORE DATASETS HERE
            # if using the api, you can just provide the mapping between relation to ner types (above)
            # however this is not necessary to send
            if dataset_name == "tacred":
                relation_ner_types = TACRED_ENTITY_TYPES

    torch.manual_seed(random_state)
    random.seed(random_state)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    # you can either send us tokens
    # load_spacy_to_custom_dataset_ner_mapping <- remember to say how to update this
    custom_vocab= {}
    if "custom_vocab_tokens" in payload:
        tokens = payload["custom_vocab_tokens"]
        custom_vocab = build_custom_vocab("", len(vocab), tokens)
    # or if this is an "re" task and you have sent ner_labels, we will create a custom
    # vocab for you of [SUBJ-ner_label, OBJ-ner_label] for each label in ner_labels
    # we do this due to parsing. Look at tokenize() in:
    # model_api/model_training/next_framework/training/util_functions.py
    else:
        if len(ner_labels) > 0 and task == "re":
            custom_vocab = build_custom_vocab("", len(vocab), ner_labels, "re")
        else:
            # finally, if the task is not "re", and you just want to set a custom vocab
            # we allow you to do so
            custom_vocab = build_custom_vocab(dataset_name, len(vocab))
    
    custom_vocab_length = len(custom_vocab)

    if payload["leanlife"]:
        update_model_training(experiment_name, -1, epochs, -1, -1, "starting training pipeline")

    # Step 2.
    if build_data:
        text_data = payload["training_data"]
        explanation_data = payload["explanation_data"]
        build_datasets_from_text(text_data, vocab_path, explanation_data, custom_vocab, save_string, label_map, task=task, dataset=dataset_name)

        if "eval_data" in payload:
            eval_data = payload["eval_data"]
            build_labeled_dataset([tup[0] for tup in eval_data], 
                                  [tup[1] for tup in eval_data],
                                  vocab, save_string, "dev", label_map, custom_vocab)
        
        if payload["leanlife"]:
            time_spent = time.time() - start_time
            update_model_training(experiment_name, -1, epochs, time_spent, -1, "built training data")
    
    # Step 3.
    with open(PATH_TO_PARENT + "../next_framework/data/training_data/unlabeled_data_{}.p".format(save_string), "rb") as f:
        unlabeled_data = pickle.load(f)
    
    with open(PATH_TO_PARENT + "../next_framework/data/training_data/matched_data_{}.p".format(save_string), "rb") as f:
        strict_match_data = pickle.load(f)
    
    with open(PATH_TO_PARENT + "../next_framework/data/training_data/labeling_functions_{}.p".format(save_string), "rb") as f:
        soft_labeling_functions_dict = dill.load(f)

    with open(PATH_TO_PARENT + "../next_framework/data/training_data/query_tokens_{}.p".format(save_string), "rb") as f:
        tokenized_queries = pickle.load(f)
    
    with open(PATH_TO_PARENT + "../next_framework/data/training_data/word2idx_{}.p".format(save_string), "rb") as f:
        quoted_words_to_index = pickle.load(f)
    
    if "eval_data" in payload:
        eval_path = PATH_TO_PARENT + "../next_framework/data/training_data/dev_data_{}.p".format(save_string)
    else:
        eval_path = ""
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    epochs = epochs
    epoch_string = str(epochs)

    loss_per_epoch = []
    if len(eval_path):
        dev_epoch_f1_scores = []
        best_dev_f1_score = -1
    best_loss = 1e30

    if load_model:
        clf.load_state_dict(torch.load(PATH_TO_PARENT + "../next_framework/data/saved_models/Next-Clf_{}.p".format(experiment_name)))
        logging.info("loaded model")
        
        with open(PATH_TO_PARENT + "../next_framework/data/result_data/train_loss_per_epoch_Next-Clf_{}.csv".format(experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                loss_per_epoch.append(row)
                if float(row[0]) < best_loss:
                    best_loss = float(row[0])
        
        if len(eval_path):
            with open(PATH_TO_PARENT + "../next_framework/data/result_data/dev_f1_per_epoch_Next-Clf_{}.csv".format(experiment_name)) as f:
                reader=csv.reader(f)
                next(reader)
                for row in reader:
                    dev_epoch_f1_scores.append(row)
                    if float(row[-1]) > best_dev_f1_score:
                        best_dev_f1_score = float(row[-1])
        
        logging.info("loaded past results")

    # Get queries ready for Find Module
    lfind_query_tokens, _ = BaseVariableLengthDataset.variable_length_batch_as_tensors(tokenized_queries, pad_idx)
    lfind_query_tokens = lfind_query_tokens.to(device).detach()

    # Preping Soft Labeling Function Data
    soft_labeling_functions = soft_labeling_functions_dict["function_pairs"]
    soft_labeling_function_labels = soft_labeling_functions_dict["labels"]
    
    h0 = torch.empty(n_layer_x_n_directions, full_batch_size, hidden_dim).to(device)
    c0 = torch.empty(n_layer_x_n_directions, full_batch_size, hidden_dim).to(device)
    nn.init.xavier_normal_(h0)
    nn.init.xavier_normal_(c0)
    
    # Step 4.
    if args.build_data:
        find_module = Find_Module.Find_Module(vocab.vectors, pad_idx, emb_dim, find_module_hidden_dim,
                                              torch.cuda.is_available(), custom_token_count=custom_vocab_length)
        
        find_module.load_state_dict(torch.load(args.find_module_path))
        find_module = find_module.to(device)
        find_module.eval()
        function_scores = {}
        for i, batch in enumerate(tqdm(unlabeled_data.as_batches(batch_size=full_batch_size, shuffle=False))):
            unlabeled_tokens, unlabeled_token_lengths, phrases, _ = batch
            unlabeled_tokens = unlabeled_tokens.to(device)
            phrase_input = build_phrase_input(phrases, pad_idx, task).to(device).detach()
            b_size, seq_length = unlabeled_tokens.shape
            mask_mat = build_mask_mat_for_batch(seq_length).to(device).detach()
            with torch.no_grad():
                lfind_output = find_module.soft_matching_forward(unlabeled_tokens.detach(), lfind_query_tokens, lower_bound).detach() # B x seq_len x Q

                for j, pair in enumerate(soft_labeling_functions):
                    func, rel = pair
                    batch_scores = func(lfind_output, quoted_words_to_index, mask_mat)(phrase_input).detach() # 1 x B

                    type_restrict_multiplier = batch_type_restrict_re(rel, phrase_input, relation_ner_types).detach() # 1 x B
                    final_scores = batch_scores * type_restrict_multiplier # 1 x B
                    final_scores = final_scores.cpu()
                    if j in function_scores:
                        function_scores[j] = torch.cat([function_scores[j], final_scores], dim=1)
                    else:
                        function_scores[j] = final_scores
        
        soft_scores = torch.cat([function_scores[key] for key in function_scores]).permute(1, 0) # len(data) x number_of_explanations

        with open("../data/training_data/soft_scores_{}.p".format(experiment_name), "wb") as f:
            pickle.dump(soft_scores, f)
    
    else:
        with open("../data/training_data/soft_scores_{}.p".format(experiment_name), "rb") as f:
            soft_scores = pickle.load(f)
    
    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, emb_dim, hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)    
    clf = clf.to(device)
    optimizer = SGD(clf.parameters(), lr=learning_rate)

    del vocab

    # define loss functions
    strict_match_loss_function  = nn.CrossEntropyLoss()
    soft_match_loss_function = nn.CrossEntropyLoss(reduction='none')

    if payload["leanlife"]:
        time_spent = time.time() - start_time
        update_model_training(experiment_name, -1, epochs, time_spent, -1, "starting training")
    
    start_time = time.time()
    
    # Step 5. TRAINING
    for epoch in range(start_epoch, start_epoch+epochs):
        logging.info('\n Epoch {:} / {:}'.format(epoch + 1, start_epoch+epochs))

        total_loss, strict_total_loss, soft_total_loss = 0, 0, 0
        batch_count = 0
        clf.train()

        for step, batch_pair in enumerate(tqdm(zip(strict_match_data.as_batches(batch_size=full_batch_size, seed=epoch),\
                                                   unlabeled_data.as_batches(batch_size=full_batch_size, seed=epoch*seed)))):
            
            # prepping batch data
            strict_match_data_batch, unlabeled_data_batch = batch_pair

            strict_match_tokens, strict_match_lengths, strict_match_labels = strict_match_data_batch

            strict_match_tokens = strict_match_tokens.to(device)
            strict_match_labels = strict_match_labels.to(device)

            unlabeled_tokens, unlabeled_token_lengths, phrases, batch_indices = unlabeled_data_batch
            tensor_indices = torch.tensor(batch_indices)
            batch_soft_scores = torch.index_select(soft_scores, 0, tensor_indices).to(device)
            unlabeled_tokens = unlabeled_tokens.to(device)

            pseudo_labels = torch.index_select(soft_labeling_function_labels, 0, torch.argmax(batch_soft_scores, dim=1))
            bound = torch.max(batch_soft_scores, dim=1).values
            unlabeled_label_weights = nn.functional.softmax(10 * bound, dim=0)

            # clear previously calculated gradients 
            clf.zero_grad()
            
            strict_match_predictions = clf.forward(strict_match_tokens, strict_match_lengths, h0, c0)
            soft_match_predictions = clf.forward(unlabeled_tokens, unlabeled_token_lengths, h0, c0)

            strict_match_loss = strict_match_loss_function(strict_match_predictions, strict_match_labels)
            soft_match_loss = torch.sum(soft_match_loss_function(soft_match_predictions, pseudo_labels) * weight)
            combined_loss = strict_match_loss + gamma * soft_match_loss

            strict_total_loss = strict_total_loss + strict_match_loss.item()
            soft_total_loss = soft_total_loss + soft_match_loss.item()
            total_loss = total_loss + combined_loss.item()
            batch_count += 1
            
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), clip_gradient_size)

            optimizer.step()

        # compute the training loss of the epoch
        train_avg_loss = total_loss / batch_count
        train_avg_strict_loss = strict_total_loss / batch_count
        train_avg_soft_loss = soft_total_loss / batch_count

        logging.info("Train Losses")
        loss_tuples = ("%.5f" % train_avg_loss, "%.5f" % train_avg_strict_loss, "%.5f" % train_avg_soft_loss)
        logging.info("Avg Train Total Loss: {}, Avg Train Strict Loss: {}, Avg Train Soft Loss: {}".format(*loss_tuples))
        
        loss_per_epoch.append((train_avg_loss, train_avg_strict_loss, train_avg_soft_loss))
        
        # Step 6.
        if len(eval_path):
            dev_results = evaluate_next_clf(eval_path, clf, strict_match_loss_function, number_of_classes, batch_size=eval_batch_size, none_label_id=none_label_id)
        
            avg_loss, avg_dev_ent_f1_score, avg_dev_val_f1_score, total_dev_class_probs, no_relation_thresholds = dev_results

            logging.info("Eval Results")
            dev_tuple = ("%.5f" % avg_loss, "%.5f" % avg_dev_ent_f1_score, "%.5f" % avg_dev_val_f1_score, str(no_relation_thresholds))
            logging.info("Avg Eval Loss: {}, Avg Eval Entropy F1 Score: {}, Avg Eval Max Value F1 Score: {}, Thresholds: {}".format(*dev_tuple))

            dev_epoch_f1_scores.append((avg_loss, avg_dev_ent_f1_score, avg_dev_val_f1_score, max(avg_dev_ent_f1_score, avg_dev_val_f1_score)))

            if best_dev_f1_score < max(avg_dev_ent_f1_score, avg_dev_val_f1_score):
                logging.info("Saving Model")
                dir_name = PATH_TO_PARENT + "../next_framework/data/saved_models/"
                torch.save(clf.state_dict(), "{}Next-Clf_{}.p".format(dir_name, experiment_name))
                with open(PATH_TO_PARENT + "../next_framework/data/result_data/eval_predictions_Next-Clf_{}.csv".format(experiment_name), "wb") as f:
                    pickle.dump(total_dev_class_probs, f)
                if none_label_id > 0:
                    with open(PATH_TO_PARENT + "../next_framework/data/result_data/thresholds.p", "wb") as f:
                        pickle.dump({"thresholds" : no_relation_thresholds}, f)
                
                best_dev_f1_score = max(avg_dev_ent_f1_score, avg_dev_val_f1_score)
            
            logging.info("Best Test F1: {}".format("%.5f" % best_dev_f1_score))
            logging.info(dev_epoch_f1_scores[-3:])
        
        else:
            logging.info("Saving Model")
            dir_name = PATH_TO_PARENT + "../next_framework/data/saved_models/"
            torch.save(clf.state_dict(), "{}Next-Clf_{}.p".format(dir_name, experiment_name))
            if none_label_id > 0:
                with open(PATH_TO_PARENT + "../next_framework/data/result_data/thresholds.p", "wb") as f:
                    pickle.dump({"thresholds" : no_relation_thresholds}, f)
        
        if payload["leanlife"]:
            time_spent = time.time() - start_time
            update_model_training(experiment_name, epoch+1, epochs, time_spent, train_avg_loss, "training")
    
    # Step 7.
    with open(PATH_TO_PARENT + "../next_framework/data/result_data/train_loss_per_epoch_Next-Clf_{}.csv".format(experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_avg_loss', 'train_avg_strict_loss', 'train_avg_soft_loss'])
        for row in loss_per_epoch:
            writer.writerow([row])
    
    if len(eval_path):
        with open(PATH_TO_PARENT + "../next_framework/data/result_data/eval_f1_per_epoch_Next-Clf_{}.csv".format(experiment_name), "w") as f:
            writer=csv.writer(f)
            writer.writerow(['avg_loss, entropy_f1_score','max_value_f1_score', 'max'])
            for row in dev_epoch_f1_scores:
                writer.writerow(row)
    
    save_path = "../model_training/next_framework/data/saved_models/Next-Clf_{}".format(experiment_name)

    if payload["leanlife"]:
        best_train_loss = min([row[0] for row in loss_per_epoch])
        file_size = os.path.getsize(save_path)
        send_model_metadata(experiment_name, save_path, best_train_loss, file_size)
    
    return save_path

def strict_match_pipeline(payload):
    """
        Takes in explanation data, and a source of unlabeled data and will annotate the unlabeled data
        with the explanations provided. Converts explanations into binary labeling functions and if an
        explanation applies to a datapoint the label associated with the explanation is now associated
        with the datapoint.

        Returns:
            arr, arr : first array is of tuples (text, label),
                       second array is of tuples (index_of_text in unlabeled data, index_of_explanation in explanatation data)
    """
    text_data = payload["unlabeled_text"]
    explanation_data = payload["explanation_triples"]
    task = payload["task"]
    matched_tuples, matched_indices = apply_strict_matching(text_data, explanation_data, task)
    
    return matched_tuples, matched_indices

def evaluate_next_clf_pipeline(payload):
    """
        Given a tuples of text and labels, and an experiment name, we will load the appropriate
        saved model and evaluate it against the data provided.
    """
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    train_dataset_size = payload["train_dataset_size"]
    task = payload["task"]
    label_map = payload["label_map"]
    eval_data = payload["eval_data"]

    embeddings = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "embeddings")
    emb_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "emb_dim")
    hidden_dim = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "hidden_dim")
    none_label_key = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "none_label_key")
    pre_training_size = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_training_size")
    eval_batch_size = _check_or_load_defaults(payload, BILSTM_DEFAULTS, "eval_batch_size")

    number_of_classes = len(label_map)
    if none_label_key != None:
        none_label_id = label_map[none_label_key]
    else:
        none_label_id = -1

    find_module_sample_rate = pre_training_size / train_dataset_size

    if find_module_sample_rate > 1:
        find_module_sample_rate = -1.0 # no sampling
    
    save_string = generate_save_string(dataset_name, embeddings, find_module_sample_rate)
    vocab_path = PATH_TO_PARENT + "../next_framework/data/vocabs/vocab_{}.p".format(save_string)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    # you can either send us tokens
    if "custom_vocab_tokens" in payload:
        tokens = payload["custom_vocab_tokens"]
        custom_vocab = build_custom_vocab("", len(vocab), tokens)
    # or if this is an "re" task and you have sent ner_labels, we will create a custom
    # vocab for you of [SUBJ-ner_label, OBJ-ner_label] for each label in ner_labels
    # we do this due to parsing. Look at tokenize() in:
    # model_api/model_training/next_framework/training/util_functions.py
    else:
        if len(ner_labels) > 0 and task == "re":
            custom_vocab = build_custom_vocab("", len(vocab), ner_labels, "re")
        else:
            # finally, if the task is not "re", and you just want to set a custom vocab
            # we allow you to do so
            custom_vocab = build_custom_vocab(dataset_name, len(vocab))
    
    custom_vocab_length = len(custom_vocab)

    build_labeled_dataset([tup[0] for tup in eval_data], 
                          [tup[1] for tup in eval_data],
                          vocab, save_string, "evaluate_next", label_map, custom_vocab)
    logging.info("built eval dataset")
    
    eval_path = PATH_TO_PARENT + "../next_framework/data/training_data/evaluate_next_data_{}.p".format(save_string)
    strict_match_loss_function  = nn.CrossEntropyLoss()
    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, emb_dim, hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)
    clf.load_state_dict(torch.load(PATH_TO_PARENT + "../next_framework/data/saved_models/Next-Clf_{}.p".format(experiment_name)))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    clf = clf.to(device)

    logging.info("loaded model")

    eval_results = evaluate_next_clf(eval_path, clf, strict_match_loss_function, number_of_classes, batch_size=eval_batch_size, none_label_id=none_label_id)
    avg_loss, avg_eval_ent_f1_score, avg_eval_val_f1_score, _, no_relation_thresholds = eval_results

    logging.info("Eval Results")
    dev_tuple = ("%.5f" % avg_loss, "%.5f" % avg_eval_ent_f1_score, "%.5f" % avg_eval_val_f1_score, str(no_relation_thresholds))
    logging.info("Avg Eval Loss: {}, Avg Eval Entropy F1 Score: {}, Avg Eval Max Value F1 Score: {}, Thresholds: {}".format(*dev_tuple))

    return dev_tuple[0], avg_eval_ent_f1_score, avg_eval_val_f1_score, no_relation_thresholds