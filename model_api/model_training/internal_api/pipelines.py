import sys
sys.path.append(".")
sys.path.append("../")
from api.defaults import FIND_MODULE_DEFAULTS, TRAINING_DEFAULTS, BILSTM_DEFAULTS
from next_framework.training.find_util_functions import build_pre_train_find_datasets, \
                                         evaluate_find_module
from next_framework.training.train_util_functions import batch_type_restrict_re, build_phrase_input, build_mask_mat_for_batch,\
                                          build_datasets_from_text, evaluate_next_clf
from next_framework.training.util_functions import similarity_loss_function, generate_save_string, build_custom_vocab,\
                                    set_re_dataset_ner_label_space, apply_strict_matching
from next_framework.training.util_classes import BaseVariableLengthDataset
from next_framework.training.constants import TACRED_ENTITY_TYPES
from next_framework.models import Find_Module, BiLSTM_Att_Clf
import torch
from transformers import AdamW
import pickle
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import random
import csv

def _check_or_load_defaults(payload, default, key):
    if key in payload:
        return payload[key]
    else:
        return default[key]

def pre_train_find_module_pipeline(payload):
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

    build_data = payload["pre_train_build_data"]
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    dataset_size = payload["dataset_size"]

    pos_weight = FIND_MODULE_DEFAULTS["pos_weight"]
    clip_gradient_size = FIND_MODULE_DEFAULTS["clip_gradients"]
    lower_bound = FIND_MODULE_DEFAULTS["lower_bound"]

    sample_rate = training_size / dataset_size

    if sample_rate > 1:
        sample_rate = -1.0 # no sampling
    
    save_string = generate_save_string(dataset_name, embeddings, sample=sample_rate)

    torch.manual_seed(random_state)
    random.seed(random_state)

    if build_data:
        text_data = payload["training_data"]
        explanation_data = payload["explanation_data"]
        build_pre_train_find_datasets(text_data, explanation_data, save_string, embeddings, random_state, dataset_name, sample_rate)
        # UPDATE REQUEST
    
    with open("../data/pre_train_data/train_data_{}.p".format(save_string), "rb") as f:
        train_dataset = pickle.load(f)
    
    primary_eval_path = "../data/pre_train_data/rq_data_{}.p".format(save_string)
    
    # optional secondary eval, can set this to the empty string
    secondary_eval_path = "../data/pre_train_data/dev_data_{}.p".format(save_string)
    
    with open("../data/vocabs/vocab_{}.p".format(save_string), "rb") as f:
        vocab = pickle.load(f)
    
    with open("../data/pre_train_data/sim_data_{}.p".format(save_string), "rb") as f:
        sim_data = pickle.load(f)
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
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
        model.load_state_dict(torch.load("../data/saved_models/Find-Module-pt_{}.p".format(experiment_name)))
        print("loaded model")

        with open("../data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                epoch_losses.append(row)
                if float(row[-1]) > best_f1_score:
                    best_f1_score = float(row[-1])
                if float(row[3]) < best_dev_loss:
                    best_dev_loss = float(row[3])
        
        with open("../data/result_data/dev_2_loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                dev_2_epoch_losses.append(row)
                if float(row[-1]) > best_dev_2_f1_score:
                    best_dev_2_f1_score = float(row[-1])
        
        print("loaded past results")
    
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

    # UPDATE Request

    for epoch in range(start_epoch, start_epoch+epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, start_epoch+epochs))

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

            if batch_count % 100 == 0 and batch_count > 0:
                print((find_total_loss, sim_total_loss, total_loss, batch_count))

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

        # UPDATE Request

        print("Starting Primary Evaluation")
        eval_results = evaluate_find_module(primary_eval_path, real_query_tokens, query_index_matrix, neg_query_index_matrix, lower_bound,
                                            model, find_loss_function, sim_loss_function, eval_batch_size, gamma)
        dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score, total_og_scores, total_new_scores = eval_results
        print("Finished Primary Evaluation")
        
        if dev_f1_score > best_f1_score or (dev_f1_score == best_f1_score and dev_avg_loss < best_dev_loss):
            print("Saving Model")
            dir_name = "../data/saved_models/"
            torch.save(model.state_dict(), "{}Find-Module-pt_{}.p".format(dir_name, experiment_name))
            with open("../data/result_data/best_dev_total_og_scores_{}.p".format(experiment_name), "wb") as f:
                pickle.dump(total_og_scores, f)
            with open("../data/result_data/best_dev_total_new_scores_{}.p".format(experiment_name), "wb") as f:
                pickle.dump(total_new_scores, f)
            best_f1_score = dev_f1_score
            best_dev_loss = dev_avg_loss

            # UPDATE Request

        epoch_losses.append((train_avg_loss, train_avg_find_loss, train_avg_sim_loss,
                             dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score))
        print("Best Primary F1: {}".format(str(best_f1_score)))
        print(epoch_losses[-3:])
        
        if len(secondary_eval_path) > 0:
            print("Starting Secondary Evaluation")
            eval_results = evaluate_find_module(secondary_eval_path, real_query_tokens, query_index_matrix, neg_query_index_matrix, lower_bound,
                                                model, find_loss_function, sim_loss_function, eval_batch_size, gamma)
            dev_2_avg_loss, dev_2_avg_find_loss, dev_2_avg_sim_loss, dev_2_f1_score, total_og_scores, total_new_scores = eval_results
            print("Finished Secondary Evaluation")

            if dev_2_f1_score > best_dev_2_f1_score:
                best_dev_2_f1_score = dev_2_f1_score
                with open("../data/result_data/best_dev_2_total_og_scores_{}.p".format(experiment_name), "wb") as f:
                    pickle.dump(total_og_scores, f)
                with open("../data/result_data/best_dev_2_total_new_scores_{}.p".format(experiment_name), "wb") as f:
                    pickle.dump(total_new_scores, f)
            
            dev_2_epoch_losses.append((dev_2_avg_loss, dev_2_avg_find_loss, dev_2_avg_sim_loss,
                                       dev_2_f1_score))
            print("Best Secondary F1: {}".format(str(best_dev_2_f1_score)))
            print(dev_2_epoch_losses[-3:])
        
        if best_f1_score > 0.9:
            break

    with open("../data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_loss','train_find_loss', 'train_sim_loss', 'dev_loss', 'dev_find_loss', 'dev_sim_loss', 'dev_f1_score'])
        for row in epoch_losses:
            writer.writerow(row)
    
    if len(secondary_eval_path) > 0:
        with open("../data/result_data/dev_2_loss_per_epoch_Find-Module-pt_{}.csv".format(experiment_name), "w") as f:
            writer=csv.writer(f)
            writer.writerow(["dev_2_avg_loss", "dev_2_avg_find_loss", "dev_2_avg_sim_loss", "dev_2_f1_score"])
            for row in dev_2_epoch_losses:
                writer.writerow(row)
    
    # UPDATE Request

def strict_match_pipeline(payload):
    text_data = payload["training_data"]
    explanation_data = payload["explanation_data"]
    task = payload["task"]
    matched_tuples, matched_indices = apply_strict_matching(text_data, explanation_data, task)
    
    return matched_tuples, matched_indices

def train_next_bilstm_pipeline(payload):    
    if payload["stage"] == "both":
        # UPDATE Request
        pre_train_find_module_pipeline(payload)

    # SETUP
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

    load_model = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "load_model")
    start_epoch = _check_or_load_defaults(payload, TRAINING_DEFAULTS, "start_epoch")

    build_data = payload["build_data"]
    experiment_name = payload["experiment_name"]
    dataset_name = payload["dataset_name"]
    dataset_size = payload["dataset_size"]

    clip_gradient_size = BILSTM_DEFAULTS["clip_gradients"]
    lower_bound = FIND_MODULE_DEFAULTS["lower_bound"]
    find_module_hidden_dim = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_hidden_dim")
    find_module_path =  "../data/saved_models/Find-Module-pt_{}.p".format(experiment_name)
    
    pre_training_size = _check_or_load_defaults(payload, FIND_MODULE_DEFAULTS, "pre_train_training_size")
    find_module_sample_rate = pre_training_size / dataset_size

    if find_module_sample_rate > 1:
        find_module_sample_rate = -1.0 # no sampling
    
    save_string = generate_save_string(dataset_name, embeddings, find_module_sample_rate)
    vocab_path = "../data/vocabs/vocab_{}.p".format(save_string)

    label_map = payload["label_map"]
    number_of_classes = len(label_map)
    task = payload["task"]

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

    if build_data:
        # UPDATE Request
        text_data = payload["training_data"]
        explanation_data = payload["explanation_data"]
        build_datasets_from_text(text_data, vocab_path, explanation_data, save_string, label_map, task=task, dataset=dataset_name)
    
    with open("../data/training_data/unlabeled_data_{}.p".format(save_string), "rb") as f:
        unlabeled_data = pickle.load(f)
    
    with open("../data/training_data/matched_data_{}.p".format(save_string), "rb") as f:
        strict_match_data = pickle.load(f)
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    with open("../data/training_data/labeling_functions_{}.p".format(save_string), "rb") as f:
        soft_labeling_functions_dict = dill.load(f)

    with open("../data/training_data/query_tokens_{}.p".format(save_string), "rb") as f:
        tokenized_queries = pickle.load(f)
    
    with open("../data/training_data/word2idx_{}.p".format(save_string), "rb") as f:
        quoted_words_to_index = pickle.load(f)
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # HERE IS WHERE YOU CAN DEFINE CUSTOM TOKENS FOR YOUR VOCAB
    if "custom_vocab" in payload:
        custom_vocab = payload["custom_vocab"]
    else:
        custom_vocab = build_custom_vocab(dataset_name, len(vocab))
    
    custom_vocab_length = len(custom_vocab)
    
    find_module = Find_Module.Find_Module(vocab.vectors, pad_idx, emb_dim, find_module_hidden_dim,
                                          torch.cuda.is_available(), custom_token_count=custom_vocab_length)
        
    find_module.load_state_dict(torch.load(find_module_path))

    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, emb_dim, hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)
    
    del vocab

    epochs = epochs
    epoch_string = str(epochs)

    loss_per_epoch = []
    best_loss = 1e30

    if load_model:
        clf.load_state_dict(torch.load("../data/saved_models/Next-Clf_{}.p".format(experiment_name)))
        print("loaded model")
        
        with open("../data/result_data/train_loss_per_epoch_Next-Clf_{}.csv".format(experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                loss_per_epoch.append(row)
                if float(row[0]) < best_loss:
                    best_loss = float(row[0])
        
        print("loaded past results")
    
    clf = clf.to(device)
    find_module = find_module.to(device)

    # Get queries ready for Find Module
    lfind_query_tokens, _ = BaseVariableLengthDataset.variable_length_batch_as_tensors(tokenized_queries, pad_idx)
    lfind_query_tokens = lfind_query_tokens.to(device).detach()

    # Preping Soft Labeling Function Data
    soft_labeling_functions = soft_labeling_functions_dict["function_pairs"]
    soft_labeling_function_labels = soft_labeling_functions_dict["labels"]

    optimizer = SGD(clf.parameters(), lr=learning_rate)
    
    h0_hard = torch.empty(4, match_batch_size, hidden_dim).to(device)
    c0_hard = torch.empty(4, match_batch_size, hidden_dim).to(device)
    nn.init.xavier_normal_(h0_hard)
    nn.init.xavier_normal_(c0_hard)

    h0_soft = torch.empty(4, unlabeled_batch_size, hidden_dim).to(device)
    c0_soft = torch.empty(4, unlabeled_batch_size, hidden_dim).to(device)
    nn.init.xavier_normal_(h0_soft)
    nn.init.xavier_normal_(c0_soft)
    
    # define loss functions
    strict_match_loss_function  = nn.CrossEntropyLoss()
    soft_match_loss_function = nn.CrossEntropyLoss(reduction='none')
    
    # TRAINING
    # UPDATE Request
    for epoch in range(start_epoch, start_epoch+epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, start_epoch+epochs))

        total_loss, strict_total_loss, soft_total_loss = 0, 0, 0
        batch_count = 0
        clf.train()
        find_module.eval()

        for step, batch_pair in enumerate(tqdm(zip(strict_match_data.as_batches(batch_size=match_batch_size, seed=epoch),\
                                                   unlabeled_data.as_batches(batch_size=unlabeled_batch_size, seed=epoch*seed)))):
            
            # prepping batch data
            strict_match_data_batch, unlabeled_data_batch = batch_pair

            strict_match_tokens, strict_match_lengths, strict_match_labels = strict_match_data_batch

            strict_match_tokens = strict_match_tokens.to(device)
            strict_match_labels = strict_match_labels.to(device)

            unlabeled_tokens, unlabeled_token_lengths, phrases = unlabeled_data_batch
            unlabeled_tokens = unlabeled_tokens.to(device)

            phrase_input = build_phrase_input(phrases, pad_idx, task).to(device).detach()
            _, seq_length = unlabeled_tokens.shape
            mask_mat = build_mask_mat_for_batch(seq_length).to(device).detach()

            # clear previously calculated gradients 
            clf.zero_grad()  
            
            with torch.no_grad():
                lfind_output = find_module.soft_matching_forward(unlabeled_tokens.detach(), lfind_query_tokens, lower_bound).detach() # B x seq_len x Q
            
                function_batch_scores = []
                for pair in soft_labeling_functions:
                    func, rel = pair
                    try:
                        batch_scores = func(lfind_output, quoted_words_to_index, mask_mat)(phrase_input).detach() # 1 x B
                    except:
                        batch_scores = torch.zeros((1, unlabeled_batch_size)).to(device).detach()
                    
                    if relation_ner_types != None:
                        type_restrict_multiplier = batch_type_restrict_re(rel, phrase_input, relation_ner_types).detach() # 1 x B
                        final_scores = batch_scores * type_restrict_multiplier # 1 x B
                    else:
                        final_scores = batch_scores
                    function_batch_scores.append(final_scores)
            
                function_batch_scores_tensor = torch.cat(function_batch_scores, dim=0).detach().permute(1,0) # B x Q
                unlabeled_label_index = torch.argmax(function_batch_scores_tensor, dim=1) # B
                
                unlabeled_labels = torch.tensor([soft_labeling_function_labels[index] for index in unlabeled_label_index]).to(device).detach() # B
                unlabeled_label_weights = nn.functional.softmax(torch.amax(function_batch_scores_tensor, dim=1), dim=0) # B
            
            strict_match_predictions = clf.forward(strict_match_tokens, strict_match_lengths, h0_hard, c0_hard)
            soft_match_predictions = clf.forward(unlabeled_tokens, unlabeled_token_lengths, h0_soft, c0_soft)

            strict_match_loss = strict_match_loss_function(strict_match_predictions, strict_match_labels)
            soft_match_loss = torch.sum(soft_match_loss_function(soft_match_predictions, unlabeled_labels) * unlabeled_label_weights)
            combined_loss = strict_match_loss + gamma * soft_match_loss

            strict_total_loss = strict_total_loss + strict_match_loss.item()
            soft_total_loss = soft_total_loss + soft_match_loss.item()
            total_loss = total_loss + combined_loss.item()
            batch_count += 1

            if batch_count % 50 == 0 and batch_count > 0:
                print((total_loss, strict_total_loss, soft_total_loss,  batch_count))
            
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), clip_gradient_size)

            optimizer.step()

        # compute the training loss of the epoch
        train_avg_loss = total_loss / batch_count
        train_avg_strict_loss = strict_total_loss / batch_count
        train_avg_soft_loss = soft_total_loss / batch_count

        print("Train Losses")
        loss_tuples = ("%.5f" % train_avg_loss, "%.5f" % train_avg_strict_loss, "%.5f" % train_avg_soft_loss)
        print("Avg Train Total Loss: {}, Avg Train Strict Loss: {}, Avg Train Soft Loss: {}".format(*loss_tuples))
        
        loss_per_epoch.append((train_avg_loss, train_avg_strict_loss, train_avg_soft_loss))
        
        train_path = "../data/training_data/{}_data_{}.p".format("matched", save_string)
        train_results = evaluate_next_clf(train_path, clf, strict_match_loss_function, number_of_classes, batch_size=eval_batch_size)
        avg_loss, avg_train_ent_f1_score, avg_train_val_f1_score, total_train_class_probs, no_relation_thresholds = train_results
        print("Train Results")
        train_tuple = ("%.5f" % avg_loss, "%.5f" % avg_train_ent_f1_score, "%.5f" % avg_train_val_f1_score, str(no_relation_thresholds))
        print("Avg Train Loss: {}, Avg Train Entropy F1 Score: {}, Avg Train Max Value F1 Score: {}, Thresholds: {}".format(*train_tuple))
        # UPDATE Request
    
    with open("../data/result_data/train_loss_per_epoch_Next-Clf_{}.csv".format(experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_avg_loss', 'train_avg_strict_loss', 'train_avg_soft_loss'])
        for row in loss_per_epoch:
            writer.writerow([row])
