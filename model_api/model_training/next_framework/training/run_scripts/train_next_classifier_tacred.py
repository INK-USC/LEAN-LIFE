from torch.optim import Adagrad, SGD
import torch
import sys
sys.path.append(".")
sys.path.append("../")
from training.train_util_functions import batch_type_restrict_re, build_phrase_input, build_mask_mat_for_batch,\
                                          build_datasets_from_splits, evaluate_next_clf
from training.util_functions import similarity_loss_function, generate_save_string, build_custom_vocab,\
                                    set_re_dataset_ner_label_space
from training.util_classes import BaseVariableLengthDataset
from training.constants import TACRED_LABEL_MAP, FIND_MODULE_HIDDEN_DIM, TACRED_ENTITY_TYPES, TACRED_NERS
from models import BiLSTM_Att_Clf, Find_Module
import pickle
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import random
import csv
import pdb
import dill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_data",
                        action="store_true",
                        help="Whether to build data.")
    parser.add_argument("--train_path",
                        type=str,
                        default="../data/tacred_train.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="../data/tacred_dev.json",
                        help="Path to dev data.")
    parser.add_argument("--test_path",
                        type=str,
                        default="../data/tacred_test.json",
                        help="Path to train data.")
    parser.add_argument("--explanation_data_path",
                        type=str,
                        default="../data/tacred_explanations.json",
                        help="Path to explanation data.")
    parser.add_argument("--find_module_path",
                        type=str,
                        default="../data/saved_models/Find-Module-pt_official.p",
                        help="Path to pretrained find module")
    parser.add_argument("--vocab_path",
                        type=str,
                        default="../data/vocabs/vocab_glove.840B.300d_-1_0.6.p",
                        help="Path to vocab created in Pre-training")
    parser.add_argument("--match_batch_size",
                        default=50,
                        type=int,
                        help="Match batch size for train.")
    parser.add_argument("--unlabeled_batch_size",
                        default=100,
                        type=int,
                        help="Unlabeled batch size for train.")
    parser.add_argument("--eval_batch_size",
                        default=50,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=0.1,
                        type=float,
                        help="The initial learning rate.")
    parser.add_argument("--epochs",
                        default=75,
                        type=int,
                        help="Number of Epochs for training")
    parser.add_argument('--embeddings',
                        type=str,
                        default="glove.840B.300d",
                        help="initial embeddings to use")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        help="weight of soft_matching_loss")
    parser.add_argument('--emb_dim',
                        type=int,
                        default=300,
                        help="embedding vector size")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden vector size of lstm (really 2*hidden_dim, due to bilstm)")
    parser.add_argument('--model_save_dir',
                        type=str,
                        default="",
                        help="where to save the model")
    parser.add_argument('--experiment_name',
                        type=str,
                        help="what to save the model file as")
    parser.add_argument('--load_clf_model',
                        action='store_true',
                        help="Whether to load a trained classifier model")
    parser.add_argument('--start_epoch',
                         type=int,
                         default=0,
                         help="start_epoch")

    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    lower_bound = -20.0
    dataset = "tacred"
    save_string = generate_save_string(dataset, args.embeddings)
    number_of_classes = len(TACRED_LABEL_MAP)
    none_label_id = TACRED_LABEL_MAP["no_relation"]
    set_re_dataset_ner_label_space(dataset, TACRED_NERS)
    task = "re"
    relation_ner_types = TACRED_ENTITY_TYPES

    if args.build_data:
        # vocab_ = {
        #     "embedding_name" : "glove.840B.300d",
        #     "save_string" : "glove.840B.300d_-1_0.6"
        # }
        build_datasets_from_splits(args.train_path, args.dev_path, args.test_path, args.vocab_path,
                                   args.explanation_data_path, save_string, TACRED_LABEL_MAP,
                                   task=task, dataset=dataset)
    

    with open("../data/training_data/unlabeled_data_{}.p".format(save_string), "rb") as f:
        unlabeled_data = pickle.load(f)
    
    with open("../data/training_data/matched_data_{}.p".format(save_string), "rb") as f:
        strict_match_data = pickle.load(f)
    
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    with open("../data/training_data/labeling_functions_{}.p".format(save_string), "rb") as f:
        soft_labeling_functions_dict = dill.load(f)

    with open("../data/training_data/query_tokens_{}.p".format(save_string), "rb") as f:
        tokenized_queries = pickle.load(f)
    
    with open("../data/training_data/word2idx_{}.p".format(save_string), "rb") as f:
        quoted_words_to_index = pickle.load(f)
    
    dev_path = "../data/training_data/dev_data_{}.p".format(save_string)
    test_path = "../data/training_data/test_data_{}.p".format(save_string)
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    custom_vocab = build_custom_vocab("tacred", len(vocab))
    custom_vocab_length = len(custom_vocab)

    epochs = args.epochs
    epoch_string = str(epochs)
    test_epoch_f1_scores = []
    dev_epoch_f1_scores = []
    best_test_f1_score = -1
    best_dev_f1_score = -1

    loss_per_epoch = []

    if args.load_clf_model:
        clf.load_state_dict(torch.load("../data/saved_models/Next-Clf_{}.p".format(args.experiment_name)))
        print("loaded model")

        with open("../data/result_data/test_f1_per_epoch_Next-Clf_{}.csv".format(args.experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                test_epoch_f1_scores.append(row)
                if float(row[-1]) > best_test_f1_score:
                    best_test_f1_score = float(row[-1])
        
        with open("../data/result_data/dev_f1_per_epoch_Next-Clf_{}.csv".format(args.experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                dev_epoch_f1_scores.append(row)
                if float(row[-1]) > best_dev_f1_score:
                    best_dev_f1_score = float(row[-1])
        
        print("loaded past results")

    # Get queries ready for Find Module
    lfind_query_tokens, _ = BaseVariableLengthDataset.variable_length_batch_as_tensors(tokenized_queries, pad_idx)
    lfind_query_tokens = lfind_query_tokens.to(device).detach()

    # Preping Soft Labeling Function Data
    soft_labeling_functions = soft_labeling_functions_dict["function_pairs"]
    soft_labeling_function_labels = soft_labeling_functions_dict["labels"].to(device)
    
    full_batch_size = args.match_batch_size + args.unlabeled_batch_size
    n_layer_x_n_directions = 4

    h0 = torch.empty(n_layer_x_n_directions, full_batch_size, args.hidden_dim).to(device)
    c0 = torch.empty(n_layer_x_n_directions, full_batch_size, args.hidden_dim).to(device)
    nn.init.xavier_normal_(h0)
    nn.init.xavier_normal_(c0)

    if args.build_data:
        find_module = Find_Module.Find_Module(vocab.vectors, pad_idx, args.emb_dim, FIND_MODULE_HIDDEN_DIM,
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

        with open("../data/training_data/soft_scores_{}.p".format(args.experiment_name), "wb") as f:
            pickle.dump(soft_scores, f)
        
        del find_module
    
    else:
        with open("../data/training_data/soft_scores_{}.p".format(args.experiment_name), "rb") as f:
            soft_scores = pickle.load(f)

    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, args.emb_dim, args.hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)

    del vocab

    clf = clf.to(device)

    optimizer = SGD(clf.parameters(), lr=args.learning_rate)

    # define loss functions
    strict_match_loss_function  = nn.CrossEntropyLoss()
    soft_match_loss_function = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(args.start_epoch, args.start_epoch+epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, args.start_epoch+epochs))

        total_loss, strict_total_loss, soft_total_loss, sim_total_loss = 0, 0, 0, 0
        batch_count = 0
        clf.train()

        for step, batch_pair in enumerate(tqdm(zip(strict_match_data.as_batches(batch_size=full_batch_size, seed=epoch),\
                                                   unlabeled_data.as_batches(batch_size=full_batch_size, seed=epoch*args.seed)))):            
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
            soft_match_loss = torch.sum(soft_match_loss_function(soft_match_predictions, pseudo_labels) * unlabeled_label_weights)
            combined_loss = strict_match_loss + args.gamma * soft_match_loss

            strict_total_loss = strict_total_loss + strict_match_loss.item()
            soft_total_loss = soft_total_loss + soft_match_loss.item()
            total_loss = total_loss + combined_loss.item()
            batch_count += 1

            if batch_count % 50 == 0 and batch_count > 0:
                print((total_loss, strict_total_loss, soft_total_loss, sim_total_loss,  batch_count))
            
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 5.0)

            optimizer.step()

        # compute the training loss of the epoch
        train_avg_loss = total_loss / batch_count
        train_avg_strict_loss = strict_total_loss / batch_count
        train_avg_soft_loss = soft_total_loss / batch_count
        train_avg_sim_loss = sim_total_loss / batch_count

        print("Train Losses")
        loss_tuples = ("%.5f" % train_avg_loss, "%.5f" % train_avg_strict_loss, "%.5f" % train_avg_soft_loss, "%.5f" % train_avg_sim_loss)
        print("Avg Train Total Loss: {}, Avg Train Strict Loss: {}, Avg Train Soft Loss: {}, Avg Train Sim Loss: {}".format(*loss_tuples))
        
        loss_per_epoch.append((train_avg_loss, train_avg_strict_loss, train_avg_soft_loss))
        
        train_path = "../data/training_data/{}_data_{}.p".format("matched", save_string)
        train_results = evaluate_next_clf(train_path, clf, strict_match_loss_function, number_of_classes, batch_size=args.eval_batch_size, none_label_id=none_label_id)
        avg_loss, avg_train_ent_f1_score, avg_train_val_f1_score, total_train_class_probs, no_relation_thresholds = train_results
        print("Train Results")
        train_tuple = ("%.5f" % avg_loss, "%.5f" % avg_train_ent_f1_score, "%.5f" % avg_train_val_f1_score, str(no_relation_thresholds))
        print("Avg Train Loss: {}, Avg Train Entropy F1 Score: {}, Avg Train Max Value F1 Score: {}, Thresholds: {}".format(*train_tuple))

        dev_results = evaluate_next_clf(dev_path, clf, strict_match_loss_function, number_of_classes, batch_size=args.eval_batch_size, none_label_id=none_label_id)
        
        avg_loss, avg_dev_ent_f1_score, avg_dev_val_f1_score, total_dev_class_probs, no_relation_thresholds = dev_results

        print("Dev Results")
        dev_tuple = ("%.5f" % avg_loss, "%.5f" % avg_dev_ent_f1_score, "%.5f" % avg_dev_val_f1_score, str(no_relation_thresholds))
        print("Avg Dev Loss: {}, Avg Dev Entropy F1 Score: {}, Avg Dev Max Value F1 Score: {}, Thresholds: {}".format(*dev_tuple))

        dev_epoch_f1_scores.append((avg_loss, avg_dev_ent_f1_score, avg_dev_val_f1_score, max(avg_dev_ent_f1_score, avg_dev_val_f1_score)))
        
        if max(avg_dev_ent_f1_score, avg_dev_val_f1_score) > best_dev_f1_score:
            best_dev_f1_score = max(avg_dev_ent_f1_score, avg_dev_val_f1_score)
            print("Updated Dev F1 Score")
        
        test_results = evaluate_next_clf(test_path, clf, strict_match_loss_function, number_of_classes,\
                                         none_label_thresholds=no_relation_thresholds,\
                                         batch_size=args.eval_batch_size, none_label_id=none_label_id)
        
        avg_loss, avg_test_ent_f1_score, avg_test_val_f1_score, total_test_class_probs, _ = test_results

        print("Test Results")
        test_tuple = ("%.5f" % avg_loss, "%.5f" % avg_test_ent_f1_score, "%.5f" % avg_test_val_f1_score, str(no_relation_thresholds))
        print("Avg Test Loss: {}, Avg Test Entropy F1 Score: {}, Avg Test Max Value F1 Score: {}, Thresholds: {}".format(*test_tuple))

        test_epoch_f1_scores.append((avg_loss, avg_test_ent_f1_score, avg_test_val_f1_score, max(avg_test_ent_f1_score, avg_test_val_f1_score)))

        if best_test_f1_score < max(avg_test_ent_f1_score, avg_test_val_f1_score):
            print("Saving Model")
            if len(args.model_save_dir) > 0:
                dir_name = args.model_save_dir
            else:
                dir_name = "../data/saved_models/"
            torch.save(clf.state_dict(), "{}Next-Clf_{}.p".format(dir_name, args.experiment_name))
            with open("../data/result_data/test_predictions_Next-Clf_{}.csv".format(args.experiment_name), "wb") as f:
                pickle.dump(total_test_class_probs, f)
            with open("../data/result_data/dev_predictions_Next-Clf_{}.csv".format(args.experiment_name), "wb") as f:
                pickle.dump(total_dev_class_probs, f)
            with open("../data/result_data/thresholds.p", "wb") as f:
                pickle.dump({"thresholds" : no_relation_thresholds}, f)
            
            best_test_f1_score = max(avg_test_ent_f1_score, avg_test_val_f1_score)
        
        print("Best Test F1: {}".format("%.5f" % best_test_f1_score))
        print(test_epoch_f1_scores[-3:])
    
    with open("../data/result_data/train_loss_per_epoch_Next-Clf_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_avg_loss', 'train_avg_strict_loss', 'train_avg_soft_loss'])
        for row in loss_per_epoch:
            writer.writerow([row])
    
    with open("../data/result_data/dev_f1_per_epoch_Next-Clf_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['avg_loss, entropy_f1_score','max_value_f1_score', 'max'])
        for row in dev_epoch_f1_scores:
            writer.writerow(row)
    
    with open("../data/result_data/test_f1_per_epoch_Next-Clf_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['avg_loss, entropy_f1_score','max_value_f1_score', 'max'])
        for row in test_epoch_f1_scores:
            writer.writerow(row)

if __name__ == "__main__":
    main()