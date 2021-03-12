from transformers import AdamW
from torch.optim import Adagrad, SGD
import torch
import sys
sys.path.append(".")
sys.path.append("../")
from training.train_util_functions import build_datasets_from_splits, evaluate_next_clf                             
from training.util_functions import similarity_loss_function, generate_save_string, build_custom_vocab,\
                                    set_re_dataset_ner_label_space
from training.util_classes import BaseVariableLengthDataset
from training.constants import TACRED_LABEL_MAP, FIND_MODULE_HIDDEN_DIM, TACRED_ENTITY_TYPES
from models import BiLSTM_Att_Clf, Find_Module
import pickle
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import random
import csv
import dill
import pdb

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
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs",
                        default=60,
                        type=int,
                        help="Number of Epochs for training")
    parser.add_argument('--embeddings',
                        type=str,
                        default="glove.840B.300d",
                        help="initial embeddings to use")
    parser.add_argument('--seed',
                        type=int,
                        default=7698,
                        help="random seed for initialization")
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
    parser.add_argument('--use_adagrad',
                        action='store_true',
                        help="use adagrad optimizer")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    lower_bound = -20.0
    dataset = "tacred"
    save_string = generate_save_string(dataset, args.embeddings)
    number_of_classes = len(TACRED_LABEL_MAP)
    none_label_id = TACRED_LABEL_MAP["no_relation"]
    set_re_dataset_ner_label_space(dataset)
    task = "re"

    if args.build_data:
        build_datasets_from_splits(args.train_path, args.dev_path, args.test_path, args.vocab_path,
                                   TACRED_LABEL_MAP, args.explanation_data_path, save_string,
                                   task=task, dataset=dataset)
    
    with open("../data/training_data/{}_data_{}.p".format("matched", save_string), "rb") as f:
        strict_match_data = pickle.load(f)
    
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
     
    dev_path = "../data/training_data/dev_data_{}.p".format(save_string)
    test_path = "../data/training_data/test_data_{}.p".format(save_string)
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    tacred_vocab = build_custom_vocab("tacred", len(vocab))
    custom_vocab_length = len(tacred_vocab)

    clf = BiLSTM_Att_Clf.BiLSTM_Att_Clf(vocab.vectors, pad_idx, args.emb_dim, args.hidden_dim,
                                        torch.cuda.is_available(), number_of_classes,
                                        custom_token_count=custom_vocab_length)
    
    del vocab

    epochs = args.epochs
    epoch_string = str(epochs)
    test_epoch_f1_scores = []
    dev_epoch_f1_scores = []
    best_test_f1_score = -1
    best_dev_f1_score = -1

    strict_loss_epoch = []

    if args.load_clf_model:
        clf.load_state_dict(torch.load("../data/saved_models/Clf_{}.p".format(args.experiment_name)))
        print("loaded model")

        with open("../data/result_data/test_f1_per_epoch_Clf_{}.csv".format(args.experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                test_epoch_f1_scores.append(row)
                if float(row[-1]) > best_test_f1_score:
                    best_test_f1_score = float(row[-1])
        
        with open("../data/result_data/dev_f1_per_epoch_Clf_{}.csv".format(args.experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                dev_epoch_f1_scores.append(row)
                if float(row[-1]) > best_dev_f1_score:
                    best_dev_f1_score = float(row[-1])
        
        print("loaded past results")
    
    clf = clf.to(device)

    if args.use_adagrad:
        optimizer = Adagrad(clf.parameters(), lr=args.learning_rate)
    else:
        optimizer = SGD(clf.parameters(), lr=args.learning_rate)

    h0 = torch.empty(4, args.match_batch_size, args.hidden_dim).to(device)
    c0 = torch.empty(4, args.match_batch_size, args.hidden_dim).to(device)
    nn.init.xavier_normal_(h0)
    nn.init.xavier_normal_(c0)
    
    # define loss functions
    strict_match_loss_function  = nn.CrossEntropyLoss()

    for epoch in range(args.start_epoch, args.start_epoch+epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, args.start_epoch+epochs))

        total_loss, strict_total_loss, soft_total_loss, sim_total_loss = 0, 0, 0, 0
        batch_count = 0
        clf.train()

        for step, batch in enumerate(tqdm(strict_match_data.as_batches(batch_size=args.match_batch_size, seed=epoch))):
            
            # prepping batch data
            strict_match_tokens, strict_match_lengths, strict_match_labels = batch

            strict_match_tokens = strict_match_tokens.to(device)
            strict_match_labels = strict_match_labels.to(device)
            
            strict_match_predictions = clf.forward(strict_match_tokens, strict_match_lengths, h0, c0)

            strict_match_loss = strict_match_loss_function(strict_match_predictions, strict_match_labels)

            strict_total_loss = strict_total_loss + strict_match_loss.item()
            batch_count += 1

            if batch_count % 50 == 0 and batch_count > 0:
                print((total_loss, strict_total_loss, soft_total_loss, sim_total_loss,  batch_count))
            
            strict_match_loss.backward()
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

        strict_loss_epoch.append(train_avg_strict_loss)
        
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
                                         no_relation_thresholds=no_relation_thresholds,\
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
            torch.save(clf.state_dict(), "{}Clf_{}.p".format(dir_name, args.experiment_name))
            with open("../data/result_data/test_predictions_Clf_{}.csv".format(args.experiment_name), "wb") as f:
                pickle.dump(total_test_class_probs, f)
            with open("../data/result_data/dev_predictions_Clf_{}.csv".format(args.experiment_name), "wb") as f:
                pickle.dump(total_dev_class_probs, f)
            with open("../data/result_data/thresholds.p", "wb") as f:
                pickle.dump({"thresholds" : no_relation_thresholds}, f)
            
            best_test_f1_score = max(avg_test_ent_f1_score, avg_test_val_f1_score)
        
        print("Best Test F1: {}".format("%.5f" % best_test_f1_score))
        print(test_epoch_f1_scores[-3:])
    
    with open("../data/result_data/train_strict_loss_per_epoch_Clf_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train loss'])
        for row in strict_loss_epoch:
            writer.writerow([row])

    with open("../data/result_data/dev_f1_per_epoch_Clf_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['avg_loss, entropy_f1_score','max_value_f1_score', 'max'])
        for row in dev_epoch_f1_scores:
            writer.writerow(row)
    
    with open("../data/result_data/test_f1_per_epoch_Clf_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['avg_loss, entropy_f1_score','max_value_f1_score', 'max'])
        for row in test_epoch_f1_scores:
            writer.writerow(row)

if __name__ == "__main__":
    main()