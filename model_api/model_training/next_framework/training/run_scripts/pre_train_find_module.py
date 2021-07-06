import torch
from torch.optim import Adagrad
from transformers import AdamW
import sys
sys.path.append(".")
sys.path.append("../")
from training.find_util_functions import build_pre_train_find_datasets_from_splits, \
                                         evaluate_find_module
from training.util_functions import similarity_loss_function, generate_save_string, build_custom_vocab
from training.util_classes import BaseVariableLengthDataset
from models import Find_Module
import pickle
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import random
import csv
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_pre_train",
                        action='store_true',
                        help="Whether to build Pre-Train data.")
    parser.add_argument("--train_path",
                        type=str,
                        default="../data/tacred_train.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="../data/tacred_dev.json",
                        help="Path to unlabled data.")
    parser.add_argument("--test_path",
                        type=str,
                        default="../data/tacred_test.json",
                        help="Path to unlabled data.")
    parser.add_argument("--explanation_data_path",
                        type=str,
                        default="../data/tacred_explanations.json",
                        help="Path to explanation data.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for train.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs",
                        default=25, # will train for 24, stopping criteria of 0.9 f1
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
                        default=0.5,
                        help="weight of sim_loss")
    parser.add_argument('--emb_dim',
                        type=int,
                        default=300,
                        help="embedding vector size")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=300,
                        help="hidden vector size of lstm (really 2*hidden_dim, due to bilstm)")
    parser.add_argument('--model_save_dir',
                        type=str,
                        default="",
                        help="where to save the model")
    parser.add_argument('--experiment_name',
                        type=str,
                        default="official",
                        help="what to save the model file as")
    parser.add_argument('--load_model',
                        action='store_true',
                        help="Whether to load a model")
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
    sample_rate = 0.6
    lower_bound = -20.0
    dataset = "tacred"
    
    if args.build_pre_train:
        build_pre_train_find_datasets_from_splits(args.train_path, args.dev_path, args.test_path,
                                                  args.explanation_data_path, embedding_name=args.embeddings,
                                                  sample_rate=sample_rate, dataset=dataset)

    save_string = generate_save_string(dataset, args.embeddings, sample=sample_rate)

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
    
    custom_vocab = build_custom_vocab(dataset, len(vocab))
    custom_vocab_length = len(custom_vocab)

    model = Find_Module.Find_Module(emb_weight=vocab.vectors, padding_idx=pad_idx, emb_dim=args.emb_dim,
                                    hidden_dim=args.hidden_dim, cuda=torch.cuda.is_available(),
                                    custom_token_count=custom_vocab_length)
    del vocab

    # prepping variables for storing training progress
    epochs = args.epochs
    epoch_string = str(epochs)
    epoch_losses = []
    dev_2_epoch_losses = []
    best_f1_score = -1
    best_dev_2_f1_score = -1
    best_dev_loss = float('inf') 
    
    if args.load_model:
        model.load_state_dict(torch.load("../data/saved_models/Find-Module-pt_{}.p".format(args.experiment_name)))
        print("loaded model")

        with open("../data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(args.experiment_name)) as f:
            reader=csv.reader(f)
            next(reader)
            for row in reader:
                epoch_losses.append(row)
                if float(row[-1]) > best_f1_score:
                    best_f1_score = float(row[-1])
                if float(row[3]) < best_dev_loss:
                    best_dev_loss = float(row[3])
        
        with open("../data/result_data/dev_2_loss_per_epoch_Find-Module-pt_{}.csv".format(args.experiment_name)) as f:
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
    if args.use_adagrad:
        optimizer = Adagrad(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
         
    # define loss functions
    find_loss_function  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    sim_loss_function = similarity_loss_function

    for epoch in range(args.start_epoch, args.start_epoch+epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, args.start_epoch+epochs))

        total_loss, find_total_loss, sim_total_loss = 0, 0, 0
        batch_count = 0
        model.train()
        # iterate over batches
        for step, batch in enumerate(tqdm(train_dataset.as_batches(batch_size=args.train_batch_size, seed=epoch))):
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
            string_loss = find_loss + args.gamma * sim_loss

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()

        # compute the training loss of the epoch
        train_avg_loss = total_loss / batch_count
        train_avg_find_loss = find_total_loss / batch_count
        train_avg_sim_loss = sim_total_loss / batch_count


        print("Starting Primary Evaluation")
        eval_results = evaluate_find_module(primary_eval_path, real_query_tokens, query_index_matrix, neg_query_index_matrix, lower_bound,
                                            model, find_loss_function, sim_loss_function, args.eval_batch_size, args.gamma)
        dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score, total_og_scores, total_new_scores = eval_results
        print("Finished Primary Evaluation")
        
        if dev_f1_score > best_f1_score or (dev_f1_score == best_f1_score and dev_avg_loss < best_dev_loss):
            print("Saving Model")
            if len(args.model_save_dir) > 0:
                dir_name = args.model_save_dir
            else:
                dir_name = "../data/saved_models/"
            torch.save(model.state_dict(), "{}Find-Module-pt_{}.p".format(dir_name, args.experiment_name))
            with open("../data/result_data/best_dev_total_og_scores_{}.p".format(args.experiment_name), "wb") as f:
                pickle.dump(total_og_scores, f)
            with open("../data/result_data/best_dev_total_new_scores_{}.p".format(args.experiment_name), "wb") as f:
                pickle.dump(total_new_scores, f)
            best_f1_score = dev_f1_score
            best_dev_loss = dev_avg_loss

        epoch_losses.append((train_avg_loss, train_avg_find_loss, train_avg_sim_loss,
                             dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score))
        print("Best Primary F1: {}".format(str(best_f1_score)))
        print(epoch_losses[-3:])
        
        if len(secondary_eval_path) > 0:
            print("Starting Secondary Evaluation")
            eval_results = evaluate_find_module(secondary_eval_path, real_query_tokens, query_index_matrix, neg_query_index_matrix, lower_bound,
                                                model, find_loss_function, sim_loss_function, args.eval_batch_size, args.gamma)
            dev_2_avg_loss, dev_2_avg_find_loss, dev_2_avg_sim_loss, dev_2_f1_score, total_og_scores, total_new_scores = eval_results
            print("Finished Secondary Evaluation")

            if dev_2_f1_score > best_dev_2_f1_score:
                best_dev_2_f1_score = dev_2_f1_score
                with open("../data/result_data/best_dev_2_total_og_scores_{}.p".format(args.experiment_name), "wb") as f:
                    pickle.dump(total_og_scores, f)
                with open("../data/result_data/best_dev_2_total_new_scores_{}.p".format(args.experiment_name), "wb") as f:
                    pickle.dump(total_new_scores, f)
            
            dev_2_epoch_losses.append((dev_2_avg_loss, dev_2_avg_find_loss, dev_2_avg_sim_loss,
                                       dev_2_f1_score))
            print("Best Secondary F1: {}".format(str(best_dev_2_f1_score)))
            print(dev_2_epoch_losses[-3:])
        
        if best_f1_score > 0.9:
            break

    with open("../data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_loss','train_find_loss', 'train_sim_loss', 'dev_loss', 'dev_find_loss', 'dev_sim_loss', 'dev_f1_score'])
        for row in epoch_losses:
            writer.writerow(row)
    
    if len(secondary_eval_path) > 0:
        with open("../data/result_data/dev_2_loss_per_epoch_Find-Module-pt_{}.csv".format(args.experiment_name), "w") as f:
            writer=csv.writer(f)
            writer.writerow(["dev_2_avg_loss", "dev_2_avg_find_loss", "dev_2_avg_sim_loss", "dev_2_f1_score"])
            for row in dev_2_epoch_losses:
                writer.writerow(row)

if __name__ == "__main__":
    main()
