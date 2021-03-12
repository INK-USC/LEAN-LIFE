import torch
from transformers import AdamW
import sys
sys.path.append(".")
sys.path.append("../")
from util_functions import build_pre_train_find_datasets, similarity_loss_function, evaluate_find_module
from util_classes import PreTrainingFindModuleDataset
from models import Find_Module
import pickle
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import random
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_pre_train",
                        action='store_true',
                        help="Whether to build Pre-Train data.")
    parser.add_argument("--unlabeled_data_path",
                        type=str,
                        default="../data/carer.json",
                        help="Path to unlabled data.")
    parser.add_argument("--explanation_data_path",
                        type=str,
                        default="../data/ec_explanations.json",
                        help="Path to explanation data.")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for train.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs",
                        default=10,
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
                        default=200,
                        help="hidden vector size of lstm (really 2*hidden_dim, due to bilstm)")
    parser.add_argument('--encoding_dim',
                        type=int,
                        default=300,
                        help="final encoding dimension")
    parser.add_argument('--embedding_dropout',
                        type=float,
                        default=0.9,
                        help="embedding dropout")
    parser.add_argument('--model_save_dir',
                        type=str,
                        default="",
                        help="where to save the model")
    parser.add_argument('--experiment_name',
                        type=str,
                        help="what to save the model file as")
    parser.add_argument('--load_model',
                      action='store_true',
                      help="Whether to build load a model")
    parser.add_argument('--start_epoch',
                      type=int,
                      default=0,
                      help="start_epoch")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.build_pre_train:
        build_pre_train_find_datasets(args.unlabeled_data_path, args.explanation_data_path, embedding_name=args.embeddings, 
                                      random_state=args.seed, label_filter=["surprise", "trust", "anticipation"])

    with open("../data/pre_train_data/train_data_{}_{}.p".format(args.embeddings, str(args.seed)), "rb") as f:
        train_dataset = pickle.load(f)
    
    dev_path = "../data/pre_train_data/dev_data_{}_{}.p".format(args.embeddings, str(args.seed))
    
    with open("../data/pre_train_data/vocab_{}_{}.p".format(args.embeddings, str(args.seed)), "rb") as f:
        vocab = pickle.load(f)
    
    with open("../data/pre_train_data/sim_data_{}_{}.p".format(args.embeddings, str(args.seed)), "rb") as f:
        sim_data = pickle.load(f)
    
    pad_idx = vocab["<pad>"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Find_Module.Find_Module(emb_weight=vocab.vectors, padding_idx=pad_idx, emb_dim=args.emb_dim,
                                    hidden_dim=args.hidden_dim, encoding_dim=args.encoding_dim,
                                    cuda=torch.cuda.is_available(), embedding_dropout=args.embedding_dropout)
    
    if args.load_model:
        model.load_state_dict(torch.load("../data/saved_models/Find-Module-pt_{}.pt".format(args.experiment_name)))
        print("loaded model")
    
    del vocab
    model = model.to(device)

    real_query_tokens = PreTrainingFindModuleDataset.variable_length_batch_as_tensors(sim_data["queries"], pad_idx)
    real_query_tokens = real_query_tokens.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)   

    # define loss functions
    find_loss_function  = nn.BCEWithLogitsLoss()
    sim_loss_function = similarity_loss_function

    # number of training epochs
    epochs = args.epochs

    epoch_losses = []
    best_eval_loss = float('inf') 

    for epoch in range(args.start_epoch-1,epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

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
            token_scores = model.find_forward(tokens, queries)
            pos_scores, neg_scores = model.sim_forward(real_query_tokens, sim_data["labels"])

            # compute the loss between actual and predicted values
            find_loss = find_loss_function(token_scores, labels)
            sim_loss = sim_loss_function(pos_scores, neg_scores)
            string_loss = find_loss + args.gamma * sim_loss

            # add on to the total loss
            find_total_loss = find_total_loss  + find_loss.item()
            sim_total_loss = sim_total_loss + sim_loss.item()
            total_loss = total_loss + string_loss.item()
            batch_count += 1

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


        print("Starting Evaluation")
        eval_results = evaluate_find_module(dev_path, real_query_tokens, sim_data["labels"], model, find_loss_function, sim_loss_function, args.eval_batch_size, args.gamma)
        dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_scores = eval_results
        print("Finished Evaluation")
        
        if dev_avg_loss < best_eval_loss:
            print("Saving Model")
            if len(args.model_save_dir) > 0:
                dir_name = args.model_save_dir
            else:
                dir_name = "../data/saved_models/"
            torch.save(model.state_dict(), "{}Find-Module-pt_{}.pt".format(dir_name, args.experiment_name))
            best_eval_loss = dev_avg_loss

        epoch_losses.append((train_avg_loss, train_avg_find_loss, train_avg_sim_loss, dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_scores))

        print(epoch_losses)

    epoch_string = str(epochs)
    with open("../data/result_data/loss_per_epoch_Find-Module-pt_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_loss','train_find_loss', 'train_sim_loss', 'dev_loss', 'dev_find_loss', 'dev_sim_loss', 'dev_f1_score'])
        for row in epoch_losses:
            writer.writerow(row)

if __name__ == "__main__":
    main()
