import torch
from transformers import AdamW, DistilBertTokenizerFast
import sys
sys.path.append(".")
sys.path.append("../")
from transformer_util_functions import create_pre_training_data, build_l_find_data_loader, evaluate_find_module
from models import Find_Module_3
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
                        default=32,
                        type=int,
                        help="Total batch size for train.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Number of Epochs for training")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help="weight of sim_loss")
    parser.add_argument('--encoding_dim',
                        type=int,
                        default=300,
                        help="embedding vector size")
    parser.add_argument('--encoding_dropout',
                        type=float,
                        default=0.1,
                        help="encoding dropout")
    parser.add_argument('--model_save_dir',
                        type=str,
                        default="",
                        help="where to save the model")
    parser.add_argument('--experiment_name',
                        type=str,
                        help="what to save the model file as")
    parser.add_argument('--load_model',
                        action='store_true',
                        help="Whether to load a model")
    parser.add_argument('--start_epoch',
                         type=int,
                         default=0,
                         help="start_epoch")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.build_pre_train:
        create_pre_training_data(tokenizer, 
                                 args.train_path,
                                 args.dev_path,
                                 args.test_path,
                                 args.explanation_data_path)

    train_dataloader = build_l_find_data_loader("../data/pre_train_data/tacred_transformer_train.p", batch_size=args.train_batch_size)
    
    dev_path = "../data/pre_train_data/tacred_transformer_ziqi_eval.p"
    train_eval_path = "../data/pre_train_data/tacred_transformer_train_eval.p"
    
    with open("../data/pre_train_data/tacred_transformer_sim_data.p", "rb") as f:
        sim_data = pickle.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Find_Module_3.Find_Module(encoding_dim=args.encoding_dim, cuda=torch.cuda.is_available(),
                                      freeze_model=False, encoding_dropout=args.encoding_dropout)
    
    if args.load_model:
        model.load_state_dict(torch.load("../data/saved_models/Find-Module-Transformer-pt_{}.p".format(args.experiment_name)))
        print("loaded model")
    
    model = model.to(device)

    for key in sim_data:
        sim_data[key] = sim_data[key].to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)   

    # define loss functions
    find_loss_function  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    sim_loss_function = nn.TripletMarginLoss(margin=1.0, p=2)

    # number of training epochs
    epochs = args.epochs

    epoch_losses = []
    train_epoch_losses = []
    best_f1_score = -1
    best_train_f1_score = -1
    best_dev_loss = float('inf') 

    for epoch in range(args.start_epoch, args.start_epoch+epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, args.start_epoch+epochs))

        total_loss, find_total_loss, sim_total_loss = 0, 0, 0
        batch_count = 0
        model.train()
        # iterate over batches
        for step, batch in enumerate(tqdm(train_dataloader)):
            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            seq, seq_mask, query, query_mask, labels = batch

            # clear previously calculated gradients 
            model.zero_grad()        

            # get model predictions for the current batch
            token_scores = model.find_forward(seq, seq_mask, query, query_mask)
            anchor_vectors = model.get_normalized_pooled_encodings(sim_data["anchor_queries"],
                                                                   sim_data["anchor_query_masks"]).squeeze(1)
            positive_vectors = model.get_normalized_pooled_encodings(sim_data["positive_queries"],
                                                                     sim_data["positive_query_masks"]).squeeze(1)
            negative_vectors = model.get_normalized_pooled_encodings(sim_data["negative_queries"],
                                                                     sim_data["negative_query_masks"]).squeeze(1)


            # compute the loss between actual and predicted values
            find_loss = find_loss_function(token_scores, labels)
            sim_loss = sim_loss_function(anchor_vectors, positive_vectors, negative_vectors)
            string_loss = find_loss + args.gamma * sim_loss

            # add on to the total loss
            find_total_loss = find_total_loss  + find_loss.item()
            sim_total_loss = sim_total_loss + sim_loss.item()
            total_loss = total_loss + string_loss.item()
            batch_count += 1

            if batch_count % 300 == 0 and batch_count > 0:
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


        print("Starting Evaluation")
        eval_results = evaluate_find_module(dev_path, sim_data, model,
                                            find_loss_function, sim_loss_function,
                                            args.eval_batch_size, args.gamma)
        dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score, total_og_scores, total_new_scores = eval_results
        print("Finished Evaluation")
        
        if dev_f1_score > best_f1_score or (dev_f1_score == best_f1_score and dev_avg_loss < best_dev_loss):
            print("Saving Model")
            if len(args.model_save_dir) > 0:
                dir_name = args.model_save_dir
            else:
                dir_name = "../data/saved_models/"
            torch.save(model.state_dict(), "{}Find-Module-Transformer-pt_{}.p".format(dir_name, args.experiment_name))
            with open("../data/result_data/best_dev_total_og_scores_{}.p".format(args.experiment_name), "wb") as f:
                pickle.dump(total_og_scores, f)
            with open("../data/result_data/best_dev_total_new_scores_{}.p".format(args.experiment_name), "wb") as f:
                pickle.dump(total_new_scores, f)
            best_f1_score = dev_f1_score
            best_dev_loss = dev_avg_loss

        epoch_losses.append((train_avg_loss, train_avg_find_loss, train_avg_sim_loss,
                             dev_avg_loss, dev_avg_find_loss, dev_avg_sim_loss, dev_f1_score))
        print("Best Dev F1: {}".format(str(best_f1_score)))
        print(epoch_losses[-3:])

        print("Starting Train Evaluation")
        eval_results = evaluate_find_module(train_eval_path, sim_data, model,
                                            find_loss_function, sim_loss_function,
                                            args.eval_batch_size, args.gamma)
        train_eval_avg_loss, train_eval_avg_find_loss, train_eval_avg_sim_loss, train_eval_f1_score, total_og_scores, total_new_scores = eval_results
        print("Finished Evaluation")

        if train_eval_f1_score > best_train_f1_score:
            best_train_f1_score = train_eval_f1_score
            with open("../data/result_data/best_train_eval_total_og_scores_{}.p".format(args.experiment_name), "wb") as f:
                pickle.dump(total_og_scores, f)
            with open("../data/result_data/best_train_eval_total_new_scores_{}.p".format(args.experiment_name), "wb") as f:
                pickle.dump(total_new_scores, f)
        
        train_epoch_losses.append((train_eval_avg_loss, train_eval_avg_find_loss, train_eval_avg_sim_loss,
                                   train_eval_f1_score))
        print("Best Train F1: {}".format(str(best_train_f1_score)))
        print(train_epoch_losses[-3:])

    epoch_string = str(epochs)
    with open("../data/result_data/loss_per_epoch_Find-Module-Transformer-pt_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(['train_loss','train_find_loss', 'train_sim_loss', 'dev_loss', 'dev_find_loss', 'dev_sim_loss', 'dev_f1_score'])
        for row in epoch_losses:
            writer.writerow(row)
    
    with open("../data/result_data/train_eval_loss_per_epoch_Find-Module-Transformer-pt_{}.csv".format(args.experiment_name), "w") as f:
        writer=csv.writer(f)
        writer.writerow(["train_eval_avg_loss", "train_eval_avg_find_loss", "train_eval_avg_sim_loss", "train_eval_f1_score"])
        for row in train_epoch_losses:
            writer.writerow(row)

if __name__ == "__main__":
    main()
