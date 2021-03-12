import sys
sys.path.append(".")
sys.path.append("../")
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import random
import pickle
from training.util_classes import PreTrainingFindModuleDataset
from training.util_functions import find_array_start_position, generate_save_string, tokenize,\
                             build_vocab, convert_text_to_tokens, extract_queries_from_explanations,\
                             build_custom_vocab
from tqdm import tqdm
import re
import numpy as np
import pdb

possible_embeddings = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

def build_synthetic_pretraining_triples(data, vocab, tokenize_fn, custom_vocab={}):
    """
        As per the NExT paper, we build a pre-training dataset from a dataset of unlabeled text.
        The process is as follows per sequence of text (Seq):
            1. Tokenize the text
            2. Convert tokens into token_ids
            3. Select a random number (N) between 1 and 5 for the number of tokens that make up a query (Q)
            4. Select a starting position in the sequence (S)
            5. Extract the tokens [S:S+N], this is our query sequence Q
            6. Label each token in Seq with a 1 or 0, indicating whether the token is in Q or not

        As a result of this process we build the triple (Seq, Q, labels) that will be used in pre-training

        Arguments:
            data              (arr) : sequences of text
            vocab (torchtext.vocab) : vocabulary object
            tokenize_fun (function) : function to use to break up text into tokens
        
        Returns:
            tokenized seqs, queries, labels : triplet where each element is a list of equal length
                                              containing the information described above
    """
    token_seqs = convert_text_to_tokens(data, vocab, tokenize_fn, custom_vocab)
    token_seqs = [token_seq for token_seq in token_seqs if len(token_seq) > 3]
    queries = []
    labels = []
    for i, token_seq in enumerate(token_seqs):
        num_tokens = random.randint(1, min(len(token_seq), 5))
        starting_position = random.randint(0, len(token_seq)-num_tokens)
        end_position = starting_position + num_tokens
        queries.append(token_seq[starting_position:end_position])
        label_seq = []
        for i in range(len(token_seq)):
            if i >= starting_position and i < end_position:
                label_seq.append(1.0)
            else:
                label_seq.append(0.0)
        labels.append(label_seq)
    
    return token_seqs, queries, labels

def build_real_pretraining_triples(sentences, queries, vocab, tokenize_fn, custom_vocab={}):
    """
        To evaluate a model against real explanations

        Arguments:
            sentences (arr) : original sentences that an explanation is about
            queries   (arr) : queries from explanations
            vocab (torchtext.vocab) : vocabulary object
            tokenize_fun (function) : function to use to break up text into tokens
        
        Returns:
            tokenized seqs, queries, labels : triplet where each element is a list of equal length
                                              containing similar information to `build_synthetic_pretraining_triples`
    """
    tokenized_sentences = convert_text_to_tokens(sentences, vocab, tokenize_fn, custom_vocab)
    tokenized_queries = convert_text_to_tokens(queries, vocab, tokenize_fn)
    labels = []
    indices_to_delete = []
    for i, tokenized_sentence in enumerate(tokenized_sentences):
        tokenized_query = tokenized_queries[i]
        sent_labels = [0] * len(tokenized_sentence)
        start_position = find_array_start_position(tokenized_sentence, tokenized_query)
        if start_position > 0:
            sent_labels[start_position:start_position+len(tokenized_query)] = [1] * len(tokenized_query)
            labels.append(sent_labels)
        else:
            indices_to_delete.append(i)
    
    indices_to_delete.reverse()
    
    for i in indices_to_delete:
        del tokenized_sentences[i]
        del tokenized_queries[i]

    return tokenized_sentences, tokenized_queries, labels

def build_variable_length_text_pre_training_dataset(data, vocab, split_name, save_string, custom_vocab={}):
    """
        Given a split of data (train, dev, test) this function builds a PreTrainingFindModuleDataset and saves
        it to disk. A VariableLegnthTextDataset object handles batching sequences together and ensuring
        every input is of the same length (the length of the max sequence length in a batch).

        Arguments:
            data              (arr) : split of data that needs to be processed
            vocab (torchtext.vocab) : vocab object used for conversion between text token and token_id
            split_name        (str) : name of split (used for naming)
            save_string       (str) : string to indicate some of the hyper-params used to create the vocab
    """
    pad_idx = vocab["<pad>"]
    token_seqs, queries, labels = build_synthetic_pretraining_triples(data, vocab, tokenize, custom_vocab)
    dataset = PreTrainingFindModuleDataset(token_seqs, queries, labels, pad_idx)

    print("Finished building {} dataset of size: {}".format(split_name, str(len(token_seqs))))

    file_name = "../data/pre_train_data/{}_data_{}.p".format(split_name, save_string)

    with open(file_name, "wb") as f:
        pickle.dump(dataset, f)


def tokenize_explanation_queries(explanation_data, vocab, label_filter, save_string):
    """
        Given a list of explanations for labeling decisions, we find those explanations that include phrases
        that must exist in a text-sequence for a label to be applied to the text sequence.

            Ex: The text contains the phrase "xyz"

        We then tokenize and convert the phrases within quotes (queries) into sequence of token_ids that will
        be used at training time to try and push embeddings of queries associated with the same label closer
        together.

        Arguments:
            explanation_data  (arr) : array of natural language explanations for labeling decisions
            vocab (torchtext.vocab) : vocab object used for conversion between text token and token_id
            label_filter      (arr) : labels to consider when extracting queries from explanations
                                      (allows user to ignore explanations associated with certain labels)
            save_string       (str) : string to indicate some of the hyper-params used to create the vocab
    """
    queries = []
    labels = []
    for entry in explanation_data:
        explanation = entry["explanation"]
        label = entry["label"]
        if label_filter is None or label in label_filter:
            possible_queries = extract_queries_from_explanations(explanation)
            for query in possible_queries:
                queries.append(query)
                labels.append(label)

    tokenized_queries = convert_text_to_tokens(queries, vocab, tokenize)

    print("Finished tokenizing actual queries, count: {}".format(str(len(tokenized_queries))))

    file_name = "../data/pre_train_data/sim_data_{}.p".format(save_string)

    with open(file_name, "wb") as f:
        pickle.dump({"queries" : tokenized_queries, "labels" : labels}, f)

def build_real_query_eval_dataset(explanation_data, vocab, label_filter, dataset_name, save_string, custom_vocab={}):
    """
        As an evaluation set, we take an real natural language explanation and check if it has a
        quoted phrase in it. If it does, then we build an evaluation based on the sentence the 
        explanation is about. The evaluation is to see if the machine can detect the quoted phrase of
        the explanation in the sentence its quoted from.

        Arguments:
            explanation_data  (arr) : array of natural language explanations for labeling decisions
            vocab (torchtext.vocab) : vocab object used for conversion between text token and token_id
            label_filter      (arr) : labels to consider when extracting queries from explanations
                                      (allows user to ignore explanations associated with certain labels)
            dataset_name      (str) : original dataset name, indicating what these explanations are
                                      associated with
            save_string       (str) : string to indicate some of the hyper-params used to create the vocab 
    """
    sentences = []
    queries = []
    for entry in explanation_data:
        sentence = entry["sent"]
        explanation = entry["explanation"]
        label = entry["label"]
        if label_filter is None or label in label_filter:
            possible_queries = extract_queries_from_explanations(explanation)
            for query in possible_queries:
                queries.append(query)
                sentences.append(sentence)
    
    output = build_real_pretraining_triples(sentences, queries, vocab, tokenize, custom_vocab)

    tokenized_sentences, tokenized_queries, labels = output
    
    eval_dataset_2 = PreTrainingFindModuleDataset(tokenized_sentences, tokenized_queries, labels, vocab["<pad>"])

    file_name = "../data/pre_train_data/rq_data_{}.p".format(save_string)

    with open(file_name, "wb") as f:
        pickle.dump(eval_dataset_2, f)

def build_pre_train_find_datasets(text_data, explanation_data, save_string, embedding_name="glove.840B.300d",
                                  random_state=42, dataset="tacred", sample_rate=-1.0,
                                  label_filter=None):
    """
        As per the NExT paper, we build train, dev and test datasets to allow for the pre-training and
        evaluation of the FIND module.

        Steps taken:
            1. Load unlabeled data
            2. If only a sample of the data is to be used, we sample the data
            3. Split data into train and dev splits
            4. Build a vocabulary object using the train split
            5. Build needed datasets for computing L_find loss (for each split)
            6. Build needed datasets for computing L_sim loss
        
        Arguments:
            text_data        (arr) : array of text data
            explanation_data (arr) : array of dictionaries holding explanaiton data
            embedding_name   (str) : name of pre-trained embeddings being used in vocab
            random_state     (int) : random seed to use when splitting data into train, dev, test splits
            label_filter     (arr) : labels to consider when extracting queries from explanations
                                     (allows user to ignore explanations associated with certain labels)
            sample_rate    (float) : percentage of unlabeled data to use when building datasets for L_find
            dataset          (str) : name of the dataset explanations come from

    """
    if not embedding_name in possible_embeddings:
        print("Not Valid Embedding Option")
        return
    
    if sample_rate > 0:
        sample_number = int(len(text_data) * sample_rate)
        text_data = random.sample(text_data, sample_number)
    
    train, dev = train_test_split(text_data, train_size=0.8, random_state=random_state)

    vocab = build_vocab(train, embedding_name, save_string)

    custom_vocab = build_custom_vocab(dataset, vocab_length=len(vocab))

    build_variable_length_text_pre_training_dataset(train, vocab, "train", save_string, custom_vocab)

    build_variable_length_text_pre_training_dataset(dev, vocab, "dev", save_string, custom_vocab)

    tokenize_explanation_queries(explanation_data, vocab, label_filter, save_string)

    build_real_query_eval_dataset(explanation_data, vocab, label_filter, dataset, save_string, custom_vocab)

def build_pre_train_find_datasets_from_splits(train_path, dev_path, test_path, explanation_path,
                                              embedding_name="glove.840B.300d", label_filter=None, sample_rate=-1.0,
                                              dataset="tacred"):
    """
        Provided pre-split data, follow the steps taken in build_pre_train_find_datasets

        Splits are assumed to be json file, where the only element is an array of text

        Arguments:
            train_path       (str) : path to training split of data
            dev_path         (str) : path to dev split of data
            test_path        (str) : path to test split of data
            explanation_path (str) : path to explanation data
            embedding_name   (str) : name of pre-trained embeddings being used in vocab
            label_filter     (arr) : labels to consider when extracting queries from explanations
                                     (allows user to ignore explanations associated with certain labels)
            sample_rate    (float) : percentage of unlabeled data to use when building datasets for L_find
            dataset          (str) : name of the dataset explanations come from
    """

    with open(train_path) as f:
        train = json.load(f)
        train = [ent["text"] for ent in train]
    
    train_sample = None
    if sample_rate > 0:
        sample_number = int(len(train) * sample_rate)
        train_sample = random.sample(train, sample_number)
    
    save_string = generate_save_string(dataset, embedding_name, sample=sample_rate)

    vocab = build_vocab(train, embedding_name, save_string)

    custom_vocab = build_custom_vocab(dataset, vocab_length=len(vocab))

    if train_sample:
        build_variable_length_text_pre_training_dataset(train_sample, vocab, "train", save_string, custom_vocab)
    else:
        build_variable_length_text_pre_training_dataset(train, vocab, "train", save_string, custom_vocab)

    with open(dev_path) as f:
        dev = json.load(f)
        dev = [ent["text"] for ent in dev]

    build_variable_length_text_pre_training_dataset(dev, vocab, "dev", save_string, custom_vocab)

    with open(test_path) as f:
        test = json.load(f)
        test = [ent["text"] for ent in test]
    
    build_variable_length_text_pre_training_dataset(test, vocab, "test", save_string, custom_vocab)

    with open(explanation_path) as f:
        explanation_data = json.load(f)

    tokenize_explanation_queries(explanation_data, vocab, label_filter, save_string)

    build_real_query_eval_dataset(explanation_data, vocab, label_filter, dataset, save_string, custom_vocab)

def evaluate_find_module(data_path, act_queries, query_index_matrix, neg_query_index_matrix, lower_bound,
                         model, find_loss_fn, sim_loss_fn, batch_size=128, gamma=0.5):
    """
        Evaluates a Find Module model against a dataset

        Arguments:
            data_path            (str) : path to PreTrainingFindModuleDataset that the model should be
                                         evaluated against
            act_queries (torch.tensor) : queries to be used for computing L_sim, dim: (n, max_len)
            query_labels         (arr) : labels associated with queries
            model        (Find_Module) : model to use in evaluation
            find_loss_fn        (func) : loss function for L_find
            sim_loss_fn         (func) : loss function for L_sim
            batch_size           (int) : size of batch to use when computing L_find
            gamma              (float) : weight associated with L_sim
        
        Returns:
            avg_loss, avg_find_loss, avg_sim_loss, avg_f1_score : average of metrics computed per batch
    """
    with open(data_path, "rb") as f:
        eval_dataset = pickle.load(f)
    
    # deactivate dropout layers
    model.eval()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    total_loss, total_find_loss, total_sim_loss, total_f1_score = 0, 0, 0, 0
    total_og_scores = []
    total_new_scores = []
    batch_count = 0

    # iterate over batches
    for step, batch in enumerate(tqdm(eval_dataset.as_batches(batch_size=batch_size, shuffle=False))):
        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        tokens, queries, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            token_scores = model.find_forward(tokens, queries, lower_bound)
            pos_scores, neg_scores = model.sim_forward(act_queries, query_index_matrix, neg_query_index_matrix)

            # compute the validation loss between actual and predicted values
            find_loss = find_loss_fn(token_scores, labels)
            sim_loss = sim_loss_fn(pos_scores, neg_scores)
            string_loss = find_loss + gamma * sim_loss
            
            scores = token_scores.detach().cpu().numpy().flatten()
            new_scores = [1 if score > 0  else 0 for score in scores]
            f1_labels = labels.detach().cpu().numpy().flatten()

            total_loss = total_loss + string_loss.item()
            total_find_loss = total_find_loss + find_loss.item()
            total_sim_loss = total_sim_loss + sim_loss.item()
            total_f1_score = total_f1_score + f1_score(f1_labels, new_scores)
            batch_count += 1
            total_og_scores.append(scores)
            total_new_scores.append(new_scores)

    # compute the validation loss of the epoch
    avg_loss = total_loss / batch_count
    avg_find_loss = total_find_loss / batch_count
    avg_sim_loss = total_sim_loss / batch_count
    avg_f1_score = total_f1_score / batch_count
    total_og_scores  = np.concatenate(total_og_scores, axis=0)
    total_new_scores  = np.concatenate(total_new_scores, axis=0)

    return avg_loss, avg_find_loss, avg_sim_loss, avg_f1_score, total_og_scores, total_new_scores