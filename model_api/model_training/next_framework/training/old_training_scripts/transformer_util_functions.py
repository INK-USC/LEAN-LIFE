import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import util_functions as util_f
import json
import random

MAX_QUERY_LENGTH = 5
BOS_TOKEN = 101
EOS_TOKEN = 102

def batch_encode_text(text, tokenizer, max_length):
    output = tokenizer.batch_encode_plus(
        text,
        max_length = max_length,
        padding=True,
        truncation=True
    )

    tokens = torch.tensor(output['input_ids'])
    attn_mask = torch.tensor(output['attention_mask'])

    return tokens, attn_mask

def extract_query_from_token_seqs(token_seqs):
    queries = []
    labels = []
    for i, token_seq in enumerate(token_seqs):
        num_tokens = random.randint(1, min(len(token_seq)-2, MAX_QUERY_LENGTH)) # don't want to select BOS or EOS token
        starting_position = random.randint(1, len(token_seq)-num_tokens-1) # don't want to select BOS or EOS token
        end_position = starting_position + num_tokens
        queries.append(token_seq[starting_position:end_position])
        label_seq = []
        for i in range(len(token_seq)):
            if i >= starting_position and i < end_position:
                label_seq.append(1.0)
            else:
                label_seq.append(0.0)
        labels.append(label_seq)
    
    query_masks = []
    for i, query in enumerate(queries):
        updated_query = [BOS_TOKEN]
        query_mask = [1]
        j = 0
        while j < len(query):
            query_mask.append(1)
            updated_query.append(query[j])
            j += 1
        query_mask.append(1)
        updated_query.append(EOS_TOKEN)

        while j < MAX_QUERY_LENGTH+2:
            query_mask.append(0)
            updated_query.append(0)
            j += 1
        
        queries[i] = updated_query
        query_masks.append(query_mask)
    
    queries = torch.tensor(queries)
    query_masks = torch.tensor(query_masks)
    labels = torch.tensor(labels)
    
    return queries, query_masks, labels

def create_l_find_data(tokenizer, max_length, split, split_name):
    seqs, attn_masks = batch_encode_text(split, tokenizer, max_length)
    queries, query_attn_masks, labels = extract_query_from_token_seqs(seqs)
    
    print("{} Data: {}".format(split_name, str(len(seqs))))

    with open("../data/pre_train_data/tacred_transformer_{}.p".format(split_name), "wb") as f:
        data = {
            "seq" : seqs,
            "seq_mask" : attn_masks,
            "query" : queries,
            "query_mask" : query_attn_masks,
            "labels" : labels
        }
        pickle.dump(data, f)

def create_l_sim_data(tokenizer, explanation_data):
    queries = []
    labels = []
    for entry in explanation_data:
        explanation = entry["explanation"]
        label = entry["label"]
        possible_queries = util_f.extract_queries_from_explanations(explanation)
        for query in possible_queries:
            queries.append(query)
            labels.append(label)
    
    queries, query_attn_masks = batch_encode_text(queries, tokenizer, max_length=MAX_QUERY_LENGTH+2)

    label_to_index = {}
    for i, label in enumerate(labels):
        if label in label_to_index:
            label_to_index[label].append(i)
        else:
            label_to_index[label] = [i]
    
    all_labels = list(label_to_index.keys())
    
    anchor_queries = []
    anchor_query_masks = []
    positive_queries = []
    positive_query_masks = []
    negative_queries = []
    negative_query_masks = []

    for label in label_to_index:
        indices = label_to_index[label]
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                anchor_index = indices[i]
                positive_index = indices[j]
                anchor_queries.append(queries[anchor_index].tolist())
                anchor_query_masks.append(query_attn_masks[anchor_index].tolist())
                positive_queries.append(queries[positive_index].tolist())
                positive_query_masks.append(query_attn_masks[positive_index].tolist())
                while(True):
                    random_label = random.sample(all_labels, 1)[0]
                    if random_label != label:
                        negative_index = random.sample(label_to_index[random_label], 1)[0]
                        negative_queries.append(queries[negative_index].tolist())
                        negative_query_masks.append(query_attn_masks[negative_index].tolist())
                        break
    
    anchor_queries = torch.tensor(anchor_queries)
    anchor_query_masks = torch.tensor(anchor_query_masks)
    positive_queries = torch.tensor(positive_queries)
    positive_query_masks = torch.tensor(positive_query_masks)
    negative_queries = torch.tensor(negative_queries)
    negative_query_masks = torch.tensor(negative_query_masks)

    print("Sim Data: {}".format(str(len(anchor_queries))))

    with open("../data/pre_train_data/tacred_transformer_sim_data.p", "wb") as f:
        data = {
            "anchor_queries" : anchor_queries,
            "anchor_query_masks" : anchor_query_masks,
            "positive_queries" : positive_queries,
            "positive_query_masks" : positive_query_masks,
            "negative_queries" : negative_queries,
            "negative_query_masks" : negative_query_masks
        }
        pickle.dump(data, f)

def create_ziqi_eval_data(tokenizer, explanation_data, max_length):
    sentences = []
    queries = []
    for entry in explanation_data:
        sentence = entry["sent"]
        explanation = entry["explanation"]
        label = entry["label"]
        possible_queries = util_f.extract_queries_from_explanations(explanation)
        for query in possible_queries:
            queries.append(query)
            sentences.append(sentence)
    
    sentence_tokens, sentence_attn_mask = batch_encode_text(sentences, tokenizer, max_length)
    query_tokens, query_attn_mask = batch_encode_text(queries, tokenizer, MAX_QUERY_LENGTH+2)

    stripped_query_tokens = [[tok for tok in query if tok != 0] for query in query_tokens]
    stripped_query_tokens = [query[1:len(query)-1] for query in stripped_query_tokens]
    
    labels = []
    for i, sentence in enumerate(sentence_tokens):
        seq_label = []
        start_position = util_f.find_array_start_position(sentence, stripped_query_tokens[i])
        end_position = start_position + len(stripped_query_tokens[i])
        for j in range(len(sentence)):
            if j >= start_position and j < end_position:
                seq_label.append(1.0)
            else:
                seq_label.append(0.0)
        labels.append(seq_label)

    labels = torch.tensor(labels)
    
    print("Ziqi Eval Dataset: {}".format(str(len(sentence_tokens))))

    with open("../data/pre_train_data/tacred_transformer_ziqi_eval.p", "wb") as f:
        data = {
            "seq" : sentence_tokens,
            "seq_mask" : sentence_attn_mask,
            "query" : query_tokens,
            "query_mask" : query_attn_mask,
            "labels" : labels
        }
        pickle.dump(data, f)

def create_pre_training_data(tokenizer, train_path, dev_path, test_path, explanation_path,
                             max_length=110, clean=False, train_pct=0.6):
    with open(train_path, encoding='utf-8') as f:
        train_text = json.load(f)

    with open(dev_path, encoding='utf-8') as f:
        dev_text = json.load(f)

    with open(test_path, encoding='utf-8') as f:
        test_text = json.load(f)
    
    with open(explanation_path, encoding='utf-8') as f:
        explanation_data = json.load(f)
    
    if clean == True:
        train_text = [" ".join(util_f.tokenize(sent)) for sent in train_text if len(sent) > 3]
        dev_text = [" ".join(util_f.tokenize(sent)) for sent in dev_text if len(sent) > 3]
        test_text = [" ".join(util_f.tokenize(sent)) for sent in test_text if len(sent) > 3]
    else:
        train_text = [" ".join(sent) for sent in train_text if len(sent) > 3]
        dev_text = [" ".join(sent) for sent in dev_text if len(sent) > 3]
        test_text = [" ".join(sent) for sent in test_text if len(sent) > 3]
    
    train_sample = random.sample(train_text, k=int(len(train_text)*train_pct))

    eval_train_sample = random.sample(train_sample, k=int(len(train_sample)*0.1))
    
    create_l_find_data(tokenizer, max_length, train_sample, "train")
    create_l_find_data(tokenizer, max_length, dev_text, "dev")
    create_l_find_data(tokenizer, max_length, test_text, "test")
    create_l_find_data(tokenizer, max_length, eval_train_sample, "train_eval")
    
    create_ziqi_eval_data(tokenizer, explanation_data, max_length)

    create_l_sim_data(tokenizer, explanation_data)
     
def build_l_find_data_loader(path, batch_size, train=True):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    # wrap tensors
    dataset = TensorDataset(data_dict["seq"],
                            data_dict["seq_mask"],
                            data_dict["query"],
                            data_dict["query_mask"],
                            data_dict["labels"])

    # sampler for sampling the data during training
    if train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # dataLoader for train set
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader

def evaluate_find_module(data_path, explanation_data, model, find_loss_fn, sim_loss_fn, batch_size=32, gamma=0.5):
    
    dataloader = build_l_find_data_loader(data_path, batch_size=batch_size, train=False)
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
    for step, batch in enumerate(tqdm(dataloader)):

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        seq, seq_mask, query, query_mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            token_scores = model.find_forward(seq, seq_mask, query, query_mask)
            anchor_vectors = model.create_pooled_encodings(explanation_data["anchor_queries"],
                                                           explanation_data["anchor_query_masks"]).squeeze(1)
            positive_vectors = model.create_pooled_encodings(explanation_data["positive_queries"],
                                                             explanation_data["positive_query_masks"]).squeeze(1)
            negative_vectors = model.create_pooled_encodings(explanation_data["negative_queries"],
                                                             explanation_data["negative_query_masks"]).squeeze(1)

            # compute the validation loss between actual and predicted values
            find_loss = find_loss_fn(token_scores, labels)
            sim_loss = sim_loss_fn(anchor_vectors, positive_vectors, negative_vectors)
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