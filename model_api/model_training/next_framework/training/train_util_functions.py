import pickle
import json
import random
from training.util_functions import generate_save_string, convert_text_to_tokens, tokenize, extract_queries_from_explanations, clean_text, build_vocab, build_custom_vocab, load_spacy_to_custom_dataset_ner_mapping
from training.constants import PARSER_TRAIN_SAMPLE, UNMATCH_TYPE_SCORE, TACRED_LABEL_MAP, DEV_F1_SAMPLE
from training.util_classes import BaseVariableLengthDataset, UnlabeledTrainingDataset, TrainingDataset
import sys
sys.path.append(".")
sys.path.append("../")
from CCG.parser import CCGParserTrainer
from CCG.utils import generate_phrase
from CCG.soft_grammar_functions import NER_LABEL_SPACE
import spacy
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
import dill
import pdb
from tqdm import tqdm
import numpy as np

nlp = spacy.load("en_core_web_sm")
MODEL_API_PATH = "../model_training/next_framework/"

def batch_type_restrict_re(relation, phrase_inputs, relation_ner_types):
    entity_types = relation_ner_types[relation]
    entity_ids = (NER_LABEL_SPACE[entity_types[0]], NER_LABEL_SPACE[entity_types[1]])
    restrict_subj = torch.eq(torch.reshape(phrase_inputs[:,-2],[1,-1]), entity_ids[0]).float()
    restrict_obj = torch.eq(torch.reshape(phrase_inputs[:,-1],[1,-1]), entity_ids[1]).float()
    restrict_subj = restrict_subj+(1.0-restrict_subj)*UNMATCH_TYPE_SCORE
    restrict_obj = restrict_obj+(1.0-restrict_obj)*UNMATCH_TYPE_SCORE
    
    return restrict_subj*restrict_obj

def build_phrase_input(phrases, pad_idx, task):
    tokens = [phrase.tokens for phrase in phrases]
    ners = [phrase.ners for phrase in phrases]
    subj_posis = torch.tensor([phrase.subj_posi for phrase in phrases]).unsqueeze(1)
    obj_posis =  torch.tensor([phrase.obj_posi for phrase in phrases]).unsqueeze(1)
    if task == "re":
        lengths = torch.tensor([len(token_seq) for token_seq in tokens]).unsqueeze(1)
        subj_bool = subj_posis < lengths
        obj_bool = obj_posis < lengths
        assert sum(subj_bool).item() == len(phrases)
        assert sum(obj_bool).item() == len(phrases)
        subj =  torch.tensor([phrase.ners[phrase.subj_posi] for phrase in phrases]).unsqueeze(1) # has to be NERs due to type check
        obj = torch.tensor([phrase.ners[phrase.obj_posi] for phrase in phrases]).unsqueeze(1)
    else:
        no_ner_id = NER_LABEL_SPACE[""]
        subj = torch.full((len(phrases),1), no_ner_id)
        obj = torch.full((len(phrases),1), no_ner_id)

    ner_pad = NER_LABEL_SPACE["<PAD>"]

    tokens, _ = BaseVariableLengthDataset.variable_length_batch_as_tensors(tokens, pad_idx)
    ners, _ = BaseVariableLengthDataset.variable_length_batch_as_tensors(ners, ner_pad)

    assert tokens.shape == ners.shape

    phrase_input = torch.cat([tokens, ners, subj_posis, obj_posis, subj, obj], dim=1)

    return phrase_input

def build_mask_mat_for_batch(seq_length):
    mask_mat = torch.zeros((seq_length, seq_length, seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            mask_mat[i,j,i:j+1] = 1
    mask_mat = mask_mat.float()

    return mask_mat

def _prepare_labels(labels, label_map):
    converted_labels = []
    for label in labels:
        converted_labels.append(label_map[label])
    return converted_labels

def _prepare_unlabeled_data(unlabeled_data_phrases, vocab, spacy_to_custom_ner={}, custom_vocab={}):
    seq_tokens = []
    seq_phrases = []
    for phrase in unlabeled_data_phrases:
        tokens = phrase.tokens
        ners = phrase.ners

        if phrase.subj_posi != None:
            tokens[phrase.subj_posi] = "SUBJ-{}".format(ners[phrase.subj_posi])
        if phrase.obj_posi != None:
            tokens[phrase.obj_posi] = "OBJ-{}".format(ners[phrase.obj_posi])

        ners = [spacy_to_custom_ner[ner] if ner in spacy_to_custom_ner else ner for ner in ners]
        
        tokens = [custom_vocab[token] if token in custom_vocab else vocab[token] for token in tokens]
        ners = [NER_LABEL_SPACE[ner] for ner in ners]
        
        seq_tokens.append(tokens)
        phrase.update_tokens_and_ners(tokens, ners) # its okay to update with custom vocab, hard-matching is done
        seq_phrases.append(phrase)
    
    return seq_tokens, seq_phrases

def create_parser(parser_training_data, explanation_path, task="re", explanation_data=None):
    parser_trainer = None
    
    if task == "re":
        parser_trainer = CCGParserTrainer(task explanation_path, "", parser_training_data, explanation_data)
    elif task == "sa":
        parser_trainer = CCGParserTrainer(task, explanation_path, "", parser_training_data, explanation_data)
    
    parser_trainer.train()
    parser = parser_trainer.get_parser()

    with open(MODEL_API_PATH + "data/training_data/parser_debug.p", "wb") as f:
        dill.dump(parser, f)

    return parser

def match_training_data(labeling_functions, train, task, function_ner_types={}):

    phrases = [generate_phrase(entry, nlp) for entry in train]

    with open(MODEL_API_PATH + "data/training_data/train_phrases_debug.p", "wb") as f:
        pickle.dump(phrases, f)

    # with open(MODEL_API_PATH + "data/training_data/train_phrases_debug.p", "rb") as f:
    #     phrases = pickle.load(f)

    matched_data_tuples = []
    matched_indices = []
    unlabeled_data_phrases = []

    for i, phrase in enumerate(phrases):
        not_matched = True
        for j, function in enumerate(labeling_functions):
            try:
                if function(phrase):
                    if task == "re":
                        subj_type = phrase.ners[phrase.subj_posi].lower()
                        obj_type = phrase.ners[phrase.obj_posi].lower()
                        if function_ner_types[function][0] == subj_type and function_ner_types[function][1] == obj_type:
                            sentence = phrase.sentence.replace("subj", "SUBJ-{}".format(phrase.ners[phrase.subj_posi]))
                            sentence = sentence.replace("obj", "OBJ-{}".format(phrase.ners[phrase.obj_posi]))
                            matched_data_tuples.append((sentence, labeling_functions[function]))
                            matched_indices.append((i, j))
                            not_matched = False
                            break
                    else:
                        matched_data_tuples.append((phrase.sentence, labeling_functions[function]))
                        not_matched = False
                        break
            except:
                continue
        if not_matched:
            unlabeled_data_phrases.append(phrase)
    
    with open(MODEL_API_PATH + "data/training_data/matched_data_tuples_debug.p", "wb") as f:
        pickle.dump(matched_data_tuples, f)

    with open(MODEL_API_PATH + "data/training_data/matched_indices_debug.p", "wb") as f:
        pickle.dump(matched_indices, f)
    
    with open(MODEL_API_PATH + "data/training_data/unlabeled_data_debug.p", "wb") as f:
        pickle.dump(unlabeled_data_phrases, f)

    return matched_data_tuples, unlabeled_data_phrases, matched_indices

def build_unlabeled_dataset(unlabeled_data_phrases, vocab, save_string, spacy_to_custom_ner={}, custom_vocab={}): # check spacy_to_custom_ner
    pad_idx = vocab["<pad>"]
    seq_tokens, seq_phrases = _prepare_unlabeled_data(unlabeled_data_phrases, vocab, spacy_to_custom_ner, custom_vocab)

    dataset = UnlabeledTrainingDataset(seq_tokens, seq_phrases, pad_idx)

    print("Finished building unlabeled dataset of size: {}".format(str(len(seq_tokens))))

    file_name = MODEL_API_PATH + "data/training_data/unlabeled_data_{}.p".format(save_string)

    with open(file_name, "wb") as f:
        pickle.dump(dataset, f)

def build_labeled_dataset(sentences, labels, vocab, save_string, split, label_map, custom_vocab={}): # check build_labeled_dataset
    pad_idx = vocab["<pad>"]
    seq_tokens = convert_text_to_tokens(sentences, vocab, tokenize, custom_vocab)
    
    label_ids = _prepare_labels(labels, label_map)

    dataset = TrainingDataset(seq_tokens, label_ids, pad_idx)

    print("Finished building {} dataset of size: {}".format(split, str(len(seq_tokens))))

    file_name = MODEL_API_PATH + "data/training_data/{}_data_{}.p".format(split, save_string)

    with open(file_name, "wb") as f:
        pickle.dump(dataset, f)

def build_word_to_idx(raw_explanations, vocab, save_string):
    quoted_words = []
    for i, key in enumerate(raw_explanations):
        explanation = raw_explanations[key]
        queries = extract_queries_from_explanations(explanation)
        for query in queries:
            query = " ".join(tokenize(query)).strip()
            quoted_words.append(query)
    
    tokenized_queries = convert_text_to_tokens(quoted_words, vocab, lambda x: x.split())

    print("Finished tokenizing actual queries, count: {}".format(str(len(tokenized_queries))))

    file_name = MODEL_API_PATH + "data/training_data/query_tokens_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        pickle.dump(tokenized_queries, f)

    quoted_words_to_index = {}
    for i, quoted_word in enumerate(quoted_words):
        quoted_words_to_index[quoted_word] = i
    
    file_name = MODEL_API_PATH + "data/training_data/word2idx_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        pickle.dump(quoted_words_to_index, f)

def apply_strict_matching(text_data, explanation_data, task):
    parser_training_data = random.sample(train, min(PARSER_TRAIN_SAMPLE, len(train)))
    
    parser = create_parser(parser_training_data, "", task, explanation_data)

    strict_labeling_functions = parser.labeling_functions

    function_ner_types = parser.ner_types
    
    text_sample = None
    if sample_rate > 0:
        sample_number = int(len(text_data) * sample_rate)
        text_sample = random.sample(text_data, sample_number)

    if text_sample:
        matched_data_tuples, _, matched_indices = match_training_data(strict_labeling_functions, text_sample, task, function_ner_types)
    else:
        matched_data_tuples, _, matched_indices = match_training_data(strict_labeling_functions, text_data, task, function_ner_types)

    return matched_data_tuples, 


def build_datasets_from_text(text_data, vocab_, explanation_data, save_string, label_map, label_filter=None,
                             sample_rate=-1.0, task="re", dataset="tacred"):
    if type(vocab_) == str:
        with open(vocab_, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = build_vocab(train, vocab_["embedding_name"], vocab_["save_string"])
    
    parser_training_data = random.sample(text_data, min(PARSER_TRAIN_SAMPLE, len(train)))
    
    parser = create_parser(parser_training_data, "", task, explanation_data)

    strict_labeling_functions = parser.labeling_functions

    function_ner_types = parser.ner_types
    
    text_sample = None
    if sample_rate > 0:
        sample_number = int(len(text_data) * sample_rate)
        text_sample = random.sample(text_data, sample_number)

    if text_sample:
        matched_data_tuples, unlabeled_data_phrases, _ = match_training_data(strict_labeling_functions, text_sample, task, function_ner_types)
    else:
        matched_data_tuples, unlabeled_data_phrases, _ = match_training_data(strict_labeling_functions, text_data, task, function_ner_types)
    
    custom_vocab = build_custom_vocab(dataset, vocab_length=len(vocab))
    spacy_to_custom_ner_mapping = load_spacy_to_custom_dataset_ner_mapping(dataset)

    build_unlabeled_dataset(unlabeled_data_phrases, vocab, save_string, spacy_to_custom_ner_mapping, custom_vocab)

    build_labeled_dataset([tup[0] for tup in matched_data_tuples], 
                          [tup[1] for tup in matched_data_tuples],
                          vocab, save_string, "matched", label_map, custom_vocab)
    
    filtered_raw_explanations = parser.filtered_raw_explanations

    build_word_to_idx(filtered_raw_explanations, vocab, save_string)

    soft_matching_functions = parser.soft_labeling_functions

    function_labels = _prepare_labels([entry[1] for entry in soft_matching_functions], label_map)

    file_name = MODEL_API_PATH + "data/training_data/labeling_functions_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        dill.dump({"function_pairs" : soft_matching_functions,
                     "labels" : function_labels}, f)

def build_datasets_from_splits(train_path, dev_path, test_path, vocab_, explanation_path, save_string, label_map,
                               label_filter=None, sample_rate=-1.0, task="re", dataset="tacred"):
    
    with open(train_path) as f:
        train = json.load(f)
        train = [entry["text"] for entry in train]
    
    if type(vocab_) == str:
        with open(vocab_, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = build_vocab(train, vocab_["embedding_name"], vocab_["save_string"])

    parser_training_data = random.sample(train, min(PARSER_TRAIN_SAMPLE, len(train)))
    
    # parser = create_parser(parser_training_data, explanation_path, task)

    with open("../data/training_data/parser_debug.p", "rb") as f:
        parser = dill.load(f)
    
    strict_labeling_functions = parser.labeling_functions

    function_ner_types = parser.ner_types
    
    train_sample = None
    if sample_rate > 0:
        sample_number = int(len(train) * sample_rate)
        train_sample = random.sample(train, sample_number)

    # if train_sample:
    #     matched_data_tuples, unlabeled_data_phrases, _ = match_training_data(strict_labeling_functions, train_sample, task, function_ner_types)
    # else:
    #     matched_data_tuples, unlabeled_data_phrases, _ = match_training_data(strict_labeling_functions, train, task, function_ner_types)
    
    with open("../data/training_data/matched_data_tuples_debug.p", "rb") as f:
        matched_data_tuples = pickle.load(f)
    
    with open("../data/training_data/unlabeled_data_debug.p", "rb") as f:
        unlabeled_data_phrases = pickle.load(f)

    custom_vocab = build_custom_vocab(dataset, vocab_length=len(vocab))
    spacy_to_custom_ner_mapping = load_spacy_to_custom_dataset_ner_mapping(dataset)

    build_unlabeled_dataset(unlabeled_data_phrases, vocab, save_string, spacy_to_custom_ner_mapping, custom_vocab)

    build_labeled_dataset([tup[0] for tup in matched_data_tuples], 
                          [tup[1] for tup in matched_data_tuples],
                          vocab, save_string, "matched", label_map, custom_vocab)

    with open(dev_path) as f:
        dev = json.load(f)
    
    build_labeled_dataset([ent["text"] for ent in dev], 
                          [ent["label"] for ent in dev],
                          vocab, save_string, "dev", label_map, custom_vocab)
    
    with open(test_path) as f:
        test = json.load(f)
    
    build_labeled_dataset([ent["text"] for ent in test], 
                          [ent["label"] for ent in test],
                          vocab, save_string, "test", label_map, custom_vocab)

    filtered_raw_explanations = parser.filtered_raw_explanations

    build_word_to_idx(filtered_raw_explanations, vocab, save_string)

    soft_matching_functions = parser.soft_labeling_functions

    function_labels = _prepare_labels([entry[1] for entry in soft_matching_functions], label_map)

    file_name = "../data/training_data/labeling_functions_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        dill.dump({"function_pairs" : soft_matching_functions,
                     "labels" : function_labels}, f)

def _apply_none_label(values, preds, none_label_id, threshold, entropy=True):
    if entropy:
        none_label_mask = values > threshold
    else:
        none_label_mask = values < threshold

    final_preds = [none_label_id if mask else preds[i] for i, mask in enumerate(none_label_mask)]

    return final_preds

def evaluate_next_clf(data_path, model, strict_loss_fn, num_labels, none_label_thresholds=None,
                      batch_size=128, hidden_dim=100, none_label_id=-1):

    with open(data_path, "rb") as f:
        eval_dataset = pickle.load(f)
    
    # deactivate dropout layers
    model.eval()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    h0 = torch.empty(4, batch_size, hidden_dim).to(device)
    c0 = torch.empty(4, batch_size, hidden_dim).to(device)
    nn.init.xavier_normal_(h0)
    nn.init.xavier_normal_(c0)
    
    if none_label_thresholds == None:
        none_label_thresholds = prep_and_tune_none_label_threshold(model, h0, c0, eval_dataset, device, batch_size,\
                                                                     none_label_id, num_labels)
    
    entropy_threshold, max_value_threshold = none_label_thresholds

    strict_total_loss = 0
    total_class_probs = []
    batch_count = 0
    true_labels = []
    entropy_predictions = []
    max_val_predictions = []
    for step, batch in enumerate(tqdm(eval_dataset.as_batches(batch_size=batch_size, shuffle=False))):
        tokens, seq_lengths, batch_labels = batch
        tokens = tokens.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            preds = model.forward(tokens, seq_lengths, h0, c0)
            class_probs = nn.functional.softmax(preds, dim=1)

            strict_loss = strict_loss_fn(preds, batch_labels)

            max_probs = torch.max(class_probs, dim=1).values.cpu().numpy()
            entropy = torch.sum(class_probs * -1.0 * torch.log(torch.clamp(class_probs, 1e-5, 1.0)), axis=1).cpu().numpy()
            class_preds = torch.argmax(class_probs, dim=1).cpu().numpy()
            
            entropy_final_class_preds = _apply_none_label(entropy, class_preds, none_label_id, entropy_threshold)

            max_value_final_class_preds = _apply_none_label(max_probs, class_preds, none_label_id,\
                                                                   max_value_threshold, False)
            f1_labels = batch_labels.cpu().numpy()
            entropy_predictions.append(entropy_final_class_preds)
            max_val_predictions.append(max_value_final_class_preds)
            true_labels.append(f1_labels)
            
            strict_total_loss += strict_loss.item()
            batch_count += 1
            total_class_probs.append(class_probs.cpu().numpy())

    _, _, avg_ent_f1_score = f1_eval_function(list(np.concatenate(entropy_predictions)), list(np.concatenate(true_labels)), none_label_id)
    _, _, avg_val_f1_score = f1_eval_function(list(np.concatenate(max_val_predictions)), list(np.concatenate(true_labels)), none_label_id)
    total_class_probs = np.concatenate(total_class_probs, axis=0)
    avg_strict_loss = strict_total_loss / batch_count

    return avg_strict_loss, avg_ent_f1_score, avg_val_f1_score, total_class_probs, none_label_thresholds

def prep_and_tune_none_label_threshold(model, h0, c0, eval_dataset, device, batch_size, none_label_id, num_labels):
    
    # if there isn't a none label, thresholds are set so that predictions are left alone
    if none_label_id < 0:
        return 2*int(-1.0 * np.log(1/num_labels)), -1.0
    
    entropy_values = []
    max_prob_values = []
    predict_labels = []
    labels = []
    
    sample_number = min(DEV_F1_SAMPLE, eval_dataset.length)

    model.eval()

    for step, batch in enumerate(tqdm(eval_dataset.as_batches(batch_size=batch_size, shuffle=False, sample=sample_number))):
        tokens, seq_lengths, batch_labels = batch
        tokens = tokens.to(device)

        with torch.no_grad():
            preds = model.forward(tokens, seq_lengths, h0, c0) # b x c
            class_probs = nn.functional.softmax(preds, dim=1)
            max_probs = torch.max(class_probs, dim=1).values
            entropy = torch.sum(class_probs * -1.0 * torch.log(torch.clamp(class_probs, 1e-5, 1.0)), axis=1)
            class_preds = torch.argmax(class_probs, dim=1)
            
            entropy_values.append(entropy.cpu().numpy())
            max_prob_values.append(max_probs.cpu().numpy())
            predict_labels.append(class_preds.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    entropy_values = np.concatenate(entropy_values).ravel()
    max_prob_values = np.concatenate(max_prob_values).ravel()
    predict_labels = np.concatenate(predict_labels).ravel()
    labels = np.concatenate(labels).ravel()

    none_label_threshold_entropy, best_f1 = tune_none_label_threshold(entropy_values, predict_labels, labels,\
                                                                        none_label_id, num_labels)
    print(best_f1)

    none_label_threshold_max_value, best_f1 = tune_none_label_threshold(max_prob_values, predict_labels, labels,\
                                                                          none_label_id, num_labels, False)
    print(best_f1)

    return none_label_threshold_entropy, none_label_threshold_max_value
    
def tune_none_label_threshold(values, preds, labels, none_label_id, num_labels, entropy=True):
    step = 0.001
    if entropy:
        max_entropy_cut_off = int(-1.0 * np.log(1/num_labels) / step) + 1
        thresholds = [step * i for i in range(1, max_entropy_cut_off)]
    else:
        max_prob_cut_off = int(1/step) + 1
        thresholds = [step * i for i in range(1, max_prob_cut_off)]
    
    best_f1 = 0
    best_threshold = -1
    for threshold in thresholds:
        final_preds = _apply_none_label(values, preds, none_label_id, threshold, entropy)
        _, _, f1_score = f1_eval_function(final_preds, labels, none_label_id)

        if f1_score > best_f1:
            best_threshold = threshold
            best_f1 = f1_score
    
    return best_threshold, best_f1

def f1_eval_function(pred, labels, none_label_id):
    correct_by_relation = 0
    guessed_by_relation = 0
    gold_by_relation = 0

    # Loop over the data to compute a score
    for idx in range(len(pred)):
        gold = labels[idx]
        guess = pred[idx]

        if gold == none_label_id and guess == none_label_id:
            pass
        elif gold == none_label_id and guess != none_label_id:
            guessed_by_relation += 1
        elif gold != none_label_id and guess == none_label_id:
            gold_by_relation += 1
        elif gold != none_label_id and guess != none_label_id:
            guessed_by_relation += 1
            gold_by_relation += 1
            if gold == guess:
                correct_by_relation += 1

    prec = 0.0
    if guessed_by_relation > 0:
        prec = float(correct_by_relation/guessed_by_relation)
        recall = 0.0
    if gold_by_relation > 0:
        recall = float(correct_by_relation/gold_by_relation)
        f1 = 0.0
    if prec + recall > 0:
        f1 = 2.0 * prec * recall / (prec + recall)

    return prec, recall, f1