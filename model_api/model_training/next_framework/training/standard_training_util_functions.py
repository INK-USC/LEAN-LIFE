import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import pathlib
import sys

PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")

from training.next_util_functions import build_vocab
from training.next_training_util_functions import build_labeled_dataset, f1_eval_function


def build_datasets_from_text(labeled_data, vocab_, dev_data, custom_vocab, save_string, label_map, sample_rate=-1.0):
    """
        Builds all required datastructures for training

        Arguments:
            labeled_data     (arr) : array of dictionaries, each dictionary has the following two keys:
                                        * text, * label
            vocab_      (str|dict) : if string, then vocab is the path to a torch.text.Vocab object, else it's dictionary with two keys:
                                      * embedding_name, * save_string --> needed to create a vocab if one already hasn't been built
            dev_data         (arr) : array of dictionaries, each dictionary has the following two keys:
                                        * text, * label
            custom_vocab    (dict) : key - string, value - token_id (can be empty, {}) 
            save_string      (str) : string to indicate some of the hyper-params used to create the vocab
            label_map       (dict) : key - label name, value - label_id
            sample_rate    (float) : percentage of unlabeled data to use when building datasets
    """
    if type(vocab_) == str:
        with open(vocab_, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = build_vocab(list(map(lambda x: x['text'], labeled_data)), vocab_["embedding_name"],
                            vocab_["save_string"])

    if sample_rate > 0:
        sample_number = int(len(labeled_data) * sample_rate)
        labeled_data = random.sample(labeled_data, sample_number)

    for data, name in [(labeled_data, "labeled"), (dev_data, "dev")]:
        sentences = []
        labels = []

        for row in data:
            sentences.append(row["text"])
            labels.append(row["label"])

        build_labeled_dataset(sentences, labels, vocab, save_string, name, label_map, custom_vocab)


def evaluate_clf(data_path, model, strict_loss_fn, batch_size=128, hidden_dim=100, none_label_id=-1, dir_x_lay=4):
    """
        Function that evaluates a BiLSTM+Att model against an dataset

        If `none_label_thresholds` == None, we always try to find threshold values, even if none_label_id=-1
        However, if none_labels=-1, we set the thresholds so that they will never apply.

        Arguments:
            data_path               (str) : path to eval dataset
            model                (Object) : model that needs to be evaluated
            strict_loss_fn         (func) : loss function to use
            batch_size              (int) : _
            hidden_dim              (int) : of BiLSTM
            none_label_id           (int) : -1 -> indicates no null/none/neutral label, else the label_id
            dir_x_lay               (int) : number_of_directions x num_layers in the BiLSTM

        Returns:
            tup : avg_strict_loss, avg_f1_score, total_class_probs, none_label_thresholds
    """

    with open(data_path, "rb") as f:
        eval_dataset = pickle.load(f)

    # deactivate dropout layers
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    h0 = torch.empty(dir_x_lay, batch_size, hidden_dim).to(device)
    c0 = torch.empty(dir_x_lay, batch_size, hidden_dim).to(device)
    nn.init.xavier_normal_(h0)
    nn.init.xavier_normal_(c0)

    strict_total_loss = 0
    total_class_probs = []
    batch_count = 0
    true_labels = []
    class_predictions = []

    for step, batch in enumerate(tqdm(eval_dataset.as_batches(batch_size=batch_size, shuffle=False))):
        tokens, seq_lengths, batch_labels = batch
        tokens = tokens.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            preds = model.forward(tokens, seq_lengths, h0, c0)
            class_probs = nn.functional.softmax(preds, dim=1)

            strict_loss = strict_loss_fn(preds, batch_labels)

            class_preds = torch.argmax(class_probs, dim=1).cpu().numpy()

            f1_labels = batch_labels.cpu().numpy()

            class_predictions.append(class_preds)
            true_labels.append(f1_labels)

            strict_total_loss += strict_loss.item()
            batch_count += 1
            total_class_probs.append(class_probs.cpu().numpy())

    _, _, avg_f1_score = f1_eval_function(list(np.concatenate(class_predictions)),
                                          list(np.concatenate(true_labels)), none_label_id)
    total_class_probs = np.concatenate(total_class_probs, axis=0)
    avg_strict_loss = strict_total_loss / batch_count

    return avg_strict_loss, avg_f1_score, total_class_probs
