import pickle
import random

import pathlib
import sys

PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")

from training.next_util_functions import build_vocab
from training.next_training_util_functions import build_labeled_dataset


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
        vocab = build_vocab(list(map(lambda x: x['text'], labeled_data)), vocab_["embedding_name"], vocab_["save_string"])

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
