import numpy as np
from typing import List, Optional
import torch
from torchtext.vocab import pretrained_aliases, Vectors
from enum import Enum
import logging
import pathlib
import pickle
from .instance import Instance
from .defaults import TRIG_MODEL_DEFAULTS, TRIG_TRAINING_DEFAULTS, IOBES

PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/../../generated_data/"

START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"


def _check_or_load_defaults(payload, default, key):
    if key in payload and payload[key] is not None:
        return payload[key]
    else:
        return default[key]


class ContextEmb(Enum):
    none = 0
    elmo = 1
    # bert = 2  # not supported yet
    # flair = 3  # not supported yet


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Predefined label string.
        self.PAD = PAD
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = UNK
        self.unk_id = -1

        # Model hyper parameters
        self.embedding_name = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "embeddings")
        self.embedding_dim = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "emb_dim")
        self.embedding = self.read_pretrained_embedding()
        self.context_emb = ContextEmb[_check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "content_emb")]
        self.context_emb_size = 0
        self.word_embedding = None
        self.seed = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "seed")
        self.digit2zero = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "digit2zero")
        self.hidden_dim = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "hidden_dim")
        self.rep_hidden_dim = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "hidden_dim")
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "dropout")
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "use_char_rnn")
        self.use_crf_layer = _check_or_load_defaults(args, TRIG_MODEL_DEFAULTS, "use_crf_layer")

        # Data specification
        self.percentage = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "percentage")
        self.dataset_name = args["dataset_name"]
        self.experiment_name = args["experiment_name"]
        self.label2idx = {}
        self.idx2labels = []
        self.label_size = 0
        self.char2idx = {}
        self.idx2char = []
        self.word2idx = dict()
        self.idx2word = []
        self.num_char = 0
        self.start_label_id = -1
        self.stop_label_id = -1

        # Training hyperparameter
        # self.model_folder = args.model_folder
        self.optimizer = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "optimizer").lower()
        self.trig_optimizer = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "trig_optimizer").lower()
        self.learning_rate = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "learning_rate")
        self.momentum = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "momentum")
        self.l2 = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "l2")
        self.num_epochs = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "num_epochs")
        self.num_epochs_soft = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "pre_train_num_epochs")
        self.use_dev = True
        self.batch_size = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "batch_size")
        self.clip = 5
        self.lr_decay = _check_or_load_defaults(args, TRIG_TRAINING_DEFAULTS, "lr_decay")

        # self.device = torch.device(args.device)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # self.ds_setting = args.ds_setting
        # print(self.ds_setting)

        self.is_lean_life = args.get("leanlife", False)
        self.project_id = args.get("project_id", None)

        self.SAVE_PATH_PARENT = PATH_TO_PARENT
        self.MODEL_PATH = self.SAVE_PATH_PARENT + "saved_models/"
        self.VOCAB_PATH = self.SAVE_PATH_PARENT + "vocabs/"
        self.TRAINING_DATA_PATH = self.SAVE_PATH_PARENT + "training_data/"
        self.RESULT_DATA_PATH = self.SAVE_PATH_PARENT + "result_data/"
        self.save_string = self.generate_save_string()

    def generate_save_string(self, sample=-1.0):
        """
            To allow for multiple datasets to exist at once, we add this string to identify which dataset a run
            script should load.
            Arguments:
                sample       (float) : percentage of possible data used for training
        """
        # dataset        (str) : name of dataset
        # embedding_name (str) : name of pre-trained embedding to use
        #                        (possible names can be found in possible_embeddings)
        # random_state   (int) : random state used to split data into train, dev, test (if applicable)
        return "_".join([self.dataset_name, self.embedding_name, str(self.seed), str(sample)])

    def generate_training_data_path(self, label: str):
        return self.TRAINING_DATA_PATH + "{}_data_{}.p".format(label, self.save_string)

    def generate_vocab_path(self, label: str):
        return self.VOCAB_PATH + "{}_data_{}.p".format(label, self.save_string)

    def generate_model_path(self, label: str):
        return self.MODEL_PATH + "{}_{}".format(label, self.save_string)

    def build_label_idx(self, insts: List[Instance]) -> None:
        """
        Build the mapping from label to index and index to labels.
        :param insts: list of instances.
        :return:
        """
        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)
                # if label not in [START,STOP,PAD,UNK,'O','T'] and label.split('-')[1] not in self.triggerlabel:
                #    self.triggerlabel[label.split('-')[1]] = len(self.triggerlabel)

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        # self.trig_label_size = len(self.triggerlabel)
        self.start_label_id = self.label2idx[self.START_TAG]
        self.stop_label_id = self.label2idx[self.STOP_TAG]
        logging.info("#labels: {}".format(self.label_size))
        logging.info("label 2idx: {}".format(self.label2idx))

    def build_word_idx(self, train_insts: List[Instance], dev_insts: List[Instance] = None,
                       test_insts: List[Instance] = None) -> None:
        """
        Build the vocab 2 idx for all instances
        :param train_insts:
        :param dev_insts:
        :param test_insts:
        :return:
        """
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = 1
        self.unk_id = 1
        self.idx2word.append(self.UNK)

        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = 1
        self.idx2char.append(self.UNK)

        # extract char on train, dev, test
        whole_sets = [] + train_insts
        if dev_insts is not None:
            whole_sets += dev_insts
        if test_insts is not None:
            whole_sets += test_insts

        for inst in whole_sets:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        # extract char only on train (doesn't matter for dev and test)
        for inst in train_insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)

    def read_pretrained_embedding(self) -> Optional[Vectors]:
        """
            Read pretrained embedding based on embedding_name
        Returns:
        """
        logging.info("reading the pretrained embedding: %s" % self.embedding_name)
        if self.embedding_name not in pretrained_aliases:
            logging.info("pretrain embedding_name not found, using random embedding")
            return None
        return pretrained_aliases[self.embedding_name]()

    def build_emb_table(self) -> None:
        """
        build the embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
        :return:
        """
        logging.info("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            logging.info("[Info] Use the pretrained word embedding to initialize: %d x %d" % (
                len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding.stoi:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding.stoi:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale,
                                                                                    [1, self.embedding_dim])
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def dump_label_word_emb_data(self, label_length=None):
        """
            Dump all data calculated by build_label_idx, build_word_idx, build_emb_table to files
        Returns:
        """
        # write labelIdx data to file
        path = self.generate_vocab_path("label_idx")
        with open(path, 'wb') as f:
            pickle.dump({
                'label2idx': self.label2idx,
                'idx2labels': self.idx2labels,
                'label_length': label_length,
            }, f)

        # write wordIdx data to file
        path = self.generate_vocab_path("word_idx")
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'char2idx': self.char2idx,
                'idx2char': self.idx2char,
            }, f)

        # write word_embedding data to file
        path = self.generate_vocab_path("word_embedding")
        with open(path, 'wb') as f:
            pickle.dump(self.word_embedding, f)

    def read_label_word_emd_data(self):
        # read labelIdx data from file
        path = self.generate_vocab_path("label_idx")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.label2idx = data['label2idx']
            self.idx2labels = data['idx2labels']
            label_length = data['label_length']

        self.label_size = len(self.label2idx)
        # self.trig_label_size = len(self.triggerlabel)
        self.start_label_id = self.label2idx[self.START_TAG]
        self.stop_label_id = self.label2idx[self.STOP_TAG]
        logging.info("#labels: {}".format(self.label_size))
        logging.info("label 2idx: {}".format(self.label2idx))

        # read wordIdx data from file
        path = self.generate_vocab_path("word_idx")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.char2idx = data['char2idx']
            self.idx2char = data['idx2char']

        self.unk_id = 1
        self.num_char = len(self.idx2char)

        # read word_embedding data to file
        path = self.generate_vocab_path("word_embedding")
        with open(path, 'rb') as f:
            self.word_embedding = pickle.load(f)

        return label_length

    def map_insts_ids(self, insts: List[Instance], dataset_label: str):
        """
        Create id for word, char and label in each instance.
        :param insts:
        :param dataset_label: str
        :return:
        """
        for inst in insts:
            words = inst.input.words
            inst.word_ids = []
            inst.char_ids = []
            inst.output_ids = [] if inst.output else None
            for word in words:
                if word in self.word2idx:
                    inst.word_ids.append(self.word2idx[word])
                else:
                    inst.word_ids.append(self.word2idx[self.UNK])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.UNK])
                inst.char_ids.append(char_id)
            if inst.output:
                for label in inst.output:
                    if label in self.label2idx:
                        inst.output_ids.append(self.label2idx[label])
                    else:
                        inst.output_ids.append(self.label2idx[IOBES.O])

        # write data to file
        path = self.generate_training_data_path(dataset_label)
        with open(path, 'wb') as f:
            pickle.dump(insts, f)
