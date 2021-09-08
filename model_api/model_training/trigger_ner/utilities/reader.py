from tqdm import tqdm
from .defaults import IOBES
from .instance import Instance
from .sentence import Sentence
from typing import List, Dict, Tuple
import re
import logging
import random

# TODO: check seed setting
random.seed(1337)


class Reader:
    def __init__(self, digit2zero: bool = True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero

    def use_iobes(self, insts: List[Instance]) -> None:
        """
        Use IOBES tagging schema to replace the IOB tagging schema in the instance
        :param insts:
        :return:
        """
        for inst in insts:
            output = inst.output
            if not output:
                continue
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(IOBES.B):
                        output[pos] = curr_entity.replace(IOBES.B, IOBES.S)
                    elif curr_entity.startswith(IOBES.I):
                        output[pos] = curr_entity.replace(IOBES.I, IOBES.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(IOBES.B):
                        if next_entity.startswith(IOBES.O) or next_entity.startswith(IOBES.B):
                            output[pos] = curr_entity.replace(IOBES.B, IOBES.S)
                    elif curr_entity.startswith(IOBES.I):
                        if next_entity.startswith(IOBES.O) or next_entity.startswith(IOBES.B):
                            output[pos] = curr_entity.replace(IOBES.I, IOBES.E)

    def build_data(self, dataset: Dict[str, str], dataset_label: str, use_iobes=True) -> List[Instance]:
        logging.info("Reading dataset: " + dataset_label)
        insts = []

        for row in tqdm(dataset):
            sentence: str = row['text']
            words = sentence.split(' ')

            if 'label' in row:
                label: str = row['label']
                labels = label.split(' ')
            else:
                labels = None

            if self.digit2zero:
                words = list(map(
                    lambda word: re.sub(r'\d', '0', word),  # replace digit with 0.
                    words
                ))

            insts.append(Instance(Sentence(words), labels))

        if use_iobes:
            self.use_iobes(insts)

        logging.info("number of sentences: {}".format(len(insts)))
        return insts

    def build_trigger_data(self, dataset: Dict[str, str], use_iobes=True) -> Tuple[List[Instance], int, int]:
        logging.info("Reading trigger dataset")
        label_vocab = dict()
        insts = []
        max_length = 0
        for row in tqdm(dataset):
            trigger_label = None

            sentence: str = row['text']
            words = sentence.split(' ')

            labels = row['label'].split(' ')

            for label in labels:
                if label.startswith("B-") or label.startswith("I-"):
                    key = label.split("-")[1]
                    if key not in label_vocab:
                        label_vocab[key] = len(label_vocab)
                    trigger_label = label_vocab[key]

            explanation: str = row['explanation']
            triggers = explanation.split(' ')

            assert len(words) == len(labels) == len(triggers)

            entity_dict = dict()
            for index, ent in enumerate(triggers):
                if ent.startswith("T-"):
                    key = ent.split("-")[1]
                    if key in entity_dict:
                        entity_dict[key].append((words[index], index))
                    else:
                        entity_dict[key] = [(words[index], index)]

            trigger_positions = []
            trigger_keys = []

            for key in entity_dict:
                trigger_positions.append([i[1] for i in entity_dict[key]])
                trigger_keys.append(" ".join(i[0] for i in entity_dict[key]))

            if self.digit2zero:
                words = list(map(
                    lambda word: re.sub(r'\d', '0', word),  # replace digit with 0.
                    words
                ))

            for trigger_position, trigger_key in zip(trigger_positions, trigger_keys):
                insts.append(Instance(Sentence(words), labels, None, trigger_label, trigger_position, trigger_key))

        if use_iobes:
            self.use_iobes(insts)

        logging.info("number of sentences: {}".format(len(insts)))
        return insts, max_length, len(label_vocab)

    def merge_labels(self, dataset: List[Instance]):
        inst_dictionary = dict()

        for inst in dataset:
            key = " ".join(word for word in inst.input.words)
            if key not in inst_dictionary:
                inst_dictionary[key] = [inst]
            else:
                inst_dictionary[key].append(inst)

        for key in inst_dictionary:
            candidate_labels = []
            for inst in inst_dictionary[key]:
                candidate_labels.append(inst.output)

            final_labels = []
            for j in range(len(candidate_labels[0])):
                is_entity = False
                for i in range(len(candidate_labels)):
                    if candidate_labels[i][j] != 'O':
                        final_labels.append(candidate_labels[i][j])
                        is_entity = True
                        break
                if is_entity == False:
                    final_labels.append('O')

            for inst in inst_dictionary[key]:
                inst.output = final_labels

    def trigger_percentage(self, dataset: List[Instance], percentage: float):
        inst_dictionary = dict()

        for inst in dataset:
            key = " ".join(word for word in inst.input.words)
            if key not in inst_dictionary:
                inst_dictionary[key] = [inst]
            else:
                inst_dictionary[key].append(inst)

        numbers = int(len(inst_dictionary) * percentage / 100)
        new_inst_keys = list(inst_dictionary.keys())

        random.shuffle(new_inst_keys)
        new_inst_keys = new_inst_keys[:numbers]

        new_inst = []
        for key in new_inst_keys:
            new_inst.extend(inst_dictionary[key])

        return new_inst
