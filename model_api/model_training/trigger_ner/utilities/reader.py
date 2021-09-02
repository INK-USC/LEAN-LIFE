from tqdm import tqdm
from .defaults import IOBES
from .instance import Instance
from .sentence import Sentence
from typing import List, Dict
import re
import logging
import pickle


# import random
# TODO: check seed setting
# random.seed(1337)

class Reader:
    def __init__(self, digit2zero: bool = True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        # TODO: check usage
        # self.vocab = set()

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

            # self.vocab.add(words)
            insts.append(Instance(Sentence(words), labels))

        if use_iobes:
            self.use_iobes(insts)

        logging.info("number of sentences: {}".format(len(insts)))
        return insts
