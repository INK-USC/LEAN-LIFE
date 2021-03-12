import numpy as np
import json
from CCG import constants
from CCG import utils
from CCG import util_classes as classes
import os
import pickle
import torch.nn as nn
import torch
import copy
from collections import namedtuple
from nltk.ccg import chart, lexicon
import spacy
import random
import numpy as np
import pdb
import dill

nlp = spacy.load("en_core_web_sm")

class TrainedCCGParser():
    """
        A wrapper around an NLTK CCG Chart Parser
        Prepares the data for the parser and builds the grammar for the parser as well

        Attributes:
            
    """
    def __init__(self, low_end_filter_count=3, high_end_filter_pct=0.2):
        self.loaded_data = None
        self.grammar = None
        self.semantic_reps = None
        self.labeling_functions = None
        self.soft_labeling_functions = None
        self.filtered_raw_explanations = None
        self.ner_types = None
        self.soft_label_function_to_semantic_map = None
        self.low_end_filter_count = low_end_filter_count
        self.high_end_filter_pct = high_end_filter_pct

    def load_data(self, data):
        self.loaded_data = data
    
    def create_and_set_grammar(self, init_grammar=constants.RAW_GRAMMAR):
        """
            Function that takes initial fixed grammar and adds some loaded_data specific rules to the grammar
            Rules associated with words found in explanations from loaded_data are added to the grammar
            After rules are added the class property grammar is set.

            Arguments:
                init_grammar (str) : initial grammar to use
        """
        quote_words = {}
        for i, triple in enumerate(self.loaded_data):
            explanation = triple.raw_explanation
            if len(explanation):
                chunks = utils.clean_and_chunk(explanation, nlp)
                self.loaded_data[i].chunked_explanation = chunks
                for chunk in chunks:
                    terminals = utils.convert_chunk_to_terminal(chunk)
                    if terminals:
                        if terminals[0].startswith("\"") and terminals[0].endswith("\""):
                            quote_words[terminals[0]] = 1
        
        self.grammar = utils.add_rules_to_grammar(quote_words, init_grammar)

    def tokenize_explanations(self):
        """
            Assuming data has been loaded, and grammar and parser have been created, we start the process of
            parsing explanations into Labeling Functions. Steps performed by this function are:
                1. Explanation gets chunked (if needed)
                2. Explanations get tokenized
        """
        for i, datapoint in enumerate(self.loaded_data):
            if len(datapoint.raw_explanation):
                if datapoint.chunked_explanation:
                    chunks = datapoint.chunked_explanation
                else:
                    explanation = datapoint.raw_explanation
                    chunks = utils.clean_and_chunk(explanation, nlp)
                    self.loaded_data[i].chunked_explanation = chunks
                tokenizations = [[]]
                for chunk in chunks:
                    predicates = utils.convert_chunk_to_terminal(chunk)
                    if predicates:
                        if predicates[0].startswith("\"") and predicates[0].endswith("\""):
                            predicates[0] = utils.prepare_token_for_rule_addition(predicates[0])
                        if len(predicates) == 1:
                            for tokenization in tokenizations:
                                tokenization.append(predicates[0])
                        else:
                            temp_tokenizations = []
                            for tokenization in tokenizations:
                                for possible in predicates:
                                    tokenization_copy = tokenization[:]
                                    tokenization_copy.append(possible)
                                    temp_tokenizations.append(tokenization_copy)
                            tokenizations = temp_tokenizations

                self.loaded_data[i].tokenized_explanations = tokenizations

        
    def build_labeling_rules(self, verbose=True):
        """
            Assuming explanations have already been tokenized, and beam=False, we convert token sequences
            into labeling functions. Several token sequences are often mapped to the same labeling function,
            hence why we store semantic representations and labeling functions as keys in a dictionary. We 
            keep track of counts for training purposes. Steps performed by this function are:
                1. Token Sequences -> Parse Trees
                2. Parse Trees -> Semantic Representation
                3. Semantic Representation -> Labeling Function
        """
        # cut_off = int(len(self.loaded_data) * 0.2)
        # for i, datapoint in enumerate(self.loaded_data):
        #     if len(datapoint.raw_explanation):
        #         tokenizations = self.loaded_data[i].tokenized_explanations
        #         logic_forms = []
        #         for tokenization in tokenizations:
        #             try:
        #                 parses = list(utils.parse_tokens(tokenization, self.grammar))
        #                 logic_forms += parses
        #             except:
        #                 continue
        #         semantic_counts = {}
        #         if len(logic_forms):
        #             semantic_counts = {}
        #             for parse in logic_forms:
        #                 semantic_repr = utils.create_semantic_repr(parse)
        #                 if semantic_repr:
        #                     if semantic_repr in semantic_counts:
        #                         semantic_counts[semantic_repr] += 1
        #                     else:
        #                         semantic_counts[semantic_repr] = 1
                    
        #             if len(semantic_counts) > 1:
        #                 semantic_counts = utils.check_clauses_in_parse_filter(semantic_counts)

        #         self.loaded_data[i].semantic_counts = semantic_counts

        #         labeling_functions = {}
        #         if len(semantic_counts):
        #             for key in semantic_counts:
        #                 labeling_function = utils.create_labeling_function(key)
        #                 if labeling_function:
        #                     try:
        #                         if labeling_function(datapoint.sentence): # filtering out labeling functions that don't even apply on their own datapoint
        #                             labeling_functions[key] = labeling_function
        #                     except:
        #                         continue                                    

        #         self.loaded_data[i].labeling_functions = labeling_functions

        #     if verbose:
        #         if i > 0 and i % cut_off == 0:
        #             print("Parser: 20% more explanations parsed")
        
        # with open("loaded_data.p", "wb") as f:
        #     dill.dump(self.loaded_data, f)
        
        with open("../training/loaded_data.p", "rb") as f:
            self.loaded_data = dill.load(f)
        
    def matrix_filter(self, unlabeled_data, task="re"):
        """
            Version of BabbleLabbel's filter bank concept. Label Functions that don't apply to the original
            sentence that the explanation was written about have already been filtered out in build_labeling_rules.

            Filters out all functions that apply to more than high_end_filter_pct of datapoints
            Filters out all functions that don't apply to more than n number of datapoints

            For functions with the same output signature on the datapoints, we pick the first one, and filter out
            the rest.

            The remaining functions are then stored alongside their labels.
        """
        labeling_functions = []
        semantic_reps = []
        raw_explanations = []
        function_label_map = {}

        if task == "re":
            ner_types = []

        for i, datapoint in enumerate(self.loaded_data):
            labeling_functions_dict = datapoint.labeling_functions
            for key in labeling_functions_dict:
                function = labeling_functions_dict[key]
                labeling_functions.append(labeling_functions_dict[key])
                semantic_reps.append(key)
                raw_explanations.append(datapoint.raw_explanation)
                function_label_map[function] = datapoint.label

                if task == "re":
                    original_phrase = datapoint.sentence
                    subj_type = original_phrase.ners[original_phrase.subj_posi].lower()
                    obj_type = original_phrase.ners[original_phrase.obj_posi].lower()
                    ner_types.append((subj_type, obj_type))

        
        matrix = [[] for i in range(len(labeling_functions))]

        for i, function in enumerate(labeling_functions):
            for entry in unlabeled_data:
                try:
                    if function(entry):
                        if task == "re":
                            subj_type = entry.ners[entry.subj_posi].lower()
                            obj_type = entry.ners[entry.obj_posi].lower()
                            if ner_types[i][0] == subj_type and ner_types[i][1] == obj_type:
                                matrix[i].append(1)
                            else:
                                matrix[i].append(0)
                        else:
                            matrix[i].append(1)
                    else:
                        matrix[i].append(0)
                except:
                    matrix[i].append(0)
        
        matrix = np.array(matrix, dtype=np.int32)

        row_sums = np.sum(matrix, axis=1)

        total_data_points = len(unlabeled_data)

        functions_to_delete = []
        for i, r_sum in enumerate(row_sums):
            if r_sum/total_data_points > self.high_end_filter_pct or r_sum < self.low_end_filter_count:
                functions_to_delete.append(i)
        
        # print("Total Hits {}".format(sum(row_sums)))
        
        # print("Count Filter {}".format(functions_to_delete))

        matrix = np.delete(matrix, functions_to_delete, 0)
        functions_to_delete.sort(reverse=True)
        for index in functions_to_delete:
            del labeling_functions[index]
            del semantic_reps[index]
            del raw_explanations[index]
            if task == "re":
                del ner_types[index]
        
        hashes = {}
        functions_to_delete = []
        for i, row in enumerate(matrix):
            row_hash = hash(str(list(row))) # same as babble-labbel
            if row_hash in hashes:
                functions_to_delete.append(i)
                # print("{} conflicted with {}".format(i, hashes[row_hash]))
            else:
                hashes[row_hash] = i

        # print("Hash Filter {}".format(functions_to_delete))
        functions_to_delete.sort(reverse=True)
        for index in functions_to_delete:
            del labeling_functions[index]
            del semantic_reps[index]
            del raw_explanations[index]
            if task == "re":
                del ner_types[index]

        self.labeling_functions = {}
        self.semantic_reps = {}
        self.filtered_raw_explanations = {}
        if task == "re":
            self.ner_types = {}
        for i, function in enumerate(labeling_functions):
            self.labeling_functions[function] = function_label_map[function]
            self.semantic_reps[semantic_reps[i]] = function
            self.filtered_raw_explanations[semantic_reps[i]] = raw_explanations[i]
            
            if task == "re":
                self.ner_types[function] = ner_types[i]
        
    def set_final_datastructures(self, task="re"):
        self.labeling_functions = {}
        self.semantic_reps = {}
        self.filtered_raw_explanations = {}
        if task == "re":
            self.ner_types = {}
        for i, datapoint in enumerate(self.loaded_data):
            labeling_functions_dict = datapoint.labeling_functions
            for key in labeling_functions_dict:
                function = labeling_functions_dict[key]
                self.labeling_functions[function] = datapoint.label
                self.semantic_reps[key] = function
                self.filtered_raw_explanations[key] = datapoint.raw_explanation

                if task == "re":
                    original_phrase = datapoint.sentence
                    subj_type = original_phrase.ners[original_phrase.subj_posi].lower()
                    obj_type = original_phrase.ners[original_phrase.obj_posi].lower()
                    self.ner_types[function] = (subj_type, obj_type)


    def build_soft_labeling_functions(self):
        self.soft_labeling_functions = []
        self.soft_label_function_to_semantic_map = {}
        soft_filtered_raw_explanations = {}
        for key in self.semantic_reps:
            soft_labeling_function = utils.create_soft_labeling_function(key)
            if soft_labeling_function:
                self.soft_labeling_functions.append((soft_labeling_function, self.labeling_functions[self.semantic_reps[key]]))
                self.soft_label_function_to_semantic_map[soft_labeling_function] = key
                soft_filtered_raw_explanations[key] = self.filtered_raw_explanations[key]
        
        self.filtered_raw_explanations = soft_filtered_raw_explanations

class CCGParserTrainer():
    """
        Wrapper object to train a TrainedCCGParser object

        Attributes:
            params             (dict) : dictionary holding hyperparameters for training
            parser (TrainedCCGParser) : a TrainedCCGParser instance
    """
    def __init__(self, task, explanation_file="", unlabeled_data_file="", unlabeled_data=None, explanation_data=None):
        self.params = {}
        self.params["explanation_file"] = explanation_file
        self.params["unlabeled_data_file"] = unlabeled_data_file
        self.params["task"] = task
        # Temporary until data coming in standard
        self.text_key = "text"
        self.exp_key = "explanation"
        self.label_key = "label"
        self.parser = TrainedCCGParser()
        self.unlabeled_data = unlabeled_data if unlabeled_data != None else []
        self.explanation_data = explanation_data
    
    def load_data(self, path):
        if len(path):
            with open(path) as f:
                data = json.load(f)
        else:
            data = self.explanation_data
        
        processed_data = []
        for dic in data:
            text = dic[self.text_key]
            explanation = dic[self.exp_key]
            phrase_for_text = utils.generate_phrase(text, nlp)
            label = dic[self.label_key]
            data_point = classes.DataPoint(phrase_for_text, label, explanation)
            processed_data.append(data_point)
        
        self.parser.load_data(processed_data)
    
    def prepare_unlabeled_data(self, path=""):
        if len(self.unlabeled_data) == 0 :
            with open(path) as f:
                data = json.load(f)
        else:
            data = self.unlabeled_data
            self.unlabeled_data = []
        for entry in data:
            phrase_for_text = utils.generate_phrase(entry, nlp)
            self.unlabeled_data.append(phrase_for_text)
        
        # with open("training_phrases.p", "wb") as f:
        #     pickle.dump(self.unlabeled_data, f)
        # with open("training_phrases.p", "rb") as f:
        #     self.unlabeled_data = pickle.load(f)

    def train(self, matrix_filter=False, build_soft_functions=True, verbose=True):
        self.load_data(self.params["explanation_file"])
        if verbose:
            print("Parser: Loaded explanation data")
        self.parser.create_and_set_grammar()
        if verbose:
            print("Parser: Created and Set Grammar")
        self.parser.tokenize_explanations()
        if verbose:
            print("Parser: Tokenized Explanations")
        self.parser.build_labeling_rules()
        if verbose:
            print("Parser: Built Labeling Rules")
        if matrix_filter:
            self.prepare_unlabeled_data(self.params["unlabeled_data_file"])
            if verbose:
                print("Parser: Prepared unlabeled data")
            self.parser.matrix_filter(self.unlabeled_data, self.params["task"])
            if verbose:
                print("Parser: Filtered out bad explanations")
        else:
            self.parser.set_final_datastructures(self.params["task"])
            if verbose:
                print("Parser: Set labeling functions")
        if build_soft_functions:
            self.parser.build_soft_labeling_functions()
            if verbose:
                print("Parser: Built soft labeling functions")
        print("Parser: Done")
    
    def get_parser(self):
        return self.parser
