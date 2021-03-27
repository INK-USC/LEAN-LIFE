import logging
import pathlib
import pickle
import re
import sys
import spacy
import torch
from torchtext.data import Field, Example, Dataset
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")
from CCG.soft_grammar_functions import NER_LABEL_SPACE
from training.next_constants import TACRED_NERS, SPACY_TO_TACRED, SPACY_NERS

nlp = spacy.load("en_core_web_sm")

def load_spacy_to_custom_dataset_ner_mapping(dataset):
    """
        Function used to load dictionary that maps SPACY NERs to Custom NERs if user has defined one.

        After creating constant value in `next_constants.py`, please update this function so that the
        appropriate map is loaded

        Arguments:
            dataset (str) : name of dataset
        Returns:
            dict : key - spacy NER, value - Custom NER
    """
    if dataset == "tacred":
        return SPACY_TO_TACRED
    return {}

def set_ner_label_space(labels):
    """
        In order for certain soft-matching functions to work a global NER_LABEL_SPACE parameter must be set.
        This function handles the setting of that param and ensures that ids of labels don't overlap and
        duplicates aren't made.

        Arguments:
            labels (arr) : array of label names to add to the NER_LABEL_SPACE
    """
    if len(NER_LABEL_SPACE) > 0:
        largest_key = max(list(NER_LABEL_SPACE.values()))
    else:
        largest_key = -1
    current_key = largest_key + 1
    for label in labels:
        if label not in NER_LABEL_SPACE:
            NER_LABEL_SPACE[label] = current_key
            current_key += 1

def set_re_dataset_ner_label_space(dataset, custom_ners=[]):
    """
        A wrapper function that for an RE tasks sets the global NER_LABEL_SPACE

        Arguments:
            dataset       (str) : name of dataset
            customer_ners (arr) : array of ners provided by user
    """
    temp = SPACY_NERS[:]
    
    custom_mapping = load_spacy_to_custom_dataset_ner_mapping(dataset)
   
    for i, entry in enumerate(temp):
        if entry in custom_mapping:
            temp[i] = custom_mapping[entry]
    
    temp.append("<PAD>")
    set_ner_label_space(temp)
    set_ner_label_space(custom_ners)

def _build_custom_vocab(tokens, vocab_length):
    """
        Helper function for building custom vocab
    """
    custom_vocab = {}
    cur_key = vocab_length
    for token in tokens:
        custom_vocab[token] = cur_key
        cur_key += 1
    
    return custom_vocab

def _build_re_custom_tokens(ner_labels):
    """
        For RE tasks we employ the strategy below to create custom-tokens for our vocab:
            Per each ner_label, create two tokens SUBJ-NER_LABEL, OBJ-NER_LABEL
            These tokens make up the custom vocab
        
        Arguments:
            ner_labels (arr) : array of ner label names
    """
    tokens  = []
    for label in ner_labels:
        tokens.append("SUBJ-{}".format(label))
        tokens.append("OBJ-{}".format(label))
    return tokens

def _build_tacred_custom_vocab(vocab_length):
    """
        For the TACRED dataset we defined a function to build its custom vocab
        You can also define similar functions to be used in `build_custom_vocab`
    """
    cur_key = vocab_length
    tokens = _build_re_custom_tokens(list(TACRED_NERS.keys()))
    custom_vocab = _build_custom_vocab(tokens, vocab_length)
        
    return custom_vocab

def build_custom_vocab(dataset, vocab_length, tokens=[], type_str=""):
    """
        We offer three different ways to set a custom vocab:
            1. Define a function above and call it by sending in you datasets name (as is done for tacred)
            2. If your the task is "re", we create the re custom vocab out out ner labels
               check `_build_re_custom_tokens()` for more
            3. Otherwise if you send in tokens we will create a custom_vocab out of them
        
        Arguments:
            dataset      (str) : name of dataset
            vocab_length (int) : non-custom vocab length
            tokens       (arr) : array of tokens to build a custom vocab out of
            type_str     (str) : indicating name of task
        
        Returns:
            dict : key - token, value - token_id
    """
    custom_vocab = {}
    
    if len(tokens) == 0:
        if dataset == "tacred":
            custom_vocab = _build_tacred_custom_vocab(vocab_length)
        # elif...
    else:
        if type_str == "re":
            custom_vocab = _build_custom_vocab(_build_re_custom_tokens(tokens), vocab_length)
        else:
            custom_vocab = _build_custom_vocab(tokens, vocab_length)
        
    return custom_vocab

def find_array_start_position(big_array, small_array):
    """
        Find the starting index of a sub_array inside of a larger array

        Returns -1 if the small_array is not contrained within the larger array

        Arguments:
            big_array   (arr) : the larger array to search through
            small_array (arr) : the smaller array to find
        
        Returns:
            int : start position of small_array if small_array is within big_array, else -1
    """
    small_array_len = len(small_array)
    cut_off = len(big_array) - small_array_len
    for i, elem in enumerate(big_array):
        if i <= cut_off:
            if elem == small_array[0]:
                if big_array[i:i+small_array_len] == small_array:
                    return i
        else:
            break

    return -1

def generate_save_string(dataset, embedding_name, random_state=-1, sample=-1.0):
    """
        To allow for multiple datasets to exist at once, we add this string to identify which dataset a run
        script should load.

        Arguments:
            dataset        (str) : name of dataset
            embedding_name (str) : name of pre-trained embedding to use
                                   (possible names can be found in possible_embeddings)
            random_state   (int) : random state used to split data into train, dev, test (if applicable)
            sample       (float) : percentage of possible data used for training
    """
    return "_".join([dataset, embedding_name, str(random_state), str(sample)])

def clean_text(text):
    text = text.strip()
    text = text.lower()
    text = text.replace('\n', '')

    return text

def tokenize(sentence, tokenizer=nlp):
    """
        Simple tokenizer function that is needed to build a vocabulary
        Uses spaCy's en model to tokenize

        Arguments:
            sentence          (str) : input to be tokenized
            tokenizer (spaCy Model) : spaCy model to use for tokenization purposes

        Returns:
            arr : list of tokens
    """

    sbj_replacement = None
    obj_replacement = None

    if "SUBJ-" in sentence and "OBJ-" in sentence:
        sbj_replacement = re.search(r"SUBJ-[A-Z_'s,0-9]+", sentence).group(0).strip()
        sbj_replacement = sbj_replacement.replace("'s", "")
        sbj_replacement = sbj_replacement.replace(",", "")
        obj_replacement = re.search(r"OBJ-[A-Z_'s,0-9]+", sentence).group(0).strip()
        obj_replacement = obj_replacement.replace("'s", "")
        obj_replacement = obj_replacement.replace(",", "")
        sentence = re.sub(r"SUBJ-[A-Z_0-9]+", "SUBJ", sentence)
        sentence = re.sub(r"SUBJ-[A-Z_'s0-9]+", "SUBJ's", sentence)
        sentence = re.sub(r"SUBJ-[A-Z_0-9]+,", "SUBJ,", sentence)
        sentence = re.sub(r"OBJ-[A-Z_0-9]+", "OBJ", sentence)
        sentence = re.sub(r"OBJ-[A-Z_'s0-9]+", "OBJ's ", sentence)
        sentence = re.sub(r"OBJ-[A-Z_0-9]+,", "OBJ,", sentence)
    
    sentence = clean_text(sentence)

    spacy_tokens = [tok.text for tok in tokenizer.tokenizer(sentence)]

    if sbj_replacement != None:
        sbj_index = spacy_tokens.index("subj")
        assert sbj_index > -1
        spacy_tokens[sbj_index] = sbj_replacement
    if obj_replacement != None:
        obj_index = spacy_tokens.index("obj")
        assert obj_index > -1
        spacy_tokens[obj_index] = obj_replacement
    
    return spacy_tokens


def build_vocab(train, embedding_name, save_string="", save=True):
    """
        Note: Using the Field class will be deprecated soon by TorchText, however at the time of writing the
              new methodology for creating a vocabulary has not been released.
        
        Function that takes in training data and builds a TorchText Vocabulary object, which couples two
        important datastructures:
            1. Mapping from text token to token_id
            2. Mapping from token_id to vector

        Function expects a pre-computed set of vectors to be used in the mapping from token_id to vector

        Arguments:
            train          (arr) : array of text sequences that make up one's training data
            embedding_name (str) : name of pre-trained embedding to use 
                                   (possible names can be found in possible_embeddings)
            save_string    (str) : string to indicate some of the hyper-params used to create the vocab
        Returns:
            torchtext.vocab : vocab object
    """
    # text_field = Field(tokenize=tokenize, init_token = '<bos>', eos_token='<eos>')
    text_field = Field(tokenize=tokenize)
    fields = [("text", text_field)]
    train_examples = []
    for text in train:
        train_examples.append(Example.fromlist([text], fields))
    
    train_dataset = Dataset(train_examples, fields=fields)
    
    text_field.build_vocab(train_dataset, vectors=embedding_name)
    vocab = text_field.vocab

    logging.info("Finished building vocab of size {}".format(str(len(vocab))))

    if save:
        file_name = PATH_TO_PARENT + "data/vocabs/vocab_{}.p".format(save_string)

        with open(file_name, "wb") as f:
            pickle.dump(vocab, f)

    return vocab

def convert_text_to_tokens(data, base_vocab, tokenize_fn, custom_vocab={}):
    """
        Converts sequences of text to sequences of token ids per the provided vocabulary

        Arguments:
            data                   (arr) : sequences of text
            base_vocab (torchtext.vocab) : vocabulary object
            tokenize_fun      (function) : function to use to break up text into tokens

        Returns:
            arr : array of arrays, each inner array is a token_id representation of the text passed in

    """
    word_seqs = [tokenize_fn(seq) for seq in data]
    token_seqs = [[custom_vocab[word] if word in custom_vocab else base_vocab[word] for word in word_seq] for word_seq in word_seqs]

    return token_seqs

def extract_queries_from_explanations(explanation):
    """
        Checks for the existence of a quoted phrase within an explanation
        Three types of quotes are accepted
        
        Arguments:
            explanation (str) : explanation text for a labeling decision
        
        Returns:
            arr : an array of quoted phrases or an empty array
    """
    possible_queries = re.findall('"[^"]+"', explanation)
    if len(possible_queries):
        possible_queries = [query[1:len(query)-1] for query in possible_queries]
        return possible_queries
    
    possible_queries = re.findall("'[^']+'", explanation)
    if len(possible_queries):
        possible_queries = [query[1:len(query)-1] for query in possible_queries]
        return possible_queries

    possible_queries = re.findall("`[^`]+`", explanation)
    if len(possible_queries):
        possible_queries = [query[1:len(query)-1] for query in possible_queries]
        return possible_queries

    return []

def similarity_loss_function(pos_scores, neg_scores):
    """
        L_sim in the NExT Paper

        Arguments:
            pos_scores (torch.tensor) : per query the max value of (tau - cos(q_li_j, q_li_k))^2
                                        dims: (n,)
            neg_scores (torch.tensor) : per query the max value of (cos(q_li_j, q_lk_m))^2
                                        dims: (n,)
        
        Returns:
            torch.tensor : average of the sum of scores per query, dims: (1,)
    """
    return torch.mean(pos_scores + neg_scores)