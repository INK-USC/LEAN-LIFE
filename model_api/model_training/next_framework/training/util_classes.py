
import torch
import random
from abc import ABC, abstractmethod

class BaseVariableLengthDataset(ABC):
    @abstractmethod
    def as_batches(self):
        pass

    @staticmethod
    def variable_length_batch_as_tensors(batch, fill_value, dtype=torch.long):
        """
            Static function that ensures each batch is of fixed length. The fixed legnth is set to be the max
            sequence length of the batch. The function fills the rest of values with a passed in value.

            Arguments:
                batch      (arr) : batch of data
                fill_value (any) : what shorter sequences should be padded with
                dtype      (any) : the type of data the fill_value is
            
            Returns:
                torch.tensor : tensor that represents the batch sent in, 
                               dims : (n, max_seq_len)
        """
        n_ex = len(batch)
        max_len = max(len(seq) for seq in batch)
        seqs_tensor = torch.full(size=(n_ex, max_len), fill_value=fill_value, dtype=dtype)
        lengths = []
        for i, seq in enumerate(batch):
            seqs_tensor[i, 0:len(seq)] = torch.tensor(seq)
            lengths.append(len(seq))
        return seqs_tensor, lengths

class PreTrainingFindModuleDataset(BaseVariableLengthDataset):
    """
        Dataset for couping together token sequences, the queries extracted from the token sequences and the
        labels that indicate where from the sequence the query is extracted from. This dataset is needed to
        pre-train the Find Module, per the NExT paper.

        Methods:
            as_batches -- breaks data into batches, shuffles data if needed, pads the batches as well
            variable_length_batch_as_tensors -- pads a batch, so that elements in the batch are of equal
                                                length
    """
    def __init__(self, tokens, queries, labels, pad_idx):
        """
            Arguments:
                tokens  (arr) : dataset of token_ids, can be of variable length
                queries (arr) : corresponding query sequence for each instance in the dataset
                labels  (arr) : labels indiciating where the query was extracted from in the original instance
                pad_idx (int) : index that should be used to pad token sequences and query seqeunces to
                                ensure all instances in a batch are of the same length
        """
        self.tokens = tokens
        self.queries = queries
        self.labels = labels
        self.pad_idx = pad_idx
        assert len(self.tokens) == len(self.queries)
        assert len(self.tokens) == len(self.labels)
        print("Dataset built, count: {}".format(str(len(self.tokens))))
    
    def as_batches(self, batch_size, seed=0, shuffle=True):
        """
            Takes data passed in during creation time, and creates batches out of them of size batch_size
            Shuffles data if needed (for training -- shuffles per epoch via seed)
            Ensures elements of each batch are of the same length

            Arguments:
                batch_size (int) : size of each batch
                seed       (int) : seed to use when shuffling data (won't override overall random seed)
                shuffle   (bool) : whether to shuffle data before batching

            Returns:
                batch_tokens, batch_queries, batch_labels : per batch the tokens, queries and labels
                                                            needed for find pre-training
        """
        if shuffle:
            temp_data = list(zip(self.tokens, self.queries, self.labels))
            random.Random(seed).shuffle(temp_data)
            self.tokens, self.queries, self.labels = zip(*temp_data)
        for i in range(0, len(self.tokens), batch_size): # i incrememt by batch_size
            batch_tokens = self.tokens[i: i+batch_size] # slice
            batch_tokens, _ = self.variable_length_batch_as_tensors(batch_tokens, self.pad_idx)
            batch_queries = self.queries[i: i+batch_size]
            batch_queries, _ = self.variable_length_batch_as_tensors(batch_queries, self.pad_idx)
            batch_labels = self.labels[i: i+batch_size]
            batch_labels, _ = self.variable_length_batch_as_tensors(batch_labels, 0.0, torch.float)
            yield (batch_tokens, batch_queries, batch_labels)
    
class TrainingDataset(BaseVariableLengthDataset):
    """
        Dataset for couping together token sequences and the sequences' labels.

        Methods:
            as_batches -- breaks data into batches, shuffles data if needed, pads the batches as well
            variable_length_batch_as_tensors -- pads a batch, so that elements in the batch are of equal
                                                length
    """
    def __init__(self, tokens, labels, pad_idx):
        """
            Arguments:
                tokens  (arr) : dataset of token_ids, can be of variable length
                labels  (arr) : labels indiciating where the query was extracted from in the original instance
                pad_idx (int) : index that should be used to pad token sequences and query seqeunces to
                                ensure all instances in a batch are of the same length
        """
        self.tokens = tokens
        self.labels = labels
        self.pad_idx = pad_idx
        assert len(self.tokens) == len(self.labels)
        self.length = len(self.tokens)
        print("Dataset built, count: {}".format(str(self.length)))
    
    def as_batches(self, batch_size, seed=0, shuffle=True, sample=-1):
        """
            Takes data passed in during creation time, and creates batches out of them of size batch_size
            Shuffles data if needed (for training -- shuffles per epoch via seed)
            Ensures elements of each batch are of the same length

            Arguments:
                batch_size (int) : size of each batch
                seed       (int) : seed to use when shuffling data (won't override overall random seed)
                shuffle   (bool) : whether to shuffle data before batching

            Returns:
                batch_tokens, batch_queries, batch_labels : per batch the tokens, queries and labels
                                                            needed for find pre-training
        """
        if shuffle:
            temp_data = list(zip(self.tokens, self.labels))
            random.Random(seed).shuffle(temp_data)
            self.tokens, self.labels = zip(*temp_data)
        
        tokens = self.tokens
        labels = self.labels

        if sample > 0:
            temp_data = list(zip(tokens, labels))
            sampled_data = random.Random(seed).sample(temp_data, sample)
            tokens, labels = zip(*sampled_data)
        
        for i in range(0, len(tokens), batch_size): # i incrememt by batch_size
            batch_tokens = tokens[i: i+batch_size] # slice
            batch_tokens, batch_lengths = self.variable_length_batch_as_tensors(batch_tokens, self.pad_idx)
            batch_labels = torch.tensor(labels[i: i+batch_size])
            yield (batch_tokens, batch_lengths, batch_labels)

class UnlabeledTrainingDataset(BaseVariableLengthDataset):
    """
        Dataset for couping together token sequences and the sequences' labels.

        Methods:
            as_batches -- breaks data into batches, shuffles data if needed, pads the batches as well
            variable_length_batch_as_tensors -- pads a batch, so that elements in the batch are of equal
                                                length
    """
    def __init__(self, tokens, phrases, pad_idx):
        """
            Arguments:
                tokens  (arr) : dataset of token_ids, can be of variable length
                phrases  (arr) :
                pad_idx (int) : index that should be used to pad token sequences and query seqeunces to
                                ensure all instances in a batch are of the same length
        """
        self.tokens = tokens
        self.phrases = phrases
        self.pad_idx = pad_idx
        assert len(self.tokens) == len(self.phrases)
        print("Dataset built, count: {}".format(str(len(self.tokens))))
    
    def as_batches(self, batch_size, seed=0, shuffle=True):
        """
            Takes data passed in during creation time, and creates batches out of them of size batch_size
            Shuffles data if needed (for training -- shuffles per epoch via seed)
            Ensures elements of each batch are of the same length

            Arguments:
                batch_size (int) : size of each batch
                seed       (int) : seed to use when shuffling data (won't override overall random seed)
                shuffle   (bool) : whether to shuffle data before batching

            Returns:
                batch_tokens, batch_queries, batch_labels : per batch the tokens, queries and labels
                                                            needed for find pre-training
        """
        if shuffle:
            temp_data = list(zip(self.tokens, self.phrases))
            random.Random(seed).shuffle(temp_data)
            self.tokens, self.phrases = zip(*temp_data)
        for i in range(0, len(self.tokens), batch_size): # i incrememt by batch_size
            batch_tokens = self.tokens[i: i+batch_size] # slice
            batch_tokens, batch_lengths = self.variable_length_batch_as_tensors(batch_tokens, self.pad_idx)
            batch_phrases= self.phrases[i: i+batch_size]
            yield (batch_tokens, batch_lengths, batch_phrases)