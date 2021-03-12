import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoModel 

ENCODING_STATE_DIM = 768

class Find_Module(nn.Module):
    """
        Find Module per the NExT paper's description
    """
    def __init__(self, encoding_dim, cuda, freeze_model, encoding_dropout, padding_score=-1*1e30, compression_dim=600):
        """
            Arguments:
                encoding_dim        (int) : final encoding dimension for a vector representation of a token
                cuda               (bool) : is a gpu available for usage
                freeze_model       (bool) : whether to freeze the encoder model
                encoding_dropout  (float) : percentage of vector's representation to be randomly zeroed
                                            out before pooling
                padding_score       (int) : cosine score for padding positions when applying self-attention
                                            pooling
                compression_dim     (int) : intermediate dimension before applying non-linear activation 
        """
        super(Find_Module, self).__init__()

        self.padding_score = padding_score
        self.encoding_dim = encoding_dim
        self.compression_dim = compression_dim
        self.cuda = cuda

        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased', return_dict=True)
        
        # "freezes" all parameters associated with distilbert
        if freeze_model:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        self.encoding_dropout = nn.Dropout(p=encoding_dropout)

        self.compression_matrix_1 = nn.Linear(ENCODING_STATE_DIM, self.compression_dim)

        diagonal_vector = torch.zeros(self.compression_dim, 1)
        nn.init.xavier_uniform_(diagonal_vector)
        diagonal_vector = diagonal_vector.squeeze(1)
        self.feature_weight_matrix = nn.Parameter(torch.diag(input=diagonal_vector), requires_grad=True)
        
        # Non-Linear Activation
        self.relu =  nn.ReLU()

        self.compression_matrix_2 = nn.Linear(self.compression_dim, self.encoding_dim)

        self.attention_matrix = nn.Linear(self.encoding_dim, self.encoding_dim)
        temp_att_vector = torch.zeros(self.encoding_dim, 1)
        nn.init.xavier_uniform_(temp_att_vector)
        self.attention_vector = nn.Parameter(temp_att_vector, requires_grad=True)
        self.attn_softmax = nn.Softmax(dim=2)

        self.pooled_linear_transform = nn.Linear(self.encoding_dim, self.encoding_dim)

        diagonal_vector_2 = torch.zeros(self.encoding_dim, 1)
        nn.init.xavier_uniform_(diagonal_vector_2)
        diagonal_vector_2 = diagonal_vector_2.squeeze(1)
        self.feature_weight_matrix_cosine = nn.Parameter(torch.diag(input=diagonal_vector_2), requires_grad=True)

        self.pooled_project_up = nn.Linear(self.encoding_dim, self.compression_dim)
        self.pooled_project_down = nn.Linear(self.compression_dim, self.encoding_dim)
    
    def attention_pooling(self, hidden_states, padding_indexes):
        """
            Pools hidden states together using an trainable attention matrix and query vector
            Arguments:
                hidden_states   (torch.tensor) : N x seq_len x encoding_dim
                padding_indexes (torch.tensor) : N x seq_len
            Returns:
                (torch.tensor) : N x 1 x encoding_dim
        """
        padding_scores = self.padding_score * padding_indexes # N x seq_len
        linear_transform = self.attention_matrix(hidden_states) # linear_transform = N x seq_len x encoding_dim
        tanh_tensor = torch.tanh(linear_transform) # element wise tanh
        batch_dot_products = torch.matmul(tanh_tensor, self.attention_vector) # batch_dot_product = batch x seq_len x 1
        updated_batch_dot_products = batch_dot_products + padding_scores.unsqueeze(2) # making sure score of padding_idx tokens is incredibly low
        updated_batch_dot_products = updated_batch_dot_products.permute(0,2,1) # batch x 1 x seq_len
        batch_soft_max = self.attn_softmax(updated_batch_dot_products) #apply softmax along row
        pooled_rep = torch.bmm(batch_soft_max, hidden_states) # pooled_rep = batch x 1 x encoding_dim --> one per x :)

        return pooled_rep

    def encode_tokens(self, input_ids, input_mask):
        """
            Convert a sequence of tokens into their vector representations
            Arguments:
                input_ids  (torch.tensor) : N x seq_len
                input_mask (torch.tensor) : N x seq_len
            Returns:
                (torch.tensor) : N x seq_len x ENCODING_STATE_DIM
        """
        output =  self.encoder(input_ids, input_mask) # hidden_states =  N x seq_len x ENCODING_STATE_DIM
        hidden_states = output.last_hidden_state
        hidden_states = self.encoding_dropout(hidden_states)
        
        return hidden_states
    
    def create_encodings(self, input_ids, input_mask):
        hidden_states = self.encode_tokens(input_ids, input_mask)
        compressed_hidden_states = self.compression_matrix_1(hidden_states)
        normalized_hidden_states = f.normalize(compressed_hidden_states, p=2, dim=2)
        updated_hidden_states = torch.matmul(normalized_hidden_states, self.feature_weight_matrix)
        non_linear_hidden_states = self.relu(updated_hidden_states)
        final_hidden_states = self.compression_matrix_2(non_linear_hidden_states)

        return final_hidden_states
    
    def create_pooled_encodings(self, input_ids, input_mask):
        hidden_states = self.create_encodings(input_ids, input_mask)
        padding_indexes = torch.tensor([[1.0 if ind == 0 else 0.0 for ind in mask] for mask in input_mask])
        if self.cuda:
            device = torch.device("cuda")
            padding_indexes = padding_indexes.to(device)
        pooled_encodings = self.attention_pooling(hidden_states, padding_indexes)
        return pooled_encodings
    
    def get_normalized_pooled_encodings(self, input_ids, input_mask):
        pooled_encodings = self.create_pooled_encodings(input_ids, input_mask)
        normalized_encodings = f.normalize(pooled_encodings, p=2, dim=2)

        return normalized_encodings
        
    def find_forward(self, seq_token_ids, seq_attn_mask, query_token_ids, query_attn_mask):
        """
            Forward function for computing L_find when pre-training Find Module

            Arguments:
                seqs    (torch.tensor) : N x seq_len_i, token sequences for current batch
                queries (torch.tensor) : N x seq_len_j, token sequences for corresponding queries
            
            Returns:
                (torch.tensor) : N x seq_len_i, similarity scores between each token in a sequence and
                                 the corresponding query
        """
        seq_encodings = self.create_encodings(seq_token_ids, seq_attn_mask) # N x seq_len x encoding_dim
        pooled_query_encodings = self.create_pooled_encodings(query_token_ids, query_attn_mask) # N x 1 x encoding_dim
        
        large_pooled_encodings = self.pooled_project_up(pooled_query_encodings) # N x 1 x compression_dim
        non_linear_pooled_encodings = self.relu(large_pooled_encodings)
        small_pooled_encodings = self.pooled_project_down(non_linear_pooled_encodings) # N x 1 x encoding_dim
        
        normalized_seq_encodings = f.normalize(seq_encodings, p=2, dim=2)
        normalized_pooled_encodings = f.normalize(small_pooled_encodings, p=2, dim=2)

        updated_seq_encodings = torch.matmul(normalized_seq_encodings, self.feature_weight_matrix_cosine) # N x seq_len x encoding_dim
        updated_query_encodings = torch.matmul(normalized_pooled_encodings, self.feature_weight_matrix_cosine) # N x 1 x encoding_dim

        updated_query_encodings = updated_query_encodings.permute(0, 2, 1) # N x encoding_dim x 1

        seq_similarities = torch.matmul(updated_seq_encodings, updated_query_encodings).squeeze(2) # N x seq_len

        return seq_similarities