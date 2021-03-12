import torch
import torch.nn as nn
import torch.nn.functional as f

class BiLSTM_Att_Clf(nn.Module):
    def __init__(self, emb_weight, padding_idx, emb_dim, hidden_dim, cuda, number_of_classes, custom_token_count=0,
                 n_layers=2, encoding_dropout=0.5, padding_score=-1e30):
        """
            Arguments:
                emb_weight (torch.tensor) : created vocabulary's vector representation for each token, where
                                            vector_i corresponds to token_i
                                            dims : (vocab_size, emb_dim)
                padding_idx         (int) : index of pad token in vocabulary
                emb_dim             (int) : legnth of each vector representing a token in the vocabulary
                hidden_dim          (int) : size of hidden representation emitted by lstm
                                            (we are using a bi-lstm, final hidden_dim will be 2*hidden_dim)
                cuda               (bool) : is a gpu available for usage
                number_of_classes   (int) : number of classes to predict over
                n_layers            (int) : number of layers for the bi-lstm that encodes n-gram representations
                                            of a token 
                encoding_dropout  (float) : percentage of vector's representation to be randomly zeroed
                                            out before pooling
                padding_score     (float) : score of padding tokens during attention calculation
        """
        super(BiLSTM_Att_Clf, self).__init__()

        self.padding_idx = padding_idx
        self.padding_score = padding_score
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = 2*hidden_dim
        self.number_of_classes = number_of_classes
        self.custom_token_count = custom_token_count

        if self.custom_token_count:
            custom_vocab_embeddings = nn.init.normal_(torch.empty(self.custom_token_count, self.emb_dim), -1., 1.0)
            emb_weight = torch.cat([emb_weight, custom_vocab_embeddings])

        self.embeddings = nn.Embedding.from_pretrained(emb_weight, freeze=False, padding_idx=self.padding_idx)
        self.encoding_bilstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=n_layers,
                                       bidirectional=True, batch_first=True)
        
        self.attention_matrix = nn.Linear(self.encoding_dim, self.encoding_dim)
        self.attention_activation = nn.Tanh()
        self.attention_vector = nn.Linear(self.encoding_dim, 1, bias=False)
        self.attn_softmax = nn.Softmax(dim=2)
        nn.init.xavier_uniform_(self.attention_matrix.weight)
        nn.init.constant_(self.attention_matrix.bias, 0.)
        nn.init.constant_(self.attention_vector.weight, 0.)

        self.weight_final_layer = nn.Linear(self.encoding_dim, self.number_of_classes, bias=False)
        nn.init.kaiming_uniform_(self.weight_final_layer.weight, a=0.01, mode='fan_in')

        self.embedding_dropout = nn.Dropout(p=0.04)
        self.encoding_dropout = nn.Dropout(p=encoding_dropout)
    
    def get_attention_weights(self, hidden_states, padding_indexes=None):
        """
            Calculates attention weights for each token in each sequence passed in
                * heavily discounts the importance of padding_tokens, when indices representing which
                  tokens are padding and which aren't
            Arguments:
                hidden_states   (torch.tensor) : N x seq_len x encoding_dim
                padding_indexes (torch.tensor) : N x seq_len
            Returns:
                (torch.tensor) : N x 1 x seq_len
        """
        linear_transform = self.attention_matrix(hidden_states) # linear_transform = N x seq_len x encoding_dim
        tanh_tensor = self.attention_activation(linear_transform) # element wise tanh
        batch_dot_products = self.attention_vector(tanh_tensor) # batch_dot_product = batch x seq_len x 1
        if padding_indexes != None:
            padding_scores = self.padding_score * padding_indexes # N x seq_len
            batch_dot_products = batch_dot_products + padding_scores.unsqueeze(2) # making sure score of padding_idx tokens is incredibly low
        updated_batch_dot_products = batch_dot_products.permute(0,2,1) # batch x 1 x seq_len
        batch_soft_max = self.attn_softmax(updated_batch_dot_products) #apply softmax along row

        return batch_soft_max

    def attention_pooling(self, hidden_states, padding_indexes):
        """
            Pools hidden states together using a trainable attention matrix and query vector
            
            Arguments:
                hidden_states   (torch.tensor) : N x seq_len x encoding_dim
                padding_indexes (torch.tensor) : N x seq_len
            
            Returns:
                (torch.tensor) : N x 1 x encoding_dim
        """
        batch_soft_max = self.get_attention_weights(hidden_states, padding_indexes) # batch_soft_max = batch x 1 x seq_len
        pooled_rep = torch.bmm(batch_soft_max, hidden_states) # pooled_rep = batch x 1 x encoding_dim

        return pooled_rep
    
    def get_embeddings(self, seqs):
        """
            Convert tokens into vectors. Also figures out what tokens are padding.
            Arguments:
                seqs (torch.tensor) : N x seq_len
            
            Returns:
                seq_embs, padding_indexes : N x seq_len x embedding_dim, N x seq_len
        """
        padding_indexes = seqs == self.padding_idx # N x seq_len
        padding_indexes = padding_indexes.float()

        seq_embs = self.embeddings(seqs)
                
        return seq_embs, padding_indexes
    
    def encode_tokens(self, seqs, seq_lengths, h0, c0):
        """
            Create raw encodings for a sequence of tokens
            Arguments:
                seqs (torch.tensor) : N x seq_len
            
            Returns:
                seq_embs, padding_indexes : N x seq_len x encoding_dim, N x seq_len
        """
        seq_embs, padding_indexes = self.get_embeddings(seqs) # N x seq_len x embedding_dim, N, seq_len
        seq_embs = self.embedding_dropout(seq_embs)
        seq_embs = nn.utils.rnn.pack_padded_sequence(seq_embs, seq_lengths, enforce_sorted=False, batch_first=True)
        seq_encodings, _ = self.encoding_bilstm(seq_embs, (h0,c0)) # N x seq_len, encoding_dim
        seq_encodings, _ = nn.utils.rnn.pad_packed_sequence(seq_encodings, batch_first=True)
        seq_encodings = self.encoding_dropout(seq_encodings)
        
        return seq_encodings, padding_indexes
    
    def classification_head(self, pooled_vectors):
        """
            MLP head on top of encoded representation to be classified
            Arguments:
                pooled_vectors (torch.tensor) : N x encoding_dim
            
            Returns:
                (torch.tensor) : N x number_of_classes
        """
        classification_scores = self.weight_final_layer(pooled_vectors) # N x number_of_classes

        return classification_scores

    def forward(self, seqs, seq_lengths, h0, c0):
        seq_encodings, padding_indexes = self.encode_tokens(seqs, seq_lengths,  h0, c0) # N x seq_len x encoding_dim, N x seq_len
        pooled_encodings = self.attention_pooling(seq_encodings, padding_indexes).squeeze(1) # N x encoding_dim
        classification_scores = self.classification_head(pooled_encodings)

        return classification_scores # N x number_of_classes
