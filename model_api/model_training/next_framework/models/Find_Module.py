import torch
import torch.nn as nn
import torch.nn.functional as f

class Find_Module(nn.Module):
    """
        Find Module - PyTorch implementation. Heavily Influenced by the NExT paper, but with the following
        differences from the source code found here: https://github.com/Rahul-Khanna/NExT/blob/master/models/soft_match.py#L215
            1. hidden_dim = 300, original = 100
            2. encoding_dropout = 0.1, original = 0.5
            3. bilstm + MLP head used to sequence tag,
                original = single linear layer combining the similarities into a single score
            4. Different weight initializations
            5. Added lower bound on how negative a final score can be (helps with data imbalance problem)
        
        In training there are the following differences:
            1. AdamW used, original = Adagrad
            2. lr = 0.001, original = 0.1
            3. weight of 20 used in BCELoss, original = 1
            4. Clip gradients to magnitude of 1.0, original = 5.0
        
        Training Script can be found here: https://github.com/Rahul-Khanna/NExT/blob/master/rahul_code/training/pre_train_find_module.py
    """
    def __init__(self, emb_weight, padding_idx, emb_dim, hidden_dim, cuda,
                 n_layers=2, encoding_dropout=0.1, sliding_win_size=3,
                 padding_score=-1e30, custom_token_count=0):
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
                n_layers            (int) : number of layers for the bi-lstm that encodes n-gram representations
                                            of a token 
                encoding_dropout  (float) : percentage of vector's representation to be randomly zeroed
                                            out before pooling
                sliding_win_size    (int) : size of window to consider when building pooled representations
                                            for a token (as of now this is fixed at 3)
                padding_score     (float) : score of padding tokens during attention calculation
        """
        super(Find_Module, self).__init__()

        self.padding_idx = padding_idx
        self.padding_score = padding_score
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = 2*hidden_dim
        self.sliding_win_size = sliding_win_size
        self.number_of_cosines = sum([i+1 for i in range(self.sliding_win_size)])
        self.cuda = cuda
        self.custom_token_count = custom_token_count

        if self.custom_token_count:
            custom_vocab_embeddings = nn.init.normal_(torch.empty(self.custom_token_count, self.emb_dim), -1., 1.0)
            emb_weight = torch.cat([emb_weight, custom_vocab_embeddings])

        self.embeddings = nn.Embedding.from_pretrained(emb_weight, freeze=False, padding_idx=self.padding_idx)
        self.encoding_bilstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=n_layers,
                                       bidirectional=True, batch_first=True)
        self.encoding_dropout = nn.Dropout(p=encoding_dropout)
        
        self.attention_matrix = nn.Linear(self.encoding_dim, self.encoding_dim)
        nn.init.xavier_uniform_(self.attention_matrix.weight)
        self.attention_activation = nn.Tanh()
        self.attention_vector = nn.Linear(self.encoding_dim, 1, bias=False)
        nn.init.kaiming_uniform_(self.attention_vector.weight, mode='fan_in')
        self.attn_softmax = nn.Softmax(dim=2)

        self.cosine_bilstm = nn.LSTM(self.number_of_cosines, 32, num_layers=2, bidirectional=True, batch_first=True)

        self.weight_linear_layer_1 = nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.weight_linear_layer_1.weight, a=0.01, mode='fan_in')
        self.weight_linear_layer_2 = nn.Linear(32, 16)
        nn.init.kaiming_uniform_(self.weight_linear_layer_2.weight, a=0.01, mode='fan_in')
        self.weight_linear_layer_3 = nn.Linear(16, 8)
        nn.init.kaiming_uniform_(self.weight_linear_layer_3.weight, a=0.01, mode='fan_in')
        
        self.weight_activation_function = nn.LeakyReLU()
        self.mlp_dropout = nn.Dropout(p=0.1)

        self.weight_final_layer = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.weight_final_layer.weight)

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
        
        seq_embs = self.embeddings(seqs) # seq_embs = N x seq_len x embedding_dim
        
        return seq_embs, padding_indexes
    
    def get_hidden_states(self, seq_embs):
        """
            Run embedding vectors through an encoder (bilstm)
            Apply a final dropout layer to the outputted hidden states
            Arguments:
                seq_embs (torch.tensor) : N x seq_len x embedding_dim
            
            Returns:
                seq_embs, padding_indexes : N x seq_len x encoding_dim
        """
        hidden_states, _ = self.encoding_bilstm(seq_embs) # N x seq_len x encoding_dim

        return hidden_states        
    
    def encode_tokens(self, seqs):
        """
            Create raw encodings for a sequence of tokens
            Arguments:
                seqs (torch.tensor) : N x seq_len
            
            Returns:
                seq_embs, padding_indexes : N x seq_len x encoding_dim, N x seq_len
        """
        seq_embs, padding_indexes = self.get_embeddings(seqs) # N x seq_len x embedding_dim, N, seq_len
        seq_encodings = self.get_hidden_states(seq_embs) # N x seq_len, encoding_dim
        seq_encodings = self.encoding_dropout(seq_encodings)
        
        return seq_encodings, padding_indexes
    
    def compute_dot_product_between_token_rep_and_query_vectors(self, token_rep, normalized_query_vectors):
        """
            Computes dot product between each sequence's tokens' representations and a pooled
            representation of the corresponding query vector. The pooled query representations
            are assumed to have already been normalized.
            Arguments:
                token_rep                (torch.tensor) : N x seq_len x encoding_dim
                normalized_query_vectors (torch.tensor) : N x encoding_dim x 1
            
            Returns:
                (torch.tensor) : N x seq_len x 1, cosine similarity between each token's representation
                                    and the query vector
        """
        normalized_token_rep = f.normalize(token_rep + 1e-5, p=2, dim=2) # normalizing rows of each matrix in the batch
        cosine_sim = torch.matmul(normalized_token_rep, normalized_query_vectors) # N x seq_len x 1
        
        return cosine_sim
    
    def _build_unigram_hidden_states(self, seq_embs):
        """
            Build unigram representations for each token from each token's embedding representation
            Each sequence becomes length 1, so encoder doesn't work off context of neighboring tokens
                * (possible) justification for this is the LSTM is only trained on very short sequences
            Arguments:
                seq_embs (torch.tensor) : N x seq_len x embedding_dim
            
            Returns:
                (torch.tensor) : N x seq_len x encoding_dim
        """
        
        batch_size, seq_len, embedding_dim = seq_embs.shape
        new_batch_size = batch_size * seq_len
        new_seq_len = 1
        
        unigrams = torch.reshape(seq_embs, (new_batch_size, new_seq_len, embedding_dim))

        unigram_hidden_states = self.get_hidden_states(unigrams)

        needed_size =  (batch_size, seq_len, self.encoding_dim)

        unigram_hidden_states = torch.reshape(unigram_hidden_states, needed_size)

        return unigram_hidden_states
    
    def _build_bigram_hidden_states(self, seq_embs, padding_indexes):
        """
            Build bigram representations for each token from each token's embedding representation
            Each token_i is represented by two bigrams, [token_(i-1), token_(i)] and [token_(i), token_(i+1)]
            These bigrams are then encoded by an encoder (bilstm)
            And then finally pooled together to create a single representation for each bigram
            At the end, for each token we have two vectors representing it
            Arguments:
                seq_embs        (torch.tensor) : N x seq_len x embedding_dim
                padding_indexes (torch.tensor) : N x seq_len
            
            Returns:
                fwd_pooled_reps, bwd_pooled_reps : both are N x seq_len x encoding_dim
        """
        batch_size, seq_len, embedding_dim = seq_embs.shape
        new_batch_size = batch_size * (seq_len+1)
        new_seq_len = 2

        padding = torch.zeros(batch_size, 1, embedding_dim) # batch_size x 1 x embedding_dim
        padding_i = torch.ones(batch_size, 1) # batch_size x 1
        if self.cuda:
            device = torch.device("cuda")
            padding = padding.to(device)
            padding_i = padding_i.to(device)

        padded_embeddings = torch.cat((padding, seq_embs, padding), 1).unsqueeze(2) # batch_size x seq_len+2 x 1 x embedding_dim
        bigrams = torch.cat((padded_embeddings[:,0:-1,:], padded_embeddings[:,1:,:]), 2) # batch_size x seq_len+1 x 2 x embedding_dim
        needed_size = (new_batch_size, new_seq_len, embedding_dim) 
        bigrams = torch.reshape(bigrams, needed_size) # batch_size * (seq_len+1) x 2 x embedding_dim

        padded_padding_indexes = torch.cat((padding_i, padding_indexes, padding_i), 1).unsqueeze(2) # batch_size x seq_len+2 x 1
        bigram_paddings = torch.cat((padded_padding_indexes[:,0:-1,:], padded_padding_indexes[:,1:,:]), 2) # batch_size x seq_len+1 x 2
        needed_size = (new_batch_size, new_seq_len)
        bigram_paddings = torch.reshape(bigram_paddings, needed_size) # batch_size * (seq_len+1) x 2

        bigram_hidden_states = self.get_hidden_states(bigrams) # batch_size * (seq_len+1) x 2 x encoding_dim
        bigram_hidden_states_d = self.encoding_dropout(bigram_hidden_states) # batch_size * (seq_len+1) x 2 x encoding_dim
        
        bigram_softmax_weights = self.get_attention_weights(bigram_hidden_states_d, bigram_paddings) # batch_size * (seq_len+1) x 1 x 2
        bigram_pooled_reps = torch.bmm(bigram_softmax_weights, bigram_hidden_states) # batch_size * (seq_len+1) x 1 x encoding_dim
        
        needed_size = (batch_size, seq_len+1, 1, self.encoding_dim)
        bigram_pooled_reps = torch.reshape(bigram_pooled_reps, needed_size).squeeze(2) # batch_size x (seq_len+1) x encoding_dim

        return bigram_pooled_reps

    def _build_trigram_hidden_states(self, seq_embs, padding_indexes):
        """
            Build trigram representations for each token from each token's embedding representation
            Each token_i is represented by three trigrams:
                1. [token_(i-2), token_(i-1), token_(i)]
                2. [token_(i-1), token_(i), token_(i+1)]
                3. [token_(i), token_(i+1), token_(i+2)]
            These trigrams are then encoded by an encoder (bilstm)
            And then finally pooled together to create a single representation for each trigram
            At the end, for each token we have three vectors representing it
            Arguments:
                seq_embs        (torch.tensor) : N x seq_len x embedding_dim
                padding_indexes (torch.tensor) : N x seq_len
            
            Returns:
                fwd_pooled_reps, mid_pooled_reps, bwd_pooled_reps : all are N x seq_len x encoding_dim
        """
        batch_size, seq_len, embedding_dim = seq_embs.shape
        new_batch_size = batch_size * (seq_len+2)
        new_seq_len = 3

        padding = torch.zeros(batch_size, 2, embedding_dim) # batch_size x 2 x embedding_dim
        padding_i = torch.ones(batch_size, 2) # batch_size x 2
        if self.cuda:
            device = torch.device("cuda")
            padding = padding.to(device)
            padding_i = padding_i.to(device)

        padded_embeddings = torch.cat((padding, seq_embs, padding), 1).unsqueeze(2) # batch_size x seq_len+4 x 1 x embedding_dim
        trigrams = torch.cat((padded_embeddings[:,0:-2,:], padded_embeddings[:,1:-1,:], padded_embeddings[:,2:,:]), 2) # batch_size x seq_len+2 x 3 x embedding_dim
        needed_size = (new_batch_size, new_seq_len, embedding_dim)
        trigrams = torch.reshape(trigrams, needed_size) # batch_size * (seq_len+2) x 3 x embedding_dim

        padded_padding_indexes = torch.cat((padding_i, padding_indexes, padding_i), 1).unsqueeze(2) # batch_size x seq_len+4 x 1
        trigram_paddings = torch.cat((padded_padding_indexes[:,0:-2,:], padded_padding_indexes[:,1:-1,:], padded_padding_indexes[:,2:,:]), 2) # batch_size x seq_len+2 x 3
        needed_size = (new_batch_size, new_seq_len)
        trigram_paddings = torch.reshape(trigram_paddings, needed_size) #  batch_size * (seq_len+2) x 3

        trigram_hidden_states = self.get_hidden_states(trigrams) # batch_size * (seq_len+2) x 3 x encoding_dim
        trigram_hidden_states_d = self.encoding_dropout(trigram_hidden_states) # batch_size * (seq_len+2) x 3 x encoding_dim
        trigram_softmax_weights = self.get_attention_weights(trigram_hidden_states_d, trigram_paddings) # batch_size * (seq_len+2) x 1 x 3
        trigram_pooled_reps = torch.bmm(trigram_softmax_weights, trigram_hidden_states) # batch_size * (seq_len+2) x 1 x encoding_dim
        
        needed_size = (batch_size, seq_len+2, 1, self.encoding_dim)
        trigram_pooled_reps = torch.reshape(trigram_pooled_reps, needed_size).squeeze(2) # batch_size x (seq_len+2) x encoding_dim

        return trigram_pooled_reps
        
    def compute_unigram_similarities(self, seq_embs, normalized_query_vectors):
        """
            Compute similarities between unigram representations and pooled query representations
            
            Arguments:
                seq_embs                 (torch.tensor) : N x seq_len x embedding_dim
                normalized_query_vectors (torch.tensor) : N x encoding_dim x 1
            
            Returns:
                (torch.tensor) :  N x seq_len x 1
        """
        unigram_hidden_states = self._build_unigram_hidden_states(seq_embs) # N x seq_len x encoding_dim
        unigram_similarities = self.compute_dot_product_between_token_rep_and_query_vectors(unigram_hidden_states,
                                                                                            normalized_query_vectors) # N x seq_len x 1
        
        return unigram_similarities
    
    def compute_bigram_similarities(self, seq_embs, padding_indexes, normalized_query_vectors):
        """
            Compute similarities between bigram representations and pooled query representations

            Arguments:
                seq_embs                 (torch.tensor) : N x seq_len x embedding_dim
                padding_indexes          (torch.tensor) : N x seq_len
                normalized_query_vectors (torch.tensor) : N x encoding_dim x 1
            
            Returns:
                fwd_similarities, bwd_similarities :  N x seq_len x 1
        """
        _, seq_len, _ = seq_embs.shape
        bigram_hidden_states = self._build_bigram_hidden_states(seq_embs, padding_indexes) # N x seq_len+1 x encoding_dim
        bigram_similarities = self.compute_dot_product_between_token_rep_and_query_vectors(bigram_hidden_states,
                                                                                            normalized_query_vectors) # N x seq_len+1 x 1
        bwd_similarities = bigram_similarities[:,:seq_len,:] # batch_size x seq_len x 1
        fwd_similarities = bigram_similarities[:,1:,:] # batch_size x seq_len x 1

        return fwd_similarities, bwd_similarities
    
    def compute_trigram_similarities(self, seq_embs, padding_indexes, normalized_query_vectors):
        """
            Compute similarities between trigram representations and pooled query representations

            Arguments:
                seq_embs                 (torch.tensor) : N x seq_len x embedding_dim
                padding_indexes          (torch.tensor) : N x seq_len
                normalized_query_vectors (torch.tensor) : N x encoding_dim x 1
            
            Returns:
                fwd_similarities, mid_similarities, bwd_similarities :  N x seq_len x 1
        """
        _, seq_len, _ = seq_embs.shape
        trigram_hidden_states = self._build_trigram_hidden_states(seq_embs, padding_indexes) # N x seq_len+2 x encoding_dim
        trigram_similarities = self.compute_dot_product_between_token_rep_and_query_vectors(trigram_hidden_states,
                                                                                            normalized_query_vectors) # N x seq_len+2 x 1
        bwd_similarities = trigram_similarities[:,:seq_len,:] # batch_size x seq_len x 1
        mid_similarities = trigram_similarities[:,1:seq_len+1,:] # batch_size x seq_len x 1
        fwd_similarities = trigram_similarities[:,2:,:] # batch_size x seq_len x 1

        return fwd_similarities, mid_similarities, bwd_similarities
    
    def similarity_head(self, encoded_cosines):
        """
            MLP head on top of encoded representation of cosine similarities

            Arguments:
                combined_cosines (torch.tensor) : N x seq_len x 64
            
            Returns:
                (torch.tensor) : N x seq_len x 1
        """        
        compressed_cosines = self.weight_linear_layer_1(encoded_cosines) # N x seq_len x 32
        compressed_cosines = self.weight_activation_function(compressed_cosines)
        compressed_cosines = self.mlp_dropout(compressed_cosines)
        
        compressed_cosines = self.weight_linear_layer_2(compressed_cosines) # N x seq_len x 16
        compressed_cosines = self.weight_activation_function(compressed_cosines)
        compressed_cosines = self.mlp_dropout(compressed_cosines)
        
        compressed_cosines = self.weight_linear_layer_3(compressed_cosines) # N x seq_len x 8
        compressed_cosines = self.weight_activation_function(compressed_cosines)
        compressed_cosines = self.mlp_dropout(compressed_cosines)
            
        similarity_scores = self.weight_final_layer(compressed_cosines) # N x seq_len x 1

        return similarity_scores

    def pre_train_get_similarity(self, seq_embeddings, padding_indexes, query_vectors):
        """
            Compute similarity between each token in a sequence and the corresponding query_vector per the
            NExT paper's specification. Steps followed:
                1. Get a token's encoder representation (unigram)
                2. Construct pooled representation for the bigrams that include a given token
                3. Construct pooled representation for the trigrams that include a given token
                4. Compute dot product between query vector and a tokens's encoder representation
                5. Compute dot product between query vector and bigrams that include a given token
                6. Compute dot product between query vector and trigrams that include a given token
                7. Compute a similarity vector containing the cosine between the token and a query using
                   all 6 representations
                8. Send cosine vectors through a bi-lstm
                9. Send output of bi-lstm through MLP head to produce final matching score
            
            Arguments:
                seq_embeddings  (torch.tensor) : N x seq_len x embedding_dim
                padding_indexes (torch.tensor) : N x seq_len
                query_vectors   (torch.tensor) : N x 1 x encoding_dim
            
            Returns:
                (torch.tensor) : N x seq_len x 1
        """        
        batch_size, seq_len, _ = seq_embeddings.shape

        normalized_query_vectors = f.normalize(query_vectors + 1e-5, p=2, dim=2)
        normalized_query_vectors = normalized_query_vectors.permute(0, 2, 1) # arranging query_vectors to be N x encoding_dim x 1
        
        if self.sliding_win_size == 3:
            uni_cosine = self.compute_unigram_similarities(seq_embeddings, normalized_query_vectors) # N x seq_len x 1
            bigram_cosines = self.compute_bigram_similarities(seq_embeddings, padding_indexes, normalized_query_vectors)
            fwd_bigram_hs_cosine, bwd_bigram_hs_cosine = bigram_cosines # N x seq_len x 1 (all)
            trigram_cosines = self.compute_trigram_similarities(seq_embeddings, padding_indexes, normalized_query_vectors)
            fwd_trigram_hs_cosine, mid_trigram_hs_cosine, bwd_trigram_hs_cosine = trigram_cosines # N x seq_len x 1 (all)

            combined_cosines = torch.cat((uni_cosine,
                                          fwd_bigram_hs_cosine,
                                          bwd_bigram_hs_cosine,
                                          fwd_trigram_hs_cosine,
                                          mid_trigram_hs_cosine,
                                          bwd_trigram_hs_cosine), 2) # combined_cosines = N x seq_len x number_of_cosines

            encoded_cosines, _ = self.cosine_bilstm(combined_cosines) # N x seq_len x 64
            encoded_cosines = self.encoding_dropout(encoded_cosines)
            
            similarity_scores = self.similarity_head(encoded_cosines) # N x seq_len x 1
            
            return similarity_scores
        
    def train_get_similarity(self, seq_embeddings, padding_indexes, query_vectors):
        """
            Compute similarity between each token in a sequence and the corresponding query_vector per the
            NExT paper's specification. Steps followed:
                1. Get a token's encoder representation (unigram)
                2. Construct pooled representation for the bigrams that include a given token
                3. Construct pooled representation for the trigrams that include a given token
                4. Compute dot product between query vector and a tokens's encoder representation
                5. Compute dot product between query vector and bigrams that include a given token
                6. Compute dot product between query vector and trigrams that include a given token
                7. Compute a similarity vector containing the cosine between the token and a query using
                   all 6 representations
                8. Send cosine vectors through a bi-lstm
                9. Send output of bi-lstm through MLP head to produce final matching score
            
            Arguments:
                seq_embeddings  (torch.tensor) : N x seq_len x embedding_dim
                padding_indexes (torch.tensor) : N x seq_len
                query_vectors   (torch.tensor) : Q x 1 x encoding_dim
            
            Returns:
                (torch.tensor) : N x seq_len x 1
        """        
        batch_size, seq_len, _ = seq_embeddings.shape
        number_of_queries, _, _ = query_vectors.shape

        normalized_query_vectors = f.normalize(query_vectors + 1e-5, p=2, dim=2)
        normalized_query_vectors = normalized_query_vectors.permute(2, 0, 1).squeeze(2) # arranging query_vectors to be encoding_dim x Q
        
        if self.sliding_win_size == 3:
            unigram_reps = self._build_unigram_hidden_states(seq_embeddings) # N x seq_len x encoding_dim
            bigram_reps = self._build_bigram_hidden_states(seq_embeddings, padding_indexes) # batch_size x (seq_len+1) x encoding_dim
            fwd_bigram_reps = bigram_reps[:,1:,:] # batch_size x seq_len x encoding_dim
            bwd_bigram_reps = bigram_reps[:,:seq_len,:] # batch_size x seq_len x encoding_dim
            trigram_reps = self._build_trigram_hidden_states(seq_embeddings, padding_indexes) # batch_size x (seq_len+2) x encoding_dim
            fwd_trigram_reps = trigram_reps[:,2:,:] # batch_size x seq_len x encoding_dim
            mid_trigram_reps = trigram_reps[:,1:seq_len+1,:] # batch_size x seq_len x encoding_dim
            bwd_trigram_reps = trigram_reps[:,:seq_len,:] # batch_size x seq_len x encoding_dim

            reps = torch.cat((unigram_reps,
                              fwd_bigram_reps,
                              bwd_bigram_reps,
                              fwd_trigram_reps,
                              mid_trigram_reps,
                              bwd_trigram_reps), 1) # combined_cosines = N x seq_len*number_of_cosines x encoding_dim
            
            reps = torch.reshape(reps, (batch_size, self.number_of_cosines, seq_len, self.encoding_dim))

            normalized_reps = f.normalize(reps + 1e-5, p=2, dim=3)

            all_cosines = torch.matmul(normalized_reps, normalized_query_vectors).permute(3, 0, 2, 1) # Q x N x seq_len x number_of_cosines

            encoded_cosines = torch.cat([self.cosine_bilstm(query_cosine)[0] for query_cosine in all_cosines], dim=0) # Q*N x seq_len x 64
            encoded_cosines = self.encoding_dropout(encoded_cosines)
            
            similarity_scores = self.similarity_head(encoded_cosines).squeeze(2) # Q*N x seq_len

            similarity_scores = torch.reshape(similarity_scores, (number_of_queries, batch_size , seq_len)) # Q x N x seq_len
            
            return similarity_scores
        
    def find_forward(self, seqs, queries, lower_bound):
        """
            Forward function for computing L_find when pre-training Find Module

            Arguments:
                seqs        (torch.tensor) : N x seq_len_i, token sequences for current batch
                queries     (torch.tensor) : N x seq_len_j, token sequences for corresponding queries
                lower_bound (torch.tensor) : N x seq_len_i, constant tensor with lower_bound for token
                                                            similarity score
            
            Returns:
                (torch.tensor) : N x seq_len_i, similarity scores between each token in a sequence and
                                 the corresponding query
        """
        query_encodings, query_padding_indexes = self.encode_tokens(queries) # N x seq_len_j x encdoing_dim, N x seq_len_j
        seq_embeddings, seq_padding_indexes = self.get_embeddings(seqs) # N x seq_len_i x embedding_dim, N x seq_len_i

        if self.cuda:
            device = torch.device("cuda")
            query_padding_indexes = query_padding_indexes.to(device)

        pooled_query_encodings = self.attention_pooling(query_encodings, query_padding_indexes) # N x 1 x encoding_dim
        seq_similarities = self.pre_train_get_similarity(seq_embeddings, seq_padding_indexes, pooled_query_encodings).squeeze(2) # N x seq_len

        seq_similarities = torch.clamp(seq_similarities, min=lower_bound)

        return seq_similarities

    def sim_forward(self, queries, query_index_matrix, neg_query_index_matrix, tau=0.9):
        """
            Forward function for computing L_sim when pre-training Find Module

            Remember if distance between two vectors are small, then cosine between those vectors is close to
            1. So by taking max of cosines, you're actually finding min distance between vectors. Hence, why
            tau is introduced when computing max distance between a query and queries of the same class.

            Arguments:
                queries                (torch.tensor) : N x seq_len, token sequences each query
                query_index_matrix     (torch.tensor) : N x N, indicates per query the queries that share the same label
                neg_query_index_matrix (torch.tensor) : N x N, indicates per query the queries that don't share the same label
                tau            (float) : constant used in NExT paper
            
            Returns:
                pos_scores, neg_scores : per query the maximum distance score between the query and queires
                                         of the same class, per query the minimum distance score between the
                                         query and queires of the same class
        """
        query_embeddings, query_padding_indexes = self.get_embeddings(queries) # N x q_len x embedding_dim, N x q_len
        query_encodings = self.get_hidden_states(query_embeddings) # N x q_len x encoding_dim
        query_encodings_d = self.encoding_dropout(query_encodings) # N x q_len x encoding_dim

        query_attention_weights = self.get_attention_weights(query_encodings_d, query_padding_indexes) # N x 1 x q_len
        pooled_query = torch.bmm(query_attention_weights, query_encodings) # N x 1 x encoding_dim
        pooled_query_d = torch.bmm(query_attention_weights, query_encodings_d) # N x 1 x encoding_dim

        pooled_query = f.normalize(pooled_query + 1e-5, p=2, dim=2).squeeze(1) # N x encoding_dim
        pooled_query_d = f.normalize(pooled_query_d + 1e-5, p=2, dim=2).squeeze(1).permute(1,0) # encoding_dim x N

        query_similarities = torch.mm(pooled_query, pooled_query_d) # N x N

        query_pos_similarities = torch.square(torch.clamp(tau - query_similarities, min=0.0))
        query_neg_similarities = torch.square(torch.clamp(query_similarities, min=0.0))

        pos_scores = torch.max(query_pos_similarities - 1e30*neg_query_index_matrix, axis=1).values
        pos_scores = torch.clamp(pos_scores, min=0.0) # incase a class only has one example, no loss
        neg_scores = torch.max(query_neg_similarities - 1e30*query_index_matrix, axis=1).values
        neg_scores = torch.clamp(neg_scores, min=0.0)

        return pos_scores, neg_scores
    
    def soft_matching_forward(self, seqs, queries, lower_bound):
        """
            Forward function for computing similarity between each token in a sequence to each query

            Arguments:
                seqs        (torch.tensor) : N x seq_len_i, token sequences for current batch
                queries     (torch.tensor) : Q x seq_len_j, token sequences for queries
                lower_bound        (float) : lower_bound for token similarity score
            Returns:
                tup(torch.tensor) : N x seq_len_i x Q, N x seq_len_i; similarity scores between each token in a
                                    sequence and each query, along with an indication of padding indices
        """
        query_encodings, query_padding_indexes = self.encode_tokens(queries) # Q x seq_len_j x encdoing_dim, Q x seq_len_j
        seq_embeddings, seq_padding_indexes = self.get_embeddings(seqs) # N x seq_len_i x embedding_dim, N x seq_len_i

        if self.cuda:
            device = torch.device("cuda")
            query_padding_indexes = query_padding_indexes.to(device)

        pooled_query_encodings = self.attention_pooling(query_encodings, query_padding_indexes) # Q x 1 x encoding_dim
        seq_similarities = self.train_get_similarity(seq_embeddings, seq_padding_indexes, pooled_query_encodings) # Q x N x seq_len

        seq_similarities = torch.clamp(seq_similarities, min=lower_bound)

        seq_padding_indexes = 1.0 - seq_padding_indexes
        seq_similarities = torch.sigmoid(seq_similarities) * seq_padding_indexes
        seq_similarities = seq_similarities.permute(1, 2, 0)

        return seq_similarities
