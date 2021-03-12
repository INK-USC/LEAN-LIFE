import torch
import torch.nn as nn
import torch.nn.functional as f

class Find_Module(nn.Module):
    """
        Find Module per the NExT paper's description
    """
    def __init__(self, emb_weight, padding_idx, emb_dim, hidden_dim, cuda,
                 n_layers=2, embedding_dropout=0.04, encoding_dropout=0.5, sliding_win_size=3,
                 padding_score=-1*1e30):
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
                embedding_dropout (float) : percentage of initial embedding's components to be randomly
                                            zeroed out 
                encoding_dropout  (float) : percentage of vector's representation to be randomly zeroed
                                            out before pooling
                sliding_win_size    (int) : size of window to consider when building pooled representations
                                            for a token (as of now this is fixed at 3)
        """
        super(Find_Module, self).__init__()

        self.padding_idx = padding_idx
        self.padding_score = padding_score
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = 2 * hidden_dim
        self.sliding_win_size = sliding_win_size
        self.number_of_cosines = sum([i+1 for i in range(self.sliding_win_size)])
        self.cuda = cuda

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.embeddings = nn.Embedding.from_pretrained(emb_weight, freeze=False, padding_idx=self.padding_idx)
        self.bilstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.encoding_dropout = nn.Dropout(p=encoding_dropout)
                
        self.attention_matrix = nn.Linear(self.encoding_dim, self.encoding_dim)
        temp_att_vector = torch.zeros(self.encoding_dim, 1)
        nn.init.xavier_uniform_(temp_att_vector)
        self.attention_vector = nn.Parameter(temp_att_vector, requires_grad=True)
        self.attn_softmax = nn.Softmax(dim=2)

        diagonal_vector = torch.zeros(self.encoding_dim, 1)
        nn.init.xavier_uniform_(diagonal_vector)
        diagonal_vector = diagonal_vector.squeeze(1)
        self.feature_weight_matrix = nn.Parameter(torch.diag(input=diagonal_vector), requires_grad=True)

        self.weight_linear_layer = nn.Linear(self.number_of_cosines, 1)
    
    
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

    def encode_tokens(self, seqs):
        """
            Convert a sequence of tokens into their vector representations
            Arguments:
                seqs (torch.tensor) : N x seq_len
            Returns:
                (torch.tensor) : N x seq_len x encoding_dim
        """
        padding_indexes = seqs == self.padding_idx
        padding_indexes = padding_indexes.float()
        if self.cuda:
            device = torch.device("cuda")
            padding_indexes = padding_indexes.to(device)
        seq_embs = self.embeddings(seqs) # seq_embs = N x seq_len x hidden_dim
        seq_embs = self.embedding_dropout(seq_embs)
        hidden_states, _ =  self.bilstm(seq_embs) # hidden_states =  N x seq_len x 2*hidden_dim
        hidden_states = self.encoding_dropout(hidden_states)

        return padding_indexes, hidden_states
    
    def build_forward_backward_bigram_hidden_states(self, hidden_states, padding_indexes):
        """
            Construct pooled representations for the bigrams surround a token w_i

            Arguments:
                hidden_states   (torch.tensor) : N x seq_len x encoding_dim
                padding_indexes (torch.tensor) : N x seq_len x encoding_dim
            
            Returns:
                forward_bigram_hidden_states, backward_bigram_hidden_states : both are N x seq_len x encoding_dim
        """
        batch_size, seq_len, encoding_dim = hidden_states.shape
        
        updated_hidden_states = torch.zeros(batch_size, seq_len+2,encoding_dim) 
        updated_hidden_states[:,1:seq_len+1,:] = hidden_states
        updated_padding_indexes = torch.empty(batch_size, seq_len+2).fill_(1.)
        updated_padding_indexes[:,1:seq_len+1] = padding_indexes
        if self.cuda:
            device = torch.device("cuda")
            updated_hidden_states = updated_hidden_states.to(device)
            updated_padding_indexes = updated_padding_indexes.to(device)

        forward_bigram_hidden_states = torch.zeros(batch_size, seq_len, encoding_dim) #geting pooled hidden states for w_i, w_i+1
        backward_bigram_hidden_states = torch.zeros(batch_size, seq_len, encoding_dim) #geting pooled hidden states for w_i-1, w_i
        
        for i in range(batch_size):
            for j in range(1, seq_len+1):
                bigram_hidden_states = updated_hidden_states[i, j:j+2, :].unsqueeze(0)
                bigram_padding_indexes = updated_padding_indexes[i, j:j+2].unsqueeze(0)
                forward_bigram_hidden_states[i, j-1, :] = self.attention_pooling(bigram_hidden_states, bigram_padding_indexes).squeeze(0)
                
                bigram_hidden_states = updated_hidden_states[i, j-1:j+1, :].unsqueeze(0)
                bigram_padding_indexes = updated_padding_indexes[i, j-1:j+1].unsqueeze(0)
                backward_bigram_hidden_states[i, j-1, :] = self.attention_pooling(bigram_hidden_states, bigram_padding_indexes).squeeze(0)
        
        return forward_bigram_hidden_states, backward_bigram_hidden_states
    
    def build_forward_backward_trigram_hidden_states(self, hidden_states, padding_indexes):
        """
            Construct pooled representations for the trigrams surrounding a token w_i

            Arguments:
                hidden_states   (torch.tensor) : N x seq_len x encoding_dim
                padding_indexes (torch.tensor) : N x seq_len x encoding_dim
            
            Returns:
                forward_trigram_hidden_states  : N x seq_len x encoding_dim
                middle_trigram_hidden_states   : N x seq_len x encoding_dim
                backward_trigram_hidden_states : N x seq_len x encoding_dim
        """
        batch_size, seq_len, encoding_dim = hidden_states.shape
        
        updated_hidden_states = torch.zeros(batch_size, seq_len+4,encoding_dim) 
        updated_hidden_states[:,2:seq_len+2,:] = hidden_states
        updated_padding_indexes = torch.empty(batch_size, seq_len+4).fill_(1.)
        updated_padding_indexes[:,2:seq_len+2] = padding_indexes
        if self.cuda:
            device = torch.device("cuda")
            updated_hidden_states = updated_hidden_states.to(device)
            updated_padding_indexes = updated_padding_indexes.to(device)
        
        forward_trigram_hidden_states = torch.zeros(batch_size, seq_len, encoding_dim) #geting pooled hidden states for w_i, w_i+1, w_i+2
        middle_trigram_hidden_states = torch.zeros(batch_size, seq_len, encoding_dim) #geting pooled hidden states for w_i-1, w_i, w_i+1
        backward_trigram_hidden_states = torch.zeros(batch_size, seq_len, encoding_dim) #geting pooled hidden states for w_i-2, w_i-1, w_i
        
        for i in range(batch_size):
            for j in range(2, seq_len+2):
                trigram_hidden_states = updated_hidden_states[i, j:j+3, :].unsqueeze(0)
                trigram_padding_indexes = updated_padding_indexes[i, j:j+3].unsqueeze(0)
                forward_trigram_hidden_states[i, j-2, :] = self.attention_pooling(trigram_hidden_states, trigram_padding_indexes).squeeze(0)
                
                trigram_hidden_states = updated_hidden_states[i, j-1:j+2, :].unsqueeze(0)
                trigram_padding_indexes = updated_padding_indexes[i, j-1:j+2].unsqueeze(0)
                middle_trigram_hidden_states[i, j-2, :] = self.attention_pooling(trigram_hidden_states, trigram_padding_indexes).squeeze(0)

                trigram_hidden_states = updated_hidden_states[i, j-2:j+1, :].unsqueeze(0)
                trigram_padding_indexes = updated_padding_indexes[i, j-2:j+1].unsqueeze(0)
                backward_trigram_hidden_states[i, j-2, :] = self.attention_pooling(trigram_hidden_states, trigram_padding_indexes).squeeze(0)
        
        return forward_trigram_hidden_states, middle_trigram_hidden_states, backward_trigram_hidden_states
    
    def compute_dot_product_between_token_rep_and_query_vectors(self, token_rep, normalized_query_vectors):
        """
            Computes dot product between each sequence's tokens' representations and a pooled
            representation of the corresponding query vector. The pooled query representations
            are assumed to have already been normalized.

            Arguments:
                token_rep                (torch.tensor) : N x seq_len x encoding_dim
                normalized_query_vectors (torch.tensor) : N x encoding_dim x 1
            
            Returns:
                (torch.tensor) : N x seq_len, cosine similarity between each token's representation
                                    and the query vector
        """
        normalized_token_rep = f.normalize(token_rep, p=2, dim=2) # normalizing rows of each matrix in the batch
        cosine_sim = torch.matmul(normalized_token_rep, normalized_query_vectors).squeeze(2)
        
        return cosine_sim
    
    def pre_train_get_similarity(self, hidden_states, padding_indexes, query_vectors):
        """
            Compute similarity between each token in a sequence and the corresponding query_vector per the
            NExT paper's specification. Steps followed:
                1. Construct pooled representation for the bigram that starts with a token
                2. Construct pooled representation for the bigram that ends with a token
                3. Compute dot product between query vector and a tokens's original representation
                4. Compute dot product between query vector and bigram that starts with token
                5. Compute dot product between query vector and bigram that ends with token
                6. Average dot product scores per token by tunable weight vector
            
            Arguments:
                hidden_states (torch.tensor) : N x seq_len x encoding_dim
                query_vectors (torch.tensor) : N x 1 x encoding_dim
            
            Returns:
                (torch.tensor) : N x seq_len x 1
        """
        updated_query_vectors = torch.matmul(query_vectors, self.feature_weight_matrix) #updated_query_vectors = N x 1 x encoding_dim <-zqD
        batch_size, seq_len, _ = hidden_states.shape
        if self.sliding_win_size == 3:
            bigram_states = self.build_forward_backward_bigram_hidden_states(hidden_states, padding_indexes)
            forward_bigram_hidden_states, backward_bigram_hidden_states = bigram_states
            trigram_states = self.build_forward_backward_trigram_hidden_states(hidden_states, padding_indexes)
            forward_trigram_hidden_states, middle_trigram_hidden_states, backward_trigram_hidden_states = trigram_states
            if self.cuda:
                device = torch.device("cuda")
                forward_bigram_hidden_states = forward_bigram_hidden_states.to(device)
                backward_bigram_hidden_states = backward_bigram_hidden_states.to(device)
                forward_trigram_hidden_states = forward_trigram_hidden_states.to(device)
                middle_trigram_hidden_states = middle_trigram_hidden_states.to(device)
                backward_trigram_hidden_states = backward_trigram_hidden_states.to(device)
            
            updated_hs = torch.matmul(hidden_states, self.feature_weight_matrix) #updated_hs = N x seq_len x encoding_dim
            updated_bigram_forward_hs = torch.matmul(forward_bigram_hidden_states, self.feature_weight_matrix) #updated_forward_hs = N x seq_len x encoding_dim
            updated_bigram_backward_hs = torch.matmul(backward_bigram_hidden_states, self.feature_weight_matrix) #updated_backward_hs = N x seq_len x encoding_dim
            update_trigram_forward_hs = torch.matmul(forward_trigram_hidden_states, self.feature_weight_matrix)
            update_trigram_middle_hs = torch.matmul(middle_trigram_hidden_states, self.feature_weight_matrix)
            update_trigram_backward_hs = torch.matmul(backward_trigram_hidden_states, self.feature_weight_matrix)

            normalized_query_vectors = f.normalize(updated_query_vectors, p=2, dim=2) # normalizing rows of each matrix in the batch
            normalized_query_vectors = normalized_query_vectors.permute(0, 2, 1) # arranging query_vectors to be N x encoding_dim x 1

            hs_cosine = self.compute_dot_product_between_token_rep_and_query_vectors(updated_hs,
                                                                                     normalized_query_vectors)
            fwd_bigram_hs_cosine = self.compute_dot_product_between_token_rep_and_query_vectors(updated_bigram_forward_hs, 
                                                                                                normalized_query_vectors)
            bwd_bigram_hs_cosine = self.compute_dot_product_between_token_rep_and_query_vectors(updated_bigram_backward_hs,
                                                                                                normalized_query_vectors)	
            fwd_trigram_hs_cosine = self.compute_dot_product_between_token_rep_and_query_vectors(update_trigram_forward_hs, 
                                                                                                 normalized_query_vectors)
            mid_trigram_hs_cosine = self.compute_dot_product_between_token_rep_and_query_vectors(update_trigram_middle_hs, 
                                                                                                 normalized_query_vectors)
            bwd_trigram_hs_cosine = self.compute_dot_product_between_token_rep_and_query_vectors(update_trigram_backward_hs,
                                                                                                 normalized_query_vectors)	 

            combined_cosines = torch.zeros(batch_size, seq_len, self.number_of_cosines) # building cosine similarity of sliding window
            combined_cosines[:,:,0] = hs_cosine
            combined_cosines[:,:,1] = fwd_bigram_hs_cosine
            combined_cosines[:,:,2] = bwd_bigram_hs_cosine
            combined_cosines[:,:,3] = fwd_trigram_hs_cosine
            combined_cosines[:,:,4] = mid_trigram_hs_cosine
            combined_cosines[:,:,5] = bwd_trigram_hs_cosine

            if self.cuda:
                device = torch.device("cuda")
                combined_cosines = combined_cosines.to(device) # cobined_cosines = N x seq_len x sliding_win_size
            
            similarity_scores = self.weight_linear_layer(combined_cosines) # similarity_scores = N x seq_len x 1
            
            return similarity_scores
        
    def find_forward(self, seqs, queries):
        """
            Forward function for computing L_find when pre-training Find Module

            Arguments:
                seqs    (torch.tensor) : N x seq_len_i, token sequences for current batch
                queries (torch.tensor) : N x seq_len_j, token sequences for corresponding queries
            
            Returns:
                (torch.tensor) : N x seq_len_i, similarity scores between each token in a sequence and
                                 the corresponding query
        """
        query_padding_indexes, query_encodings = self.encode_tokens(queries) # N x seq_len x encoding_dim
        seq_padding_indexes, seq_encodings = self.encode_tokens(seqs) # N x seq_len x encoding_dim

        pooled_query_encodings = self.attention_pooling(query_encodings, query_padding_indexes) # N x 1 x encoding_dim
        seq_similarities = self.pre_train_get_similarity(seq_encodings, seq_padding_indexes, pooled_query_encodings).squeeze(2) # N x seq_len

        return seq_similarities
    
    def compute_sim_query(self, query_vector, pos_tensor, neg_tensor, tau=0.9):
        """
            Given a query vector, the set of all query vectors that are of the same class, and the set of all
            query vectors of a different class, compute the min similarity between the query vector and 
            query vectors of the same class and the max similarity between the query vector and query vectors
            of a different class.

            Remember if distance between two vectors are small, then cosine between those vectors is close to
            1. So by taking max of cosines, you're actually finding min distance between vectors. Hence, why
            tau is introduced when computing max distance between a query and queries of the same class.

            Arguments:
                query_vector (torch.tensor) : 1 x encoding_dim
                pos_tensor   (torch.tensor) : pos_count x encoding_dim, all query vectors of the same class
                neg_tensor   (torch.tensor) : neg_count x encoding_dim, all query vectors of different classes
                tau                 (float) : constant used in NExT paper
            
            Returns:
                pos_score, neg_score : float representing max distance score for queries of the same class,
                                       and min distance score for queries of a different class
        """
        updated_query_vector = torch.matmul(query_vector, self.feature_weight_matrix) # 1 x encoding_dim
        updated_pos_tensor = torch.matmul(pos_tensor, self.feature_weight_matrix) # pos_count x encoding_dim
        updated_neg_tensor = torch.matmul(neg_tensor, self.feature_weight_matrix) # neg_count x encoding_dim
        
        normalized_query_vector = f.normalize(updated_query_vector, p=2, dim=1).permute(1,0) # encoding_dim x 1
        normalized_pos_tensor = f.normalize(updated_pos_tensor, p=2, dim=1)
        normalized_neg_tensor = f.normalize(updated_neg_tensor, p=2, dim=1)

        pos_scores = tau - torch.matmul(normalized_pos_tensor, normalized_query_vector).squeeze(1) # pos_count 
        neg_scores = torch.matmul(normalized_neg_tensor, normalized_query_vector).squeeze(1) # neg_count

        pos_score = torch.max(pos_scores**2)
        neg_score = torch.max(neg_scores**2)

        return pos_score, neg_score

    def sim_forward(self, queries, labels):
        """
            Forward function for computing L_sim when pre-training Find Module

            Arguments:
                queries (torch.tensor) : N x seq_len, token sequences each query
                labels           (arr) : corresponding label for each query
            
            Returns:
                pos_scores, neg_scores : per query the maximum distance score between the query and queires
                                         of the same class, per query the minimum distance score between the
                                         query and queires of the same class	                         
        """
        queries_by_label = {}
        query_padding_indexes, query_encodings = self.encode_tokens(queries) # N x seq_len x encoding_dim
        pooled_reps = self.attention_pooling(query_encodings, query_padding_indexes).squeeze(1) # N  x encoding_dim
        for i, label in enumerate(labels):
            if label in queries_by_label:
                queries_by_label[label][i] = pooled_reps[i]
            else:
                queries_by_label[label] = {i : pooled_reps[i]}
        
        pos_scores = torch.zeros(len(labels))
        neg_scores = torch.zeros(len(labels))

        for i, label in enumerate(labels):
            query_rep = pooled_reps[i].unsqueeze(0)
            pos_tensor_array = [queries_by_label[label][j] for j in list(queries_by_label[label].keys()) if j != i]
            neg_tensor_array = []
            for label_2 in queries_by_label:
                if label_2 != label:
                    for key in queries_by_label[label]:
                        neg_tensor_array.append(queries_by_label[label][key])
            
            if len(pos_tensor_array) and len(neg_tensor_array):
                pos_tensor = torch.stack(pos_tensor_array) # pos_count x encoding_dim
                neg_tensor = torch.stack(neg_tensor_array) # neg_count x encoding_dim
                pos_score, neg_score = self.compute_sim_query(query_rep, pos_tensor, neg_tensor)
                pos_scores[i] = pos_score
                neg_scores[i] = neg_score
        
        if self.cuda:
            device = torch.device("cuda")
            pos_scores = pos_scores.to(device)
            neg_scores = neg_scores.to(device)

        return pos_scores, neg_scores