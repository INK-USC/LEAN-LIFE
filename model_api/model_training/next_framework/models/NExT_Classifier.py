import torch
import torch.nn as nn
from models.BiLSTM_Att_Clf import BiLSTM_Att_Clf
from models.Find_Module import Find_Module

class NExT_Classifier(nn.Module):
    def __init__(self, emb_weight, padding_idx, emb_dim, hidden_dim, cuda, number_of_classes,
                 find_module_path, n_layers=2, clf_encoding_dropout=0.5, find_encoding_dropout=0.1,
                 sliding_window_size=3, padding_score=-1e30, ):

        super(NExT_Classifier, self).__init__()

        self.clf = BiLSTM_Att_Clf(emb_weight, padding_idx, emb_dim, hidden_dim, cuda, number_of_classes,
                                  n_layers, clf_encoding_dropout, padding_score)
        
        self.find_module = Find_Module(emb_weight, padding_idx, emb_dim, hidden_dim, cuda, n_layers,
                                       find_encoding_dropout, sliding_window_size, padding_score)
        
        self.find_module.load_state_dict(torch.load(find_module_path))
    
    def build_query_score_matrix(self, seqs, queries, lower_bound):
        lfind_output, seq_padding_indexes = self.find_module.soft_matching_forward(seqs, queries, lower_bound) # Q x B x seq_len, B x seq_len
        seq_padding_indexes = 1.0 - seq_padding_indexes
        lfind_output = torch.sigmoid(lfind_output) * seq_padding_indexes
        lfind_output = lfind_output.permute(1, 2, 0) # B x seq_len x Q

        return lfind_output

    
    def sim_forward(self, queries, query_index_matrix, neg_query_index_matrix, tau=0.9):
        return self.find_module.sim_forward(queries, query_index_matrix, neg_query_index_matrix, tau)
    
    def predictions(self, seqs):
        return self.clf(seqs)




