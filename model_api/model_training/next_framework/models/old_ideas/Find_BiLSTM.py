import torch
import torch.nn as nn
import pdb

# NOT USED FOR NOW
class FIND_BiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, layers, cuda):
        super(FIND_BiLSTM, self).__init__()
        self.dropout_pct = dropout
        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cuda = cuda
        
        temp_forward_lstms = []
        # temp_forward_linears = []
        temp_backward_lstms = []
        # temp_backward_linears = []
        
        temp_forward_lstms.append(nn.LSTM(self.input_dim, self.output_dim, batch_first=True))
        temp_backward_lstms.append(nn.LSTM(self.input_dim, self.output_dim, batch_first=True))
        
        # temp_forward_linears.append(nn.Linear(self.input_dim, self.input_dim))
        # temp_backward_linears.append(nn.Linear(self.input_dim, self.input_dim))
        
        if self.layers > 1:
            for i in range(1, layers):
                temp_forward_lstms.append(nn.LSTM(self.output_dim, self.output_dim, batch_first=True))
                temp_backward_lstms.append(nn.LSTM(self.output_dim, self.output_dim, batch_first=True))
                # temp_forward_linears.append(nn.Linear(2*self.output_dim, 2*self.output_dim))
                # temp_backward_linears.append(nn.Linear(2*self.output_dim, 2*self.output_dim))
        
        self.forward_lstms = nn.ModuleList(temp_forward_lstms)
        # self.forward_linears = nn.ModuleList(temp_forward_linears)
        self.backward_lstms = nn.ModuleList(temp_backward_lstms)
        # self.backward_linears = nn.ModuleList(temp_backward_linears)

        if self.cuda:
            device = torch.device("cuda")
            for i in range(layers):
                self.forward_lstms[i] = self.forward_lstms[i].to(device)
                # self.forward_linears[i] = self.forward_linears[i].to(device)
                self.backward_lstms[i] = self.backward_lstms[i].to(device)
                # self.backward_linears[i] = self.backward_linears[i].to(device)

        self.dropout = nn.Dropout(p=self.dropout_pct)
    
    def forward(self, input_seqs):
        dropout = self.dropout_pct < 1.0
        next_input = (input_seqs, input_seqs)
        for layer, fwd_lstm in enumerate(self.forward_lstms):
            if dropout:
                next_input = (self.dropout(next_input[0]), self.dropout(next_input[1]))

            # fwd_input = self.forward_linears[layer](next_input)
            fwd_hidden_states, _ = fwd_lstm(next_input[0])
            
            reversed_next_input = torch.flip(next_input[1], (2,))
            # bwd_input = self.backward_linears[layer](reversed_next_input)
            bwd_hidden_states, _ = self.backward_lstms[layer](reversed_next_input)
            reversed_bwd_hidden_states = torch.flip(bwd_hidden_states, (2,))
            
            next_input = (fwd_hidden_states, reversed_bwd_hidden_states)

        return torch.cat([next_input[0], next_input[1]], 2)


