import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, device, bidirectional,
                 dropout=(0.4, 0.0, 0.0)):
        super(LSTMClassifier, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout[1])
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.dropout2 = nn.Dropout(p=dropout[2])
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)

    def disable_dropout(self):
        self.lstm.dropout = .0
        self.dropout1.p = .0
        self.dropout2.p = .0

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device))

    def forward(self, batch, lengths=None):
        self.hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        # pack sequence if lengths available(during training)
        if lengths:
            batch = pack_padded_sequence(batch, lengths, batch_first=True)
        # hidden - [1, batch_size, hidden_dim]
        output, self.hidden = self.lstm(batch, self.hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        # [batch_size,seq_len, hidden_dim]
        output = self.bn1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        output = self.hidden2out(output)
        return output