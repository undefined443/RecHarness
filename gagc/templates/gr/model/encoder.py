import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)
        return x.unsqueeze(0).contiguous()
