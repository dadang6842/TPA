import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMBackbone(nn.Module):
    def __init__(self, in_ch=3, d_model=128, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.projection = nn.Linear(hidden_dim * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.projection(lstm_out) 
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out