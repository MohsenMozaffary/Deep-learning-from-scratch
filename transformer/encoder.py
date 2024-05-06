from transformer_main import Transformer
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, seq_len, feature_size):
        super(Encoder, self).__init__()
        self.features_size = feature_size
        self.position_embedding = nn.Embedding(seq_len, seq_len)

        self.transformer = Transformer(feature_size, feature_size)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        b, seq_len = x.shape[:2]
        positions = torch.arange(0, seq_len)
        positions = positions.repeat(b,1)

        out = x + self.position_embedding(positions)
        out = self.dropout(out)

        out = self.transformer(out)

        return out
