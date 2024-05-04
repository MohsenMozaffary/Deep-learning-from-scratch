from Attention import SelfAttention
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, feature_size, n_heads):
        super(Transformer, self).__init__()
        self.attention = SelfAttention(feature_size, n_heads)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.fc1 = nn.Linear(feature_size, 4*feature_size)
        self.fc2 = nn.Linear(4*feature_size, feature_size)
        self.activation = nn.Relu()
        nn.dropout = nn.Dropout(0.25)

    def forward(self, query, key, value):
        attention = self.attention(query, key, value)
        x = attention + query # skip connection
        x = self.dropout(self.norm1(x))
        linear = self.fc1(x)
        linear = self.activation(linear)
        linear = self.fc2(linear)
        linear = self.dropout(self.norm2(linear + x)) # skip connection

        return linear