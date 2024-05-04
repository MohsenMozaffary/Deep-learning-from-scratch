import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, feature_size, n_heads):
        # feature_size needs to be divisible by n_heads
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.feature_size = feature_size
        self.head_dim = feature_size // n_heads

        self.value = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc = nn.Linear(feature_size, feature_size)

    def forward(self, query, key, value):
        #query: batch x seq_len x feature
        #key: batch x seq_len x feature
        #value: batch x seq_len x feature
        b, seq_len = query.shape[0:2]
        # make value, key, and query ready for multihead attention
        query = query.reshape(b, seq_len, self.n_heads, self.head_dim)
        key = key.reshape(b, seq_len, self.n_heads, self.head_dim)
        value = value.reshape(b, seq_len, self.n_heads, self.head_dim)
        # energy: batch x n_heads x seq_len x seq_len
        energy = torch.einsum("bqnd, bknd -> bnqk", [query, key])

        attention = torch.softmax(energy / torch.sqrt(self.head_dim), dim = 3)
        # selfAttention: batch x seq_len x n_heads, feature
        selfAttention = torch.einsum("bnqk, bvnd -> bqnd", [attention, value])
        # selfAttention: batch x seq_len x feature
        selfAttention = selfAttention.reshape(b, seq_len, self.n_heads*self.head_dim)
        # output: batch x seq_len x feature
        output = self.fc(selfAttention)

        return output