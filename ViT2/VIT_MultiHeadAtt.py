import torch.nn as nn
import torch

class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, proj_dim, n_heads):
        super(MultiHeadSelfAttentionLayer, self).__init__()

        self.proj_dim = proj_dim
        self.n_heads = n_heads
        assert proj_dim % n_heads == 0, "proj_dim must be divisible by n_heads"

        self.head_dim = proj_dim // n_heads

        self.query = nn.Linear(proj_dim, proj_dim)
        self.key = nn.Linear(proj_dim, proj_dim)
        self.value = nn.Linear(proj_dim, proj_dim)

        self.out_proj = nn.Linear(proj_dim, proj_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, proj_dim = x.size()

        Q = self.query(x)  # (batch_size, seq_len, proj_dim)
        K = self.key(x)    # (batch_size, seq_len, proj_dim)
        V = self.value(x)  # (batch_size, seq_len, proj_dim)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.softmax(attn_weights)  # (batch_size, n_heads, seq_len, seq_len)

        attention_output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, head_dim)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.proj_dim)  # (batch_size, seq_len, proj_dim)

        output = self.out_proj(attention_output)  # (batch_size, seq_len, proj_dim)

        return output