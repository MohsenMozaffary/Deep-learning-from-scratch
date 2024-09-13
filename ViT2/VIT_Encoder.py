import torch.nn as nn
from VIT_MultiHeadAtt import MultiHeadSelfAttentionLayer

class Encoder(nn.Module):

    def __init__(self, proj_dim, n_heads):
        super(Encoder).__init__()

        self.proj_dim = proj_dim
        self.n_heads = n_heads
        self.linear_mapping_dim = int(4*self.proj_dim)

        self.layer_norm1 = nn.LayerNorm(self.proj_dim)
        self.multi_head_attention = MultiHeadSelfAttentionLayer(self.proj_dim, self.n_heads)
        self.layer_norm2 = nn.LayerNorm(self.proj_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.proj_dim, self.linear_mapping_dim),
            nn.GELU(),
            nn.Linear(self.linear_mapping_dim, self.proj_dim)
        )

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        out = x + self.linear_layers(self.layer_norm2(x))

        return out