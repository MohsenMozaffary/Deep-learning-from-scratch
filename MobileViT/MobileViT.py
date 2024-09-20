# MobileViT implementation

import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_dim, projection_dim, kernel_size = 3, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, projection_dim, kernel_size = kernel_size, stride = stride )
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))

        return x
    
class inverted_resudual_block(nn.Module):
    def __init__(self, input_channels, expanded_channels, output_channels, stride = 1):
        super().__init__()
        self.irb = nn.Sequential(
        nn.Conv2d(input_channels, expanded_channels, 1, padding=1, bias=False),
        nn.BatchNorm2d(expanded_channels),
        nn.SiLU(),
        nn.Conv2d(expanded_channels, expanded_channels, kernel_size = 3, padding=1, 
                  stride = stride, groups = expanded_channels, bias = False),
        nn.BatchNorm2d(expanded_channels),
        nn.SiLU(),
        nn.Conv2d(expanded_channels, output_channels, kernel_size = 1, stride = stride,
                  padding = 1, bias = False),
        nn.BatchNorm2d(output_channels)
        )
        
    def forward(self, x):

        return x + self.irb(x)
    
class transformer_block(nn.Module):
    def __init__(self, projection_dim, num_heads = 2, dropout = 0.1):
        super().__init__()
        
        self.layerNorm1 = nn.LayerNorm(projection_dim)
        self.attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads,
                                               dropout=dropout)
        self.layerNorm2 = nn.LayerNorm(projection_dim)
        self.mlp = nn.Sequential(
                nn.Linear(projection_dim, projection_dim*2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim*2, projection_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.layerNorm1(x)
        out, _ = self.attention(out, out, out)
        out = out + x
        out2 = self.layerNorm2(out)
        out2 = self.mlp(out2),
        final_out = out + out2

        return final_out
    
from einops import rearrange

class mobileViT_block(nn.Module):
    def __init__(self, input_dim, projection_dim, n_transformer = 4, stride = 1, patch_size = 8 ):
        super().__init__()

        self.conv1 = conv_block(input_dim, projection_dim)
        self.conv2 = conv_block(projection_dim, projection_dim, kernel_size = 1)
        self.patch_size = patch_size
        self.n_transformer = n_transformer
        self.transformer = nn.ModuleList(
            [transformer_block(projection_dim) for _ in range(n_transformer)]
        )
        self.conv3 = conv_block(projection_dim, projection_dim, kernel_size = 1)
        self.conv4 = conv_block(2*projection_dim, projection_dim, kernel_size = 1)



    def forward(self, x):
        y = x.clone()
        b, c, w, h = x.shape
        assert w == h, "width and height in MobileViT_block are not with the same dimensions"
        assert w%self.patch_size == 0, "feature dimension of MobileViT_block input is not divisible by patch_size"

        x = self.conv1(x)
        x = self.conv2(x)

        x = rearrange(x, 'b c (h s) (w s) -> b (s s) (h w) c', s = self.patch_size)
        for block in range(self.n_transformer):
            x = self.transformer[block](x)
        
        x = rearrange(x, 'b (s s) (h w) c -> b c (h s) (w s)', s = self.patch_size, h = h//self.patch_size)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)

        return x

import torch.nn.functional as F


class MobileViT(nn.Module):
    def __init__(self, in_dim = 1, num_classes = 5):
        super().__init__()

        self.in_dim = in_dim
        self.downsample = nn.Sequential(
            conv_block(self.in_dim, 16),
            inverted_resudual_block(16, 32, 16),
            inverted_resudual_block(16, 32, 24),
            inverted_resudual_block(24, 48, 24),
            inverted_resudual_block(24, 48, 24)           
        )
        
        self.MV2_1 = nn.Sequential(
            inverted_resudual_block(24, 48, 48),
            mobileViT_block(48, 64, n_transformer = 2)

        )

        self.MV2_2 = nn.Sequential(
            inverted_resudual_block(64, 128, 64),
            mobileViT_block(64, 80, n_transformer = 4)
        )

        self.MV2_3 = nn.Sequential(
            inverted_resudual_block(80, 160, 80),
            mobileViT_block(80, 96, n_transformer = 3)
        )

        self.pointWise_Conv = conv_block(96, 320, kernel_size=1)
        
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(320, num_classes)

    def forward(self, x):

        x = self.downsample(x)
        x = self.MV2_1(x)
        x = self.MV2_2(x)
        x = self.MV2_3(x)
        x = self.pointWise_Conv(x)
        x = self.pooling(x)
        x = self.linear(x)

        return F.softmax(x, dim = 1)