import numpy as np

import torch
import torch.nn as nn
from einops import rearrange
from VIT_Encoder import Encoder

class ViT(nn.Module):
    def __init__(self, mapping_dim = 16, n_patch = 8, input_frame_size = (1, 64, 64), n_encoder = 5, n_heads = 4, final_dim = 16):
        super(ViT, self).__init__()

        self.n_patch = n_patch
        self.total_n_patch = int(n_patch*n_patch + 1)
        self.mapping_dim = mapping_dim
        self.ch, self.w, self.h = input_frame_size
        self.n_heads = n_heads
        self.n_encoder = n_encoder
        self.final_dim = final_dim
        assert self.w%n_patch == 0, "width of frame is not divisible by number of patches (n_patch), image needs to be reshaped"
        assert self.h%n_patch == 0, "height of frame is not divisible by number of patches (n_patch), image needs to be reshaped"

        self.patch_w = self.w//self.n_patch
        self.patch_h = self.h//self.n_patch

        self.patch_dim = int(self.ch*self.patch_w*self.patch_h)
        self.Linear_patch = nn.Linear(self.patch_dim, self.mapping_dim)
        self.cls_token = nn.Parameter(torch.randn(1, self.mapping_dim))

        embedding = torch.ones(self.total_n_patch, self.mapping_dim)

        for patch  in range(self.total_n_patch):
            for d in range(self.mapping_dim):
                if d%2 == 0:
                    embedding[patch, d] = np.sin(patch/(10000**(d/self.mapping_dim)))
                else:
                    embedding[patch, d] = np.cos(patch/(10000**((d-1)/self.mapping_dim)))
        
        self.pos_embd = embedding
        self.encoder_blocks = nn.ModuleList([Encoder(self.mapping_dim, self.n_heads) for _ in range(self.n_encoder)])

        self.final_linear = nn.Sequential(
            nn.Linear(self.mapping_dim, self.final_dim),
            nn.Softmax(dim = -1)
        )

    
    def forward(self, x):
        b, c, w, h = x.shape
        assert c == self.ch, "channel dimension of the input data is not the same as what defined in the instance of VIT"
        assert w == self.w, "width dimension of the input data is not the same as what defined in the instance of VIT"
        assert h == self.h, "height dimension of the input data is not the same as what defined in the instance of VIT"
        p = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1 = self.n_patch, p2 = self.n_patch)
        mapped_patches = self.Linear_patch(p)
        cls_tokens = self.cls_token.expand(b, 1, self.proj_dim)

        expanded_mapped_patches = torch.cat((mapped_patches, cls_tokens), dim = 1)
        pos_embed = self.pos_embd.repeat(b, 1, 1)

        out = expanded_mapped_patches + pos_embed

        for en_blocks in self.encoder_blocks:
            out = en_blocks(out)

        out = self.final_linear(out[:,0])

        return out