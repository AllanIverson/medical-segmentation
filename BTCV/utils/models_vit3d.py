from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from utils.patch_embed import PatchEmbed3D

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,img_size=224,patch_size=16,temp_stride=16,num_frames=224,in_chans=1,**kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans,kwargs['embed_dim'],num_frames,temp_stride)
        embed_dim = kwargs['embed_dim']

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], embed_dim))
        

    def forward(self, x):
        
        B = x.shape[0]
        x = self.patch_embed(x)

        pos_embed = self.pos_embed_spatial.repeat(1, self.patch_embed.grid_size[0], 1) + \
                    torch.repeat_interleave(self.pos_embed_temporal, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], dim=1)
        pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)

        x = x + pos_embed[:, 1:, :]

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)

        outcome=[]
        for blk in self.blocks:
            x = blk(x)
            outcome.append(x[:,1:,:])
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome.append(self.fc_norm(x))

        else:
            x = self.norm(x)
            outcome.append(x[:, 1:, :])

        return outcome[-1],outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,global_pool=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 
def vit_160_patch16(**kwargs):
    model = VisionTransformer(
        img_size=160,num_frames=192,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,global_pool=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def vit_224_large(**kwargs):
    model = VisionTransformer(
        img_size=224,num_frames=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,global_pool=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_96_large(**kwargs):
    model = VisionTransformer(
        img_size=96,num_frames=96,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,global_pool=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_160_large(**kwargs):
    model = VisionTransformer(
        img_size=160,num_frames=192,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,global_pool=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_160_base(**kwargs):
    model = VisionTransformer(
        img_size=160,num_frames=160,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,global_pool=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae3d_vit_base_patch16 = vit_base_patch16
mae3d_160_base_patch16 = vit_160_patch16
mae3d_224_large = vit_224_large
mae3d_96_large = vit_96_large
mae3d_160_large = vit_160_large
mae3d_160_base = vit_160_base
