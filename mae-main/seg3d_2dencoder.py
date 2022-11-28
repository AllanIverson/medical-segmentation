from functools import partial

import torch
import torch.nn as nn
# from util.diceloss import SoftDiceLoss
from util.diceloss import DiceLoss
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class SegViT3D_Encoder(nn.Module):
    """ 自己用来分割的mae(医学图像,输入通道为1),backbone是timm库的ViT,仿照models_mae.py写的
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,lossforpatch = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.lossforpatch = lossforpatch
        # --------------------------------------------------------------------------
        # my_seg_mae encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)  # use convlution 16*16
        num_patches = self.patch_embed.num_patches #196
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1 , embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)  # delete qk_sacle=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss  #work for pretrain 

        # self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        把图像转换成patch但是不是卷积的方式,只是形状变换
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

   #------------------------------------------------------

   #------------------------------------------------------

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)  # use kernel size=16  -> x (N,196,768)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # return x, mask, ids_restore
        return x


def mae_vit_medical_base_patch16_encoder(**kwargs):
    model = SegViT3D_Encoder(
    patch_size=16, embed_dim=256, depth=12, num_heads=16,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medical_large_patch16_encoder(**kwargs):
    model = SegViT3D_Encoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medical_huge_patch14_encoder(**kwargs):
    model = SegViT3D_Encoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medical_half_patch16_encoder(**kwargs):
    model = SegViT3D_Encoder(
    patch_size=16, embed_dim=768, depth=6, num_heads=16,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = SegViT3D_Encoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_block9_base_patch16_dec512d8b(**kwargs):
    model = SegViT3D_Encoder(
        patch_size=16, embed_dim=768, depth=9, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
# set recommended archs
mae_vit_medical_base_patch16 = mae_vit_medical_base_patch16_encoder  # decoder: 512 dim, 8 blocks
mae_vit_medical_large_patch16 = mae_vit_medical_large_patch16_encoder  # decoder: 512 dim, 8 blocks
mae_vit_medical_huge_patch14 = mae_vit_medical_huge_patch14_encoder  # decoder: 512 dim, 8 blocks
mae_vit_medical_half_patch16 = mae_vit_medical_half_patch16_encoder
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_block9_base_patch16 = mae_vit_block9_base_patch16_dec512d8b