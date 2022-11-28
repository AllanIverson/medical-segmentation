from functools import partial

import torch
import torch.nn as nn
# from util.diceloss import SoftDiceLoss
from util.diceloss import DiceLoss
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

import seg3d_2dencoder
from util.pos_embed import interpolate_pos_embed
from swin_transformer import *
from email.mime import image
from turtle import forward
from util.diceloss import DiceLoss
from einops import rearrange

class SegVit3D(nn.Module):
    """
    encoder: 预训练2dmae的encoder参数(冻结权重)
    decoder: 采用3D滑窗transformer_decoder
    """
    def __init__(self, img_size=224,img_deep=160, patch_size=16, in_chans=1,embed_dim=256,
                 decoder_embed_dim=256,decoder_depth=[2,2,2,2],decoder_num_heads=[4,8,16,32],
                 encoder:nn.Module=None,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 window_size = 4, shift_size = 1,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 act_layer=nn.GELU,
                 fused_window_process=False,
                 lossforpatch = True,
                 add_encoder = False,
                 ):
        r"""SegVit3D.
        Args:
            img_size(int):the size of one slide of 3D subject
            img_deep(int):the slides number of 3D subject
            in_chans(int):the channel of 3D subject
            embed_dim (int):the dim of encoder patch_embed
            decoder_dim(int):the dim of decoder patch_embed
            decoder_depth(int):the decoder-block number 
            decoder_num_heads (int): Number of decoder attention heads.
            encoder(nn.Module):mae encoder(pretrain)
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
            fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
            lossforpatch:choose target->patch or pred->image
        """
        super().__init__()
        #-----------------------------------------------------
        self.encoder_dim = embed_dim
        self.decoder_dim = decoder_embed_dim
        self.img_size = img_size
        self.norm = norm_layer(decoder_embed_dim)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_deep = img_deep
        self.mlp_ratio = mlp_ratio
        self.encoder_norm = norm_layer(embed_dim)

        #whether add  extra encoder for downstream tasks
        self.add_encoder = add_encoder
        self.extra_encoder = nn.ModuleList([
            Block(embed_dim, 12, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#修改了qk_sacle=None
            for i in range(2)])

        self.decoder_embed = nn.Linear(self.encoder_dim, self.decoder_dim, bias=True)
        self.decoder_pred = nn.Linear(self.decoder_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # MaeSeg3D encoder
        self.encoder = encoder
        # ---------------------------------------------------------------------
        # MaeSeg3D decoder 
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(decoder_depth))]  # stochastic depth decay rule

        self.num_layers = len(decoder_depth)

        # build layers
        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(decoder_embed_dim),
                               input_resolution=(int((img_size//patch_size)**2) ,
                                                 img_deep ),
                               depth=decoder_depth[i_layer],
                               num_heads=decoder_num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop, attn_drop=attn_drop,
                               drop_path=dpr[sum(decoder_depth[:i_layer]):sum(decoder_depth[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample= None,
                               use_checkpoint=False,
                               fused_window_process=fused_window_process)
            self.blocks.append(layer)
        self.apply(self._init_weights)
        self.lossforpatch = lossforpatch

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}



    def encoder_forward(self,x):
        """
        input:
        x_in:(N,1,D,H,W) #(1,1,160,224,224)
        return:
        x:(N,L,patch_size**2*1) #(1,196*160,256)
        """
        N,C,D,H,W = x.shape

        
        x = x.squeeze(0).permute(1,0,2,3)

        
        x = self.encoder(x)
        dim = self.encoder.embed_dim
        num_patches = self.encoder.num_patches
        

        if self.add_encoder == True:
            for blk in self.extra_encoder:
                x = blk(x)
            x = self.encoder_norm(x)
            
            x = x[:,1:,:] #(160,196,768)
            x = x.reshape(N,D*num_patches,dim) #(1,160*196,768)
            return x

        else:
            x = x[:,1:,:] #(160,196,768)
            x = x.reshape(N,D*num_patches,dim) #(1,160*196,768)
            return x


    def decoder_forward(self,x):
        """
        x_in:(B,patch_num*img_deep,256)  #(B,196*160,256)
        """
        x = self.decoder_embed(x)

        for layer in self.blocks:
            x = layer(x)
        
        x = self.norm(x) # B L C

        x = self.decoder_pred(x)
        return x

    def forward_loss(self,pred,target):
        """
        target:[B,1,H,W,C]
        pred:[B,L,256]
        """
        LOSS = DiceLoss()
        if self.lossforpatch:
            target = self.patchify3D(target)  #[N,196,256]
            assert pred.shape == target.shape
            loss = LOSS(pred,target)
        else:
            pred = self.unpatchify3D(pred)  #[N,1,224,224]
            assert pred.shape == target.shape
            loss = LOSS(pred,target)
        return loss
    
    def forward(self,imgs,label):
        latent = self.encoder_forward(imgs)
        pred = self.decoder_forward(latent)  # [N, L, p*p*1] (N,196*160,256)
        loss = self.forward_loss(pred, label)
        return loss, pred
        

    def patchify3D(self, imgs):
        """
        imgs: (N, 1, T, H, W)
        x: (N, L, patch_size**2 *1 *temp_stride)
        """
        x = rearrange(imgs, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=1, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b n p c -> b n (p c)')
        return x

    def unpatchify3D(self, x):
        """
        x: (N, L, patch_size**2 *1 *temp_stride)
        imgs: (N, 1, T, H, W)
        """
        x = rearrange(x, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, c=self.in_chans, h=int(self.img_size//self.patch_size), w=int(self.img_size//self.patch_size))
        return x

def mae_vit_medical_base_patch16(**kwargs):
    model = SegVit3D(
        img_size=224,img_deep=160,patch_size=16,in_chans=1,embed_dim=256,
        decoder_embed_dim=256,decoder_depth=[2,2,2,2],decoder_num_heads=[4,8,16,32],
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
        window_size = 4, shift_size = 1,
        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
        act_layer=nn.GELU,
        fused_window_process=False,
        lossforpatch = True,
        **kwargs
)
    return model

def mae_vit_medical_256_block6(**kwargs):
    model = SegVit3D(
        img_size=224,img_deep=160,patch_size=16,in_chans=1,embed_dim=768,
        decoder_embed_dim=256,decoder_depth=[2,2,6,2],decoder_num_heads=[4,8,16,32],
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
        window_size = 4, shift_size = 1,
        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
        act_layer=nn.GELU,
        fused_window_process=False,
        lossforpatch = True,
        **kwargs
)
    return model   

def mae_vit_medical_512_block6(**kwargs):
    model = SegVit3D(
        img_size=224,img_deep=160,patch_size=16,in_chans=1,embed_dim=768,
        decoder_embed_dim=512,decoder_depth=[2,2,6,2],decoder_num_heads=[4,8,16,32],
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
        window_size = 4, shift_size = 1,
        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
        act_layer=nn.GELU,
        fused_window_process=False,
        lossforpatch = True,
        **kwargs
)
    return model 


def mae_vit_medical_extra_512_block6(**kwargs):
    model = SegVit3D(
        img_size=224,img_deep=160,patch_size=16,in_chans=1,embed_dim=768,
        decoder_embed_dim=512,decoder_depth=[2,2,6,2],decoder_num_heads=[4,8,16,32],
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
        window_size = 4, shift_size = 1,
        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
        act_layer=nn.GELU,
        fused_window_process=False,
        lossforpatch = True,
        add_encoder= True,
        **kwargs
)
    return model 

def mae_vit_medical_extra_1024_512_block6(**kwargs):
    model = SegVit3D(
        img_size=224,img_deep=160,patch_size=16,in_chans=1,embed_dim=1024,
        decoder_embed_dim=512,decoder_depth=[2,2,6,2],decoder_num_heads=[4,8,16,32],
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
        window_size = 4, shift_size = 1,
        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
        act_layer=nn.GELU,
        fused_window_process=False,
        lossforpatch = True,
        add_encoder= True,
        **kwargs
)
    return model 

mae_vit3d_medical_base_windowsize4 = mae_vit_medical_base_patch16  # encoder: 256 dim,decoder: 256 dim, 4 blocks
mae_vit3d_medical_512_windowsize4  = mae_vit_medical_256_block6      
mae_vit3d_medical_768_512_windowsize4 = mae_vit_medical_512_block6   
mae_vit3d_medical_768_512_extra_windowsize4 = mae_vit_medical_extra_512_block6
ame_vit3d_medical_1024_512_extra_windowsize4 = mae_vit_medical_extra_1024_512_block6