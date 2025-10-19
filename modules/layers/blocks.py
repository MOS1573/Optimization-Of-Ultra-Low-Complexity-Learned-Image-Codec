"""Reference: 
    - https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    - https://github.com/baofff/U-ViT/blob/main/libs/timm.py
    - https://github.com/bytedance/1d-tokenizer


"""

import torch
import torch.nn as nn
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']
        self.grid_size = self.image_size // self.patch_size
        self.num_latent_tokens_vq = config['num_latent_tokens_vq']
        self.num_latent_tokens_vae = config['num_latent_tokens_vae']
        self.token_size_vq = config['token_size_vq']
        self.token_size_vae = config['token_size_vae']

        self.in_channels = config['in_channels']

        self.width = config['width']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))

        self.latent_token_positional_embedding_vq = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens_vq, self.width))
        self.latent_token_positional_embedding_vae = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens_vae, self.width))
        
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))

        self.ln_post_vq = nn.LayerNorm(self.width)
        self.ln_post_vae = nn.LayerNorm(self.width)
        self.conv_out_vq = nn.Conv2d(self.width, self.token_size_vq, kernel_size=1, bias=True)
        self.conv_out_vae = nn.Conv2d(self.width, self.token_size_vae, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens_vq, latent_tokens_vae):
        batch_size = pixel_values.shape[0]
        x = pixel_values  # [B, C, H, W] -> [B, 3, 256, 256]
        x = self.patch_embed(x) # [B, width, grid, grid] -> [B, 512, 16, 16]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [B, width, grid*grid] -> [B, 512, 256]
        x = x.permute(0, 2, 1) # [B, grid*grid, width] -> [B, 256, 512]
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1) # -> class embedding ([B, 1, width]) + x ([B, 256, 512]) = [B, 257, 512]
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid*grid+1, width] ([8, 257, 512])
        

        latent_tokens_vq = _expand_token(latent_tokens_vq, x.shape[0]).to(x.dtype) # shape = [B, num_latent_tokens, width] ([2, 8, 512])
        latent_tokens_vq = latent_tokens_vq + self.latent_token_positional_embedding_vq.to(x.dtype) # shape = [B, num_latent_tokens, width] ([2, 8, 512])

        latent_tokens_vae = _expand_token(latent_tokens_vae, x.shape[0]).to(x.dtype) # shape = [B, num_latent_tokens, width] ([2, 2, 512])
        latent_tokens_vae = latent_tokens_vae + self.latent_token_positional_embedding_vae.to(x.dtype) # shape = [B, num_latent_tokens, width] ([2, 2, 512])
        x = torch.cat([x, latent_tokens_vq, latent_tokens_vae], dim=1) # shape = [B, 257+num_latent_tokens_vq + num_latent_tokens_vae, width] ([2, 271, 512])

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)

        latent_tokens_vq = x[:, 1+self.grid_size**2:1+self.grid_size**2+self.num_latent_tokens_vq] # shape = [B, num_latent_tokens, width] ([2, 8, 512])
        latent_tokens_vq = self.ln_post_vq(latent_tokens_vq)
        latent_tokens_vq = latent_tokens_vq.reshape(batch_size, self.num_latent_tokens_vq, self.width, 1).permute(0, 2, 1, 3) # shape = [B, width, num_latent_tokens, 1] ([2, 512, 8, 1])
        latent_tokens_vq = self.conv_out_vq(latent_tokens_vq) # shape = [B, token_size, num_latent_tokens, 1] ([2, 6, 8, 1])
        latent_tokens_vq = latent_tokens_vq.reshape(batch_size, self.token_size_vq, 1, self.num_latent_tokens_vq) # shape = [B, token_size, 1, num_latent_tokens] ([2, 6, 1, 8])

        latent_tokens_vae = x[:, 1+self.grid_size**2+self.num_latent_tokens_vq:] # shape = [B, num_latent_tokens, width] ([2, 2, 512])
        latent_tokens_vae = self.ln_post_vae(latent_tokens_vae)
        latent_tokens_vae = latent_tokens_vae.reshape(batch_size, self.num_latent_tokens_vae, self.width, 1).permute(0, 2, 1, 3) # shape = [B, width, num_latent_tokens, 1] ([2, 512, 2, 1])
        latent_tokens_vae = self.conv_out_vae(latent_tokens_vae) # shape = [B, token_size, num_latent_tokens, 1] ([2, 3*2, 2, 1])
        latent_tokens_vae = latent_tokens_vae.reshape(batch_size, self.num_latent_tokens_vae, self.token_size_vae, 1) # shape = [B, token_size, num_latent_tokens, 1] ([2, 3*2, 2, 1])

        return latent_tokens_vq, latent_tokens_vae

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']
        self.grid_size = self.image_size // self.patch_size
        self.num_latent_tokens_vq = config['num_latent_tokens_vq']
        self.num_latent_tokens_vae = config['num_latent_tokens_vae']
        self.token_size_vq = config['token_size_vq']
        self.token_size_vae = config['token_size_vae']
        self.token_expand_size_vae = config['token_expand_size_vae']
        self.in_channels = config['in_channels']

        self.width = config['width']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']

        self.decoder_embed_vq = nn.Linear(
            self.token_size_vq, self.width, bias=True)
        self.decoder_embed_vae = nn.Linear(
            self.token_expand_size_vae, self.width, bias=True)

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding_vq = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens_vq, self.width))
        self.latent_token_positional_embedding_vae = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens_vae*self.token_size_vae, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)


        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, self.patch_size * self.patch_size * self.in_channels, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                p1 = self.patch_size, p2 = self.patch_size),)
        self.conv_out = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1, bias=True)

    def forward(self, z_quantized_vq, z_quantized_vae):
        N, C_vq, H_vq, W_vq = z_quantized_vq.shape
        assert H_vq == 1 and W_vq == self.num_latent_tokens_vq, f"{H_vq}, {W_vq}, {self.num_latent_tokens_vq}"
        x_vq = z_quantized_vq.reshape(N, C_vq*H_vq, W_vq).permute(0, 2, 1) # NLD
        x_vq = self.decoder_embed_vq(x_vq)

        N, C_vae, H_vae, W_vae = z_quantized_vae.shape
        assert H_vae == self.token_size_vae and W_vae == self.token_expand_size_vae, f"{H_vae}, {W_vae}, {self.num_latent_tokens_vae}"
        x_vae = z_quantized_vae.reshape(N, C_vae*H_vae, W_vae)# NLD
        x_vae = self.decoder_embed_vae(x_vae)

        batchsize, seq_len_vq, _ = x_vq.shape
        _, seq_len_vae, _ = x_vae.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x_vq.dtype)

        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)

        x_vq = x_vq + self.latent_token_positional_embedding_vq[:seq_len_vq]
        x_vae = x_vae + self.latent_token_positional_embedding_vae[:seq_len_vae]
        x = torch.cat([mask_tokens, x_vq, x_vae], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x