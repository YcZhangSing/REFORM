# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



""" PyTorch Florence-2 model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import copy
import math
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import CrossEntropyLoss 
from collections import OrderedDict
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

##GNN
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
from .configuration_florence2 import Florence2Config 
from .configuration_florence2 import Florence2LanguageConfig
from .configuration_florence2 import Florence2VisionConfig


from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)

@dataclass
class Seq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    classification_logits: Optional[torch.FloatTensor] = None
    classification_logits_list: Optional[List[torch.FloatTensor]] = None
    think_decoder_hiden:Optional[torch.FloatTensor] = None
    attention_mask_reason: Optional[torch.Tensor] = None
    attention_mask_answer: Optional[torch.Tensor] = None
    
@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reason_loss: Optional[torch.FloatTensor] = None
    consistency_loss:Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_think_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    classification_logits: Optional[torch.FloatTensor] = None
    classification_logits_list: Optional[List[torch.FloatTensor]] = None
    cos_sim_list: Optional[List[torch.FloatTensor]] = None
    think_logits:Optional[torch.FloatTensor] = None


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Florence2Config"

class LearnedAbsolutePositionEmbedding2D(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256, num_pos=50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - (embedding_dim // 2))

    def forward(self, pixel_values):
        """
        pixel_values: (batch_size, height, width, num_channels) 
        returns: (batch_size, height, width, embedding_dim * 2)
        """
        if len(pixel_values.shape) != 4:
            raise ValueError('pixel_values must be a 4D tensor')
        height, width = pixel_values.shape[1:3]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        # (height, width, embedding_dim * 2)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # (embedding_dim * 2, height, width)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        # (batch_size, embedding_dim * 2, height, width)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # (batch_size, height, width, embedding_dim * 2)
        pos = pos.permute(0, 2, 3, 1)
        return pos

class PositionalEmbeddingCosine1D(nn.Module):
    """
    This class implements a very simple positional encoding. It follows closely
    the encoder from the link below:
    https://pytorch.org/tutorials/beginner/translation_transformer.html

    Args:
        embed_dim: The dimension of the embeddings.
        dropout_prob: The dropout probability.
        max_seq_len: The maximum length to precompute the positional encodings.
    """
    def __init__(
            self,
            embed_dim: int = 512,
            max_seq_len: int = 1024) -> None:
        super(PositionalEmbeddingCosine1D, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        # Generate the sinusoidal arrays.
        factor = math.log(10000)
        denominator = torch.exp(
            -factor * torch.arange(0, self.embed_dim, 2) / self.embed_dim)
        # Matrix where rows correspond to a positional embedding as a function
        # of the position index (i.e., the row index).
        frequencies = \
            torch.arange(0, self.max_seq_len) \
            .reshape(self.max_seq_len, 1) * denominator
        pos_idx_to_embed = torch.zeros((self.max_seq_len, self.embed_dim))
        # Populate uneven entries.
        pos_idx_to_embed[:, 0::2] = torch.sin(frequencies)
        pos_idx_to_embed[:, 1::2] = torch.cos(frequencies)
        # Save the positional embeddings in a constant buffer.
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embeds: The sequence embeddings in order. Allowed size:
                1. [T, D], where T is the length of the sequence, and D is the
                frame embedding dimension.
                2. [B, T, D], where B is the batch size and T and D are the
                same as above.

        Returns a tensor of with the same dimensions as the input: i.e.,
        [1, T, D] or [T, D].
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.size(-2)
        assert len_seq <= self.max_seq_len
        pos_embeds = self.pos_idx_to_embed[0:seq_embeds.size(-2), :]
        # Adapt pre-computed positional embeddings to the input.
        if shape_len == 3:
            pos_embeds = pos_embeds.view(
                (1, pos_embeds.size(0), pos_embeds.size(1)))
        return pos_embeds


class LearnedAbsolutePositionEmbedding1D(nn.Module):
    """
    Learnable absolute positional embeddings for 1D sequences.

    Args:
        embed_dim: The dimension of the embeddings.
        max_seq_len: The maximum length to precompute the positional encodings.
    """
    def __init__(
            self,
            embedding_dim: int = 512,
            num_pos: int = 1024) -> None:
        super(LearnedAbsolutePositionEmbedding1D, self).__init__()
        self.embeddings = nn.Embedding(num_pos, embedding_dim)
        self.num_pos = num_pos

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embeds: The sequence embeddings in order. Allowed size:
                1. [T, D], where T is the length of the sequence, and D is the
                frame embedding dimension.
                2. [B, T, D], where B is the batch size and T and D are the
                same as above.

        Returns a tensor of with the same dimensions as the input: i.e.,
        [1, T, D] or [T, D].
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.size(-2)
        assert len_seq <= self.num_pos
        # [T, D]
        pos_embeds = self.embeddings(torch.arange(len_seq).to(seq_embeds.device))
        # Adapt pre-computed positional embeddings to the input.
        if shape_len == 3:
            pos_embeds = pos_embeds.view(
                (1, pos_embeds.size(0), pos_embeds.size(1)))
        return pos_embeds



class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm != None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features, hidden_features)),
            ("act", act_layer()),
            ("fc2", nn.Linear(hidden_features, out_features))
        ]))

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=bias
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class ConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(
                x, 'b (h w) c -> b c h w',
                h=H, w=W
            )

        x = self.proj(x)

        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):

    def __init__(self, dim, groups=8, qkv_bias=True):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (float(N) ** -0.5)
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):

    def __init__(self, dim, groups, mlp_ratio=4., qkv_bias=True,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 conv_at_attn=True, conv_at_ffn=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, batch_size: int, window_size: int, H: int, W: int):
    B = batch_size 
    # this will cause onnx conversion failed for dynamic axis, because treated as constant
    # int(windows.shape[0] / (H * W / window_size / window_size)) 
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):

        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(
            -1, self.window_size, self.window_size, C
        )
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, conv_at_attn=True, conv_at_ffn=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    """ DaViT: Dual-Attention Transformer

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        patch_size (tuple(int)): Patch size of convolution in different stages. Default: (7, 2, 2, 2).
        patch_stride (tuple(int)): Patch stride of convolution in different stages. Default: (4, 2, 2, 2).
        patch_padding (tuple(int)): Patch padding of convolution in different stages. Default: (3, 0, 0, 0).
        patch_prenorm (tuple(bool)): If True, perform norm before convlution layer. Default: (True, False, False, False).
        embed_dims (tuple(int)): Patch embedding dimension in different stages. Default: (64, 128, 192, 256).
        num_heads (tuple(int)): Number of spatial attention heads in different stages. Default: (4, 8, 12, 16).
        num_groups (tuple(int)): Number of channel groups in different stages. Default: (4, 8, 12, 16).
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        enable_checkpoint (bool): If True, enable checkpointing. Default: False.
        conv_at_attn (bool): If True, performe depthwise convolution before attention layer. Default: True.
        conv_at_ffn (bool): If True, performe depthwise convolution before ffn layer. Default: True.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
     ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i]
            )
            convs.append(conv_embed)

            block = MySequential(
                *[
                    MySequential(OrderedDict([
                        (
                            'spatial_block', SpatialBlock(
                                embed_dims[i],
                                num_heads[i],
                                window_size,
                                drop_path_rate=dpr[depth_offset+j*2],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                            )
                        ),
                        (
                            'channel_block', ChannelBlock(
                                embed_dims[i],
                                num_groups[i],
                                drop_path_rate=dpr[depth_offset+j*2+1],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                            )
                        )
                    ])) for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i]*2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features_unpool(self, x):
        """
        forward until avg pooling 
        Args:
            x (_type_): input image tensor
        """
        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            if self.enable_checkpoint:
                x, input_size = checkpoint.checkpoint(block, x, input_size)
            else:
                x, input_size = block(x, input_size)
        return x

    def forward_features(self, x):
        x = self.forward_features_unpool(x)

        # (batch_size, num_tokens, token_dim)
        x = self.avgpool(x.transpose(1, 2))
        # (batch_size, 1, num_tokens)
        x = torch.flatten(x, 1)
        x = self.norms(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    @classmethod
    def from_config(cls, config):
        return cls(
            depths=config.depths,
            embed_dims=config.dim_embed,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            window_size=config.window_size,
        )




if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Florence2LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Florence2 is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)
        # # <----Debug-----># #
        # print("positions max:", positions.max())
        # print("embedding table size:", self.weight.size(0))
        # # <----Debug-----># #
        
        return super().forward(positions + self.offset)


class Florence2ScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class Florence2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Florence2LanguageConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Florence2FlashAttention2(Florence2Attention):
    """
    Florence2 flash attention module. This module inherits from `Florence2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Florence2FlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("Florence2FlashAttention2 attention does not support output_attentions")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        # get query proj
        query_states = self._reshape(self.q_proj(hidden_states), -1, bsz)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0].transpose(1, 2)
            value_states = past_key_value[1].transpose(1, 2)
        elif is_cross_attention:
            # cross_attentions
            key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
            value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        else:
            # self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=self.dropout
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Florence2SdpaAttention(Florence2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Florence2Model is using Florence2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


FLORENCE2_ATTENTION_CLASSES = {
    "eager": Florence2Attention,
    "sdpa": Florence2SdpaAttention,
    "flash_attention_2": Florence2FlashAttention2,
}

#############
class Florence2EncoderLayer(nn.Module):
    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FLORENCE2_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        ##QKV
        # Query通过hidden_states传过来就行
        Key_value_embeddings:Optional[torch.FloatTensor] = None,
        ###
        
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        
        if Key_value_embeddings is not None: ##使用不同模态作为Q KV
            hidden_states, attn_weights, _ = self.self_attn(
                hidden_states=hidden_states, ## Q
                key_value_states=Key_value_embeddings, ## K V
                attention_mask=attention_mask, 
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
        else: ##源代码实现
            hidden_states, attn_weights, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Florence2DecoderLayer(nn.Module):
    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FLORENCE2_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = FLORENCE2_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        
        
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)
            
        return outputs



class Florence2LanguagePreTrainedModel(PreTrainedModel):
    config_class = Florence2LanguageConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"Florence2EncoderLayer", r"Florence2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class Florence2Encoder(Florence2LanguagePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Florence2EncoderLayer`].

    Args:
        config: Florence2LanguageConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: Florence2LanguageConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = Florence2ScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([Florence2EncoderLayer(config) for _ in range(config.encoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        
        ##增加参数，可以指定image作为Q,text作为kv
        imageQ_textKV: Optional[bool] = False,
        depart_len: Optional[int] = None, ## 577+32
        ##
        
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # print("在 Florence2Encoder 类中的调用栈追溯:")
        # import traceback
        # print("".join(traceback.format_stack()))  # 打印当前调用栈

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        ### 这里报错？
        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training) #不改变hidden_states的形状

        # expand attention_mask
        if attention_mask is not None:
            
            
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._use_sdpa and head_mask is None and not output_attentions:
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        
        # 如果指定使用 image 作为 Q，text 作为 KV
        if imageQ_textKV:
            # 切分 attention_mask
            if attention_mask is not None:
                image_attention_mask = attention_mask[:, :depart_len, :]
                text_attention_mask = attention_mask[:, depart_len:, :]
                # 更新 attention_mask
                attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)

            pre_embeds = hidden_states[:, :depart_len, :]  # [batch_size, image_token_len, hidden_size]
            suff_embeds = hidden_states[:, depart_len:, :]  # [batch_size, text_token_len, hidden_size]

            hidden_states = pre_embeds  # Query (Q)
            KV_embeds = suff_embeds  # Key-Value (KV)
                 
        else:
            KV_embeds = None
            

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        Key_value_embeddings = KV_embeds,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        Key_value_embeddings = KV_embeds,
                        
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class Florence2Decoder(Florence2LanguagePreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`Florence2DecoderLayer`]

    Args:
        config: Florence2LanguageConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: Florence2LanguageConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = Florence2ScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([Florence2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self._use_sdpa and cross_attn_head_mask is None and not output_attentions:
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class Bbox_Verification(nn.Module):
    def __init__(self, embeding):
        super().__init__()
        self.norm_layer_aggr = nn.LayerNorm(embeding)
        self.aggregator = nn.MultiheadAttention(embeding, 16, dropout=0.0, batch_first=True)

        self.bbox_head = self.build_mlp(input_dim=embeding, output_dim=4)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def forward(self, query_token, img_embed):
        bs = img_embed.shape[0]
        # query_token的形状是[bs,768]变成[bs,1,768]
        cls_tokens_local = query_token.unsqueeze(1) 
        
        local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local),
                                          key=self.norm_layer_aggr(img_embed[:, :, :]),
                                          value=self.norm_layer_aggr(img_embed[:, :, :]))[0] ## 聚合器，使用nn.MultiheadAttention多头注意力机制定义
        # 坐标检测
        output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
        
        return output_coord

class GNN(nn.Module):
    '''加入了残差思想
    加入了两层BatchNorm1d归一化
    在最后加入了LayerNorml来适配Transformer
    适配Transformer的LayerNorm
    '''
    def __init__(self, hidden_dim=768, gnn_type="GATv2", num_heads=4, dropout=0.1):
        super(GNN, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        # 使用 GATv2 做可学习边权聚合
        if gnn_type != "GATv2":
            raise ValueError(f"Unsupported gnn_type: {gnn_type}. Only 'GATv2' is supported.")
        out_channels = hidden_dim // num_heads
        self.gnn1 = GATv2Conv(hidden_dim, out_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gnn2 = GATv2Conv(hidden_dim, out_channels, heads=num_heads, concat=True, dropout=dropout)
        self.dropout = dropout

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Layer Normalization (适配 Transformer)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, GATv2Conv):
            if getattr(m, "lin_l", None) is not None:
                nn.init.xavier_uniform_(m.lin_l.weight)
            if getattr(m, "lin_r", None) is not None:
                nn.init.xavier_uniform_(m.lin_r.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, batch_index):
        # 保存输入 x 作为残差项
        residual = x

        # GNN 层 1
        x = self.gnn1(x, edge_index)  
        x = self.bn1(x)  # BatchNorm
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN 层 2
        x = self.gnn2(x, edge_index)  
        x = self.bn2(x)  # BatchNorm

        # 残差连接
        x = x + residual

        # LayerNorm 适配 Transformer
        x = self.layer_norm(x)

        # 图级读出
        graph_x = global_mean_pool(x, batch_index)
        return graph_x



class Florence2LanguageModel(Florence2LanguagePreTrainedModel):
        ## 权重绑定: 让模型中的两个或多个不同的层共享完全相同的一份权重参数。_tied_weights_keys设置到本类的层级即可
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "decoder2.embed_tokens.weight"]

    def __init__(self, config: Florence2LanguageConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = Florence2Encoder(config, self.shared)
        self.decoder = Florence2Decoder(config, self.shared)
        self.d_model = config.d_model ##= 768
        # Learnable tokens (randomly initialized) 可学习token
        self.learnable_tokens_len = 32
        # 用于引导下游任务分类
        self.learnable_tokens = nn.Parameter(torch.randn(self.learnable_tokens_len, config.d_model)) #[32,768]
        
        ## 用于生成reasoning think的token
        self.learnable_tokens_2 = nn.Parameter(torch.randn(self.learnable_tokens_len, config.d_model)) #[32,768]
        
        
        #二分类器_用于learnable_token分类
        self.classifier = nn.Linear(config.d_model, 2)
        self.Dubl_alpha = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5（权重均衡）
        # self.Dubl_classifier = nn.Linear(config.d_model*2, 2) # 双分支的分类层，GNN分支和AttentionPool分支cat得到的分类token形状是[bs,2*768]
        #定义一个二分类头_用于第二次forward中两个模态的分类
        # self.Second_classifier = nn.Linear(config.d_model, 2) # config.d_model 是hiden_size 在base model 中是768

        # # 在实现2.中 Learnable Token的加权聚合——初始化注意力层
        self.AttP_hidden_dim = 256
        self.attn_linear = nn.Linear(config.d_model, self.AttP_hidden_dim) ##(768,256)
        self.attn_weight = nn.Linear(self.AttP_hidden_dim, 1) 
        
        ###定义一个Bbox检测头
        self.Bbox_Verification = Bbox_Verification(config.d_model) # projection_dim = 768
        ###定义一个GNN分类器
        self.gnn = GNN(hidden_dim=config.d_model, gnn_type="GATv2") # projection_dim = 768
        self.gnn2 = GNN(hidden_dim=config.d_model, gnn_type="GATv2") # projection_dim = 768
        self.graph_topk = min(8, self.learnable_tokens_len - 1)
        self.decoder_2 = Florence2Decoder(config, self.shared)
        
        # 先对所有已定义的标准模块进行最终初始化
        self.post_init()
        #<---!!!!!---->
        ## 防止出现问题decoder_2的权重放在train代码中，显式的从decoder复制过来
        #<---!!!!!---->
        
        # ## 复制一份decoder，为了避免初始化过程中显存占用过多，可以先搬到CPU复制，然后再放回device
        # self.decoder_2 = copy.deepcopy(self.decoder.cpu())
        # self.decoder_2.to(next(self.decoder.parameters()).device)

    def attention(self, query, key, value):
        """
        进行自注意力计算
        """
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, seq_len, seq_len)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, seq_len, hidden_size)

        return attention_output.mean(dim=1)  # 对序列维度求平均，得到最终的输出

    
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder_2.embed_tokens, self.shared)
            

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
        self.decoder_2.embed_tokens = self.shared
        

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_decoder_2(self):
        return self.decoder_2
    
    def get_encoder_output_newEmbedding(
        self,
        input_ids: torch.LongTensor = None, #*#
        attention_mask: Optional[torch.Tensor] = None, #*#
        head_mask: Optional[torch.Tensor] = None,#*#
        encoder_outputs: Optional[List[torch.FloatTensor]] = None, #就是None
        inputs_embeds: Optional[torch.FloatTensor] = None, #已设置
        output_attentions: Optional[bool] = None, #*#
        output_hidden_states: Optional[bool] = None,#*#
        return_dict: Optional[bool] = None, #已设置
        first_forward: Optional[bool] = False, ##是否是第一次forward
        image_tokens_len:Optional[int] = 577
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if encoder_outputs is None: ## 源码正确
        
            if first_forward:
                first_forward = False
                ## 冻结self.encoder的参数
                for param in self.encoder.parameters():
                    param.requires_grad = False
                
                batch_size = len(inputs_embeds)
                # 扩展 learnable_tokens, 拼接 learnable_tokens_expanded 到 input_embeddings 的577图像嵌入之后
                learnable_tokens_expanded = self.learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 32, 768]
                learnable_tokens_expanded_2 = self.learnable_tokens_2.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 32, 768]
                
                
                inputs_embeds = torch.cat((inputs_embeds[:, :image_tokens_len, :], learnable_tokens_expanded, learnable_tokens_expanded_2, inputs_embeds[:, image_tokens_len:, :]), dim=1) ##图像嵌入（577）+learnable token1（32）+ learnable token1（32) +文本嵌入（不定长）
                # 扩展 attention_mask，添加 learnable_token 部分的有效掩码
                learnable_token_mask = torch.ones(batch_size, self.learnable_tokens_len*2).to(attention_mask.device)  # [batch_size, learnable_tokens_len]
                
                attention_mask = torch.cat((attention_mask[:, :image_tokens_len], learnable_token_mask, attention_mask[:, image_tokens_len:]), dim=1)
                
                ## image向量的长度：image长度+可学习token长度
                ## 待修改
                depart_len = image_tokens_len + self.learnable_tokens_len*2
                ###image+learnable_token做Query text做KV
                ##获得输出
                first_encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                ##增加的参数
                imageQ_textKV = True,
                depart_len = depart_len,
                )
                # ##解冻self.encoder的参数
                # for param in self.encoder.parameters():
                #         param.requires_grad = True
                        
            inputs_embeds[:, image_tokens_len:image_tokens_len+self.learnable_tokens_len*2, :] = first_encoder_outputs[0][:,image_tokens_len:image_tokens_len+self.learnable_tokens_len*2,:] ## inputs_embeds:[bs,l+learnable_tokens_len,768]
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return encoder_outputs, inputs_embeds
        
    def stable_logits(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        对Logits进行数值稳定化处理，防止极端值导致计算异常，但不计算Softmax。
        适用于后续使用 `nn.CrossEntropyLoss()` 计算损失的情况。

        Args:
            logits: 原始Logits张量
            dim: 需要进行稳定化的维度（默认最后一维）

        Returns:
            经过数值稳定化处理的 logits
        """
        max_logits = torch.max(logits, dim=dim, keepdim=True).values  # 计算最大值
        return logits - max_logits  # 平移 logits，增强数值稳定性

    def create_graph_data(self, output_learnable_token, topk=8):
        """
        将 Transformer Encoder 输出的 output_learnable_token 转换为 PyG 的图数据格式。
        使用基于余弦相似度的 top-k 稀疏图，并修复 batch 维度下的 edge_index 偏移。
        """
        bs, num_tokens, hidden_dim = output_learnable_token.shape
        device = output_learnable_token.device

        # 将 batch 拉平
        x = output_learnable_token.reshape(bs * num_tokens, hidden_dim)  # [bs*num_tokens, hidden_dim]

        # 生成 batch_index
        batch_index = torch.arange(bs).repeat_interleave(num_tokens).to(device)  # [bs*num_tokens]
        if num_tokens == 1:
            nodes = torch.arange(bs, device=device, dtype=torch.long)
            edge_index = torch.stack([nodes, nodes], dim=0)
            return x, edge_index, batch_index

        # 自适应稀疏图：每个 token 连接到 top-k 相似 token
        k = max(1, min(topk, num_tokens - 1))
        token_norm = F.normalize(output_learnable_token, p=2, dim=-1)
        sim = torch.matmul(token_norm, token_norm.transpose(1, 2))  # [bs, num_tokens, num_tokens]
        diag_mask = torch.eye(num_tokens, device=device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(diag_mask, float("-inf"))
        knn_idx = torch.topk(sim, k=k, dim=-1).indices  # [bs, num_tokens, k]

        src = torch.arange(num_tokens, device=device, dtype=torch.long).view(1, num_tokens, 1).expand(bs, -1, k)
        dst = knn_idx.long()
        offsets = (torch.arange(bs, device=device, dtype=torch.long) * num_tokens).view(bs, 1, 1)
        src = (src + offsets).reshape(-1)
        dst = (dst + offsets).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)  # [2, bs*num_tokens*k]
        rev_edge_index = torch.stack([dst, src], dim=0)  # 对称边
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)

        return x, edge_index, batch_index

    def forward(
        self,
        input_ids: torch.LongTensor = None,        
        attention_mask: Optional[torch.Tensor] = None,
        first_forward: Optional[bool] = False, ##是否是第一次forward  
        ## 两个decoder的输入
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder2_input_ids: Optional[torch.LongTensor] = None,
        decoder2_attention_mask: Optional[torch.LongTensor] = None,

        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,        
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Florence2 automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            
        # if decoder2_input_ids is None and decoder_inputs_embeds is None:
        #     if input_ids is None:
        #         raise ValueError(
        #             "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
        #             "passed, `input_ids` cannot be `None`. Please pass either "
        #             "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
        #         )

        #     decoder2_input_ids = shift_tokens_right(
        #         input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        #     )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        learnable_token_logits = None
        output_coord = None
        

        
        if encoder_outputs is None: 
        
            if first_forward:
                first_forward = False
                ## 冻结self.encoder的参数
                for param in self.encoder.parameters():
                    param.requires_grad = False
                
                batch_size = len(inputs_embeds)
                # 扩展 learnable_tokens, 拼接 learnable_tokens_expanded 到 input_embeddings 的577图像嵌入之后
                learnable_tokens_expanded = self.learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 32, 768]
                learnable_tokens_expanded_2 = self.learnable_tokens_2.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 32, 768]
                
                inputs_embeds = torch.cat((inputs_embeds[:, :577, :], learnable_tokens_expanded, learnable_tokens_expanded_2, inputs_embeds[:, 577:, :]), dim=1) ##图像嵌入（577）+learnable token1（32）+ learnable token2（32）+文本嵌入（不定长）
                # 扩展 attention_mask，添加 learnable_token 部分的有效掩码
                learnable_token_mask = torch.ones(batch_size, self.learnable_tokens_len*2).to(attention_mask.device)  # [batch_size, learnable_tokens_len]
                attention_mask = torch.cat((attention_mask[:, :577], learnable_token_mask, attention_mask[:, 577:]), dim=1)
                
                ## Query向量的长度：image长度+可学习token长度
                depart_len = 577 + self.learnable_tokens_len*2
                ###image+learnable_token*2做Query text做KV, 
                # 特别注意，encoder的输出是query的长度，
                # 所以以image+learnable_token做query的话，返回的first_encoder_outputs长度也是image+learnable_token
                # 这里使用image+learnable_token的理论解释还不很明确，
                # 既然后面只使用learnable_token，可不可以用learnable_token作为query，查询image和text呢？
                first_encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                ##增加的参数
                imageQ_textKV = True,
                depart_len = depart_len,
                )
                ##解冻self.encoder的参数
                for param in self.encoder.parameters():
                        param.requires_grad = True
            
            ##提取encoder_outputs中的learnable_token
            output_learnable_token_1 = first_encoder_outputs[0][:,577:577+self.learnable_tokens_len,:] ##[bs,32,768]
                        
            ## GNN 输出的logits ##
            GNN_x, GNN_edge_index, GNN_batch_index = self.create_graph_data(
                output_learnable_token_1, topk=self.graph_topk
            )
            GNN_output_x = self.gnn(GNN_x, GNN_edge_index, GNN_batch_index)  # [bs, 768]
            mean_output_learnable_token_GNN = GNN_output_x  # [bs, 768]
            
            
            ## Attention Pool 输出的logits ##
            attn_scores = torch.relu(self.attn_linear(output_learnable_token_1))  # 对每个token做映射
            # [batch_size, num_tokens, hidden_dim] -> [batch_size, num_tokens, 1]
            attn_scores = self.attn_weight(attn_scores)
            # [batch_size, num_tokens, 1] -> [batch_size, num_tokens]
            attn_weights = F.softmax(attn_scores, dim=1)  # 对tokens的注意力得分做softmax
            # [batch_size, num_tokens, 1] * [batch_size, num_tokens, input_dim] -> [batch_size, input_dim]
            mean_learnable_token_AT = torch.sum(attn_weights * output_learnable_token_1, dim=1) 
            
            ## 两个分支并行的输出
            alpha = torch.sigmoid(self.Dubl_alpha).unsqueeze(-1) # 可学习权重alpha形状变成[1,1] 以便在下一行代码支持广播机制变成[bs,1]以支持矩阵运算
            mean_learnable_token = alpha * mean_learnable_token_AT + (1 - alpha) * mean_output_learnable_token_GNN
            learnable_token_logits = self.classifier(mean_learnable_token)  # [bs, 2]

            inputs_embeds[:, 577:577+self.learnable_tokens_len*2, :] = first_encoder_outputs[0][:,577:577+self.learnable_tokens_len*2,:] ## inputs_embeds:[bs,l+learnable_tokens_len,768]
            
            ###<---imp2的内容--->
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            ## 问题： 最后一次backbone上的decoder的时候应该也是需要是把[I,L1,L2,T]送入decoder
            ### 然后在辅助头上的decoder的时候只需要把把[L2]送入decoder得到reasoning即可
            
            
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # #<------forward2---imp1---------------------------># #
        last_hidden_state = encoder_outputs.last_hidden_state 
        
        ##<<-------GNN2------->>
        GNN2_x, GNN2_edge_index, GNN2_batch_index = self.create_graph_data(
            last_hidden_state[:,577:577+self.learnable_tokens_len,:], topk=self.graph_topk
        ) ##[bs,32,768]
        GNN2_output_x = self.gnn2(GNN2_x, GNN2_edge_index, GNN2_batch_index)  
        ####更新GNN产生生成的learnable token拼接到image上
        ##<<-------GNN2------->>
        
        ### 坐标框检测
        query_a = GNN2_output_x ##[bs,768]
        output_coord = self.Bbox_Verification(query_a,last_hidden_state[:,:577,:]) ##选取image的pathtoken，AMD是不要第一个，这次要第一个看看
        
        logits = [learnable_token_logits, output_coord]
        # #<------forward2---imp1----------------------------># #
        


    
        decoder2_outputs = None
        ## 仅在训练时使用decoder2
        if decoder2_input_ids is not None and self.training:
            decoder2_outputs = self.decoder_2(
                input_ids=decoder2_input_ids,
                attention_mask=decoder2_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                past_key_values=past_key_values, # 推理阶段缓存（训练不需要）
                head_mask=decoder_head_mask, ## 可选，屏蔽指定 self-attention 头
                cross_attn_head_mask=cross_attn_head_mask, ## 可选，屏蔽指定 cross-attention 头
                use_cache=False, ## 是否输出 past_key_values ## decoder2并不参与推理，所以这里直接明确关闭
                return_dict=return_dict, ## 控制输出细节
                
            )
        think_decoder_hidden = (decoder2_outputs.last_hidden_state if decoder2_outputs is not None else None)
        
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        ## decoder中是以自己预测的decoder_input_ids作为query，以encoder_outputs做KV
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, ## 已经生成的序列
            inputs_embeds=decoder_inputs_embeds, ## decoder的自回归输入（token id或embedding）
            attention_mask=decoder_attention_mask, ## mask掉 padding token，自注意力用
            encoder_hidden_states=encoder_outputs[0],## cross-attention的KV
            encoder_attention_mask=attention_mask,## mask掉encoder padding token，cross-attention用
            head_mask=decoder_head_mask, ## 可选，屏蔽指定 self-attention 头
            cross_attn_head_mask=cross_attn_head_mask, ## 可选，屏蔽指定 cross-attention 头
            past_key_values=past_key_values, # 推理阶段缓存（训练不需要）
            use_cache=use_cache, ## 是否输出 past_key_values
            output_attentions=output_attentions, ## 控制输出细节
            output_hidden_states=output_hidden_states, ## 控制输出细节
            return_dict=return_dict, ## 控制输出细节
        )

        attention_mask_reason = None
        attention_mask_answer = None
        if self.training:
            attention_mask_reason = (decoder2_input_ids != self.config.pad_token_id).int()
            attention_mask_answer = (decoder_input_ids != self.config.pad_token_id).int()
        
        
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            classification_logits_list = logits,
            ## 新增
            think_decoder_hiden = think_decoder_hidden,
            attention_mask_reason = attention_mask_reason,
            attention_mask_answer = attention_mask_answer,
            
        )




import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ConsistencyLossWithProjectionAndAttention(nn.Module):
    def __init__(self, hidden_size, margin=0):
        """
        :param hidden_size: 输入特征的维度 (即 768)
        :param proj_dim: 投影维度 (例如 256)
        :param margin: 余弦相似度的安全边界
        margin=0.05 < margin=0.15
        """
        super(ConsistencyLossWithProjectionAndAttention, self).__init__()

        # 定义 margin（安全边界）
        self.margin = margin



    def _masked_mean_pool(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        对 hidden_states 进行掩码均值池化
        
        Args:
            hidden_states (torch.Tensor): [bs, seq_len, hidden_size]
            mask (torch.Tensor): [bs, seq_len]，值为 0 或 1
        
        Returns:
            torch.Tensor: [bs, hidden_size]
        """
        # 1. 扩展掩码: [bs, seq_len] -> [bs, seq_len, 1]
        #    并确保它是浮点型以便乘法
        mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states).float()
        
        # 2. 乘以掩码，padding token 的向量变为 0
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # 3. 计算每个序列的真实长度（非 0 元素的数量）
        #    [bs, seq_len] -> [bs] -> [bs, 1]
        sum_mask = mask.sum(dim=1).unsqueeze(-1)
        
        # 4. 钳位 (clamp) 以防止除以 0
        sum_mask_clamped = torch.clamp(sum_mask, min=1e-9)
        
        # 5. 返回平均向量
        return sum_embeddings / sum_mask_clamped.float()


    def forward(self, 
                h_reason: torch.Tensor, 
                h_answer: torch.Tensor, 
                m_reason: torch.Tensor, 
                m_answer: torch.Tensor):
        
        # 1. 池化: [bs, len, hidden_size] -> [bs, hidden_size]
        v_reason = self._masked_mean_pool(h_reason, m_reason)
        v_answer = self._masked_mean_pool(h_answer, m_answer)
        

        # 3. 计算余弦相似度
        # F.cosine_similarity 会自动处理 L2 归一化
        # z_reason 和 z_answer 都是 [bs, proj_dim]
        ## 值域 【-1 1】
        cos_sim = F.cosine_similarity(v_reason, v_answer, dim=1)  

        # 4. 计算损失
        # 目标：只惩罚 cos_sim < margin 的情况
        # 损失 = relu(x) = max(0, margin - cos_sim)
        scale = 4.0  ## 放大一下惩罚
        loss = F.relu(self.margin - cos_sim) * scale  # loss 形状为 [bs]

        # 5. 返回
        # loss的值域是[0, (margin + 1) * scale] (如果 cos_sim 为 -1)
        # 我们返回整个批次的平均 loss，以及一些用于监控的统计数据
        return loss.mean(), cos_sim.mean(), cos_sim.min()



class Florence2LanguageForConditionalGeneration(Florence2LanguagePreTrainedModel):
    base_model_prefix = "model"
    # _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _tied_weights_keys = [
        "encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
        "decoder.embed_tokens.weight","lm_head.weight","lm_think_head.weight"
    ]
    
    # _keys_to_ignore_on_load_missing = ["final_logits_bias"] ## 不需要报错这个bias
    _keys_to_ignore_on_load_missing = ["final_logits_bias", "final_logits_bias_think"]


    def __init__(self, config: Florence2LanguageConfig):
        super().__init__(config)
        self.model = Florence2LanguageModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        ### 记得在train代码中手动copy一下self.lm_head的初始化权重
        self.lm_think_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # optional: copy weights from existing lm_head for warm-start
        self.register_buffer("final_logits_bias_think", torch.zeros((1, self.model.shared.num_embeddings)))
        
        ## 一致性投影层
        self.consistency_module = ConsistencyLossWithProjectionAndAttention(hidden_size=config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()
    
    ##传递Florence2LanguageModel中的新逻辑函数
    def get_Florence2LanguageModel_encoder_output(
        self,
        input_ids: torch.LongTensor = None, #*#
        attention_mask: Optional[torch.Tensor] = None, #*#
        head_mask: Optional[torch.Tensor] = None,#*#
        encoder_outputs: Optional[List[torch.FloatTensor]] = None, #就是None
        inputs_embeds: Optional[torch.FloatTensor] = None, #已设置
        output_attentions: Optional[bool] = None, #*#
        output_hidden_states: Optional[bool] = None,#*#
        return_dict: Optional[bool] = None, #已设置
        first_forward: Optional[bool] = False, ##是否是第一次forward
        image_tokens_len:Optional[int] = 577
        ):
        return self.model.get_encoder_output_newEmbedding(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            head_mask = head_mask,
            encoder_outputs = encoder_outputs,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            first_forward = first_forward,
            image_tokens_len = image_tokens_len
        )
    
    def get_decoder(self):
        return self.model.get_decoder()
    def get_decoder_2(self):
        return self.model.get_decoder_2()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
    #     old_num_tokens = self.final_logits_bias.shape[-1]
    #     if new_num_tokens <= old_num_tokens:
    #         new_bias = self.final_logits_bias[:, :new_num_tokens]
    #     else:
    #         extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
    #         new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
    #     self.register_buffer("final_logits_bias", new_bias)
    
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 同步扩展两个 bias
        ## 必须同时扩展/裁剪两个 final_logits_bias，否则出现 vocab_size 变化时会导致 shape 不匹配。
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
            new_bias2 = self.final_logits_bias_think[:, :new_num_tokens]
        else:
            extra = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra], dim=1)
            new_bias2 = torch.cat([self.final_logits_bias_think, extra.clone()], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        self.register_buffer("final_logits_bias_think", new_bias2)

    def get_output_embeddings(self):
        # 兼容性：保持默认返回 primary head（lm_head）
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 仍然设置 primary head
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        learnable_forward: Optional[bool] = False, ##是否是第一次prompt
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        
        decoder2_input_ids: Optional[torch.LongTensor] = None,
        decoder2_attention_mask: Optional[torch.LongTensor] = None,
        decoder2_inputs_embeds: Optional[torch.FloatTensor] = None,
    
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,

        labels: Optional[torch.LongTensor] = None,
        reason_labels:Optional[torch.LongTensor] = None, ## 新增的reason lable
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            learnable_forward = True
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        if reason_labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder2_input_ids is None and decoder2_inputs_embeds is None:
                decoder2_input_ids = shift_tokens_right(
                    reason_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

                

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask, ##
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs, #来自forward传入，没有修改的，用##标识
            decoder_attention_mask=decoder_attention_mask, ##
            head_mask=head_mask, ##
            decoder_head_mask=decoder_head_mask, ##
            cross_attn_head_mask=cross_attn_head_mask, ##
            past_key_values=past_key_values, ##
            inputs_embeds=inputs_embeds, ##
            decoder_inputs_embeds=decoder_inputs_embeds, ##
            use_cache=use_cache, 
            output_attentions=output_attentions, ##
            output_hidden_states=output_hidden_states, ##
            return_dict=return_dict,
            first_forward = learnable_forward, ##是否使用两次forward   
            decoder2_input_ids = decoder2_input_ids,
            decoder2_attention_mask = decoder2_attention_mask
        )
        
        classification_logits_list = outputs.classification_logits_list

        ## 主decoder的 language modeling loss 是 masked_lm_loss
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                
        ## 辅助decoder的 language modeling loss 是 masked_reason_lm_loss
        masked_reason_lm_loss = None
        lm_think_logits = None
        consistency_loss = None  # 新增：一致性损失
        cos_sim_mean = None
        cos_sim_min = None
        if outputs.think_decoder_hiden is not None:

            ##<[debug]>

            lm_think_logits = self.lm_think_head(outputs.think_decoder_hiden)
            # 使用独立的bias：
            lm_think_logits = lm_think_logits + self.final_logits_bias_think.to(lm_think_logits.device)

            reason_labels = reason_labels.to(lm_think_logits.device)

            # 1. 忽略 pad (-100)，避免在填充位置计算 loss # 使用标签平滑来软化标签
            # 长文本的标签和输出间的差距可能变得更加显著。大差异的预测值可能导致梯度计算中的NaN或Inf。
            ## label_smoothing 的取值范围是 [0.0, 1.0)，即一个大于等于 0.0 且小于 1.0 的浮点数。通常在 0.05 到 0.2 之间。
            ## 1.0 是一个理论上的极限值，这相当于告诉模型：“所有类别都是等可能的，你的训练标签没有任何信息价值”
            loss_fct = CrossEntropyLoss()  
  
            masked_reason_lm_loss = loss_fct(
                lm_think_logits.view(-1, self.config.vocab_size),
                reason_labels.view(-1)
            )
            
            ## reason 和 answer 是完全不同风格的两种文本，如果直接粗暴的计算余弦相似度，过于粗糙
            ## 这里引入了投影层和多头注意力机制，reason和answer映射到相同的语义空间后再计算相似度
            ## 返回的consistency_loss 还要乘上 answer 和 answer label 相似度系数，以最终组成 “正确一致性损失”
            ## consistency_loss的理论范围为[0,2]
            # return loss.mean(),cos_sim.mean(),cos_sim.min() 
            
            
            consistency_loss, cos_sim_mean,cos_sim_min  = self.consistency_module(
                                                                                outputs.think_decoder_hiden,  ## reason_hidden
                                                                                outputs.last_hidden_state,    ## answer_hidden
                                                                                outputs.attention_mask_reason, ## reason_mask
                                                                                outputs.attention_mask_answer  ## answer_mask          
                                                                                ) 
            
            # ### 降低计算量，以answer的建模损失模拟正确一致性损失的系数
            # ## 关闭梯度回传，一定程度上防止模型通过大的answer loss 来获得低的 一致性权重从而降低损失
            # with torch.no_grad():
            #     ## sigmoid()的取值范围是(0, 0.5], *2 缩放到 (0, 1]
            #     scaled_aas = 2.0 * torch.sigmoid(-masked_lm_loss.detach())  # loss 高 → 权重低
            
            ## 最终正确一致性损失的取值范围就是[0,2]    
            # consistency_loss = scaled_aas * consistency_loss
            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            reason_loss=masked_reason_lm_loss,
            logits=lm_logits,
            consistency_loss = consistency_loss,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_think_hidden_states=outputs.think_decoder_hiden,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            classification_logits_list = classification_logits_list,
            cos_sim_list = [cos_sim_mean, cos_sim_min],
            think_logits = lm_think_logits
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

@dataclass
class Florence2Seq2SeqLMOutput(ModelOutput):
    """
    Base class for Florence-2 model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size,
            num_image_tokens, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder
    """
    loss: Optional[torch.FloatTensor] = None
    reason_loss: Optional[torch.FloatTensor] = None
    consistency_loss:Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    classification_logits: Optional[torch.FloatTensor] = None
    classification_logits_list: Optional[torch.FloatTensor] = None
    cos_sim_list: Optional[torch.FloatTensor] = None
    


FLORENCE2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Florence2Config`] or [`Florence2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Florence-2 Model outputting raw hidden-states without any specific head on top.",
    FLORENCE2_START_DOCSTRING,
)
class Florence2PreTrainedModel(PreTrainedModel):
    config_class = Florence2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    @property
    def _supports_flash_attn_2(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        Flash Attention 2 or not.
        """
        return self.language_model._supports_flash_attn_2

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


FLORENCE2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`Florence2Processor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    """The FLORENCE2 vision model without any head""",
    FLORENCE2_START_DOCSTRING,
)
class Florence2VisionModel(Florence2PreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        assert config.model_type == 'davit', 'only DaViT is supported for now'
        self.vision_tower = DaViT.from_config(config=config)

        self.post_init()
    
    def forward(self, pixel_values):
        if len(pixel_values.shape) == 4:
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        return x


@add_start_docstrings(
    """The FLORENCE2 vision model with projection layer""",
    FLORENCE2_START_DOCSTRING,
)
class Florence2VisionModelWithProjection(Florence2PreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        assert config.model_type == 'davit', 'only DaViT is supported for now'
        self.vision_tower = DaViT.from_config(config=config)

        self._build_image_projection_layers(config)

        self.post_init()
    
    def _build_image_projection_layers(self, config):
        image_dim_out = config.dim_embed[-1]
        dim_projection = config.projection_dim
        self.image_projection = nn.Parameter(
            torch.empty(image_dim_out, dim_projection)
        )
        self.image_proj_norm = nn.LayerNorm(dim_projection)
        image_pos_embed_config = config.image_pos_embed
        if image_pos_embed_config['type'] == 'learned_abs_2d':
            self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
                embedding_dim=image_dim_out,
                num_pos=image_pos_embed_config['max_pos_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

        self.image_feature_source = config.image_feature_source

        # temporal embedding
        visual_temporal_embedding_config = config.visual_temporal_embedding
        if visual_temporal_embedding_config['type'] == 'COSINE':
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=image_dim_out,
                max_seq_len=visual_temporal_embedding_config['max_temporal_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

    def forward(self, pixel_values):
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            T = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        
        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = torch.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)


        return x



@add_start_docstrings(
    """The FLORENCE2 model which consists of a vision backbone and a language model.""",
    FLORENCE2_START_DOCSTRING,
)
class Florence2ForConditionalGeneration(Florence2PreTrainedModel):
    
    ## 最新版的florence-2中修改了这里的定义，直接使用_tied_weights_keys指定 而不是在__init__中继承
    ## 所以以前下载的旧版florence2这里可能不一样
    _tied_weights_keys = [
        "language_model.encoder.embed_tokens.weight",
        "language_model.decoder.embed_tokens.weight",
        "language_model.decoder_2.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.lm_think_head.weight",
    ]
    
    
    def __init__(self, config: Florence2Config):
        super().__init__(config)
        assert config.vision_config.model_type == 'davit', 'only DaViT is supported for now'
        self.vision_tower = DaViT.from_config(config=config.vision_config)
        # remove unused layers 
        del self.vision_tower.head
        del self.vision_tower.norms

        self.vocab_size = config.vocab_size
        self._attn_implementation = config._attn_implementation
        self._build_image_projection_layers(config)

        language_model = Florence2LanguageForConditionalGeneration(config=config.text_config)

        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()
    
    
    def _build_image_projection_layers(self, config):
        image_dim_out = config.vision_config.dim_embed[-1]
        dim_projection = config.vision_config.projection_dim
        self.image_projection = nn.Parameter(
            torch.empty(image_dim_out, dim_projection)
        )
        self.image_proj_norm = nn.LayerNorm(dim_projection)
        image_pos_embed_config = config.vision_config.image_pos_embed
        if image_pos_embed_config['type'] == 'learned_abs_2d':
            self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
                embedding_dim=image_dim_out,
                num_pos=image_pos_embed_config['max_pos_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

        self.image_feature_source = config.vision_config.image_feature_source

        # temporal embedding
        visual_temporal_embedding_config = config.vision_config.visual_temporal_embedding
        if visual_temporal_embedding_config['type'] == 'COSINE':
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=image_dim_out,
                max_seq_len=visual_temporal_embedding_config['max_temporal_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()
    
    def get_decoder_2(self):
        return self.language_model.get_decoder_2()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def _encode_image(self, pixel_values):
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            T = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        
        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = torch.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x 

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds 
    ):
        batch_size, image_token_length = image_features.size()[:-1]
        device = image_features.device
        image_attention_mask = torch.ones(batch_size, image_token_length, device=device)

        # task_prefix_embeds: [batch_size, padded_context_length, hidden_size]
        # task_prefix_attention_mask: [batch_size, context_length]
        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = torch.ones(batch_size, task_prefix_embeds.size(1), device=device)

        if len(task_prefix_attention_mask.shape) == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        # concat [image embeds, task prefix embeds]
        inputs_embeds = torch.cat([image_features, task_prefix_embeds], dim=1)
        attention_mask = torch.cat([image_attention_mask, task_prefix_attention_mask], dim=1)

        return inputs_embeds, attention_mask


    @add_start_docstrings_to_model_forward(FLORENCE2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Florence2Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder2_input_ids: Optional[torch.LongTensor] = None,
        decoder2_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        reason_labels: Optional[torch.LongTensor] = None, ## 新增的reason think labels
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        learnable_forward: Optional[bool] = True, ###默认是使用learnable_forward方法forward两次的
    ) -> Union[Tuple, Florence2Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Florence2ForConditionalGeneration

        >>> model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large")
        >>> processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

        >>> prompt = "<CAPTION>"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=100)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A green car parked in front of a yellow building."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = None
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                # (batch_size, num_image_tokens, hidden_size)
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)

        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
            
        outputs = self.language_model(
            attention_mask=attention_mask,
            labels=labels,
            reason_labels=reason_labels,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder2_input_ids=decoder2_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder2_attention_mask=decoder2_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            learnable_forward = learnable_forward, ##是否使用forward两次，默认为True
        )
        classification_logits_list = outputs.classification_logits_list

        logits = outputs.logits
        logits = logits.float()

        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (outputs.loss,) + output if outputs.loss is not None else output

        return Florence2Seq2SeqLMOutput(
            loss=outputs.loss,
            reason_loss=outputs.reason_loss,
            consistency_loss = outputs.consistency_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            image_hidden_states=image_features,
            classification_logits_list = classification_logits_list,
            cos_sim_list = outputs.cos_sim_list
            
        )




    
    
    @torch.no_grad()
    def _generate_with_decoder_beam_search(
        decoder,
        lm_head,
        encoder_hidden_states,
        num_beams=3,
    ):
        """
        Beam Search generation for Florence-2 style decoder.

        Args:
            decoder: nn.Module, the decoder model (e.g., Florence2Decoder)
            lm_head: nn.Linear, maps decoder output to vocab logits
            encoder_hidden_states: Tensor of shape (bs*num_beams, seq_len, hidden_dim)
            num_beams: number of beams
            max_length: maximum generation length
        """
        max_length = 580 ## 替换为image_token的时候，最多577个token嵌入，所以这里生成多了也没用，其中会包含一些无用token，所以设置580是合适的
        device = encoder_hidden_states.device
        vocab_size = lm_head.out_features
        batch_size = encoder_hidden_states.size(0) // num_beams

        bos_token_id = 0 
        eos_token_id = 2 
        pad_token_id = 1 

        # 初始化输入 (bs*num_beams, 1)
        input_ids = torch.full(
            (batch_size * num_beams, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )

        # beam分数初始化
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9  # 除第一个beam外都屏蔽
        beam_scores = beam_scores.view(-1)  # 展平为(bs*num_beams,)

        # 每个样本的完成标志
        done = [False for _ in range(batch_size)]

        for step in range(max_length):
            decoder_outputs = decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
            next_token_logits = lm_head(decoder_outputs.last_hidden_state[:, -1, :])  # 取最后一步输出
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            # 累积beam分数
            next_scores = next_token_log_probs + beam_scores[:, None]
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # 取每个样本的topk
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

            # 准备下一步输入
            next_batch_input_ids = []
            next_beam_scores = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    next_batch_input_ids.extend(
                        [torch.full((1, input_ids.size(1)), pad_token_id, dtype=torch.long, device=device)]
                        * num_beams
                    )
                    next_beam_scores.extend([0] * num_beams)
                    continue

                beam_tokens = next_tokens[batch_idx]
                beam_scores_per_batch = next_scores[batch_idx]

                beam_indices = beam_tokens // vocab_size
                token_ids = beam_tokens % vocab_size

                # 更新输入序列
                batch_input = input_ids[batch_idx * num_beams + beam_indices]
                batch_input = torch.cat([batch_input, token_ids.unsqueeze(1)], dim=1)

                # 检查eos
                for i, token_id in enumerate(token_ids):
                    if token_id.item() == eos_token_id:
                        done[batch_idx] = True

                next_batch_input_ids.extend(batch_input)
                next_beam_scores.extend(beam_scores_per_batch)

            input_ids = torch.stack(next_batch_input_ids)
            beam_scores = torch.stack(next_beam_scores)

            if all(done):
                break

        # 取每个batch第一个beam的输出
        final_seqs = input_ids.view(batch_size, num_beams, -1)[:, 0, :]
        return final_seqs

    
    @torch.no_grad()
    def _generate_with_decoder_greedy_topk(
        self,
        decoder: nn.Module,
        lm_head: nn.Linear,
        encoder_hidden_states: torch.Tensor,
        num_beams: int = 1,
        **kwargs
    ) -> torch.LongTensor:
        """
        Florence-2 自定义快速生成函数。
        支持从 encoder_hidden_states (batch*num_beams, ...) 恢复真实 batch；
        忽略 beam search，仅保留每个样本的第一个 beam 结果；
        采用贪心 + 可选 top-k 限制以平衡速度与质量。
        """

        # === 1. 生成配置 ===
        max_length = 580 ## 替换为image_token的时候，最多577个token嵌入，所以这里生成多了也没用，其中会包含一些无用token，所以设置580是合适的
        eos_token_id = kwargs.get("eos_token_id", getattr(self.config.text_config, "eos_token_id", None))
        decoder_start_token_id = kwargs.get("decoder_start_token_id", 0)
        top_k = 3

        device = encoder_hidden_states.device

        # === 2. 计算真实 batch 大小，选择每个样本的第一个 beam ===
        if num_beams > 1:
            real_batch_size = encoder_hidden_states.shape[0] // num_beams
            encoder_hidden_states = encoder_hidden_states.view(real_batch_size, num_beams, *encoder_hidden_states.shape[1:])
            encoder_hidden_states = encoder_hidden_states[:, 0, ...].contiguous()  # 仅保留第一个 beam
        else:
            real_batch_size = encoder_hidden_states.shape[0]

        # === 3. 初始化生成序列 ===
        generated_ids = torch.full(
            (real_batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        past_key_values = None

        # === 4. 自回归循环 ===
        for _ in range(max_length - 1):
            input_ids_for_step = generated_ids[:, -1:] if past_key_values else generated_ids

            outputs = decoder(
                input_ids=input_ids_for_step,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            logits = lm_head(last_hidden_state)

            # --- 轻量 top-k 贪心选取 ---
            if top_k > 1:
                topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(topk_logits, dim=-1)
                next_token_ids = topk_indices.gather(-1, torch.multinomial(probs, num_samples=1))
            else:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)
            past_key_values = outputs.past_key_values

            # --- EOS 检查（按样本独立停止） ---
            if eos_token_id is not None:
                eos_mask = next_token_ids.eq(eos_token_id)
                if eos_mask.all():
                    break

        return generated_ids



    def filter_special_tokens(self, generated_ids, special_tokens, pad_token_id=1):
        """
        过滤掉生成的 token 中的特殊 token ID。
        然后根据目标长度截断或右填充，确保最终返回的 filtered_ids 的长度为 target_length。

        :param generated_ids: 生成的 token ID 序列，形式为 [batch_size, sequence_length]
        :param special_tokens: 特殊 token 的 ID 集合
        :param target_length: 目标长度，默认是 576
        :param pad_token_id: 填充用的 token ID，默认是 1（即 <pad> 的 ID）
        :return: 过滤后并处理为目标长度的 token ID 序列
        """
        filtered_ids = []
        target_length=576
        
        for seq in generated_ids:
            # 过滤掉特殊 token
            filtered_seq = [token for token in seq if token not in special_tokens]
            
            # 截断：如果过滤后的序列长度超过目标长度，则截断
            if len(filtered_seq) > target_length:
                filtered_seq = filtered_seq[:target_length]
            
            # 填充：如果过滤后的序列长度小于目标长度，则右填充至目标长度
            while len(filtered_seq) < target_length:
                filtered_seq.append(pad_token_id)  # 使用指定的 pad_token_id 填充
            
            # 添加到最终结果
            filtered_ids.append(filtered_seq)
        
        return filtered_ids


    def generate(
        self,
        input_ids, 
        inputs_embeds=None,
        pixel_values=None,
        return_decoder2_outputs=False,  # 新增参数，控制是否返回decoder2输出
        For_RL=False,
        **kwargs
        ):
        prompt_inputs_embeds=None
        if inputs_embeds is None:
            # print('inputs_embeds is none')
            
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                prompt_inputs_embeds = inputs_embeds
            # 2. Merge text and images
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)
                
        if inputs_embeds is not None:
            # print('inputs_embeds is not none')
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        
        ## 添加的预处理
        encoder_outputs, new_input_embeds= self.language_model.get_Florence2LanguageModel_encoder_output(
                input_ids  = None, #*#
                attention_mask = attention_mask, #*#
                head_mask = None,#*#
                encoder_outputs = None, #就是None
                inputs_embeds = inputs_embeds, #已设置
                output_attentions = False, #*#
                output_hidden_states = None,#*#
                return_dict = True, 
                first_forward = True, ##是否是第一次forward
                image_tokens_len=577
            )
        ## generate返回值是 generated_ids，这是生成的 token ID 序列
        ## 和inputid的形式其实是一样的，都是词汇表的里token id
        # answer
        answer = self.language_model.generate(
            input_ids=None,
            inputs_embeds=new_input_embeds,
            encoder_outputs = encoder_outputs,
            **kwargs,
        )
        
        if not return_decoder2_outputs: ##如果不需要返回辅助decoder_2的输出就直接返回answer
            return answer
        
        if return_decoder2_outputs or For_RL: ## 仅查看reason，不用于RL，则返回reason
            ## 计算decoder2的返回
            ## 可以使用self.get_decoder_2()获得辅助decoder的实例
            # [辅助解码器生成]
            # --- 1. 获取辅助解码器及其LM Head ---
            # decoder2 是您添加的辅助解码器
            decoder2 = self.get_decoder_2() 
            
            # 通常，辅助解码器会复用主模型的LM Head（与词嵌入权重绑定）
            lm_head2 = self.language_model.lm_think_head
            
            # --- 2. 调用辅助生成方法 ---
            ## 这里只能手动实现generate算法
            ## 考虑使用场景，reason输出是用于替换image_token,所以要保证 质量和速度，并不需要多样性
            ## 质量最好，用beam算法，_generate_with_decoder_beam_search(),但是生成速度会比较慢
            ## 生成速度快，就用贪心，再加一个Top-K,K=3; _generate_with_decoder_greedy_topk(),质量也可以，速度也很快       
            decoder2_answer = self._generate_with_decoder_greedy_topk(
                decoder=decoder2,
                lm_head=lm_head2,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                **kwargs # 将所有生成参数（如max_length等）传递给辅助生成函数
            )
            
            if not For_RL:
                # --- 3. 返回两个解码器的结果 ---
                return {
                    "answer": answer,
                    "reason": decoder2_answer
                }
                
            else: ## 继续生成
                # "<s>": 0, "<pad>": 1, "</s>": 2,"<unk>": 3,
                special_tokens = {0, 1, 2, 3}  # <s> = 0, <pad> = 1, </s> = 2, <unk> = 3
                # 过滤 reason 中的特殊 token, 并将reason长度填充到577，
                # 因为是批处理，在get_Florence2LanguageModel_encoder_output需要插入learnable token
                filtered_reason = self.filter_special_tokens(decoder2_answer, special_tokens) # [bs,token_len]
                filtered_reason = torch.tensor(filtered_reason, device=decoder2_answer.device) ## 转成tensor
                ## 通过嵌入层，这里获得的嵌入层是encoder-decoder-decoder2共享的那个嵌入层
                # 获得每一个reason_id的嵌入表示，最终长度与输入的token_id长度一样 [bs,token_len,hideen_size]
                #<==----模拟DaViT的[cls]+Token----==>
                reason_embeds = self.get_input_embeddings()(filtered_reason) #  [bs, 576, hidden_size]
                global_token = reason_embeds.mean(dim=1, keepdim=True) #  [bs,1,hidden_size]
                reason_tokens = torch.cat([global_token, reason_embeds], dim=1) # [bs, 577, hidden_size]

                ## 以reason_enbeds替换图像token，inputs_embeds是输入的prompts的嵌入
                input_embeds_selfReward, attention_mask = self._merge_input_ids_with_image_features(reason_tokens, prompt_inputs_embeds)

                decoder_inputs_selfReward, new_input_embeds_selfReward = self.language_model.get_Florence2LanguageModel_encoder_output(
                    input_ids  = None, #*#
                    attention_mask = attention_mask, #*#
                    head_mask = None,#*#
                    encoder_outputs = None, #就是None
                    inputs_embeds = input_embeds_selfReward, #已设置
                    output_attentions = False, #*#
                    output_hidden_states = None,#*#
                    return_dict = True, 
                    first_forward = True, ##是否是第一次forward
                    image_tokens_len=577
                )
                
                self_reward_answer = self.language_model.generate(
                    input_ids=None,
                    inputs_embeds=new_input_embeds_selfReward,
                    encoder_outputs = decoder_inputs_selfReward,
                    **kwargs,
                )
                
                return {
                    'answer':answer,
                    'self_reward_answer':self_reward_answer,
                    'reason':decoder2_answer
                }
        

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.language_model.shift_tokens_right(labels)

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
