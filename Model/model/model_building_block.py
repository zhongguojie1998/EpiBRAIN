# Adapted from https://github.com/lucidrains/enformer-pytorch/tree/main
#
# MIT License
#
# Copyright (c) 2021 Phil Wang, 2024 Johannes Hingerl

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =========================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum, nn


def get_positional_features_central_mask(positions, features, seq_len):
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(pow_rate, torch.arange(1, features + 1, device=positions.device)).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f"feature size is not divisible by number of components ({num_components})")

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings


def fast_relative_shift(a, b):
    return (
        einsum("i d, j d -> i j", a, b)
        .flatten()
        .as_strided(size=(a.shape[0], a.shape[0]), stride=((a.shape[0] - 1) * 2, 1), storage_offset=a.shape[0] - 1)
    )


fast_relative_shift = torch.vmap(
    torch.vmap(fast_relative_shift), in_dims=(0, None)
)  # https://johahi.github.io/blog/2024/fast-relative-shift/


class Attention(nn.Module):

    def __init__(
        self, dim=1536, *, num_rel_pos_features=1, heads=8, dim_key=64, dim_value=64, dropout=0.0, pos_dropout=0.0
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.register_buffer(
            "positions",
            get_positional_embed(4096, self.num_rel_pos_features, self.to_v.weight.device),
            persistent=False,
        )  # 4096 as this should always be the seq len at this pos?

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # initialize parameters
        def he_normal_init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self.to_q.apply(he_normal_init)
        self.to_k.apply(he_normal_init)
        self.to_v.apply(he_normal_init)
        # don't change to_out!
        self.to_rel_k.apply(he_normal_init)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)

        positions = self.pos_dropout(self.positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = fast_relative_shift(q + self.rel_pos_bias, rel_k)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class FlashAttention(nn.Module):
    def __init__(
        self,
        dim=1536,
        heads=8,
        dropout=0.15,
        pos_dropout=0.15,  # Not used
        rotary_emb_base=20000.0,
        rotary_emb_scale_base=None,
    ):
        super().__init__()

        from flash_attn.modules.mha import MHA

        self.mha = MHA(
            use_flash_attn=True,
            embed_dim=dim,
            num_heads=heads,
            num_heads_kv=(heads // 2),
            qkv_proj_bias=True,  # False,
            out_proj_bias=True,
            dropout=dropout,
            softmax_scale=(dim / heads) ** -0.5,
            causal=False,
            rotary_emb_dim=128,
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base=rotary_emb_scale_base,
            fused_bias_fc=False,
        )

        nn.init.kaiming_normal_(self.mha.Wqkv.weight, nonlinearity="relu")
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)
        nn.init.ones_(self.mha.Wqkv.bias)

    def forward(self, x):
        out = self.mha(x)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f"sequence length {seq_len} is less than target length {target_len}")

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, conv_type="standard", res_link=False):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            norm = nn.Identity()
            depthwise_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, groups=in_channels, padding="same", bias=False
            )
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            activation = nn.Identity()
        else:
            norm = nn.BatchNorm1d(
                in_channels, eps=0.001
            )  # momentum default is 0.1, it is equivalent to 0.9 in tensorflow as in Borzoi
            activation = nn.GELU(approximate="tanh")
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")

        self.block = nn.Sequential(norm, activation, conv_layer)

        self.res_link = res_link
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_diff = out_channels - in_channels

    def forward(self, x):
        out = self.block(x)

        if self.res_link:
            if self.channel_diff > 0:
                # zeros shape: [B, channel_diff, L]
                zeros = x.new_zeros(x.size(0), self.channel_diff, *x.shape[2:])
                x_skipped = torch.cat([x, zeros], dim=1)  # -> (B, C_out, L)
            else:
                # if out_channels <= in_channels
                x_skipped = x[:, : self.out_channels, ...]

            out += x_skipped

        return out


class ConvDna(nn.Module):

    def __init__(self, out_channels, kernel_size=1, **kwargs):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=4, out_channels=out_channels, kernel_size=kernel_size, padding="same"
        )

    def forward(self, x):
        out = self.conv_layer(x)

        return out


class PredictionHead(nn.Module):

    def __init__(self, in_features, heads_config=None, **kwargs):
        """
        Initialize PredictionHead with multiple output heads.
        
        Args:
            in_features (int): Input feature dimension
            heads_config (dict): Configuration for all prediction heads
            
        Expected heads_config format:
        {
            # Shared parameters (optional)
            'use_cell_encoder': bool,           # Whether to use shared celltype encoder (default: False)
            'celltype_hidden_dim': int,         # Hidden dim for celltype embedding (default: in_features//2)
            
            # Individual head configs
            'head_name': {
                'task': str,                    # Required: 'regression' or 'classification'
                
                # For use_cell_encoder=True:
                'celltype_num': int,            # Required: number of cell types (must be same across heads)
                'modality_num': int,            # Required: number of modalities (can differ between heads)
                'class_num': int,               # Required for classification: number of classes
                
                # For use_cell_encoder=False:
                'track_num': int,               # Required: total number of output tracks
                'class_num': int,               # Required for classification: number of classes
            }
        }
        """
        super().__init__()

        self.in_features = in_features

        self._extract_and_validate_params(heads_config)
        self._setup_cell_encoder()
        self._setup_heads()

    def _extract_and_validate_params(self, config):
        """Extract shared params and validate head configs"""
        if not config:
            raise ValueError("heads_config cannot be empty")

        config = config.copy()

        # Step 1: Pop shared parameters
        self.use_cell_encoder = config.pop("use_cell_encoder", False)
        self.celltype_hidden_dim = config.pop("celltype_hidden_dim", None)

        # Set default celltype_hidden_dim if needed
        if self.use_cell_encoder and self.celltype_hidden_dim is None:
            self.celltype_hidden_dim = self.in_features // 2
            print(f"celltype_hidden_dim not provided, defaulting to {self.celltype_hidden_dim}")

        # Step 2: Validate and store head configs
        self.heads_config = {}
        celltype_nums = []

        for head_name, head_config in config.items():
            if "task" not in head_config:
                raise ValueError(f"Head {head_name}: invalid config, must be dict with 'task' field")

            # If use_cell_encoder, validate celltype_num exists and collect for consistency check
            if self.use_cell_encoder:
                if "celltype_num" not in head_config:
                    raise ValueError(f"Head {head_name}: celltype_num required when use_cell_encoder=True")
                celltype_nums.append(head_config["celltype_num"])

            self.heads_config[head_name] = head_config

        # Step 3: Validate celltype_num consistency if using cell encoder
        if self.use_cell_encoder:
            if len(set(celltype_nums)) > 1:
                raise ValueError(
                    f"All heads must have same celltype_num when use_cell_encoder=True. Got: {set(celltype_nums)}"
                )
            self.shared_celltype_num = celltype_nums[0]

    def _setup_cell_encoder(self):
        """Setup shared celltype encoder if needed"""
        if self.use_cell_encoder:
            self.shared_cell_encoder = nn.Linear(
                self.in_features, self.celltype_hidden_dim * self.shared_celltype_num
            )
        else:
            self.shared_cell_encoder = None

    def _validate_head_config(self, head_name, config):
        """Validate single head config and calculate derived params"""
        task = config["task"]
        if task not in ["regression", "classification"]:
            raise ValueError(f"Head {head_name}: unsupported task '{task}', must be 'regression' or 'classification'")
        track_num = config.get("track_num")
        celltype_num = config.get("celltype_num")
        modality_num = config.get("modality_num")
        class_num = config.get("class_num")

        head_config = {"task": task}

        # Determine track configuration
        if self.use_cell_encoder:
            if modality_num is None:
                raise ValueError(f"Head {head_name}: modality_num required when use_cell_encoder=True")
            head_config.update(
                {
                    "celltype_num": celltype_num,
                    "modality_num": modality_num,
                    "track_num": celltype_num * modality_num,
                }
            )
        else:
            if track_num is not None:
                if celltype_num is not None or modality_num is not None:
                    print(
                        f"Head {head_name}: track_num provided along with celltype_num/modality_num. Using track_num only."
                    )
                head_config.update(
                    {
                        "track_num": track_num,
                        "celltype_num": None,
                        "modality_num": None,
                    }
                )
            else:
                raise ValueError(f"Head {head_name}: Must provide track_num when use_cell_encoder=False")

        head_config["class_num"] = class_num

        # Calculate output channels
        if task == "classification":
            if class_num is None:
                raise ValueError(f"Head {head_name}: class_num required for classification task")
            head_config["out_channels"] = head_config["track_num"] * class_num
        elif task == "regression":
            head_config["out_channels"] = head_config["track_num"]
        else:
            raise ValueError(
                f"Head {head_name}: unsupported task '{task}', must be 'regression' or 'classification'"
            )

        return head_config

    def _setup_heads(self):
        """Build all prediction heads based on configurations"""
        self.heads = nn.ModuleDict()
        self.head_configs = {}

        for head_name, config in self.heads_config.items():
            head_config = self._validate_head_config(head_name, config)
            self.head_configs[head_name] = head_config

            if self.use_cell_encoder:
                # Use shared cell encoder, only build modality head
                if head_config["task"] == "regression":
                    self.heads[head_name] = nn.Linear(self.celltype_hidden_dim, head_config["modality_num"])
                    self.register_parameter(f'{head_name}_scale', nn.Parameter(torch.ones(head_config["track_num"])))
                else:  # classification
                    self.heads[head_name] = nn.Linear(
                        self.celltype_hidden_dim, head_config["modality_num"] * head_config["class_num"]
                    )
            else:
                # Direct linear layer
                self.heads[head_name] = nn.Linear(self.in_features, head_config["out_channels"])
                if head_config["task"] == "regression":
                    self.register_parameter(
                        f"{head_name}_scale", nn.Parameter(torch.ones(head_config["out_channels"]))
                    )

    def forward(self, x):
        """
        x: [B, L, in_features]
        returns: dict of {head_name: prediction} where prediction is
                [B, L, track_num] for regression or [B, L, track_num, class_num] for classification
        """
        B, L, _ = x.shape

        # Compute shared cell embeddings if needed
        if self.use_cell_encoder:
            shared_cell_embs = self.shared_cell_encoder(x)  # [B, L, celltype_hidden_dim * celltype_num]
            shared_cell_embs = shared_cell_embs.view(
                B, L, self.shared_celltype_num, self.celltype_hidden_dim
            )  # [B, L, C, H]

        # Compute predictions for all heads
        outputs = {}
        for head_name, head_config in self.head_configs.items():
            head_layer = self.heads[head_name]

            if self.use_cell_encoder:
                # Use shared cell embeddings
                mod_preds = head_layer(shared_cell_embs)  # [B, L, C, M] or [B, L, C, M*K]

                if head_config["task"] == "regression":
                    scale = getattr(self, f"{head_name}_scale")
                    pred = F.softplus(mod_preds.view(B, L, -1)) * F.softplus(scale)  # [B, L, C*M]
                else:  # classification
                    pred = mod_preds.view(B, L, -1, head_config["class_num"])  # [B, L, C*M, K]
            else:
                # Direct linear prediction
                pred = head_layer(x)  # [B, L, out_channels]

                if "classification" in head_config["task"]:
                    pred = pred.view(B, L, head_config["track_num"], head_config["class_num"])
                else:  # regression
                    scale = getattr(self, f"{head_name}_scale")
                    pred = F.softplus(pred) * F.softplus(scale)

            outputs[head_name] = pred

        return outputs
    