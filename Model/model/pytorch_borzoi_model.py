# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model.config_borzoi import BorzoiConfig
from model.pytorch_borzoi_transformer import Attention, FlashAttention
from model.pytorch_borzoi_utils import Residual, TargetLengthCrop, compute_channel_sizes
from transformers import PreTrainedModel

## these for AnnotatedBorzoi
# DIR = Path(__file__).parents[0]
# TRACKS_DF = pd.read_table(str(DIR / "precomputed"/ "targets.txt")).rename(columns={'Unnamed: 0':'index'})

# torch.backends.cudnn.deterministic = True

# torch.set_float32_matmul_precision('high')


def lecun_normal_init(layer):
    if isinstance(layer, nn.Conv1d):
        fan_in = layer.in_channels * layer.kernel_size[0]
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(layer.weight, mean=0, std=std)
    elif isinstance(layer, nn.Linear):
        fan_in = layer.in_features
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(layer.weight, mean=0, std=std)
    # bias are set to 0 for all layers in TF
    if isinstance(layer, (nn.Linear, nn.Conv1d)) and layer.bias is not None:
        layer.bias.data.zero_()


def conditional_recursive_he_normal_init(layer):
    # --- Stop Condition ---
    # If the current module is one of the types we want to skip,
    # we stop the recursion for this entire branch.
    if isinstance(layer, (Attention, FlashAttention)):
        return
    # --- Initialization Logic for the current module ---
    # Apply the appropriate initialization based on the module type.
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        if layer.bias is not None:
            layer.bias.data.zero_()
    # --- Recursive  ---
    for child in layer.children():
        conditional_recursive_he_normal_init(child)


def other_init(layer):
    # same as in TF
    if isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d)):
        layer.bias.data.zero_()
        layer.weight.data.fill_(1.0)


class ConvDna(nn.Module):
    def __init__(self, out_channels=None, kernel_size=1, pool_size=1):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=4, out_channels=out_channels, kernel_size=kernel_size, padding="same"
        )
        self.max_pool = nn.MaxPool1d(kernel_size=pool_size, padding=0)

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1, conv_type="standard"):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, groups=in_channels, padding="same", bias=False
            )
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(
                in_channels, eps=0.001
            )  # momentum default is 0.1, it is equivalent to 0.9 in tensorflow as in Borzoi
            self.activation = nn.GELU(approximate="tanh")
            self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x


class Borzoi(PreTrainedModel):
    config_class = BorzoiConfig
    base_model_prefix = "borzoi"

    @staticmethod
    def from_hparams(**kwargs):
        return Borzoi(BorzoiConfig(**kwargs))

    def __init__(self, config):
        super(Borzoi, self).__init__(config)

        self.flashed = config.flashed
        self.use_autocast = config.use_autocast

        # model settings
        ## initial convolutional layers
        self.conv_dna = ConvDna(**config.conv_dna_param)
        self._max_pool = nn.MaxPool1d(kernel_size=config.res_tower_param["pool_size"], padding=0)

        res_tower_filters = [config.conv_dna_param["out_channels"]] + compute_channel_sizes(
            filters_init=config.res_tower_param["filters_init"],
            filters_end=config.res_tower_param["filters_end"],
            divisible_by=config.res_tower_param["divisible_by"],
            depth=config.res_tower_param["depth"],
        )
        internal_dim = res_tower_filters[-1]

        conv_layers = []
        for i in range(len(res_tower_filters[:-1]) - 1):
            in_channels = res_tower_filters[i]
            out_channels = res_tower_filters[i + 1]
            conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.res_tower_param["kernel_size"],
                )
            )
            if i < len(res_tower_filters[:-1]) - 2:
                conv_layers.append(nn.MaxPool1d(kernel_size=config.res_tower_param["pool_size"], padding=0))

        self.res_tower = nn.Sequential(*conv_layers)
        self.unet1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=config.res_tower_param["pool_size"], padding=0),
            ConvBlock(
                in_channels=res_tower_filters[-2],
                out_channels=internal_dim,
                kernel_size=config.res_tower_param["kernel_size"],
            ),
        )

        ## transformer layers
        transformer = []
        for _ in range(config.transformer_param["depth"]):
            heads = config.transformer_param["heads"]
            dropout = config.transformer_param["dropout"]
            attn_dropout = config.transformer_param["attn_dropout"]
            pos_dropout = config.transformer_param["pos_dropout"]

            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(internal_dim, eps=0.001),
                            (
                                Attention(
                                    internal_dim,
                                    heads=heads,
                                    dim_key=config.transformer_param["attn_dim_key"],
                                    dim_value=config.transformer_param["attn_dim_value"],
                                    dropout=attn_dropout,
                                    pos_dropout=pos_dropout,
                                    num_rel_pos_features=32,
                                )
                                if not self.flashed
                                else FlashAttention(
                                    internal_dim,
                                    heads=heads,
                                    dropout=attn_dropout,
                                    pos_dropout=pos_dropout,
                                )
                            ),
                            nn.Dropout(dropout),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(internal_dim, eps=0.001),
                            nn.Linear(internal_dim, internal_dim * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(internal_dim * 2, internal_dim),
                            nn.Dropout(dropout),
                        )
                    ),
                )
            )
        self.transformer = nn.Sequential(*transformer)

        # upsample convolutional layers
        self.upsample_layer_num = config.upsample_param["upsample_layer_num"]
        if self.upsample_layer_num >= 1:
            self.horizontal_conv1 = ConvBlock(in_channels=internal_dim, out_channels=internal_dim, kernel_size=1)
            self.upsampling_unet1 = nn.Sequential(
                ConvBlock(in_channels=internal_dim, out_channels=internal_dim, kernel_size=1),
                torch.nn.Upsample(scale_factor=config.res_tower_param["pool_size"]),
            )
            self.separable1 = ConvBlock(
                in_channels=internal_dim,
                out_channels=internal_dim,
                kernel_size=config.upsample_param["kernel_size"],
                conv_type="separable",
            )
        if self.upsample_layer_num >= 2:
            self.horizontal_conv0 = ConvBlock(
                in_channels=res_tower_filters[-2], out_channels=internal_dim, kernel_size=1
            )
            self.upsampling_unet0 = nn.Sequential(
                ConvBlock(in_channels=internal_dim, out_channels=internal_dim, kernel_size=1),
                torch.nn.Upsample(scale_factor=config.res_tower_param["pool_size"]),
            )
            self.separable0 = ConvBlock(
                in_channels=internal_dim,
                out_channels=internal_dim,
                kernel_size=config.upsample_param["kernel_size"],
                conv_type="separable",
            )

        # sequence crop
        if config.crop_param["return_center_bins_only"]:
            self.crop = TargetLengthCrop(config.crop_param["bins_to_return"])
        else:
            self.crop = TargetLengthCrop(16384 - 32)  # as in Borzoi

        # output layers
        self.final_joined_convs = nn.Sequential(
            ConvBlock(
                in_channels=internal_dim,
                out_channels=config.final_joined_conv_param["out_channels"],
                kernel_size=1,
            ),
            nn.Dropout(config.final_joined_conv_param["dropout"]),
            nn.GELU(approximate="tanh"),
        )

        ## create output heads given the config.output_heads
        self.prediction_head = nn.ModuleDict()
        for head_name, track_num in config.output_heads.items():
            self.prediction_head[head_name] = nn.Conv1d(
                in_channels=config.final_joined_conv_param["out_channels"], out_channels=track_num, kernel_size=1
            )
        self.final_softplus = nn.Softplus()

    def _init_weights(self, module):
        # this is for loading from hugging face parameter
        pass


    def init_weights(self):
        """Initialize the weights"""
        # kernel_initializer = lecun_normal, for all layers except for transformer
        # kernel_initializer = he_normal, for transformer layers
        # apply lecun norm to all layers except for transformer
        self.conv_dna.apply(lecun_normal_init)
        self.res_tower.apply(lecun_normal_init)
        self.unet1.apply(lecun_normal_init)
        if self.upsample_layer_num >= 1:
            self.horizontal_conv1.apply(lecun_normal_init)
            self.upsampling_unet1.apply(lecun_normal_init)
            self.separable1.apply(lecun_normal_init)
        if self.upsample_layer_num >= 2:
            self.horizontal_conv0.apply(lecun_normal_init)
            self.upsampling_unet0.apply(lecun_normal_init)
            self.separable0.apply(lecun_normal_init)
        self.final_joined_convs.apply(lecun_normal_init)
        self.prediction_head.apply(lecun_normal_init)

        # apply he normal to transformer layers, only to the linear output layer after Attention
        # The Attention has handeled the initialization of its weights, don't overwrite them
        conditional_recursive_he_normal_init(self.transformer)

        # Other initializations
        self.apply(other_init)

    def get_embs_after_crop(self, x):
        """
        Performs the forward pass of the model until right before the final conv layers, and includes a cropping layer.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).

        Returns:
             torch.Tensor: Output of the model up to the cropping layer with shape (N, dim, crop_bin_num)
        """
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)  # resolution 32
        x_unet1 = self.unet1(x_unet0)  # resolution 64
        x = self._max_pool(x_unet1)  # resolution 128
        x = self.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        if self.upsample_layer_num >= 1:
            x_unet1 = self.horizontal_conv1(x_unet1)
            x = self.upsampling_unet1(x)
            x += x_unet1
            x = self.separable1(x)
        if self.upsample_layer_num >= 2:
            x_unet0 = self.horizontal_conv0(x_unet0)
            x = self.upsampling_unet0(x)
            x += x_unet0
            x = self.separable0(x)
        x = self.crop(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

    def forward(self, x, use_head="human", data_parallel_training=False):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).
            use_head (str, optional): Indicate which prediction head to use. Defaults to human.
            data_parallel_training (bool, optional): If True, perform forward pass specific to DDP. Defaults to False.

        Returns:
            torch.Tensor: Output tensor with shape (N, C, crop_bin_num), where C is the number of tracks.
        """
        with torch.amp.autocast("cuda", enabled=self.use_autocast):
            x = self.get_embs_after_crop(x)
            x = self.final_joined_convs(x)

        # disable autocast for more precision in final layer
        with torch.amp.autocast("cuda", enabled=False):
            if data_parallel_training:
                # we need this to get gradients for both heads if doing DDP training
                out = self.final_softplus(self.prediction_head[use_head](x.float()))
                for head_name, head in self.prediction_head.items():
                    if head != use_head:
                        out += 0 * self.prediction_head[head_name](x.float()).sum()
                return out
            else:
                return self.final_softplus(self.prediction_head[use_head](x.float()))
