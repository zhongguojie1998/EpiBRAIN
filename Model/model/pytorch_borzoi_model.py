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
from model.pytorch_borzoi_utils import Residual, TargetLengthCrop, undo_squashed_scale
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
    def __init__(self):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=15, padding="same")
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

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
        # whether to use FlashAttention, default as False
        self.flashed = config.flashed if "flashed" in config.__dict__.keys() else False
        # whether to use autocast, if using FlashAttention, always set to True, else, default as False
        self.use_autocast = (
            True if self.flashed else config.use_autocast if "use_autocast" in config.__dict__.keys() else False
        )

        # model settings
        ## initial convolutional layers
        self.conv_dna = ConvDna()
        self._max_pool = nn.MaxPool1d(kernel_size=2, padding=0)
        self.res_tower = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=608, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=608, out_channels=736, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=736, out_channels=896, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=896, out_channels=1056, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=1056, out_channels=1280, kernel_size=5),
        )
        self.unet1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=1280, out_channels=config.dim, kernel_size=5),
        )

        ## transformer layers
        transformer = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim, eps=0.001),
                            (
                                Attention(
                                    config.dim,
                                    heads=config.heads,
                                    dim_key=config.attn_dim_key,
                                    dim_value=config.attn_dim_value,
                                    dropout=config.attn_dropout,
                                    pos_dropout=config.pos_dropout,
                                    num_rel_pos_features=32,
                                )
                                if not self.flashed
                                else FlashAttention(
                                    config.dim,
                                    heads=config.heads,
                                    dropout=config.attn_dropout,
                                    pos_dropout=config.pos_dropout,
                                )
                            ),
                            nn.Dropout(0.2),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim, eps=0.001),
                            nn.Linear(config.dim, config.dim * 2),
                            nn.Dropout(config.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(config.dim * 2, config.dim),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                )
            )
        self.transformer = nn.Sequential(*transformer)

        # upsample convolutional layers
        self.upsample_layer_num = config.upsample_layer_num
        if config.upsample_layer_num >= 1:
            self.horizontal_conv1 = ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1)
            self.upsampling_unet1 = nn.Sequential(
                ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
                torch.nn.Upsample(scale_factor=2),
            )
            self.separable1 = ConvBlock(
                in_channels=config.dim, out_channels=config.dim, kernel_size=3, conv_type="separable"
            )
        if config.upsample_layer_num >= 2:
            self.horizontal_conv0 = ConvBlock(in_channels=1280, out_channels=config.dim, kernel_size=1)
            self.upsampling_unet0 = nn.Sequential(
                ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
                torch.nn.Upsample(scale_factor=2),
            )
            self.separable0 = ConvBlock(
                in_channels=config.dim, out_channels=config.dim, kernel_size=3, conv_type="separable"
            )

        # sequence crop
        if config.return_center_bins_only:
            self.crop = TargetLengthCrop(config.bins_to_return)
        else:
            self.crop = TargetLengthCrop(16384 - 32)  # as in Borzoi

        # output layers
        self.final_joined_convs = nn.Sequential(
            ConvBlock(in_channels=config.dim, out_channels=1920, kernel_size=1),
            nn.Dropout(0.1),
            nn.GELU(approximate="tanh"),
        )
        ## create output heads given the config.output_heads
        self.prediction_head = nn.ModuleDict()
        for head_name, track_num in config.output_heads.items():
            self.prediction_head[head_name] = nn.Conv1d(in_channels=1920, out_channels=track_num, kernel_size=1)
        self.final_softplus = nn.Softplus()

    def _init_weights(self):
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

    # def set_track_subset(self, track_subset):
    #     """
    #     Creates a subset of tracks by reassigning weights in the human head.

    #     Args:
    #        track_subset: Indices of the tracks to keep.

    #     Returns:
    #         None
    #     """
    #     if not hasattr(self, "human_head_bak"):
    #         self.human_head_bak = copy.deepcopy(self.human_head)
    #     else:
    #         self.reset_track_subset()
    #     self.human_head = nn.Conv1d(1920, len(track_subset), 1)
    #     self.human_head.weight = nn.Parameter(self.human_head_bak.weight[track_subset].clone())
    #     self.human_head.bias = nn.Parameter(self.human_head_bak.bias[track_subset].clone())

    # def reset_track_subset(self):
    #     """
    #     Resets the human head to the original weights.

    #     Returns:
    #         None
    #     """
    #     self.human_head = copy.deepcopy(self.human_head_bak)

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

    def predict(self, seqs, gene_slices, use_head="human", remove_squashed_scale=False):
        """
        Predicts only for bins of interest in a batched fashion
        Args:
            seqs (torch.tensor): Nx4xL tensor of one-hot sequences
            gene_slices List[torch.Tensor]: tensors indicating bins of interest
            use_head (str, optional): Indicate which prediction head to use. Defaults to human.
            removed_squashed_scale (bool, optional): whether to undo the squashed scale

        Returns:
            Tuple[torch.Tensor, list[int]]: 1xCxB tensor of bin predictions, as well as offsets that indicate where sequences begin/end
        """
        # Calculate slice offsets
        slice_list = []
        slice_length = []
        offset = self.crop.target_length
        for i, gene_slice in enumerate(gene_slices):
            slice_list.append(gene_slice + i * offset)
            slice_length.append(gene_slice.shape[0])
        slice_list = torch.concatenate(slice_list)
        # Get embedding after cropped
        with torch.amp.autocast("cuda", enabled=self.use_autocast):
            seq_embs = self.get_embs_after_crop(seqs)
        # Reshape to flatten the batch dimension (i.e. concatenate sequences)
        seq_embs = seq_embs.permute(1, 0, 2).flatten(start_dim=1).unsqueeze(0)
        # Extract the bins of interest
        seq_embs = seq_embs[:, :, slice_list]
        # Run the model head
        with torch.amp.autocast("cuda", enabled=self.use_autocast):
            seq_embs = self.final_joined_convs(seq_embs)
        with torch.amp.autocast("cuda", enabled=False):
            conved_slices = self.final_softplus(self.prediction_head[use_head](seq_embs.float()))
        if remove_squashed_scale:
            conved_slices = undo_squashed_scale(conved_slices)
        return conved_slices, slice_length

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


# class AnnotatedBorzoi(Borzoi):

#     def __init__(self, config, tracks_df=TRACKS_DF):
#         """
#         Initializes the `AnnotatedBorzoi` model.

#         Args:
#             config (BorzoiConfig): Configuration object containing model hyperparameters.
#             tracks_df (pd.DataFrame, optional): DataFrame containing track annotations. Defaults to the original targets.txt from Borzoi.

#         Returns:
#             None
#         """
#         super(AnnotatedBorzoi, self).__init__(config)
#         assert all(x in tracks_df.columns for x in ['identifier', 'file', 'clip', 'clip_soft', 'scale', 'sum_stat', 'strand_pair', 'description'])
#         tracks_df['track_transform'] = tracks_df['sum_stat'].apply(lambda x: 3/4 if x == "sum_sqrt" else 1.)
#         self._build_annotation_df(tracks_df)

#     def _build_annotation_df(self,tracks_df):
#         """
#         Builds the annotation tensors for sense and antisense, stranded and unstranded.

#         Args:
#             tracks_df (pd.DataFrame): DataFrame containing track annotations.

#         Returns:
#             None
#         """
#         # build tensor of tracks (sense, antisense, unstranded)
#         self.sense_tracks = torch.tensor(tracks_df.loc[tracks_df.identifier.str.contains('\+') | (tracks_df.index == tracks_df['strand_pair'])].index)
#         self.antisense_tracks = torch.tensor(tracks_df.loc[tracks_df.identifier.str.endswith('-') | (tracks_df.index == tracks_df['strand_pair'])].index)
#         # check that ordering of sense and antisense is meaningful
#         assert ((tracks_df.iloc[self.antisense_tracks].description.array == tracks_df.iloc[self.sense_tracks].description.array).sum() == self.sense_tracks.shape[0])
#         # remember backing dataframe
#         self.tracks_df = tracks_df
#         self.output_tracks_df = tracks_df.loc[tracks_df.identifier.str.contains('\+') | (tracks_df.index == tracks_df['strand_pair'])].reset_index(drop=True)
#         self.register_buffer('scale_values', torch.from_numpy(self.output_tracks_df.scale.values).float().unsqueeze(0).unsqueeze(-1).to(self.conv_dna.conv_layer.weight.device), persistent=False)
#         self.register_buffer('clip_values', torch.from_numpy(self.output_tracks_df.clip_soft.values).float().unsqueeze(0).unsqueeze(-1).to(self.conv_dna.conv_layer.weight.device), persistent=False)
#         self.register_buffer('track_transform', torch.from_numpy(self.output_tracks_df.track_transform.values).float().unsqueeze(0).unsqueeze(-1).to(self.conv_dna.conv_layer.weight.device), persistent=False)

#     def set_track_subset(self, track_subset):
#         if not hasattr(self, 'tracks_df_bak'):
#             tracks_df = self.tracks_df.copy()
#             self.tracks_df_bak = tracks_df
#         else:
#             tracks_df = self.tracks_df_bak.copy()
#         tracks_df = tracks_df.iloc[track_subset].reset_index(names='old_index')
#         # remap indices and strand pairs
#         oldidx_to_newidx = {x['old_index']:i for i,x in tracks_df.iterrows()}
#         try:
#             tracks_df["strand_pair"] = tracks_df['strand_pair'].apply(lambda x: oldidx_to_newidx[x])
#         except KeyError as e:
#             raise Exception("Strand pair is missing")
#         # subset head
#         super().set_track_subset(track_subset)
#         # rebuild annotation
#         tracks_df = tracks_df.drop(columns='old_index')
#         self._build_annotation_df(tracks_df)

#     def reset_track_subset(self):
#         tracks_df = self.tracks_df_bak.copy()
#         super.reset_track_subset()
#         self._build_annotation_df(tracks_df)

#     def _predict_gene_count(self, x,
#                 gene_slices,
#                 predict_antisense = False
#                ):
#         """
#         Predicts gene counts, optionally considering averaging over the antisense strand prediction.

#         Args:
#             x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).
#             gene_slices (List[torch.Tensor]): List of tensors, each containing the slice indices (bins) to extract from the output sequence.
#             predict_antisense (bool, optional): If True, predict for the antisense strand. Defaults to False.

#         Returns:
#              Tuple[torch.Tensor, list[int]]: 1xCxB tensor of bin predictions, as well as offsets that indicate where sequences begin/end
#         """
#         if predict_antisense:
#             # revcomp the input
#             x = x.flip(dims=(1,2))
#             # adapt the bins
#             gene_slices = [(self.crop.target_length - 1 - slices) for slices in gene_slices]
#         # run model
#         x, slice_length = self.predict(x, gene_slices = gene_slices)
#         if predict_antisense:
#             # extract the antisense and unstranded tracks
#             x = x[:,self.antisense_tracks,:]
#         else:
#             # extract the sense and unstranded tracks
#             x = x[:,self.sense_tracks,:]
#         return x, slice_length

#     def _undo_squashed_scale(self,x, old_transform = True, unscale = True):
#         """
#         Reverses the squashed scaling transformation applied to the output profiles.
#         Uses the annotation df to supply information how this should be done.

#         Args:
#             x (torch.Tensor): The input tensor to be unsquashed.
#             old_transform: Which version of the transform to use
#         Returns:
#             torch.Tensor: The unsquashed tensor.
#         """
#         x = x.clone()  # IMPORTANT BECAUSE OF IMPLACE OPERATIONS TO FOLLOW?

#         clip_soft = self.clip_values.expand_as(x)
#         track_transform = self.track_transform.expand_as(x)
#         if unscale:
#             scale = self.scale_values.expand_as(x)
#         else:
#             scale = 1.

#         if old_transform:
#             x = x / scale
#             unclip_mask = x > clip_soft
#             x[unclip_mask] = (x[unclip_mask] - clip_soft[unclip_mask]) ** 2 + clip_soft[unclip_mask]
#             x = x ** (1./track_transform)
#         else:
#             unclip_mask = x > clip_soft
#             x[unclip_mask] = (x[unclip_mask] - clip_soft[unclip_mask] + 1) ** 2 + clip_soft[unclip_mask] -1
#             x = (x + 1) ** (1.0 / track_transform) - 1
#             x = x / scale
#         return x

#     def predict_gene_count(self, x,
#                 gene_slices = None,
#                 average_strands = True,
#                 bin_level_transform = None,
#                 agg_fn = lambda x: torch.sum(x, dim=-1),
#                 log1p = True
#                ):
#         """
#         Predicts gene counts.

#         Args:
#             x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).
#             gene_slices (List[torch.Tensor], optional): List of tensors, each containing the slice indices to extract from the output sequence. If None, all slices are used. Defaults to None.
#             average_strands (bool, optional): If True, average predictions from the sense and antisense strands. Defaults to True.
#             bin_level_transform (function, optional): A transformation to apply at the bin level before aggregation. If None, the squashed scale transform is undone. Defaults to None
#             agg_fn (function, optional): Aggregation function (e.g. sum, mean) to use after squashed scale transform. Defaults to the sum over all bins.
#             log1p (bool, optional): If True, apply a log1p transformation to the final prediction. Defaults to True.

#         Returns:
#             torch.Tensor: Tensor of predicted gene counts with shape (N, C)
#         """
#         if gene_slices is None: # predict everything if no slices provided
#             gene_slices = [torch.tensor([x for x in range(self.crop.target_length)]) for i in range(x.shape[0])]
#         pred_sense, slice_length = self._predict_gene_count(x, gene_slices)
#         if average_strands:
#             pred_antisense, slice_length = self._predict_gene_count(x, gene_slices, predict_antisense = True)
#             pred = (pred_sense + pred_antisense)/2
#         else:
#             pred = pred_sense
#         # sum, unsquash and log1p-transform
#         if bin_level_transform is None:
#             pred = self._undo_squashed_scale(pred)
#         else:
#             pred = bin_level_transform(pred)
#         pred = torch.stack([agg_fn(x[0]) for x in torch.split(pred, slice_length, dim = 2)])
#         if log1p:
#             pred = torch.log1p(pred)
#         return pred
