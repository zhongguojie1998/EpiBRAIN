import torch
import torch.nn as nn
from model.model_building_block import (
    Attention,
    ConvBlock,
    ConvDna,
    FlashAttention,
    PredictionHead,
    Residual,
    TargetLengthCrop,
)
from model.model_config import BorzoiConfig
from model.model_utils import (
    compute_channel_sizes,
    conditional_recursive_he_normal_init,
    lecun_normal_init,
    other_init,
    std_pred_head_config,
)
from transformers import PreTrainedModel


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

        # Conv block
        filter_dims = [config.conv_dna_param["out_channels"]] + compute_channel_sizes(
            filters_init=config.res_tower_param["filters_init"],
            filters_end=config.res_tower_param["filters_end"],
            divisible_by=config.res_tower_param["divisible_by"],
            depth=config.res_tower_param["depth"],
        )
        kernel_sizes = [config.conv_dna_param["kernel_size"]] + [
            config.res_tower_param["kernel_size"]
        ] * config.res_tower_param["depth"]

        internal_dim = filter_dims[-1]
        pool_size = config.res_tower_param["pool_size"]

        self.resolution = [pool_size**i for i in range(len(filter_dims))]
        res_tower_setting = {
            f"resol_{resol}": {
                "in_channels": filter_dims[i - 1] if i >= 1 else 4,
                "out_channels": filter_dims[i],
                "kernel_size": kernel_sizes[i],
                "res_link": config.res_tower_param["res_link"] if i >= 1 else False,
            }
            for i, resol in enumerate(self.resolution)
        }

        ## currently, it's still borzoi style simple conv (AlphaGenome use more complicated conv here)
        self.res_tower = nn.ModuleDict()
        for resol in self.resolution:
            block_cls = ConvBlock if resol > 1 else ConvDna
            self.res_tower[f"resol_{resol}_conv"] = block_cls(**res_tower_setting[f"resol_{resol}"])
            self.res_tower[f"resol_{resol}_pool"] = nn.MaxPool1d(kernel_size=pool_size, padding=0)

        # transformer block
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

        # uptaking block
        self.upsample_resolution = self.resolution[::-1][: config.upsample_param["upsample_layer_num"]]

        ## currently, it's still borzoi style conv (AlphaGenome crush the channel of input x)
        self.upsample_tower = nn.ModuleDict()
        for resol in self.upsample_resolution:
            unet_channel = res_tower_setting[f"resol_{resol}"]["out_channels"]
            self.upsample_tower[f"resol_{resol}_horizon"] = ConvBlock(
                in_channels=unet_channel, out_channels=internal_dim, kernel_size=1
            )
            self.upsample_tower[f"resol_{resol}_x"] = nn.Sequential(
                ConvBlock(in_channels=internal_dim, out_channels=internal_dim, kernel_size=1),
                torch.nn.Upsample(scale_factor=pool_size),
            )
            self.upsample_tower[f"resol_{resol}_separable"] = ConvBlock(
                in_channels=internal_dim,
                out_channels=internal_dim,
                kernel_size=config.upsample_param["kernel_size"],
                conv_type="separable",
                res_link=config.upsample_param["res_link"],
            )

        # sequence crop
        if config.crop_param["return_center_bins_only"]:
            self.crop = TargetLengthCrop(config.crop_param["bins_to_return"])
        else:
            self.crop = TargetLengthCrop(-1)  # return full length

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

        ## create unified prediction head with all output heads
        heads_config, use_cell_encoder, celltype_hidden_dim = std_pred_head_config(config.output_heads)
        self.prediction_head = PredictionHead(
            in_features=config.final_joined_conv_param["out_channels"], heads_config=heads_config,
            use_cell_encoder=use_cell_encoder, celltype_hidden_dim=celltype_hidden_dim
        )

    def _init_weights(self, module):
        # this is for loading from hugging face parameter
        pass

    def init_weights(self):
        """Initialize the weights"""
        # kernel_initializer = lecun_normal, for all layers except for transformer
        # kernel_initializer = he_normal, for transformer layers
        # apply lecun norm to all layers except for transformer
        self.res_tower.apply(lecun_normal_init)
        self.upsample_tower.apply(lecun_normal_init)
        self.final_joined_convs.apply(lecun_normal_init)
        self.prediction_head.apply(lecun_normal_init)

        # apply he normal to transformer layers, only to the linear output layer after Attention
        # The Attention has handeled the initialization of its weights, don't overwrite them
        conditional_recursive_he_normal_init(self.transformer)

        # Other initializations
        self.apply(other_init)

    def sequence_encoder(self, x):
        intermediates = {}

        for resol in self.resolution:
            x = self.res_tower[f"resol_{resol}_conv"](x)
            if resol in self.upsample_resolution:
                intermediates[f"resol_{resol}"] = x
            x = self.res_tower[f"resol_{resol}_pool"](x)

        return x, intermediates

    def sequence_decoder(self, x, intermediates):
        for resol in self.upsample_resolution:
            x = self.upsample_tower[f"resol_{resol}_x"](x)
            horizon = self.upsample_tower[f"resol_{resol}_horizon"](intermediates[f"resol_{resol}"])
            x += horizon
            x = self.upsample_tower[f"resol_{resol}_separable"](x)

        return x

    def forward(self, x, use_head=None, **kwargs):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).
            use_head (str or list or None, optional): 
                - None: return all heads as dict
                - str: return single head result directly (not wrapped in dict)
                - list: return specified heads as dict

        Returns:
            torch.Tensor or dict: 
                - If use_head is str: Output tensor with shape (N, C, crop_bin_num)
                - If use_head is None or list: Dict of {head_name: output_tensor}
        """
        with torch.amp.autocast("cuda", enabled=self.use_autocast):
            x, intermediates = self.sequence_encoder(x)

        # if we are using flashattention, the transformer layer must use autocast
        with torch.amp.autocast("cuda", enabled=self.use_autocast if not self.flashed else True):
            x = self.transformer(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)

        with torch.amp.autocast("cuda", enabled=self.use_autocast):
            x = self.sequence_decoder(x, intermediates)
            x = self.crop(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            x = self.final_joined_convs(x)
            x = x.permute(0, 2, 1)

        # disable autocast for more precision in final layer
        with torch.amp.autocast("cuda", enabled=False):
            # Get all predictions from unified prediction head
            all_outputs = self.prediction_head(x)

            # Return based on use_head parameter
            if use_head is None:
                # Return all heads
                return all_outputs
            elif isinstance(use_head, str):
                # Return single head directly (not wrapped in dict)
                return all_outputs[use_head]
            elif isinstance(use_head, (list, tuple)):
                # Return specified heads as dict
                return {head_name: all_outputs[head_name] for head_name in use_head}
            else:
                raise ValueError(f"use_head must be None, str, or list/tuple, got {type(use_head)}")
