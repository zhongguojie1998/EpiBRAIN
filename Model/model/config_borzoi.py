import copy

from transformers import PretrainedConfig


def deep_update_dict(default: dict, override: dict) -> dict:

    result = copy.deepcopy(default)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_update_dict(result[key], val)
        else:
            result[key] = val
    return result


class BorzoiConfig(PretrainedConfig):
    model_type = "borzoi"

    _DEFAULT_CONV_DNA = dict(
        out_channels=512,
        kernel_size=15,
        pool_size=2,
    )
    _DEFAULT_RES_TOWER = dict(
        filters_init=608,
        filters_end=1536,
        divisible_by=32,
        kernel_size=5,
        pool_size=2,
        depth=6,
    )
    _DEFAULT_TRANSFORMER = dict(
        heads=8,
        attn_dim_key=64,
        attn_dim_value=192,
        pos_dropout=0.01,
        attn_dropout=0.05,
        dropout=0.2,
        depth=8,
    )
    _DEFAULT_CROP = dict(
        return_center_bins_only=True,
        bins_to_return=5120,
    )
    _DEFAULT_UPSAMPLE = dict(
        upsample_layer_num=2,
        kernel_size=3,
    )
    _DEFAULT_FINAL_CONV = dict(
        out_channels=1920,
        dropout=0.1,
    )

    def __init__(
        self,
        flashed: bool = False,
        use_autocast: bool = False,
        conv_dna_param: dict | None = None,
        res_tower_param: dict | None = None,
        transformer_param: dict | None = None,
        crop_param: dict | None = None,
        upsample_param: dict | None = None,
        final_joined_conv_param: dict | None = None,
        output_heads=dict(human=7611, mouse=2608),
        **kwargs,
    ):

        self.flashed = flashed
        self.use_autocast = use_autocast if not flashed else True

        self.conv_dna_param = deep_update_dict(BorzoiConfig._DEFAULT_CONV_DNA, conv_dna_param or {})
        self.res_tower_param = deep_update_dict(BorzoiConfig._DEFAULT_RES_TOWER, res_tower_param or {})
        self.transformer_param = deep_update_dict(BorzoiConfig._DEFAULT_TRANSFORMER, transformer_param or {})
        self.crop_param = deep_update_dict(BorzoiConfig._DEFAULT_CROP, crop_param or {})
        self.upsample_param = deep_update_dict(BorzoiConfig._DEFAULT_UPSAMPLE, upsample_param or {})
        self.final_joined_conv_param = deep_update_dict(BorzoiConfig._DEFAULT_FINAL_CONV, final_joined_conv_param or {})

        # prediction heads
        self.output_heads = output_heads  # set up the prediction head

        super().__init__(**kwargs)
