from transformers import PretrainedConfig


class BorzoiConfig(PretrainedConfig):
    model_type = "borzoi"

    def __init__(
        self,
        dim=1536,
        depth=8,
        heads=8,
        attn_dim_key=64,
        attn_dim_value=192,
        attn_dropout=0.05,
        pos_dropout=0.01,
        dropout_rate=0.2,
        upsample_layer_num=2,
        return_center_bins_only=True,
        bins_to_return=6144,
        output_heads=dict(human=5313, mouse=1643),
        **kwargs,
    ):
        # model hyperparameter
        self.dim = dim  # hidden layer dim for conv layer output/transformer input
        self.depth = depth  # transformer layer depth
        self.heads = heads  # atten head num
        self.attn_dropout = attn_dropout  # atten dropout
        self.attn_dim_key = attn_dim_key  # set up atten, not used in flashatten
        self.attn_dim_value = attn_dim_value  # set up atten, not used in flashatten
        self.pos_dropout = pos_dropout  # set up atten, not used in flashatten
        self.dropout_rate = dropout_rate  # transformer linear layer dp rate
        self.upsample_layer_num = upsample_layer_num  # how many upsample layers to use, deciding final resolution, 0 corresponding to 128 bp resolution, 1 -> 64, 2 -> 32

        # sequence crop hyperparameter
        self.return_center_bins_only = return_center_bins_only
        self.bins_to_return = bins_to_return  # how many (central) bins used to calculate loss

        # prediction heads
        self.output_heads = output_heads  # set up the prediction head

        super().__init__(**kwargs)
