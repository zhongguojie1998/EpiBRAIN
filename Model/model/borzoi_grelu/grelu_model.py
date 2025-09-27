import torch.nn as nn

from grelu.lightning import LightningModel

model = LightningModel.load_from_checkpoint("/home/dl3738/work/BICAN/Chk/borzoi_grelu/model.ckpt", map_location="cpu").model

class BorzoiGreluModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = model

    def forward(self, x, use_head="human"):

        return self.model(x).permute(0, 2, 1)