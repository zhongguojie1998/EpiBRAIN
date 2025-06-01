import torch.nn as nn

LOSS_DICT = {
    "poisson": nn.PoissonNLLLoss(log_input=False, reduction="mean"),
}
