import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def poisson(y_pred, y_true, eps: float = 1e-7):
    return F.poisson_nll_loss(y_pred, y_true, log_input=False, full=False, eps=eps, reduction="none")


def poisson_multinomial(
    y_pred,
    y_true,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    eps: float = 1e-7,
    rescale: bool = False,
):
    """Possion decomposition with multinomial specificity term.

    Args:
        total_weight (float): Weight of the Poisson total term.
        eps (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
    """
    seq_len = y_true.shape[1]

    if weight_range < 1:
        raise ValueError("Poisson Multinomial weight_range must be >=1")
    elif weight_range == 1:
        position_weights = torch.ones((1, seq_len, 1), device=y_true.device, dtype=y_true.dtype)
    else:
        pos_start = -(seq_len / 2 - 0.5)
        pos_end = seq_len / 2 + 0.5
        positions = torch.arange(pos_start, pos_end, device=y_true.device, dtype=torch.float32)
        sigma = -pos_start / (np.log(weight_range)) ** (1 / weight_exp)
        position_weights = torch.exp(-((positions / sigma) ** weight_exp))
        position_weights /= torch.max(position_weights)
        position_weights = position_weights.unsqueeze(0).unsqueeze(-1)

    y_true = y_true * position_weights
    y_pred = y_pred * position_weights

    # sum across lengths
    s_true = torch.sum(y_true, dim=-2)  # B x T
    s_pred = torch.sum(y_pred, dim=-2)  # B x T

    # total count poisson loss, mean across targets
    poisson_term = poisson(s_pred, s_true, eps=eps)  # B x T
    poisson_term /= torch.sum(position_weights)

    # add eps to protect against tiny values
    y_true += eps
    y_pred += eps

    # normalize to sum to one
    p_pred = y_pred / s_pred.unsqueeze(-2)  # B x L x T

    # multinomial loss
    pl_pred = torch.log(p_pred)  # B x L x T
    multinomial_dot = -y_true * pl_pred  # B x L x T
    multinomial_term = torch.sum(multinomial_dot, dim=-2)  # B x T
    multinomial_term /= torch.sum(position_weights)

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term  # B x T
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale


class PoissonMultinomialLoss(nn.Module):
    __constants__ = ["total_weight", "weight_range", "weight_exp", "eps", "rescale", "reduction"]

    def __init__(
        self,
        total_weight: float = 1.0,
        weight_range: float = 1.0,
        weight_exp: int = 4,
        eps: float = 1e-7,
        rescale: bool = False,
        reduction: str = "mean",
    ) -> None:
        super(PoissonMultinomialLoss, self).__init__()
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.eps = eps
        self.rescale = rescale
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            weight_range=self.weight_range,
            weight_exp=self.weight_exp,
            eps=self.eps,
            rescale=self.rescale,
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


LOSS_DICT = {"poisson": nn.PoissonNLLLoss, "poisson_mn": PoissonMultinomialLoss}
