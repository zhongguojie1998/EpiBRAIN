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
    dim: int = -2,
):
    """Possion decomposition with multinomial specificity term.

    Args:
        total_weight (float): Weight of the Poisson total term.
        weight_range (float): Range of position-based weighting (only used when dim=-2).
        weight_exp (int): Exponent for position-based weighting (only used when dim=-2).
        eps (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
        dim (int): Dimension to average over. -2 for length dimension, -1 for tracks dimension.
    """
    # Apply position weighting only for length dimension (dim=-2)
    if dim == -2:
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

        # transform to float32 to ensure multinomial loss precision
        y_true = y_true.to(torch.float32) * position_weights / torch.mean(position_weights)
        y_pred = y_pred.to(torch.float32) * position_weights / torch.mean(position_weights)
        divisor = torch.sum(position_weights)
    else:
        # No position weighting for other dimensions
        y_true = y_true.to(torch.float32)
        y_pred = y_pred.to(torch.float32)
        divisor = y_true.shape[dim]

    # add eps to protect against tiny values
    y_true += eps
    y_pred += eps
    
    # sum across specified dimension
    s_true = torch.sum(y_true, dim=dim)
    s_pred = torch.sum(y_pred, dim=dim)

    # total count poisson loss
    poisson_term = poisson(s_pred, s_true, eps=eps)
    poisson_term /= divisor

    # multinomial loss - use log rule to compute log odds
    pl_pred = torch.log(y_pred) - torch.log(s_pred.unsqueeze(dim))
    # Use raw y_true (not normalized) to match TensorFlow implementation
    multinomial_dot = -y_true * pl_pred
    multinomial_term = torch.sum(multinomial_dot, dim=dim)
    multinomial_term /= divisor

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale


def poisson_reverse_multinomial(
    y_pred,
    y_true,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    eps: float = 1e-7,
    rescale: bool = False,
    dim: int = -2,
):
    """Possion decomposition with reverse multinomial specificity term.

    Reverses the multinomial term to compute KL(p_pred || p_true) instead of KL(p_true || p_pred).

    Args:
        total_weight (float): Weight of the Poisson total term.
        weight_range (float): Range of position-based weighting (only used when dim=-2).
        weight_exp (int): Exponent for position-based weighting (only used when dim=-2).
        eps (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
        dim (int): Dimension to average over. -2 for length dimension, -1 for tracks dimension.
    """
    # Apply position weighting only for length dimension (dim=-2)
    if dim == -2:
        seq_len = y_true.shape[1]

        if weight_range < 1:
            raise ValueError("Poisson Reverse Multinomial weight_range must be >=1")
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

        # transform to float32 to ensure multinomial loss precision
        y_true = y_true.to(torch.float32) * position_weights / torch.mean(position_weights)
        y_pred = y_pred.to(torch.float32) * position_weights / torch.mean(position_weights)
        divisor = torch.sum(position_weights)
    else:
        # No position weighting for other dimensions
        y_true = y_true.to(torch.float32)
        y_pred = y_pred.to(torch.float32)
        divisor = y_true.shape[dim]

    # add eps to protect against tiny values
    y_true += eps
    y_pred += eps

    # sum across specified dimension
    s_true = torch.sum(y_true, dim=dim)
    s_pred = torch.sum(y_pred, dim=dim)

    # total count poisson loss
    poisson_term = poisson(s_pred, s_true, eps=eps)
    poisson_term /= divisor

    # normalize to sum to one, then log transform TRUE (reversed!)
    pl_true = torch.log(y_true) - torch.log(s_true.unsqueeze(dim))
    p_pred = y_pred / s_pred.unsqueeze(dim)
    # reverse multinomial loss: KL(p_pred || p_true)
    multinomial_dot = -p_pred * pl_true
    multinomial_term = torch.sum(multinomial_dot, dim=dim)
    multinomial_term /= divisor

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale


def poisson_combined_multinomial(
    y_pred,
    y_true,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    eps: float = 1e-7,
    rescale: bool = False,
    dim: int = -2,
):
    """Poisson decomposition with combined (symmetrized) multinomial specificity term.

    Combines forward and reverse KL divergences: 0.5 * KL(p_true || p_pred) + 0.5 * KL(p_pred || p_true).

    Args:
        total_weight (float): Weight of the Poisson total term.
        weight_range (float): Range of position-based weighting (only used when dim=-2).
        weight_exp (int): Exponent for position-based weighting (only used when dim=-2).
        eps (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
        dim (int): Dimension to average over. -2 for length dimension, -1 for tracks dimension.
    """
    # Apply position weighting only for length dimension (dim=-2)
    if dim == -2:
        seq_len = y_true.shape[1]

        if weight_range < 1:
            raise ValueError("Poisson Combined Multinomial weight_range must be >=1")
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

        # transform to float32 to ensure multinomial loss precision
        y_true = y_true.to(torch.float32) * position_weights / torch.mean(position_weights)
        y_pred = y_pred.to(torch.float32) * position_weights / torch.mean(position_weights)
        divisor = torch.sum(position_weights)
    else:
        # No position weighting for other dimensions
        y_true = y_true.to(torch.float32)
        y_pred = y_pred.to(torch.float32)
        divisor = y_true.shape[dim]

    # add eps to protect against tiny values
    y_true += eps
    y_pred += eps

    # sum across specified dimension
    s_true = torch.sum(y_true, dim=dim)
    s_pred = torch.sum(y_pred, dim=dim)

    # total count poisson loss
    poisson_term = poisson(s_pred, s_true, eps=eps)
    poisson_term /= divisor

    # normalize to sum to one, then log transform both
    pl_pred = torch.log(y_pred) - torch.log(s_pred.unsqueeze(dim))
    pl_true = torch.log(y_true) - torch.log(s_true.unsqueeze(dim))
    p_pred = y_pred / s_pred.unsqueeze(dim)

    # forward multinomial loss: use raw y_true (not normalized) to match TensorFlow
    forward_multinomial = -y_true * pl_pred
    # reverse multinomial loss: KL(p_pred || p_true)
    reverse_multinomial = -p_pred * pl_true

    # combine with equal weights (0.5 each)
    multinomial_term = 0.5 * torch.sum(forward_multinomial, dim=dim) + 0.5 * torch.sum(reverse_multinomial, dim=dim)
    multinomial_term /= divisor

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale


def poisson_js(
    y_pred,
    y_true,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    eps: float = 1e-7,
    rescale: bool = False,
    dim: int = -2,
):
    """Poisson decomposition with Jensen-Shannon divergence specificity term.

    Args:
        total_weight (float): Weight of the Poisson total term.
        weight_range (float): Range of position-based weighting (only used when dim=-2).
        weight_exp (int): Exponent for position-based weighting (only used when dim=-2).
        eps (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
        dim (int): Dimension to average over. -2 for length dimension, -1 for tracks dimension.
    """
    # Apply position weighting only for length dimension (dim=-2)
    if dim == -2:
        seq_len = y_true.shape[1]

        if weight_range < 1:
            raise ValueError("Poisson JS weight_range must be >=1")
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

        # transform to float32 to ensure loss precision
        y_true = y_true.to(torch.float32) * position_weights / torch.mean(position_weights)
        y_pred = y_pred.to(torch.float32) * position_weights / torch.mean(position_weights)
        divisor = torch.sum(position_weights)
    else:
        # No position weighting for other dimensions
        y_true = y_true.to(torch.float32)
        y_pred = y_pred.to(torch.float32)
        divisor = y_true.shape[dim]

    # add eps to protect against tiny values
    y_true = y_true + eps
    y_pred = y_pred + eps

    # sum across specified dimension
    s_true = torch.sum(y_true, dim=dim)
    s_pred = torch.sum(y_pred, dim=dim)

    # total count poisson loss
    poisson_term = poisson(s_pred, s_true, eps=eps)
    poisson_term /= divisor

    # normalize to sum to one
    p_true = y_true / s_true.unsqueeze(dim)
    p_pred = y_pred / s_pred.unsqueeze(dim)

    # compute mixture distribution M = 0.5 * (P + Q)
    p_mixture = 0.5 * (p_true + p_pred)

    # Jensen-Shannon divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    # KL(P||M) = sum(P * log(P/M)) = sum(P * (log(P) - log(M)))
    kl_true_mixture = p_true * (torch.log(p_true) - torch.log(p_mixture))
    kl_pred_mixture = p_pred * (torch.log(p_pred) - torch.log(p_mixture))

    js_term = 0.5 * torch.sum(kl_true_mixture, dim=dim) + 0.5 * torch.sum(kl_pred_mixture, dim=dim)
    js_term /= divisor

    # normalize to scale of 1:1 term ratio
    loss_raw = js_term + total_weight * poisson_term
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


class PoissonReverseMultinomialLoss(nn.Module):
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
        super(PoissonReverseMultinomialLoss, self).__init__()
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.eps = eps
        self.rescale = rescale
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_reverse_multinomial(
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


class PoissonCombinedMultinomialLoss(nn.Module):
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
        super(PoissonCombinedMultinomialLoss, self).__init__()
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.eps = eps
        self.rescale = rescale
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_combined_multinomial(
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


class PoissonJSLoss(nn.Module):
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
        super(PoissonJSLoss, self).__init__()
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.eps = eps
        self.rescale = rescale
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_js(
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


class CrossCellMultinomialLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 0.2, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            eps=self.eps,
            rescale=self.rescale,
            dim=-1,  # Average across tracks dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossCellReverseMultinomialLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 0.2, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_reverse_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            eps=self.eps,
            rescale=self.rescale,
            dim=-1,  # Average across tracks dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossCellCombinedMultinomialLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 0.2, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_combined_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            eps=self.eps,
            rescale=self.rescale,
            dim=-1,  # Average across tracks dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossCellJSLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 0.2, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_js(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            eps=self.eps,
            rescale=self.rescale,
            dim=-1,  # Average across tracks dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossBatchMultinomialLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 1.0,
                 weight_range: float = 1.0, weight_exp: int = 4, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            weight_range=self.weight_range,
            weight_exp=self.weight_exp,
            eps=self.eps,
            rescale=self.rescale,
            dim=-3,  # Average across batch dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossBatchReverseMultinomialLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 1.0,
                 weight_range: float = 1.0, weight_exp: int = 4, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_reverse_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            weight_range=self.weight_range,
            weight_exp=self.weight_exp,
            eps=self.eps,
            rescale=self.rescale,
            dim=-3,  # Average across batch dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossBatchCombinedMultinomialLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 1.0,
                 weight_range: float = 1.0, weight_exp: int = 4, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_combined_multinomial(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            weight_range=self.weight_range,
            weight_exp=self.weight_exp,
            eps=self.eps,
            rescale=self.rescale,
            dim=-3,  # Average across batch dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CrossBatchJSLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", total_weight: float = 1.0,
                 weight_range: float = 1.0, weight_exp: int = 4, rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.rescale = rescale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = poisson_js(
            y_pred,
            y_true,
            total_weight=self.total_weight,
            weight_range=self.weight_range,
            weight_exp=self.weight_exp,
            eps=self.eps,
            rescale=self.rescale,
            dim=-3,  # Average across batch dimension
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


LOSS_DICT = {
    "poisson": nn.PoissonNLLLoss,
    "poisson_mn": PoissonMultinomialLoss,
    "poisson_rmn": PoissonReverseMultinomialLoss,
    "poisson_cmn": PoissonCombinedMultinomialLoss,
    "poisson_js": PoissonJSLoss,
    "cross_cell_mn": CrossCellMultinomialLoss,
    "cross_cell_rmn": CrossCellReverseMultinomialLoss,
    "cross_cell_cmn": CrossCellCombinedMultinomialLoss,
    "cross_cell_js": CrossCellJSLoss,
    "cross_batch_mn": CrossBatchMultinomialLoss,
    "cross_batch_rmn": CrossBatchReverseMultinomialLoss,
    "cross_batch_cmn": CrossBatchCombinedMultinomialLoss,
    "cross_batch_js": CrossBatchJSLoss,
    "transcripts_poisson_mn": PoissonMultinomialLoss,
    "transcripts_poisson_rmn": PoissonReverseMultinomialLoss,
    "transcripts_poisson_cmn": PoissonCombinedMultinomialLoss,
    "transcripts_poisson_js": PoissonJSLoss,
    "transcripts_cross_cell_mn": CrossCellMultinomialLoss,
    "transcripts_cross_cell_rmn": CrossCellReverseMultinomialLoss,
    "transcripts_cross_cell_cmn": CrossCellCombinedMultinomialLoss,
    "transcripts_cross_cell_js": CrossCellJSLoss,
    "transcripts_cross_batch_mn": CrossBatchMultinomialLoss,
    "transcripts_cross_batch_rmn": CrossBatchReverseMultinomialLoss,
    "transcripts_cross_batch_cmn": CrossBatchCombinedMultinomialLoss,
    "transcripts_cross_batch_js": CrossBatchJSLoss,
}
