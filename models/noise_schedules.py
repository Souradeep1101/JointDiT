import math

import torch
from torch import Tensor


def edm_sigma_from_tuni(
    t_uni: Tensor, P_mean: float, P_std: float, sigma_min: float, sigma_max: float
) -> Tensor:
    """
    Map uniform [0,1] -> log-normal sigma; clamp to [sigma_min, sigma_max].
    t_uni: shape [B]
    """
    # approximate inverse normal CDF via erfinv
    eps = 1e-6
    u = torch.clamp(t_uni, eps, 1.0 - eps)
    z = math.sqrt(2) * torch.erfinv(2 * u - 1)
    log_sigma = P_mean + P_std * z
    sigma = torch.exp(log_sigma)
    return torch.clamp(sigma, min=sigma_min, max=sigma_max)


def ddpm_like_sigma_from_tuni(t_uni: Tensor, sigma_min: float, sigma_max: float) -> Tensor:
    """
    Stage-A smoke: approximate DDPM noise scale as a monotone mapping.
    Real DDPM uses alpha_cumprod(t); for MVP we use scalar Gaussian sigma.
    """
    return torch.clamp(sigma_min + t_uni * (sigma_max - sigma_min), min=sigma_min, max=sigma_max)


def add_gaussian_noise(x0: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    """
    x_t = x0 + sigma * eps, with eps ~ N(0,1).
    sigma can be [B] and will be broadcast to x0 shape.
    Returns (x_t, eps).
    """
    # reshape sigma to broadcast
    while sigma.ndim < x0.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (x0.ndim - sigma.ndim)))
    eps = torch.randn_like(x0)
    x_t = x0 + sigma * eps
    return x_t, eps


def mse_x0(pred_x0: Tensor, target_x0: Tensor) -> Tensor:
    return torch.mean((pred_x0 - target_x0) ** 2)


def has_nan_or_inf(*tensors: Tensor) -> bool:
    for t in tensors:
        if not torch.isfinite(t).all():
            return True
    return False
