import torch


def _rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_angles(L, dim, base=10000.0, device="cpu", dtype=torch.float32):
    # standard 1D RoPE angles for last-dim rotation (dim must be even)
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(L, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (L, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, dim)
    return emb


def apply_rope_1d(q, k):
    """
    q,k: (B, H, L, D) with D even
    returns rotated (q,k)
    """
    B, H, L, D = q.shape
    if D % 2 != 0:
        # fall back to identity if bad dim
        return q, k
    angles = _build_angles(L, D, device=q.device, dtype=q.dtype)  # (L, D)
    cos = angles.cos()[None, None, :, :]  # (1,1,L,D)
    sin = angles.sin()[None, None, :, :]
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
