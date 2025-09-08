import torch
import torch.nn as nn


class LazyAdaLN(nn.Module):
    """
    Trainable per-channel scale/shift that initializes lazily
    from the *channel* dim of the first input it sees.

    Works with NCHW or TCHW shaped latents:
      (B,C,H,W) or (T,C,H,W) or even (C,H,W).
    """

    def __init__(self, init_scale=0.0, init_shift=0.0, dtype=None, device=None):
        super().__init__()
        self.weight = None
        self.bias = None
        self.init_scale = float(init_scale)
        self.init_shift = float(init_shift)
        self._dtype = dtype
        self._device = device

    def _init_params(self, C: int):
        w = torch.full((1, C, 1, 1), self.init_scale, dtype=self._dtype, device=self._device)
        b = torch.full((1, C, 1, 1), self.init_shift, dtype=self._dtype, device=self._device)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (C,H,W), (N,C,H,W) or (T,C,H,W) etc. Channel at dim=1 if 4D, else dim=0 for 3D.
        if x.dim() == 3:
            C = x.shape[0]
            if self.weight is None:
                self._init_params(C)
            # expand to (1,C,H,W) for broadcasting
            return (x.unsqueeze(0) * (1.0 + self.weight) + self.bias).squeeze(0)
        elif x.dim() >= 4:
            C = x.shape[1]
            if self.weight is None:
                self._init_params(C)
            return x * (1.0 + self.weight) + self.bias
        else:
            # Fallback: do nothing
            return x
