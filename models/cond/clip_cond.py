# models/cond/clip_cond.py
from __future__ import annotations

from types import ModuleType
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from PIL import Image


class CLIPCondEncoder(nn.Module):
    """
    Tiny adapter that returns (B, 1, d_model) tokens for text and/or image.

    Design notes:
    - We **lazy-import `open_clip` in __init__** so importing this module does not
      require the dependency (helps CI / light envs).
    - CLIP is frozen by default; only the linear projections learn (if you choose
      to finetune them).
    """

    def __init__(
        self,
        d_model: int,
        clip_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_clip: bool = True,
        device: Union[str, torch.device] = "cuda",
        proj_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.proj_dtype = proj_dtype

        # Lazy import to avoid hard dependency at module import time
        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "open_clip is required for CLIPCondEncoder. "
                "Install with: pip install open-clip-torch"
            ) from e

        self._oc: ModuleType = open_clip

        # Create CLIP on the requested device
        model, _, preprocess = self._oc.create_model_and_transforms(
            clip_name, pretrained=pretrained, device=self.device
        )
        self.clip = model.eval()
        self.preprocess = preprocess

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # Figure out CLIP feature width (works across variants)
        try:
            width = self.clip.text_projection.shape[1]
        except Exception:
            width = self.clip.ln_final.weight.shape[0]

        # Projection layers (trainable if you want)
        self.txt_proj = nn.Linear(width, d_model, bias=True).to(self.device, dtype=self.proj_dtype)
        self.img_proj = nn.Linear(width, d_model, bias=True).to(self.device, dtype=self.proj_dtype)

        # Ensure module tensors are on the target device
        self.to(self.device)

    @torch.no_grad()
    def _encode_text(self, prompts: Sequence[str]) -> Optional[torch.Tensor]:
        if not prompts:
            return None
        toks = self._oc.tokenize(list(prompts)).to(self.device)
        x = self.clip.encode_text(toks)  # (B, width) on self.device
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x

    @torch.no_grad()
    def _encode_image(
        self, images: Sequence[Union[str, Image.Image, None]]
    ) -> Optional[torch.Tensor]:
        if not images:
            return None

        # Maintain alignment with input list; allow Nones
        pil_list: List[Optional[Image.Image]] = []
        for im in images:
            if im is None:
                pil_list.append(None)
                continue
            pil_list.append(
                Image.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
            )

        idx = [i for i, p in enumerate(pil_list) if p is not None]
        if not idx:
            return None

        batch = torch.stack([self.preprocess(pil_list[i]) for i in idx], 0).to(self.device)
        feat = self.clip.encode_image(batch)  # (K, width) on self.device
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

        out = torch.zeros(len(images), feat.shape[-1], device=self.device, dtype=feat.dtype)
        out[torch.as_tensor(idx, device=self.device)] = feat
        return out

    def forward(
        self,
        prompts: Optional[Sequence[str]] = None,
        images: Optional[Sequence[Union[str, Image.Image, None]]] = None,
        device: Optional[Union[str, torch.device]] = None,  # kept for API compat; ignored
    ):
        """
        Returns:
            t_ctx: Optional[Tensor]  (B, 1, d_model)
            i_ctx: Optional[Tensor]  (B, 1, d_model)
        """
        # We keep internal state on self.device; do not juggle devices here.
        t_ctx = i_ctx = None

        if prompts is not None:
            t = self._encode_text(prompts)
            if t is not None:
                # Cast features to proj weight dtype for safe matmul
                t = t.to(self.txt_proj.weight.dtype)
                t_ctx = self.txt_proj(t).unsqueeze(1)  # (B, 1, d_model)

        if images is not None:
            im = self._encode_image(images)
            if im is not None:
                im = im.to(self.img_proj.weight.dtype)
                i_ctx = self.img_proj(im).unsqueeze(1)  # (B, 1, d_model)

        return t_ctx, i_ctx
