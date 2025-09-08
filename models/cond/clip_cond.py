# models/cond/clip_cond.py
from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    # only for type hints; won’t run at import time in CI
    from PIL import Image as PILImage


class CLIPCondEncoder(nn.Module):
    """
    Returns (B,1,d_model) tokens for text and/or image.
    CLIP is frozen; only the linear projections learn if you finetune them.
    Lazy-imports open_clip and Pillow to keep CI light.
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

        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "open_clip is required for CLIPCondEncoder (pip install open-clip-torch)."
            ) from e
        self._oc: ModuleType = open_clip

        model, _, preprocess = self._oc.create_model_and_transforms(
            clip_name, pretrained=pretrained, device=self.device
        )
        self.clip = model.eval()
        self.preprocess = preprocess
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # feature width across variants
        try:
            width = self.clip.text_projection.shape[1]
        except Exception:
            width = self.clip.ln_final.weight.shape[0]

        self.txt_proj = nn.Linear(width, d_model, bias=True).to(self.device, dtype=self.proj_dtype)
        self.img_proj = nn.Linear(width, d_model, bias=True).to(self.device, dtype=self.proj_dtype)
        self.to(self.device)

    @torch.no_grad()
    def _encode_text(self, prompts: Sequence[str]) -> Optional[torch.Tensor]:
        if not prompts:
            return None
        toks = self._oc.tokenize(list(prompts)).to(self.device)
        x = self.clip.encode_text(toks)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def _encode_image(
        self, images: Sequence[Union[str, "PILImage", None]]
    ) -> Optional[torch.Tensor]:
        if not images:
            return None
        try:
            from PIL import Image as PILImage  # local import to avoid CI dep
        except Exception as e:  # pragma: no cover
            raise ImportError("Pillow is required for image guidance (pip install pillow).") from e

        pil_list: List[Optional[PILImage]] = []
        for im in images:
            if im is None:
                pil_list.append(None)
                continue
            pil_list.append(
                PILImage.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
            )

        idx = [i for i, p in enumerate(pil_list) if p is not None]
        if not idx:
            return None

        batch = torch.stack([self.preprocess(pil_list[i]) for i in idx], 0).to(self.device)
        feat = self.clip.encode_image(batch)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

        out = torch.zeros(len(images), feat.shape[-1], device=self.device, dtype=feat.dtype)
        out[torch.as_tensor(idx, device=self.device)] = feat
        return out

    def forward(
        self,
        prompts: Optional[Sequence[str]] = None,
        images: Optional[Sequence[Union[str, "PILImage", None]]] = None,
        device: Optional[Union[str, torch.device]] = None,  # kept for API compat; ignored
    ):
        t_ctx = i_ctx = None
        if prompts is not None:
            t = self._encode_text(prompts)
            if t is not None:
                t = t.to(self.txt_proj.weight.dtype)
                t_ctx = self.txt_proj(t).unsqueeze(1)
        if images is not None:
            im = self._encode_image(images)
            if im is not None:
                im = im.to(self.img_proj.weight.dtype)
                i_ctx = self.img_proj(im).unsqueeze(1)
        return t_ctx, i_ctx
