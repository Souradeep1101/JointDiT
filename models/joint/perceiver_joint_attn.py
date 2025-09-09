# models/joint/perceiver_joint_attn.py
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope_1d

try:
    from torch.backends.cuda import sdp_kernel

    # prefer flash / mem-efficient when available
    sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
except Exception:
    pass


def _maybe_int_env(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v) if v else default
    except Exception:
        return default


class PerceiverJointAttention(nn.Module):
    """
    Per-modality Q/K/V & out-projections, shared heads & head_dim.
    SDPA-backed with optional query chunking and K/V downsampling.
    Supports optional `extra_ctx` tokens (e.g., CLIP) appended to K/V bank.

    `mode`:
      - "full":     cross-modal (V attends to V+A[+ctx], A attends to V+A[+ctx])
      - "iso_v":    video-only attention (A path is passthrough)
      - "iso_a":    audio-only attention (V path is passthrough)
    """

    def __init__(self, d_model: int = 256, heads: int = 8, dropout: float = 0.0, rope_cfg=None):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.head_dim = d_model // heads
        self.rope_v = bool((rope_cfg or {}).get("video", {}).get("enable", True))
        self.rope_a = bool((rope_cfg or {}).get("audio", {}).get("enable", True))

        self.q_chunk_v = max(0, _maybe_int_env("JOINTDIT_Q_CHUNK_V", 0))
        self.q_chunk_a = max(0, _maybe_int_env("JOINTDIT_Q_CHUNK_A", 0))
        self.kv_downsample = max(1, _maybe_int_env("JOINTDIT_KV_DOWNSAMPLE", 1))

        # video projections
        self.qv = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, d_model, bias=True)
        self.vv = nn.Linear(d_model, d_model, bias=True)
        self.ov = nn.Linear(d_model, d_model, bias=True)

        # audio projections
        self.qa = nn.Linear(d_model, d_model, bias=True)
        self.ka = nn.Linear(d_model, d_model, bias=True)
        self.va = nn.Linear(d_model, d_model, bias=True)
        self.oa = nn.Linear(d_model, d_model, bias=True)

        # generic context projections (for CLIP tokens)
        self.kc = nn.Linear(d_model, d_model, bias=True)
        self.vc = nn.Linear(d_model, d_model, bias=True)

        self.drop = nn.Dropout(dropout)

    # shape helpers
    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.heads, self.head_dim
        return x.view(B, L, H, Dh).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _sdpa_chunked(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk: int
    ) -> torch.Tensor:
        B, H, Lq, Dh = q.shape
        _, _, Lk, _ = k.shape
        q_ = q.reshape(B * H, Lq, Dh)
        k_ = k.reshape(B * H, Lk, Dh)
        v_ = v.reshape(B * H, Lk, Dh)

        if chunk and chunk > 0 and Lq > chunk:
            outs = []
            for i in range(0, Lq, chunk):
                qi = q_[:, i : i + chunk, :]
                oi = F.scaled_dot_product_attention(qi, k_, v_, dropout_p=0.0, is_causal=False)
                outs.append(oi)
            out_ = torch.cat(outs, dim=1)
        else:
            out_ = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0, is_causal=False)

        return out_.reshape(B, H, out_.shape[1], Dh)

    def _downsample_seq(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return x
        B, H, L, Dh = x.shape
        pad = (factor - (L % factor)) % factor
        if pad:
            x = torch.cat([x, x.new_zeros(B, H, pad, Dh)], dim=2)
            L = x.shape[2]
        x_ch = x.permute(0, 1, 3, 2).reshape(B * H, Dh, L)
        x_ds = F.avg_pool1d(x_ch, kernel_size=factor, stride=factor)
        Lp = x_ds.shape[-1]
        x_ds = x_ds.reshape(B, H, Dh, Lp).permute(0, 1, 3, 2).contiguous()
        return x_ds

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        v_shape: Optional[dict] = None,
        a_shape: Optional[dict] = None,
        mode: str = "full",
        extra_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v_tokens: (B, Lv, D)   a_tokens: (B, La, D)
        extra_ctx: (B, Lc, D) optional (e.g., CLIP text/image tokens)
        """
        # Q/K/V per modality
        qv = self._reshape_heads(self.qv(v_tokens))
        kv = self._reshape_heads(self.kv(v_tokens))
        vv = self._reshape_heads(self.vv(v_tokens))

        qa = self._reshape_heads(self.qa(a_tokens))
        ka = self._reshape_heads(self.ka(a_tokens))
        va = self._reshape_heads(self.va(a_tokens))

        # RoPE
        if self.rope_v:
            qv, kv = apply_rope_1d(qv, kv)
        if self.rope_a:
            qa, ka = apply_rope_1d(qa, ka)

        # K/V context bank according to mode
        if mode == "iso_v":
            K_cat, V_cat = kv, vv
        elif mode == "iso_a":
            K_cat, V_cat = ka, va
        else:
            K_cat = torch.cat([kv, ka], dim=2)
            V_cat = torch.cat([vv, va], dim=2)

        # Append extra context (no RoPE)
        if extra_ctx is not None:
            kc = self._reshape_heads(self.kc(extra_ctx))
            vc = self._reshape_heads(self.vc(extra_ctx))
            K_cat = torch.cat([K_cat, kc], dim=2)
            V_cat = torch.cat([V_cat, vc], dim=2)

        # Optional downsample for memory
        if self.kv_downsample > 1:
            K_cat = self._downsample_seq(K_cat, self.kv_downsample)
            V_cat = self._downsample_seq(V_cat, self.kv_downsample)

        # Attend
        v_ctx = self._sdpa_chunked(qv, K_cat, V_cat, chunk=self.q_chunk_v)
        out_v = self.ov(self._merge_heads(self.drop(v_ctx)))

        if mode == "iso_v":
            out_a = a_tokens
        else:
            a_ctx = self._sdpa_chunked(qa, K_cat, V_cat, chunk=self.q_chunk_a)
            out_a = self.oa(self._merge_heads(self.drop(a_ctx)))
            if mode == "iso_a":
                out_v = v_tokens

        return out_v, out_a
