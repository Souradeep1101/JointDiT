import torch
import torch.nn as nn

from .perceiver_joint_attn import PerceiverJointAttention


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim, hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class JointBlock(nn.Module):
    """
    Tokens in d_model, per-modality LN, joint attention, per-modality FFN.
    """

    def __init__(self, d_model=256, heads=8, ff_mult=2, dropout=0.0, rope_cfg=None):
        super().__init__()
        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_a1 = nn.LayerNorm(d_model)
        self.attn = PerceiverJointAttention(
            d_model=d_model, heads=heads, dropout=dropout, rope_cfg=rope_cfg
        )
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_a2 = nn.LayerNorm(d_model)
        self.ff_v = FeedForward(d_model, mult=ff_mult)
        self.ff_a = FeedForward(d_model, mult=ff_mult)

    def forward(
        self,
        v_tokens,
        a_tokens,
        v_shape=None,
        a_shape=None,
        mode="full",
        extra_ctx: torch.Tensor | None = None,
    ):
        # attn + residual
        v_h, a_h = self.attn(
            self.ln_v1(v_tokens),
            self.ln_a1(a_tokens),
            v_shape,
            a_shape,
            mode=mode,
            extra_ctx=extra_ctx,
        )
        v_tokens = v_tokens + v_h
        a_tokens = a_tokens + a_h
        # ffn + residual
        v_tokens = v_tokens + self.ff_v(self.ln_v2(v_tokens))
        a_tokens = a_tokens + self.ff_a(self.ln_a2(a_tokens))
        return v_tokens, a_tokens
