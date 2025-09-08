import torch
import torch.nn as nn

from models.joint.joint_block import JointBlock


class JointDiT(nn.Module):
    """
    Shape-preserving skeleton with optional CLIP context tokens (text/image).
    """

    def __init__(
        self,
        d_model=256,
        heads=8,
        ff_mult=2,
        dropout=0.0,
        rope_cfg=None,
        video_in_ch=4,
        audio_in_ch=8,
        joint_blocks=2,
        svd_slicer=None,
        aldm_slicer=None,
        **kwargs,
    ):
        # back-compat alias
        if "n_blocks" in kwargs:
            joint_blocks = kwargs.pop("n_blocks")
        super().__init__()
        self.video_in_ch = video_in_ch
        self.audio_in_ch = audio_in_ch
        self.svd = svd_slicer
        self.aldm = aldm_slicer

        # channels <-> d_model
        self.v_inproj = nn.Linear(video_in_ch, d_model)
        self.a_inproj = nn.Linear(audio_in_ch, d_model)
        self.v_outproj = nn.Linear(d_model, video_in_ch)
        self.a_outproj = nn.Linear(d_model, audio_in_ch)

        self.blocks = nn.ModuleList(
            [
                JointBlock(
                    d_model=d_model,
                    heads=heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    rope_cfg=rope_cfg,
                )
                for _ in range(joint_blocks)
            ]
        )

    @staticmethod
    def _vid_flatten(x):  # (B,T,C,H,W) -> (B, L=THW, C)
        B, T, C, H, W = x.shape
        meta = {"T": T, "H": H, "W": W}
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, C, T * H * W).transpose(1, 2).contiguous()
        return x, meta

    @staticmethod
    def _vid_unflatten(tokens, meta, C):  # (B, L, C) -> (B,T,C,H,W)
        B, L, _ = tokens.shape
        T, H, W = meta["T"], meta["H"], meta["W"]
        assert L == T * H * W, "video token length mismatch"
        x = (
            tokens.transpose(1, 2)
            .contiguous()
            .view(B, C, T, H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        return x

    @staticmethod
    def _aud_flatten(x):  # (B,C,H,W) -> (B, L=HW, C)
        B, C, H, W = x.shape
        meta = {"H": H, "W": W}
        x = x.view(B, C, H * W).transpose(1, 2).contiguous()
        return x, meta

    @staticmethod
    def _aud_unflatten(tokens, meta, C):  # (B, L, C) -> (B,C,H,W)
        B, L, _ = tokens.shape
        H, W = meta["H"], meta["W"]
        assert L == H * W, "audio token length mismatch"
        x = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        return x

    def _maybe_call(self, slicer, name, x):
        if slicer is None:
            return x
        mod = getattr(slicer, name, None)
        if mod is not None and callable(mod):
            try:
                return mod(x)
            except TypeError:
                try:
                    return mod(x)
                except Exception:
                    pass
        meth = getattr(slicer, f"run_{name}", None)
        if callable(meth):
            return meth(x)
        return x

    def forward(
        self,
        v_latents,
        a_latents,
        mode="full",
        t_ctx: torch.Tensor | None = None,
        i_ctx: torch.Tensor | None = None,
    ):
        """
        v_latents: (B,T,Cv,H,W)   a_latents: (B,Ca,H,W)
        t_ctx / i_ctx: optional CLIP tokens (B,1,d_model)
        """
        Bv, Tv, Cv, Hv, Wv = v_latents.shape
        Ba, Ca, Ha, Wa = a_latents.shape
        assert Bv == Ba, "batch size mismatch video/audio"

        v = self._maybe_call(self.svd, "block1", v_latents)
        a = self._maybe_call(self.aldm, "block1", a_latents)

        v_tok, v_meta = self._vid_flatten(v)
        a_tok, a_meta = self._aud_flatten(a)
        v_tok = self.v_inproj(v_tok)
        a_tok = self.a_inproj(a_tok)

        # build shared extra context once
        extra_list = []
        if t_ctx is not None:
            extra_list.append(t_ctx)
        if i_ctx is not None:
            extra_list.append(i_ctx)
        extra_ctx = torch.cat(extra_list, dim=1) if extra_list else None

        for blk in self.blocks:
            v_tok, a_tok = blk(v_tok, a_tok, v_meta, a_meta, mode=mode, extra_ctx=extra_ctx)

        v_tok = self.v_outproj(v_tok)
        a_tok = self.a_outproj(a_tok)
        v = self._vid_unflatten(v_tok, v_meta, Cv)
        a = self._aud_unflatten(a_tok, a_meta, Ca)

        v = self._maybe_call(self.svd, "block4", v)
        v = self._maybe_call(self.svd, "block5", v)
        v = self._maybe_call(self.svd, "block6", v)
        a = self._maybe_call(self.aldm, "block4", a)
        a = self._maybe_call(self.aldm, "block5", a)
        a = self._maybe_call(self.aldm, "block6", a)

        return v, a
