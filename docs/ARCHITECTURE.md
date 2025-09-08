# Architecture

**JointDiT** (skeleton):
- Video latents `(T, C_v, H_v, W_v)` and audio latents `(1, C_a, H_a, W_a)` from VAEs.
- Flatten to tokens → per-modality LayerNorm → **Perceiver-style joint attention** (shared heads).
- Optional RoPE for audio/video.
- Output tokens → project back to latent channels → decode with VAEs.

**Conditioning (CLIP)**:
- `models/cond/clip_cond.py` encodes **text** and **image** to CLIP embeddings, projects to `d_model`.
- `t_ctx` and `i_ctx` are passed through `JointDiT(..., t_ctx=..., i_ctx=...)` (conditioning is additive in attention pre-LN).
- Inference uses JointCFG-ish guidance:
  - video/audio: full vs. iso (no cross)
  - text+: pos vs. no-text
  - text-: pos vs. neg-text
  - image: pos vs. no-image

**Schedules**
- Video (EDM-like, log-space high→low), Audio (DDPM-like proxy).
