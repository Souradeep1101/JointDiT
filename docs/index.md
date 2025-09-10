# JointDiT: Image → Sounding-Video (I2SV)

JointDiT is a compact research repo that turns a single image (plus optional text) into a **short video with synchronized audio**. It reuses the VAEs from **Stable Video Diffusion** (video) and **AudioLDM2** (audio), adds **Joint Blocks** (a cross-modal Transformer), and uses **JointCFG** for guidance at inference.

## Highlights
- **One pass, two modalities**: predict video & audio latents together (no pipeline hand-off).
- **Joint Blocks**: cross-modal attention over flattened video/audio tokens; optional CLIP text & image conditions.
- **Works on a single 48 GB GPU**: via query chunking, KV downsampling, and CUDA expandable segments.
- **Batteries included**: dataset helpers, latent caching, Stage-A/B training recipes, CLI + Gradio UI.

> **Note on current gaps vs. paper**
> The slicer “Input/Output/Expert” calls from the pretrained UNets are stubs in this codebase, and AdaLN isn’t fully wired to diffusion time/conditions yet—so quality is currently Transformer-only over VAE latents, not UNet-polished outputs. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Repo layout
```

configs/          # training & infer configs
models/           # JointDiT core + attention
scripts/data/     # dataset + latent caching
scripts/train/    # Stage A/B
scripts/infer/    # CLI + helpers (UI uses this)
scripts/ui/       # Gradio app

````

## Quickstart

```bash
# 0) env
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) fetch VAE weights (SVD + AudioLDM2)
make fetch-models

# 2) cache latents (mini)
make cache-train
make cache-val

# 3) smoke
make smoke-day06             # inference smoke
make day7-smoke              # Stage-B smoke (100 steps)

# 4) train long
make final-train-b           # uses autoscaled VRAM settings
# or: make final-train-b-keepenv JOINTDIT_* overrides

# 5) run UI
make ui
````

## What’s JointCFG?

Guidance that **subtracts the “modality-isolated” predictions** from the full joint prediction so samples obey conditions *and* stay A/V-consistent; paper also proposes JointCFG\*. &#x20;

## License / Intended use

Research & educational. See `LICENSE`. Contributions welcome.