# JointDiT — Project Overview

This repository implements a Joint Diffusion Transformer (JointDiT) pipeline that **animates a static image and generates synchronized audio** from it (I2SV: image → sounding video). It integrates pretrained video and audio diffusion experts and introduces cross‑modal interaction for temporally aligned outputs.

## Goals
- Reproduce key components: input → N×JointBlocks → output
- Validate modes: `full`, `iso_v`, `iso_a`
- Provide reproducible training/inference scripts and day‑by‑day progress via notebooks

## How it works (high level)
- **Input Block:** modality‑specific preprocessing for video/audio latents
- **Joint Blocks:** modality experts + a cross‑modal full‑attention layer for **fine‑grained V↔A interaction**
- **Output Block:** decoders back to clean video/audio latents

```mermaid
graph TD
  A[Image (frame 0)] --> B[Video VAE → latents]
  A --> C[CLIP / Audio cond.]
  B --> D[Joint Blocks: V/A Experts + Perceiver-like Joint Attention]
  C --> D
  D --> E[Video/Audio Output Blocks]
  E --> F[Sounding Video]
```

> For the conceptual foundation of JointDiT and JointCFG/JointCFG*, see the referenced paper in this repository’s docs.