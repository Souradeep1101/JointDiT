# Troubleshooting

## CLIP
- Prefer `pretrained: openai` for offline-friendly weights.
- If you see device mismatch, ensure both tokenizer/tensors and model live on the same device (we handle this in the encoder).

## Numpy / Gradio
- Some Gradio versions require `numpy<2`. If you hit resolver conflicts, pin `numpy==1.26.*` and `altair<6`.

## Audio decode
- Needs `ffmpeg` + `libsndfile`. On Ubuntu: `apt-get install -y ffmpeg libsndfile1`.
- Griffin-Lim inversion is loudness-normalized; tweak with `JOINTDIT_AUDIO_GAIN_DB`.

## Torch / CUDA
- Match your CUDA driver with the chosen torch wheels. Example (CUDA 12.1 wheels used here).