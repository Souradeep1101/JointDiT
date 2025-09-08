
# Datasets & Caching

## AVSync15 (1s subset)
Use `scripts/data/fetch_avsync15_1s.py` to download/trim:
- Places clips under `data/raw/{train,val}` and emits meta JSONs under `data/cache/meta/{split}`.

## Caching Latents
Encode video/audio latents + (optional) CLIP image embeddings:

```bash
# Video/audio VAEs are read from configs/day02_cache.yaml
make cache-train
make cache-val
````

Output tree:

```
data/cache/
  video_latents/{split}/*.pt
  audio_latents/{split}/*.pt
  img_firstframe/{split}/*.png
  img_clip/{split}/*.pt   # if clip.enabled=true
  meta/{split}/*.json
```

Each `meta` JSON includes:

* `video_latents`, `audio_latents`
* `img_firstframe` for optional image conditioning
* timing fields (`frame_count`, `src_fps`, `fps_used`, `sr`)