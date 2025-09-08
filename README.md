
## Environment & Portability (No Docker)

You can run everything with a plain Python venv:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -c requirements.lock.txt
pre-commit install  # optional: formatting/linting on commit
````

> System deps: see `system_deps.txt` (e.g., `ffmpeg`, `libsndfile1`).
> CUDA/Torch: repo targets CUDA 12.1 wheels. Adjust if your platform uses a different CUDA.

**Quick checks**

```bash
python scripts/dev/check_env.py
make test
```

**Caching / Training / Inference**

```bash
make cache-train
make cache-val
make day7-smoke     # small Stage-B smoke
make infer CFG_INFER=configs/ui_infer.yaml PROMPT="a baby laughing" USE_IMG=1
```

**CLIP + Image Guidance**

* Enable in `configs/ui_infer.yaml`:

  ```yaml
  clip:
    enabled: true
    model: ViT-B-16
    pretrained: openai
    use_image_firstframe: true   # or set image_path manually
  ```
* Or override via env: `JOINTDIT_PROMPT`, `JOINTDIT_NEG_PROMPT`, `JOINTDIT_USE_IMG=1`.