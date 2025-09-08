# Model Assets

This folder is intentionally empty in git. To populate:

- SVD VAE: downloads to `assets/models/svd/vae`
- AudioLDM2 VAE: downloads to `assets/models/audioldm2/vae`
- (Optional) CLIP checkpoints if you want offline use

Run:
    make deps-media   # installs ffmpeg + gdown (if needed)
    make fetch-models # or your existing model-fetch script/notes

You can also override paths via env:
    export JOINTDIT_ASSETS=/path/to/models
    # then set configs to use $JOINTDIT_ASSETS/...
