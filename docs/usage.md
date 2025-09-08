# Getting Started

## Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt         # your project runtime deps
pip install -r requirements-docs.txt    # docs deps
```

## Smoke / Training / Inference (Makefile)
Common targets:
```bash
make smoke
make smoke-day04
make day7-trainB
make day7-infer CKPT=... REF=...
make infer CKPT=... REF=... PROMPT="..." STEPS=40
```

## Docs
```bash
make docs-install
make docs-serve DOCS_HOST=0.0.0.0 DOCS_PORT=8000
# or
make docs-build
make docs-deploy
```