from pathlib import Path

import yaml


def _maybe(path):
    p = Path(path)
    return p if p.exists() else None


def test_configs_load():
    # Load what exists; skip missing
    candidates = [
        "configs/day02_cache.yaml",
        "configs/day05_train.yaml",
        "configs/day06_infer.yaml",
        "configs/day07_trainB.yaml",
        "configs/ui_infer.yaml",
    ]
    found = [p for p in map(_maybe, candidates) if p]
    assert found, "No config files found to test."
    for p in found:
        with open(p, "r") as f:
            y = yaml.safe_load(f)
            assert isinstance(y, dict), f"{p} did not parse to a dict"
