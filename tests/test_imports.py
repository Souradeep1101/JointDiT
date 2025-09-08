import importlib

import pytest


def test_imports():
    __import__("models.jointdit")
    __import__("models.joint.joint_block")
    __import__("models.joint.perceiver_joint_attn")
    __import__("scripts.infer.infer_joint")


def test_import_core_models():
    for mod in [
        "models.jointdit",
        "models.joint.joint_block",
        "models.joint.perceiver_joint_attn",
    ]:
        assert importlib.import_module(mod) is not None


def test_import_infer_script_optional():
    """
    Keep previous behavior (import infer script), but don't hard-fail locally
    if optional deps (e.g., open_clip) aren't installed.
    CI installs open-clip-torch so this will pass there.
    """
    try:
        importlib.import_module("scripts.infer.infer_joint")
    except Exception as e:
        pytest.skip(f"optional import skipped: {e}")
