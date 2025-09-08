import importlib
import json
import os
import shutil

info = {}


def _try(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"ERR: {e}"


# Python + CUDA stack
import torch

info["python"] = os.sys.version.split()[0]
info["torch"] = _try("torch")
info["torchvision"] = _try("torchvision")
info["torchaudio"] = _try("torchaudio")
info["cuda_available"] = torch.cuda.is_available()
info["cuda_device_count"] = torch.cuda.device_count()
info["cuda_version"] = torch.version.cuda
info["nvcc_in_path"] = bool(shutil.which("nvcc"))

# Generative stack
info["diffusers"] = _try("diffusers")
info["transformers"] = _try("transformers")
info["open_clip_torch"] = _try("open_clip")
info["gradio"] = _try("gradio")

# System tools used at runtime
info["ffmpeg_in_path"] = bool(shutil.which("ffmpeg"))
info["ffprobe_in_path"] = bool(shutil.which("ffprobe"))

print(json.dumps(info, indent=2))
