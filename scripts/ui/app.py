#!/usr/bin/env python
# scripts/ui/app.py
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Optional, Tuple

# ---------- JSON-SCHEMA SAFETY PATCH ----------
try:
    import gradio_client.utils as _gcu

    _orig__json = getattr(_gcu, "_json_schema_to_python_type", None)
    _orig_top = getattr(_gcu, "json_schema_to_python_type", None)

    def _safe__json(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        try:
            return _orig__json(schema, defs)
        except Exception as e:
            print(f"[ui][warn] schema conversion fallback -> Any ({e})")
            return "Any"

    def _safe_top(schema):
        if not isinstance(schema, dict):
            return "Any"
        try:
            return _safe__json(schema, schema.get("$defs"))
        except Exception as e:
            print(f"[ui][warn] top-level schema fallback -> Any ({e})")
            return "Any"

    if callable(_orig__json):
        _gcu._json_schema_to_python_type = _safe__json
    if callable(_orig_top):
        _gcu.json_schema_to_python_type = _safe_top
    print("[ui] Applied JSON-schema safety patch.")
except Exception as _e:
    print(f"[ui][warn] could not apply schema patch: {_e}")
# ------------------------------------------------

import gradio as gr
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DEFAULT_CFG = "configs/ui_infer.yaml"
DEFAULT_FPS = 12
DEFAULT_STEPS = 30
DEFAULT_WV = 1.5
DEFAULT_WA = 1.5
DEFAULT_WT = 1.5
DEFAULT_WNT = 0.0
DEFAULT_WI = 0.0

VRAM_PROFILES = {
    "Balanced (~48GB)": {
        "JOINTDIT_MAX_T": "12",
        "JOINTDIT_Q_CHUNK_V": "128",
        "JOINTDIT_Q_CHUNK_A": "0",
        "JOINTDIT_KV_DOWNSAMPLE": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    },
    "Conservative (~44GB)": {
        "JOINTDIT_MAX_T": "6",
        "JOINTDIT_Q_CHUNK_V": "64",
        "JOINTDIT_Q_CHUNK_A": "0",
        "JOINTDIT_KV_DOWNSAMPLE": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    },
    "Tiny GPU (~24–32GB)": {
        "JOINTDIT_MAX_T": "4",
        "JOINTDIT_Q_CHUNK_V": "32",
        "JOINTDIT_Q_CHUNK_A": "0",
        "JOINTDIT_KV_DOWNSAMPLE": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    },
}


def _apply_vram_profile(name: str):
    prof = VRAM_PROFILES.get(name, {})
    for k, v in prof.items():
        os.environ[k] = str(v)
    return prof


def _none_if_blank(s: str | None) -> Optional[str]:
    s = (s or "").strip()
    return s if s else None


def _prepare_temp_cfg(
    base_cfg_path: str,
    ckpt_path: str,
    ref_meta_path: str,
    steps: int,
    seed: int,
    fps: int,
    wv: float,
    wa: float,
    wt: float,
    wnt: float,
    wi: float,
    prompt_txt: str,
    neg_txt: str,
    use_meta_image: bool,
    image_path: Optional[str],
    # manual shapes (used when no meta path provided)
    man_enable: bool,
    man_T: int,
    man_vc: int,
    man_vh: int,
    man_vw: int,
    man_ac: int,
    man_ah: int,
    man_aw: int,
    man_fps: int,
    man_sr: int,
) -> Path:
    base = yaml.safe_load(Path(base_cfg_path).read_text())

    # Required basics
    base["ckpt"] = ckpt_path
    base["steps"] = int(steps)
    base["seeds"] = [int(seed)]
    base["mp4_fps"] = int(fps)

    # Outputs bucket
    out_dir = base.get("out_dir", base.get("out", {}).get("dir", "outputs/ui"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base["out_dir"] = str(out_dir)

    # Either meta path or manual override
    if _none_if_blank(ref_meta_path):
        base["ref_meta"] = ref_meta_path
        base.pop("manual_shapes", None)
    else:
        base.pop("ref_meta", None)
        base["manual_shapes"] = {
            "T": int(man_T),
            "video": {"c": int(man_vc), "h": int(man_vh), "w": int(man_vw)},
            "audio": {"c": int(man_ac), "h": int(man_ah), "w": int(man_aw)},
            "fps": int(man_fps),
            "sr": int(man_sr),
        }

    # Guidance
    g = base.setdefault("guidance", {})
    g["weight_v"] = float(wv)
    g["weight_a"] = float(wa)
    g["weight_txt"] = float(wt)
    g["weight_neg_txt"] = float(wnt)
    g["weight_img"] = float(wi)

    # Prompts / CLIP
    base["prompt"] = prompt_txt or ""
    base["negative_prompt"] = neg_txt or ""
    clip = base.setdefault("clip", {})
    clip["enabled"] = bool(prompt_txt or neg_txt or image_path or use_meta_image)
    clip["use_image_firstframe"] = bool(use_meta_image)
    if image_path:
        clip["image_path"] = image_path  # overrides meta first-frame

    tmp = Path(tempfile.mkdtemp(prefix="jointdit_ui_"))
    tmp_yaml = tmp / "ui_run.yaml"
    tmp_yaml.write_text(yaml.safe_dump(base))
    return tmp_yaml


def _run_infer(tmp_yaml: Path) -> Tuple[Optional[str], Optional[str], str]:
    cmd = [sys.executable, "-u", "scripts/infer/infer_joint.py", "--cfg", str(tmp_yaml)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True
    )
    logs = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            logs.append(line.rstrip())
            print(line, end="")
    rc = proc.wait()
    log_text = "\n".join(logs)
    if rc != 0:
        return None, None, f"[infer] exit code {rc}\n{log_text}"

    y = yaml.safe_load(Path(tmp_yaml).read_text())
    out_dir = Path(y["out_dir"])
    mp4 = max(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, default=None)
    wav = max(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, default=None)
    return (str(mp4) if mp4 else None), (str(wav) if wav else None), log_text


def run(
    cfg_path,
    vram_profile,
    ckpt_path,
    ref_meta_path,
    steps,
    seed,
    fps,
    wv,
    wa,
    wt,
    wnt,
    wi,
    prompt_txt,
    neg_txt,
    use_meta_image,
    image_file,
    man_enable,
    man_T,
    man_vc,
    man_vh,
    man_vw,
    man_ac,
    man_ah,
    man_aw,
    man_fps,
    man_sr,
):
    try:
        prof = _apply_vram_profile(vram_profile)
        env_info = "\n".join([f"{k}={v}" for k, v in prof.items()])

        image_path = image_file.name if image_file else None  # gr.File returns obj with .name

        tmp_yaml = _prepare_temp_cfg(
            cfg_path,
            ckpt_path,
            ref_meta_path,
            steps,
            seed,
            fps,
            wv,
            wa,
            wt,
            wnt,
            wi,
            prompt_txt,
            neg_txt,
            bool(use_meta_image),
            image_path,
            bool(man_enable),
            man_T,
            man_vc,
            man_vh,
            man_vw,
            man_ac,
            man_ah,
            man_aw,
            man_fps,
            man_sr,
        )
        mp4, wav, log_text = _run_infer(tmp_yaml)
        header = textwrap.dedent(
            f"""
        [UI] VRAM profile: {vram_profile}
        {env_info}
        [UI] Using tmp config: {tmp_yaml}
        """
        ).strip()
        logs = header + "\n" + (log_text or "")
        return mp4, wav, logs
    except Exception:
        return None, None, "[ui][error]\n" + traceback.format_exc()


# -------------------------------- UI ----------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"), analytics_enabled=False
) as demo:
    gr.Markdown(
        "## JointDiT Inference UI\n"
        "Generate **Image → Sounding-Video** with optional **Text** + **Negative text** prompts.\n"
        "Use a cached *meta JSON* OR enter shapes manually."
    )

    with gr.Row():
        cfg_path = gr.Textbox(label="Config path", value=DEFAULT_CFG, interactive=True)
        vram_prof = gr.Dropdown(
            choices=list(VRAM_PROFILES.keys()),
            value="Balanced (~48GB)",
            label="VRAM profile",
            interactive=True,
        )

    with gr.Row():
        ckpt_tb = gr.Textbox(
            label="Checkpoint (.pt)",
            placeholder="checkpoints/.../ckpt_step_xxxxx.pt",
            interactive=True,
        )
        ref_tb = gr.Textbox(
            label="Ref meta (JSON from cache)",
            placeholder="data/cache/meta/val/clip_....json",
            interactive=True,
        )

    with gr.Row():
        steps_sl = gr.Slider(1, 200, value=DEFAULT_STEPS, step=1, label="Steps")
        seed_sl = gr.Slider(0, 999999, value=0, step=1, label="Seed")
        fps_sl = gr.Slider(1, 30, value=DEFAULT_FPS, step=1, label="MP4 FPS")

    # Prompts & image
    with gr.Row():
        prompt_tb = gr.Textbox(
            label="Prompt (CLIP text)", placeholder="e.g., 'baby babbling, crying'"
        )
        neg_tb = gr.Textbox(
            label="Negative prompt (text)", placeholder="e.g., 'people, speech, sirens'"
        )
    with gr.Row():
        use_meta_img = gr.Checkbox(value=True, label="Use first-frame image from meta")
        image_up = gr.File(label="Override image (optional)", file_count="single", type="filepath")

    # Guidance weights
    with gr.Row():
        wv_sl = gr.Slider(0.0, 3.0, value=DEFAULT_WV, step=0.1, label="Guidance weight (video)")
        wa_sl = gr.Slider(0.0, 3.0, value=DEFAULT_WA, step=0.1, label="Guidance weight (audio)")
        wt_sl = gr.Slider(0.0, 3.0, value=DEFAULT_WT, step=0.1, label="Guidance weight (text)")
        wnt_sl = gr.Slider(
            0.0, 3.0, value=DEFAULT_WNT, step=0.1, label="Guidance weight (neg text)"
        )
        wi_sl = gr.Slider(0.0, 3.0, value=DEFAULT_WI, step=0.1, label="Guidance weight (image)")

    # Manual shapes panel
    with gr.Accordion("Advanced: run WITHOUT meta.json (enter shapes manually)", open=False):
        man_enable = gr.Checkbox(label="Enable manual shapes (ignore Ref meta)", value=False)
        with gr.Row():
            man_T = gr.Number(value=12, label="T (frames)", precision=0)
            man_fps = gr.Number(value=12, label="FPS", precision=0)
            man_sr = gr.Number(value=16000, label="Audio SR", precision=0)
        with gr.Row():
            man_vc = gr.Number(value=4, label="Video latent C", precision=0)
            man_vh = gr.Number(value=32, label="Video latent H", precision=0)
            man_vw = gr.Number(value=32, label="Video latent W", precision=0)
        with gr.Row():
            man_ac = gr.Number(value=8, label="Audio latent C", precision=0)
            man_ah = gr.Number(value=16, label="Audio latent H", precision=0)
            man_aw = gr.Number(value=32, label="Audio latent W", precision=0)

    run_btn = gr.Button("Run", variant="primary")

    with gr.Row():
        video_out = gr.Video(label="Video (MP4)", interactive=False)
        audio_out = gr.Audio(label="Audio (WAV)", interactive=False)

    logs_tb = gr.Textbox(label="Logs", lines=16, interactive=False)

    run_btn.click(
        fn=run,
        inputs=[
            cfg_path,
            vram_prof,
            ckpt_tb,
            ref_tb,
            steps_sl,
            seed_sl,
            fps_sl,
            wv_sl,
            wa_sl,
            wt_sl,
            wnt_sl,
            wi_sl,
            prompt_tb,
            neg_tb,
            use_meta_img,
            image_up,
            man_enable,
            man_T,
            man_vc,
            man_vh,
            man_vw,
            man_ac,
            man_ah,
            man_aw,
            man_fps,
            man_sr,
        ],
        outputs=[video_out, audio_out, logs_tb],
        api_name="predict",
    )

if __name__ == "__main__":
    # Keep API open; hide docs UI.
    demo.queue(api_open=True).launch(
        server_name="0.0.0.0", server_port=7860, share=True, show_api=False
    )
