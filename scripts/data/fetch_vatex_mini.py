# scripts/data/fetch_vatex_mini.py
import json
import subprocess
from pathlib import Path
from typing import List, Optional

import requests
from datasets import load_dataset


# ---------- tiny HTTP helper ----------
def _parquet_urls(dataset: str, config: str, split: str) -> List[str]:
    """
    Use the dataset-viewer API to get the Parquet URLs published under refs/convert/parquet.
    Docs: https://huggingface.co/docs/dataset-viewer/en/parquet
    """
    api = f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"
    r = requests.get(api, timeout=30)
    r.raise_for_status()
    data = r.json()
    urls = [
        item["url"]
        for item in data.get("parquet_files", [])
        if item.get("dataset") == dataset
        and item.get("config") == config
        and item.get("split") == split
    ]
    if not urls:
        raise RuntimeError(f"No Parquet files found for {dataset} [{config}/{split}]")
    return urls


# ---------- shell helpers ----------
def run(cmd: List[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ytdlp(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "yt-dlp",
            "-f",
            "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "-o",
            str(out),
            url,
        ]
    )


def trim_ffmpeg(src: Path, out: Path, start: float, duration: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(src),
            "-t",
            str(max(0.5, duration)),
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out),
        ]
    )


def first_frame_png(src: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(src), "-vf", "select=eq(n\\,0)", "-vframes", "1", str(out)])


# ---------- VATEX helpers ----------
def _url_from_example(ex: dict) -> Optional[str]:
    # Prefer explicit URL fields; VATEX provides `path` with YouTube URL.
    for k in ("url", "video_url", "path"):
        if ex.get(k):
            return ex[k]
    vid = ex.get("videoID") or ex.get("videoId") or ex.get("id")
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return None


def _captions_from_example(ex: dict) -> List[str]:
    for k in ("enCap", "caption_en", "captions", "caption"):
        v = ex.get(k)
        if isinstance(v, list) and v:
            return [str(s) for s in v]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    return []


def _safe_stem(ex: dict) -> str:
    vid = str(ex.get("videoID") or ex.get("videoId") or ex.get("id") or "x")
    s = int(ex.get("start", 0))
    e = int(ex.get("end", s + 10))
    return f"vatex_{vid}_{s:04d}_{e:04d}"


def _write_meta(
    split: str,
    stem: str,
    raw_mp4: Path,
    first_png: Path,
    sr: int,
    fps_used: float,
    captions: List[str],
):
    meta_dir = Path("data/cache/meta") / split
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stem": stem,
        "video_file": str(raw_mp4.resolve()),
        "img_firstframe": str(first_png.resolve()),
        "clip_file": None,
        "video_latents": f"data/cache/video_latents/{split}/{stem}.pt",
        "audio_latents": f"data/cache/audio_latents/{split}/{stem}.pt",
        "frame_count": None,
        "src_fps": None,
        "fps_used": float(fps_used),
        "sr": int(sr),
        "captions": captions,
    }
    (meta_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))


def _process_split(split: str, ds, n: int, sr: int, fps_used: float):
    ok = 0
    for ex in ds:
        if ok >= n:
            break
        url = _url_from_example(ex)
        if not url:
            continue
        start = float(ex.get("start", 0))
        end = float(ex.get("end", start + 10))
        duration = max(0.5, end - start)

        stem = _safe_stem(ex)
        raw_base = Path("data/raw") / split
        png_base = Path("data/cache/img_firstframe") / split

        src_tmp = raw_base / f"{stem}.source.mp4"
        dst_mp4 = raw_base / f"{stem}.mp4"
        png = png_base / f"{stem}.png"

        try:
            print(f"[{split}] fetching {stem}")
            ytdlp(url, src_tmp)
            trim_ffmpeg(src_tmp, dst_mp4, start, duration)
            first_frame_png(dst_mp4, png)
            _write_meta(
                split,
                stem,
                dst_mp4,
                png,
                sr=sr,
                fps_used=fps_used,
                captions=_captions_from_example(ex),
            )
            ok += 1
        except Exception as e:
            print(f"[{split}] skip {stem}: {e}")
        finally:
            if src_tmp.exists():
                try:
                    src_tmp.unlink()
                except Exception:
                    pass
    print(f"[{split}] collected {ok}/{n}")


def main():
    # tiny subset: 8 train, 2 val
    train_n, val_n = 8, 2
    sr = 16000
    fps_used = 12.0

    dataset = "HuggingFaceM4/vatex"
    config = "v1.1"  # see dataset_infos.json for available configs

    print("[load] VATEX via viewer Parquet URLs")
    train_urls = _parquet_urls(dataset, config, "train")
    val_urls = _parquet_urls(dataset, config, "validation")

    # Build a DatasetDict in one go (faster metadata step), then pull splits
    ds = load_dataset("parquet", data_files={"train": train_urls, "val": val_urls})
    train = ds["train"]
    val = ds["val"]

    _process_split("train", train, train_n, sr, fps_used)
    _process_split("val", val, val_n, sr, fps_used)


if __name__ == "__main__":
    main()
