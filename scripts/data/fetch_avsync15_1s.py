# scripts/data/fetch_avsync15_1s.py
import argparse
import json
import subprocess
import tarfile
from pathlib import Path
from typing import List, Optional

import gdown


def run(cmd: List[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ff_trim_1s(src: Path, out: Path, fps: float, sr: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-t",
            "1.0",
            "-vf",
            f"fps={fps}",
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
            "-ar",
            str(sr),
            str(out),
        ]
    )


def first_frame_png(src: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(src), "-vf", "select=eq(n\\,0)", "-vframes", "1", str(out)])


def write_meta(
    split: str,
    stem: str,
    raw_mp4: Path,
    first_png: Path,
    sr: int,
    fps_used: float,
    captions: List[str],
):
    d = Path("data/cache/meta") / split
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "stem": stem,
        "video_file": str(raw_mp4.resolve()),
        "img_firstframe": str(first_png.resolve()),
        "clip_file": None,
        "video_latents": f"data/cache/video_latents/{split}/{stem}.pt",
        "audio_latents": f"data/cache/audio_latents/{split}/{stem}.pt",
        # 1s clip -> ~fps_used frames; good enough for our tiny subset
        "frame_count": int(round(fps_used)),
        "src_fps": float(fps_used),
        "fps_used": float(fps_used),
        "sr": int(sr),
        "captions": captions,
    }
    (d / f"{stem}.json").write_text(json.dumps(meta, indent=2))


def load_list(p: Path) -> List[str]:
    lines = []
    for s in p.read_text().splitlines():
        s = s.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


def find_video(root_candidates: List[Path], rel: str) -> Optional[Path]:
    for rc in root_candidates:
        p = rc / rel
        if p.exists():
            return p
    name = Path(rel).name
    for rc in root_candidates:
        cand = list(rc.rglob(name))
        if cand:
            return cand[0]
    return None


def copy_and_index(
    split: str,
    rel_paths: List[str],
    root_candidates: List[Path],
    sr: int,
    fps_used: float,
    limit: Optional[int] = None,
):
    ok = 0
    raw_base = Path("data/raw") / split
    png_base = Path("data/cache/img_firstframe") / split
    raw_base.mkdir(parents=True, exist_ok=True)
    png_base.mkdir(parents=True, exist_ok=True)
    for rel in rel_paths:
        if limit is not None and ok >= limit:
            break
        src = find_video(root_candidates, rel)
        if not src:
            print(f"[{split}] missing {rel}")
            continue
        cls = rel.split("/")[0]
        stem = "avsync15_" + rel.replace("/", "_").replace(".mp4", "")
        dst_mp4 = raw_base / f"{stem}.mp4"
        png = png_base / f"{stem}.png"
        ff_trim_1s(src, dst_mp4, fps_used, sr)
        first_frame_png(dst_mp4, png)
        write_meta(split, stem, dst_mp4, png, sr, fps_used, [cls])
        ok += 1
    print(
        f"[{split}] collected {ok}/{len(rel_paths) if limit is None else min(len(rel_paths),limit)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder-url", required=True)
    ap.add_argument("--limit-train", type=int, default=5)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--workdir", default="data/downloads/avsync15_1s")
    args = ap.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        url=args.folder_url, output=str(work), remaining_ok=True, quiet=False, use_cookies=False
    )
    train_txt = next(work.rglob("train.txt"))
    val_txt = next(work.rglob("test.txt"))
    tar = next(work.rglob("videos.tar.gz"), None)
    extracted = work / "extracted"
    if tar:
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar, "r:gz") as tf:
            tf.extractall(extracted)
    roots = [extracted / "videos", work / "videos", extracted, work]
    train_list = load_list(train_txt)
    val_list = load_list(val_txt)
    copy_and_index("train", train_list, roots, args.sr, args.fps, args.limit_train)
    copy_and_index("val", val_list, roots, args.sr, args.fps, args.limit_val)


if __name__ == "__main__":
    main()
