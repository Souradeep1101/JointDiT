# scripts/tools/export_system_deps.py
import argparse
import importlib.util as iu
import json
import os
import re
from pathlib import Path

BIN_TOKENS = {
    "ffmpeg",
    "git-lfs",
    "aria2c",
    "sox",
    "mediainfo",
    "wget",
    "curl",
    "unzip",
    "tar",
    "7z",
    "jq",
}
BIN2APT = {
    "ffmpeg": "ffmpeg",
    "git-lfs": "git-lfs",
    "aria2c": "aria2",
    "sox": "sox",
    "mediainfo": "mediainfo",
    "wget": "wget",
    "curl": "curl",
    "unzip": "unzip",
    "tar": "tar",
    "7z": "p7zip-full",
    "jq": "jq",
}
MOD2APT = {
    "soundfile": ["libsndfile1"],
    "librosa": ["libsndfile1"],
    "cv2": ["libgl1", "libglib2.0-0"],
    "torchaudio": ["ffmpeg"],
    "av": ["ffmpeg"],
    "open3d": ["libgl1", "libglib2.0-0"],
}


def iter_text_files(root):
    for p in root.rglob("*"):
        if p.is_file() and not any(
            s in p.parts
            for s in (
                ".venv",
                "venv",
                ".git",
                "node_modules",
                "data",
                "checkpoints",
                ".mypy_cache",
                ".pytest_cache",
                "__pycache__",
                ".ipynb_checkpoints",
            )
        ):
            if p.suffix in {
                ".py",
                ".sh",
                ".bash",
                ".txt",
                ".md",
                ".yaml",
                ".yml",
                ".json",
            } or p.name.endswith(".ipynb"):
                yield p


def scan_bins(root):
    found = set()
    for p in iter_text_files(root):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for b in BIN_TOKENS:
            if b in t:
                found.add(b)
    return found


def scan_imports_py(text):
    mods = set()
    for m in re.findall(r"^\s*import\s+([A-Za-z0-9_]+)", text, flags=re.M):
        mods.add(m.split(".")[0])
    for m in re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+", text, flags=re.M):
        mods.add(m.split(".")[0])
    return mods


def scan_imports_nb(text):
    try:
        nb = json.loads(text)
    except Exception:
        return set()
    mods = set()
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        mods |= scan_imports_py(src)
    return mods


def scan_modules(root):
    mods = set()
    for p in root.rglob("*.py"):
        if any(
            s in p.parts
            for s in (
                ".venv",
                "venv",
                ".git",
                "node_modules",
                "data",
                "checkpoints",
                ".mypy_cache",
                ".pytest_cache",
                "__pycache__",
                ".ipynb_checkpoints",
            )
        ):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        mods |= scan_imports_py(t)
    for p in root.rglob("*.ipynb"):
        if any(
            s in p.parts
            for s in (
                ".venv",
                "venv",
                ".git",
                "node_modules",
                "data",
                "checkpoints",
                ".ipynb_checkpoints",
            )
        ):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        mods |= scan_imports_nb(t)
    return mods


def mod_present(name):
    try:
        return iu.find_spec(name) is not None
    except Exception:
        return False


def infer_apt_from_modules(mods):
    needed = set()
    for k, v in MOD2APT.items():
        if k in mods or mod_present(k):
            needed.update(v)
    return needed


def infer_apt_from_bins(bins):
    needed = set()
    for b in bins:
        pkg = BIN2APT.get(b)
        if pkg:
            needed.add(pkg)
    return needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="system_deps.txt")
    ap.add_argument("--install-script", default="scripts/tools/install_system_deps.sh")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--no-detect-bins", action="store_true")
    ap.add_argument("--no-detect-mods", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    bins = set()
    mods = set()
    if not args.no_detect_bins:
        bins = scan_bins(root)
    if not args.no_detect_mods:
        mods = scan_modules(root)
    pkgs = set(args.extra)
    pkgs |= infer_apt_from_bins(bins)
    pkgs |= infer_apt_from_modules(mods)
    pkgs_sorted = sorted(pkgs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        "\n".join(pkgs_sorted) + ("\n" if pkgs_sorted else ""), encoding="utf-8"
    )
    sh = Path(args.install_script)
    sh.parent.mkdir(parents=True, exist_ok=True)
    sudo_prefix = "sudo " if os.geteuid() != 0 else ""
    if pkgs_sorted:
        sh.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nif command -v apt-get >/dev/null 2>&1; then\n"
            + sudo_prefix
            + "apt-get update\n"
            + sudo_prefix
            + "apt-get install -y "
            + " ".join(pkgs_sorted)
            + "\nelif command -v yum >/dev/null 2>&1; then\n"
            + sudo_prefix
            + "yum install -y "
            + " ".join(pkgs_sorted)
            + '\nelse\necho "unsupported package manager"; exit 1\nfi\n',
            encoding="utf-8",
        )
    else:
        sh.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\necho "no system deps detected"\n',
            encoding="utf-8",
        )
    os.chmod(sh, 0o755)
    print(f"wrote {len(pkgs_sorted)} entries to {args.out}")
    print(f"wrote installer to {sh}")


if __name__ == "__main__":
    main()
