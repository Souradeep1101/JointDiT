# scripts/tools/export_requirements.py
import argparse
import ast
import importlib.metadata as md
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name as canon
except Exception:
    Requirement = None
    # canon = lambda x: x.replace("_", "-").lower()

    def canon(x: str) -> str:
        return x.replace("_", "-").lower()


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if any(
            s in p.parts
            for s in (
                ".venv",
                "venv",
                ".git",
                "__pycache__",
                ".ipynb_checkpoints",
                "data",
                "checkpoints",
                ".mypy_cache",
                ".pytest_cache",
            )
        ):
            continue
        yield p


def iter_ipynb_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.ipynb"):
        if any(
            s in p.parts
            for s in (".venv", "venv", ".git", ".ipynb_checkpoints", "data", "checkpoints")
        ):
            continue
        yield p


def imports_from_py(path: Path) -> Set[str]:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except Exception:
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name and not a.name.startswith("."):
                    names.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module and not n.level:
                names.add(n.module.split(".")[0])
    return names


def imports_from_ipynb(path: Path) -> Set[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return set()
    out = set()
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name and not a.name.startswith("."):
                        out.add(a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module and not n.level:
                    out.add(n.module.split(".")[0])
    return out


def stdlib_names() -> Set[str]:
    s = set(getattr(sys, "stdlib_module_names", set()))
    s.update({"typing_extensions"})
    return s


def map_modules_to_dists(mod_names: Set[str]) -> Dict[str, str]:
    mapping = md.packages_distributions()
    out = {}
    for m in sorted(mod_names):
        dists = mapping.get(m, [])
        if dists:
            out[m] = dists[0]
    return out


def direct_distributions(root: Path) -> Set[str]:
    mods = set()
    for p in iter_py_files(root):
        mods |= imports_from_py(p)
    for p in iter_ipynb_files(root):
        mods |= imports_from_ipynb(p)
    mods -= stdlib_names()
    mods = {m for m in mods if m not in {"__future__", "builtins"}}
    mod2dist = map_modules_to_dists(mods)
    return {mod2dist[m] for m in mod2dist}


def deps_closure(dists: Set[str]) -> Set[str]:
    seen = set()
    queue = [canon(d) for d in dists]
    while queue:
        d = queue.pop()
        if d in seen:
            continue
        seen.add(d)
        try:
            reqs = md.requires(d) or []
        except md.PackageNotFoundError:
            continue
        for r in reqs:
            name = r.split(";", 1)[0].strip()
            if "[" in name:
                name = name.split("[", 1)[0].strip()
            if " " in name:
                name = name.split(" ", 1)[0].strip()
            if name:
                queue.append(canon(name))
    return seen


def pin_versions(dists: Set[str]) -> List[str]:
    pins = []
    skip = {"pip", "setuptools", "wheel"}
    for d in sorted(dists):
        if d in skip:
            continue
        try:
            v = md.version(d)
            pins.append(f"{d}=={v}")
        except md.PackageNotFoundError:
            continue
    return pins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="requirements.txt")
    ap.add_argument("--include-transitive", action="store_true", default=True)
    ap.add_argument("--no-transitive", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    directs = direct_distributions(root)
    if args.no_transitive:
        all_dists = {canon(d) for d in directs}
    else:
        all_dists = deps_closure(directs) | {canon(d) for d in directs}
    lines = pin_versions(all_dists)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} packages to {args.out}")


if __name__ == "__main__":
    main()
