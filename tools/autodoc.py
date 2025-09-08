import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NB_DST = DOCS / "notebooks"
API_MD = DOCS / "api.md"
INDEX_MD = DOCS / "index.md"
USAGE_MD = DOCS / "usage.md"
ARCH_MD = DOCS / "architecture.md"
TL_MD = DOCS / "timeline.md"
CL_MD = DOCS / "changelog.md"
EXCLUDE = {
    ".git",
    ".venv",
    "build",
    "dist",
    "checkpoints",
    "outputs",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}


def ensure_dirs():
    DOCS.mkdir(exist_ok=True)
    NB_DST.mkdir(parents=True, exist_ok=True)


def list_notebooks():
    n = []
    for p in ROOT.rglob("*.ipynb"):
        if any(x in p.parts for x in EXCLUDE):
            continue
        if DOCS in p.parents:
            continue
        n.append(p)
    return n


def copy_notebooks(nbs):
    for nb in nbs:
        dst = NB_DST / nb.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nb, dst)


def _fix_markdown_source(cell):
    src = cell.get("source", "")
    if isinstance(src, str):
        if "\\n" in src and "\n" not in src:
            cell["source"] = src.replace("\\n", "\n")
    elif isinstance(src, list):
        changed = False
        new = []
        for s in src:
            if isinstance(s, str) and "\\n" in s and "\n" not in s:
                s = s.replace("\\n", "\n")
                changed = True
            new.append(s)
        if changed:
            cell["source"] = new


def sanitize_docs_notebooks():
    changed_any = False
    for p in sorted(NB_DST.glob("*.ipynb")):
        nb = nbf.read(p, as_version=4)
        changed = False
        for cell in nb.cells:
            # add id (future nbformat requirement)
            if not cell.get("id"):
                cell["id"] = uuid.uuid4().hex
                changed = True
            # drop invalid execution metadata; normalize tags
            md = cell.get("metadata", {}) or {}
            if "execution" in md:
                md.pop("execution", None)
                changed = True
            if isinstance(md.get("tags"), list):
                md["tags"] = [str(t) for t in md["tags"]]
            cell["metadata"] = md
            # prettify markdown sources with literal \n
            if cell.get("cell_type") == "markdown":
                before = cell.get("source", "")
                _fix_markdown_source(cell)
                if cell.get("source", "") != before:
                    changed = True
        nb["nbformat"] = 4
        nb["nbformat_minor"] = max(nb.get("nbformat_minor", 5), 5)
        if changed:
            nbf.write(nb, p)
            print(f"[fixed] {p}")
            changed_any = True
    return changed_any


def detect_pkg():
    src = ROOT / "src"
    if src.exists():
        for p in src.rglob("__init__.py"):
            if any(x in p.parts for x in EXCLUDE):
                continue
            return str(p.parent).replace(str(ROOT) + "/", "").replace("/", ".")
    for p in ROOT.rglob("__init__.py"):
        if any(x in p.parts for x in EXCLUDE):
            continue
        if "site-packages" in p.parts:
            continue
        parent = p.parent
        if any(x in parent.parts for x in ["scripts", "docs", "notebooks"]):
            continue
        rel = str(parent.relative_to(ROOT))
        if rel:
            return rel.replace("/", ".")
    return "yourpkg"


def write_api_md(pkg):
    API_MD.write_text(
        f"# API Reference\n\n::: {pkg}\n    options:\n      heading_level: 2\n      members_order: source\n"
    )


def file_tree(root: Path):
    lines = []
    for p in sorted(root.rglob("*")):
        if any(x in p.parts for x in EXCLUDE):
            continue
        if p.is_dir():
            continue
        if DOCS in p.parents:
            continue
        lines.append(f"- {p.relative_to(root)}")
    return "\n".join(lines)


def write_architecture():
    ARCH_MD.write_text(
        "\n".join(
            [
                "# Architecture",
                "",
                "## Repository Tree",
                file_tree(ROOT),
                "",
                "## Modules",
                "High-level description of core modules and data flow.",
            ]
        )
    )


def nb_entries(nbs):
    rows = []
    for nb in nbs:
        ts = datetime.fromtimestamp(nb.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        rows.append((ts, nb.name))
    rows.sort()
    return rows


def write_timeline(nbs):
    lines = ["# Timeline", "", "Entries sorted by file modification time.", ""]
    for ts, name in nb_entries(nbs):
        lines.append(f"- {ts} — {name}")
    TL_MD.write_text("\n".join(lines))


def write_changelog():
    try:
        out = subprocess.check_output(
            ["git", "log", "--pretty=format:%ad %h %s", "--date=short"], cwd=ROOT
        ).decode()
        lines = ["# Changelog", "", *["- " + x for x in out.splitlines()]]
    except Exception:
        lines = ["# Changelog", "", "git history unavailable"]
    CL_MD.write_text("\n".join(lines))


def ensure_seed_pages():
    if not INDEX_MD.exists():
        INDEX_MD.write_text("# Project Overview\n\nEdit docs/index.md.")
    if not USAGE_MD.exists():
        USAGE_MD.write_text("# Getting Started\n\nAdd setup and run instructions.")


def main():
    ensure_dirs()
    ensure_seed_pages()
    nbs = list_notebooks()
    copy_notebooks(nbs)
    sanitize_docs_notebooks()
    write_architecture()
    write_timeline(nbs)
    write_changelog()
    if API_MD.exists():
        write_api_md(detect_pkg())


if __name__ == "__main__":
    main()
