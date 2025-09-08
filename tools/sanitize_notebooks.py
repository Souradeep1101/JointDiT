from pathlib import Path

import nbformat as nbf


def sanitize(nb):
    changed = False
    for cell in nb.cells:
        md = cell.get("metadata", {})
        if isinstance(md, dict):
            # This is the field causing your crash; just drop it.
            if "execution" in md:
                md.pop("execution", None)
                changed = True
            # Be safe: ensure tags are strings
            if isinstance(md.get("tags"), list):
                md["tags"] = [str(t) for t in md["tags"] if isinstance(t, (str, int, float))]
        cell["metadata"] = md
    # Pin a sane nbformat level
    nb["nbformat"] = 4
    nb["nbformat_minor"] = max(nb.get("nbformat_minor", 5), 5)
    return changed


def main():
    root = Path("docs/notebooks")
    for p in sorted(root.glob("*.ipynb")):
        nb = nbf.read(p, as_version=4)
        if sanitize(nb):
            nbf.write(nb, p)
            print(f"[fixed] {p}")


if __name__ == "__main__":
    main()
