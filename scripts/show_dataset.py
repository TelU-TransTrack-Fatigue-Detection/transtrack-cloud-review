import sys
from pathlib import Path
from collections import defaultdict

MAX_PER_EXT = 5


def show_tree(root: Path, prefix: str = ""):
    children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    dirs  = [c for c in children if c.is_dir()]
    files = [c for c in children if c.is_file()]

    by_ext = defaultdict(list)
    for f in files:
        by_ext[f.suffix.lower()].append(f)

    for i, d in enumerate(dirs):
        connector = "└── " if (i == len(dirs) - 1 and not by_ext) else "├── "
        print(f"{prefix}{connector}{d.name}/")
        extension = "    " if connector.startswith("└") else "│   "
        show_tree(d, prefix + extension)

    for ext, ext_files in sorted(by_ext.items()):
        total   = len(ext_files)
        shown   = ext_files[:MAX_PER_EXT]
        is_last_ext = ext == sorted(by_ext)[-1]

        for j, f in enumerate(shown):
            is_last = is_last_ext and j == len(shown) - 1 and total <= MAX_PER_EXT
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{f.name}")

        if total > MAX_PER_EXT:
            remaining = total - MAX_PER_EXT
            connector = "└── " if is_last_ext else "├── "
            print(f"{prefix}{connector}... and {remaining} more {ext} files  [{total} total]")


def summarize(root: Path):
    counts = defaultdict(int)
    for f in root.rglob("*"):
        if f.is_file():
            counts[f.suffix.lower()] += 1
    return counts


if __name__ == "__main__":
    targets = [
        Path("data-test-pama_mtbu/test"),
        Path("data-test-pama_mtbu/train"),
    ]

    for target in targets:
        if not target.exists():
            print(f"[NOT FOUND] {target}")
            continue

        print(f"\n{target}/")
        show_tree(target, prefix="")

        counts = summarize(target)
        print(f"\n  Summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        print()
