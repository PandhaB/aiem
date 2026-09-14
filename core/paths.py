from __future__ import annotations

from pathlib import Path

from core.coco import IMAGE_SUFFIXES


def resolve_under(root: Path, relative: str) -> Path:
    """Resolve a user-supplied relative path so it cannot escape ``root``."""
    relative_path = Path(relative.strip() or ".")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Invalid path.")
    source = (root / relative_path).resolve()
    if not str(source).startswith(str(root.resolve())):
        raise ValueError("Invalid path.")
    return source


def list_image_folders(root: Path, *, max_depth: int = 3) -> list[dict[str, str]]:
    """Return relative folders under ``root`` that contain at least one image."""
    if not root.is_dir():
        return []
    root = root.resolve()
    found: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) > max_depth:
            continue
        if not any(child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir()):
            continue
        found.append({"relative_path": relative.as_posix(), "name": relative.as_posix()})
    return found
