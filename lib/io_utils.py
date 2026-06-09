"""Shared JSON / path / hashing helpers for the landmark dataset tools.

These helpers are intentionally dependency-light: ``jsonable`` handles NumPy
values without importing NumPy at module load, so CLIs that never touch NumPy
keep their fast startup.
"""

from __future__ import annotations

import hashlib
import json
import typing as T
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def jsonable(value: T.Any) -> T.Any:
    """Recursively convert NumPy/Path values into JSON-serializable Python types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    # Convert NumPy arrays/scalars without importing NumPy here. ``tolist`` covers
    # ndarray and NumPy scalar types.
    if type(value).__module__ == "numpy":
        to_list = getattr(value, "tolist", None)
        if to_list is not None:
            return to_list()
    return value


def read_json(path: Path) -> T.Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: T.Any, *, sort_keys: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def safe_id(value: T.Any) -> str:
    text = str(value or "sample").strip().replace("\\", "/").strip("/") or "sample"
    return "".join(ch if ch.isalnum() or ch in "._-/#" else "_" for ch in text)


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return hash_file(path, "sha256")


def sha1_file(path: Path) -> str:
    return hash_file(path, "sha1")
