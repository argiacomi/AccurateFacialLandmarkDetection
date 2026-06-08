#!/usr/bin/env python3
"""Source resolution helpers for landmark quality datasets."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import typing as T
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path(".fs_cache/landmark_quality")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
EXTRACTED_DIR_NAME = "extracted"
EXTRACTION_MARKER = ".source.json"
ANNOTATION_SUFFIXES = (".txt", ".pts", ".json", ".mat", ".npy", ".npz")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

WFLW_ANNOTATIONS_URL = "https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz"
WFLW_IMAGES_GOOGLE_DRIVE_FILE_ID = "1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC"
cofw68_COLOR_URL = "http://www.vision.caltech.edu/xpburgos/ICCV13/Data/cofw68_color.zip"
MERL_RAV_LABELS_URL = "https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip"
AFLW2000_3D_URL = "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip"
AFLW2000_3D_SHA256 = "252bc35274d65ff27b6e573aa96c2f4c116ad88452cc984fb882258c0ed6e2d8"

DEFAULT_DOWNLOAD_SOURCES: dict[str, dict[str, str | None]] = {
    "AFLW2000-3D": {
        "url": AFLW2000_3D_URL,
        "archive_name": "AFLW2000-3D.zip",
        "sha256": AFLW2000_3D_SHA256,
    },
}

OFFICIAL_SOURCE_NOTES: dict[str, str] = {
    "WFLW": (
        "Official WFLW is distributed as separate image and annotation downloads. "
        "Place both extracted parts in the cache or pass --source-dir/--source-zip."
    ),
    "cofw68": (
        "Official cofw68 color images are available separately, but this builder expects "
        "a cofw68 JSON export or complete local source bundle."
    ),
    "MERL-RAV": (
        "MERL-RAV labels are public, but AFLW images must be requested separately. "
        "Pass --source-dir/--source-zip."
    ),
    "AFLW2000-3D": f"Official AFLW2000-3D archive: {AFLW2000_3D_URL}.",
}


@dataclass(frozen=True)
class DatasetSourceSpec:
    """Known source information for one landmark dataset."""

    dataset: str
    cache_subdir: str
    canonical_archive: str | None = None
    cache_aliases: tuple[str, ...] = ()
    extracted_aliases: tuple[str, ...] = ()
    url: str | None = None
    google_drive_file_id: str | None = None
    sha256: str | None = None
    manual_hint: str = ""

    @property
    def cache_root_name(self) -> str:
        return self.cache_subdir.strip("/") or self.dataset.lower()


@dataclass(frozen=True)
class MultiSourcePart:
    """One archive/source that contributes to a multipart dataset cache."""

    name: str
    archive_name: str
    url: str | None = None
    google_drive_file_id: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class MultiDatasetSourceSpec:
    """Known source information for datasets distributed as multiple archives."""

    dataset: str
    cache_subdir: str
    parts: tuple[MultiSourcePart, ...]
    manual_hint: str = ""

    @property
    def cache_root_name(self) -> str:
        return self.cache_subdir.strip("/") or self.dataset.lower()


WFLW_OFFICIAL_SOURCE = MultiDatasetSourceSpec(
    dataset="WFLW",
    cache_subdir="wflw",
    parts=(
        MultiSourcePart(
            name="annotations",
            archive_name="WFLW_annotations.tar.gz",
            url=WFLW_ANNOTATIONS_URL,
        ),
        MultiSourcePart(
            name="images",
            archive_name="WFLW_images.tar.gz",
            google_drive_file_id=WFLW_IMAGES_GOOGLE_DRIVE_FILE_ID,
        ),
    ),
    manual_hint="Official WFLW requires both annotation and image downloads.",
)


def _progress_enabled() -> bool:
    return logger.isEnabledFor(logging.INFO)


def _default_source_value(spec: DatasetSourceSpec, key: str) -> str | None:
    value = DEFAULT_DOWNLOAD_SOURCES.get(spec.dataset, {}).get(key)
    return str(value) if value else None


def _effective_url(spec: DatasetSourceSpec, download_url: str | None) -> str | None:
    return download_url or spec.url or _default_source_value(spec, "url")


def _effective_sha256(spec: DatasetSourceSpec) -> str | None:
    return spec.sha256 or _default_source_value(spec, "sha256")


def _archive_names(spec: DatasetSourceSpec) -> tuple[str, ...]:
    names: list[str] = []
    default_archive = _default_source_value(spec, "archive_name")
    for name in (default_archive, spec.canonical_archive, *spec.cache_aliases):
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _archive_name(spec: DatasetSourceSpec) -> str:
    names = _archive_names(spec)
    archive_names = [name for name in names if any(name.lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES)]
    if archive_names:
        return archive_names[0]
    return names[0] if names else f"{spec.dataset.lower()}.zip"


def _manual_hint(spec: DatasetSourceSpec | MultiDatasetSourceSpec) -> str:
    parts = [item for item in (spec.manual_hint, OFFICIAL_SOURCE_NOTES.get(spec.dataset, "")) if item]
    return " " + " ".join(parts) if parts else ""


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_sha256(path: Path, expected_sha256: str | None, *, label: str = "archive") -> None:
    if expected_sha256 is None:
        return
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"{label} checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}")


def _download_with_progress(response: T.Any, outfile: T.BinaryIO, *, label: str) -> None:
    length = response.headers.get("Content-Length")
    total = int(length) if length and length.isdigit() else None
    with tqdm(total=total, desc=f"Download {label}", unit="B", unit_scale=True, unit_divisor=1024, disable=not _progress_enabled()) as bar:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            outfile.write(chunk)
            bar.update(len(chunk))


def download(
    url: str | None,
    destination: Path,
    *,
    force: bool = False,
    google_drive_file_id: str | None = None,
    expected_sha256: str | None = None,
    label: str = "archive",
) -> Path:
    """Download a direct URL into destination.

    Google Drive ids are reported with a clear error rather than importing the
    full faceswap Google Drive helper.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        verify_sha256(destination, expected_sha256, label=label)
        return destination
    if force and destination.exists():
        destination.unlink()
    if google_drive_file_id is not None:
        raise ValueError(
            f"Google Drive download for {label} requires manual setup. "
            f"Download file id {google_drive_file_id} to {destination}."
        )
    if url is None:
        raise ValueError("Either url or google_drive_file_id must be supplied")
    fd, tmp_name = tempfile.mkstemp(prefix=f"{destination.name}.", suffix=".part", dir=destination.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as outfile:
            _download_with_progress(response, outfile, label=label)
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise OSError(f"download produced an empty file: {tmp_path}")
        verify_sha256(tmp_path, expected_sha256, label=label)
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return destination


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def safe_zip_extractall(zf: zipfile.ZipFile, destination: str | os.PathLike[str]) -> None:
    """Extract a zip file while blocking path traversal."""
    dest = Path(destination).resolve()
    members = zf.infolist()
    for member in members:
        target = (dest / member.filename).resolve()
        if not _is_relative_to(target, dest):
            raise ValueError(f"Blocked zip path traversal member: {member.filename}")
    for member in tqdm(members, desc=f"Extract {dest.name}", unit="file", disable=not _progress_enabled()):
        zf.extract(member, dest)


def safe_tar_extractall(tf: tarfile.TarFile, destination: str | os.PathLike[str]) -> None:
    """Extract a tar file while blocking path traversal and links."""
    dest = Path(destination).resolve()
    members = tf.getmembers()
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(f"Blocked tar link member: {member.name}")
        target = (dest / member.name).resolve()
        if not _is_relative_to(target, dest):
            raise ValueError(f"Blocked tar path traversal member: {member.name}")
    for member in tqdm(members, desc=f"Extract {dest.name}", unit="file", disable=not _progress_enabled()):
        try:
            tf.extract(member, dest, filter="data")
        except TypeError:
            tf.extract(member, dest)


def _archive_format(archive_path: Path) -> str | None:
    if zipfile.is_zipfile(archive_path):
        return "zip"
    if tarfile.is_tarfile(archive_path):
        return "tar"
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        return "tar"
    return None


def _extract_archive(archive_path: Path, destination: Path) -> None:
    archive_type = _archive_format(archive_path)
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            safe_zip_extractall(zf, destination)
        return
    if archive_type == "tar":
        with tarfile.open(archive_path, "r:*") as tf:
            safe_tar_extractall(tf, destination)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


@contextlib.contextmanager
def extract_archive_to_temp(archive: str | os.PathLike[str]) -> T.Iterator[Path]:
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        _extract_archive(archive_path, destination)
        yield destination


def is_archive(path: Path) -> bool:
    return path.is_file() and any(str(path).lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def extract_archive_to_cache(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    force: bool = False,
    expected_sha256: str | None = None,
    label: str = "dataset archive",
) -> Path:
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")
    verify_sha256(archive_path, expected_sha256, label=label)
    destination_path = Path(destination)
    if force and destination_path.exists():
        shutil.rmtree(destination_path) if destination_path.is_dir() else destination_path.unlink()
    if destination_path.is_dir() and any(destination_path.iterdir()):
        return destination_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{destination_path.name}.", suffix=".part", dir=destination_path.parent))
    try:
        _extract_archive(archive_path, tmp_dir)
        if destination_path.exists():
            shutil.rmtree(destination_path) if destination_path.is_dir() else destination_path.unlink()
        os.replace(tmp_dir, destination_path)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return destination_path


def resolve_dataset_source(
    spec: DatasetSourceSpec,
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    source_dir: str | os.PathLike[str] | None = None,
    source_zip: str | os.PathLike[str] | None = None,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Resolve a dataset source using explicit args, cache, then download."""
    if source_zip is not None:
        archive = Path(source_zip)
        if not archive.is_file():
            raise FileNotFoundError(f"{spec.dataset} source archive not found: {archive}")
        return archive
    if source_dir is not None:
        directory = Path(source_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"{spec.dataset} source directory not found: {directory}")
        return directory
    cache_root = Path(cache_dir) / spec.cache_root_name
    for name in (*_archive_names(spec), *spec.extracted_aliases, EXTRACTED_DIR_NAME):
        candidate = cache_root / name
        if candidate.is_dir():
            return candidate
        if is_archive(candidate):
            return extract_archive_to_cache(candidate, cache_root / EXTRACTED_DIR_NAME, expected_sha256=_effective_sha256(spec), label=f"{spec.dataset} archive")
        if candidate.is_file():
            return candidate
    if no_download:
        raise FileNotFoundError(f"{spec.dataset} source not found in {cache_root}. Download disabled.{_manual_hint(spec)}")
    url = _effective_url(spec, download_url)
    expected_sha256 = None if download_url else _effective_sha256(spec)
    if url is None and spec.google_drive_file_id is None:
        raise FileNotFoundError(f"{spec.dataset} source not found in {cache_root}.{_manual_hint(spec)}")
    archive = download(
        url,
        cache_root / _archive_name(spec),
        force=force_download,
        google_drive_file_id=spec.google_drive_file_id,
        expected_sha256=expected_sha256,
        label=f"{spec.dataset} archive",
    )
    return extract_archive_to_cache(archive, cache_root / EXTRACTED_DIR_NAME, force=force_download, expected_sha256=expected_sha256, label=f"{spec.dataset} archive")


def resolve_wflw_official_source(
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    raise FileNotFoundError(
        "Automatic WFLW multipart source resolution is not included in this CD-ViT port. "
        "Pass --source-dir/--source-zip containing WFLW annotations and images."
    )
