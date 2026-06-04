#!/usr/bin/env python3
"""Download landmark dataset source archives used by the local manifest builders.

The source table mirrors the dataset URL/file-id constants from faceswap's
landmark dataset source helpers. Direct URLs are downloaded with urllib. Google
Drive assets require ``--include-google-drive`` and the optional ``gdown`` CLI.

Examples:

    python tools/landmarks/download_landmark_datasets.py --output-root data/landmarks
    python tools/landmarks/download_landmark_datasets.py --dataset wflw,300w --extract
    python tools/landmarks/download_landmark_datasets.py --include-google-drive --extract
    python tools/landmarks/download_landmark_datasets.py --dataset 300w --include-alternates

Google Drive downloads require:

    pip install gdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import typing as T
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


CHUNK_SIZE = 1024 * 1024
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
ALL_DATASETS = (
    "wflw",
    "cofw",
    "300w",
    "aflw",
    "aflw2000-3d",
    "merl-rav",
    "menpo2d",
    "multipie",
)


@dataclass(frozen=True)
class SourceAsset:
    """One downloadable or manual dataset asset."""

    dataset: str
    name: str
    filename: str
    url: str | None = None
    google_drive_file_id: str | None = None
    google_drive_view_url: str | None = None
    sha256: str | None = None
    sha1: str | None = None
    required_for_builder: bool = True
    extract: bool = True
    alternate: bool = False
    note: str = ""
    manual_steps: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_manual(self) -> bool:
        return self.url is None and self.google_drive_file_id is None


SOURCES: tuple[SourceAsset, ...] = (
    SourceAsset(
        dataset="wflw",
        name="WFLW annotations",
        filename="WFLW_annotations.tar.gz",
        url="https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz",
        note="Official WFLW annotations. Images are a separate Google Drive asset.",
    ),
    SourceAsset(
        dataset="wflw",
        name="WFLW images",
        filename="WFLW_images.zip",
        google_drive_file_id="1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC",
        note="Official WFLW images. Requires --include-google-drive and gdown.",
    ),
    SourceAsset(
        dataset="cofw",
        name="COFW color images",
        filename="COFW_color.zip",
        url="https://data.caltech.edu/records/bc0bf-nc666/files/COFW_color.zip?download=1",
        required_for_builder=False,
        note="COFW color image archive. Pair with COFW68 annotations/JSON for manifest building.",
    ),
    SourceAsset(
        dataset="cofw",
        name="COFW68 benchmark annotations",
        filename="cofw68-benchmark-master.zip",
        url="https://github.com/golnazghiasi/cofw68-benchmark/archive/master.zip",
        note="COFW68 benchmark annotation repository. Convert/organize with COFW images before building.",
    ),
    SourceAsset(
        dataset="300w",
        name="300W Oxford DVE tarball",
        filename="300w.tar.gz",
        url="http://www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/300w.tar.gz",
        sha1="885b09159c61fa29998437747d589c65cfc4ccd3",
        note="Default 300W source stored in faceswap, from Oxford VGG DVE / ICCV 2019.",
    ),
    SourceAsset(
        dataset="300w",
        name="300W official split part 001",
        filename="300w.zip.001",
        url="https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.001",
        required_for_builder=False,
        extract=False,
        alternate=True,
        note="Alternate original iBUG split archive part. Combine all four parts into 300w.zip before extracting.",
    ),
    SourceAsset(
        dataset="300w",
        name="300W official split part 002",
        filename="300w.zip.002",
        url="https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.002",
        required_for_builder=False,
        extract=False,
        alternate=True,
        note="Alternate original iBUG split archive part. Combine all four parts into 300w.zip before extracting.",
    ),
    SourceAsset(
        dataset="300w",
        name="300W official split part 003",
        filename="300w.zip.003",
        url="https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.003",
        required_for_builder=False,
        extract=False,
        alternate=True,
        note="Alternate original iBUG split archive part. Combine all four parts into 300w.zip before extracting.",
    ),
    SourceAsset(
        dataset="300w",
        name="300W official split part 004",
        filename="300w.zip.004",
        url="https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.004",
        required_for_builder=False,
        extract=False,
        alternate=True,
        note="Alternate original iBUG split archive part. Combine all four parts into 300w.zip before extracting.",
    ),
    SourceAsset(
        dataset="aflw",
        name="AFLW native images/package",
        filename="AFLW.zip",
        google_drive_file_id="1uSx5hTxkxm48a3No0xm26DeJKpIooqrx",
        google_drive_view_url="https://drive.google.com/file/d/1uSx5hTxkxm48a3No0xm26DeJKpIooqrx/view",
        note=(
            "Native AFLW package used by MERL-RAV native mode. Requires --include-google-drive. "
            "Faceswap also stores a drive.usercontent direct URL, but gdown is more reliable."
        ),
    ),
    SourceAsset(
        dataset="merl-rav",
        name="MERL-RAV labels",
        filename="MERL-RAV_dataset-master.zip",
        url="https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip",
        note=(
            "MERL-RAV annotations. Pair with AFLW images by imageNNNNN, or use AFLW release-2 "
            "translation if you have that older cropped release."
        ),
    ),
    SourceAsset(
        dataset="aflw2000-3d",
        name="AFLW2000-3D",
        filename="AFLW2000-3D.zip",
        url="http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip",
        sha256="252bc35274d65ff27b6e573aa96c2f4c116ad88452cc984fb882258c0ed6e2d8",
        note="AFLW2000-3D archive with image+.mat pairs.",
    ),
    SourceAsset(
        dataset="menpo2d",
        name="Menpo2D",
        filename="Menpo2D.zip",
        google_drive_file_id="1CUqs0n135lye6J6RM5FQXT_DIT45dKvP",
        google_drive_view_url="https://drive.google.com/file/d/1CUqs0n135lye6J6RM5FQXT_DIT45dKvP/view",
        note="MenpoBenchmark Menpo2D package. Requires --include-google-drive and gdown.",
    ),
    SourceAsset(
        dataset="multipie",
        name="MultiPIE",
        filename="MultiPIE.zip",
        google_drive_file_id="18JFjBTAZqthpORmEf2LuT14IuMYNyD_h",
        google_drive_view_url="https://drive.google.com/file/d/18JFjBTAZqthpORmEf2LuT14IuMYNyD_h/view",
        note="MenpoBenchmark MultiPIE package. Requires --include-google-drive and gdown.",
    ),
)


def _dataset_key(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    aliases = {
        "aflw": "aflw",
        "aflw2000": "aflw2000-3d",
        "aflw2000-3d": "aflw2000-3d",
        "merlrav": "merl-rav",
        "merl-rav": "merl-rav",
        "menpo": "menpo2d",
        "menpo2d": "menpo2d",
        "menpo-2d": "menpo2d",
        "multi-pie": "multipie",
        "multipie": "multipie",
        "w300": "300w",
        "300w": "300w",
        "wflw": "wflw",
        "cofw": "cofw",
        "all": "all",
    }
    return aliases.get(key, key)


def _selected_sources(dataset_arg: str, *, include_alternates: bool) -> list[SourceAsset]:
    requested = tuple(_dataset_key(item) for item in dataset_arg.split(",") if item.strip())
    selected = set(ALL_DATASETS if not requested or requested == ("all",) else requested)
    unknown = sorted(selected - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return [
        source
        for source in SOURCES
        if source.dataset in selected and (include_alternates or not source.alternate)
    ]


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _hash_file(path, "sha256")


def _sha1_file(path: Path) -> str:
    return _hash_file(path, "sha1")


def _verify(path: Path, *, sha256: str | None, sha1: str | None) -> None:
    if sha256 is not None:
        actual = _sha256_file(path)
        if actual.lower() != sha256.lower():
            raise ValueError(f"sha256 mismatch for {path}: expected {sha256}, got {actual}")
    if sha1 is not None:
        actual = _sha1_file(path)
        if actual.lower() != sha1.lower():
            raise ValueError(f"sha1 mismatch for {path}: expected {sha1}, got {actual}")


def _download_url(url: str, destination: Path, *, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return destination
    if force and destination.exists():
        destination.unlink()

    fd, tmp_name = tempfile.mkstemp(prefix=f"{destination.name}.", suffix=".part", dir=destination.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        print(f"Downloading {url} -> {destination}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(
                        f"  {downloaded / 1_000_000:.1f} MB / {total / 1_000_000:.1f} MB ({pct:.1f}%)",
                        end="\r",
                    )
            if total:
                print()
        if tmp_path.stat().st_size == 0:
            raise OSError(f"download produced an empty file: {tmp_path}")
        os.replace(tmp_path, destination)
        return destination
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _download_google_drive(file_id: str, destination: Path, *, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return destination
    gdown = shutil.which("gdown")
    if gdown is None:
        raise RuntimeError(
            "Google Drive download requested but gdown is not installed. Run `pip install gdown`, "
            f"or manually download file id {file_id} to {destination}."
        )
    if force and destination.exists():
        destination.unlink()
    print(f"Downloading Google Drive file {file_id} -> {destination}")
    subprocess.run([gdown, "--id", file_id, "-O", str(destination)], check=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"Google Drive download failed or produced empty file: {destination}")
    return destination


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not _is_relative_to(target, destination.resolve()):
                raise ValueError(f"blocked zip path traversal member: {member.filename}")
        zf.extractall(destination)


def _extract_tar(path: Path, destination: Path) -> None:
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"blocked tar link member: {member.name}")
            target = (destination / member.name).resolve()
            if not _is_relative_to(target, destination.resolve()):
                raise ValueError(f"blocked tar path traversal member: {member.name}")
        try:
            tf.extractall(destination, filter="data")
        except TypeError:
            tf.extractall(destination)


def _extract_archive(path: Path, destination: Path, *, force: bool) -> Path:
    if not any(str(path).lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
        return destination
    marker = destination / ".extracted_from.json"
    marker_payload = {"archive": path.name, "size": path.stat().st_size, "sha256": _sha256_file(path)}
    if destination.is_dir() and marker.is_file() and not force:
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == marker_payload:
                print(f"Using existing extraction {destination}")
                return destination
        except json.JSONDecodeError:
            pass
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {path} -> {destination}")
    if zipfile.is_zipfile(path) or path.name.lower().endswith(".zip"):
        _extract_zip(path, destination)
    elif tarfile.is_tarfile(path) or path.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
        _extract_tar(path, destination)
    else:
        raise ValueError(f"unsupported archive format: {path}")
    marker.write_text(json.dumps(marker_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _write_manual_steps(asset: SourceAsset, dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / asset.filename
    steps = asset.manual_steps
    if not steps:
        steps = (
            f"Install gdown and rerun with --include-google-drive, or manually download Google Drive file id {asset.google_drive_file_id}.",
            f"Save it as {dataset_dir / 'archives' / asset.filename}.",
        )
    lines = [f"# {asset.name}", "", asset.note or "Manual setup required.", ""]
    if asset.google_drive_view_url:
        lines.extend([f"Google Drive view URL: {asset.google_drive_view_url}", ""])
    lines.extend(["## Steps", ""])
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(steps, 1))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote manual instructions: {path}")
    return path


def _process_asset(asset: SourceAsset, args: argparse.Namespace) -> dict[str, T.Any]:
    dataset_dir = Path(args.output_root) / asset.dataset
    archive_dir = dataset_dir / "archives"
    extract_dir = dataset_dir / "extracted" / Path(asset.filename).name
    result: dict[str, T.Any] = {
        "dataset": asset.dataset,
        "name": asset.name,
        "filename": asset.filename,
        "status": "pending",
        "required_for_builder": asset.required_for_builder,
        "alternate": asset.alternate,
        "note": asset.note,
    }

    if asset.is_manual:
        path = _write_manual_steps(asset, dataset_dir)
        result.update(status="manual", path=str(path))
        return result

    if asset.google_drive_file_id and not args.include_google_drive:
        path = _write_manual_steps(asset, dataset_dir)
        result.update(
            status="manual_google_drive",
            path=str(path),
            google_drive_file_id=asset.google_drive_file_id,
            google_drive_view_url=asset.google_drive_view_url,
        )
        return result

    destination = archive_dir / asset.filename
    try:
        if asset.google_drive_file_id:
            path = _download_google_drive(asset.google_drive_file_id, destination, force=args.force)
        else:
            assert asset.url is not None
            path = _download_url(asset.url, destination, force=args.force)
        if not args.skip_checksum:
            _verify(path, sha256=asset.sha256, sha1=asset.sha1)
        result.update(
            status="downloaded",
            archive=str(path),
            sha256=_sha256_file(path),
            sha1=_sha1_file(path),
        )
        if args.extract and asset.extract:
            extracted = _extract_archive(path, extract_dir, force=args.force)
            result["extracted"] = str(extracted)
    except Exception as err:  # noqa: BLE001
        result.update(status="error", error=str(err))
        if args.keep_going:
            print(f"ERROR: {asset.name}: {err}", file=sys.stderr)
            return result
        raise
    return result


def _write_summary(results: list[dict[str, T.Any]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "download_summary.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _print_build_hints(output_root: Path) -> None:
    print("\nBuild hints:")
    print(
        "  WFLW: "
        f"python tools/landmarks/build_quality_dataset.py --dataset wflw --source-dir {output_root / 'wflw' / 'extracted'} --output-dir runs/landmarks/build_wflw"
    )
    print(
        "  COFW: "
        f"python tools/landmarks/build_quality_dataset.py --dataset cofw --source-dir {output_root / 'cofw' / 'extracted'} --output-dir runs/landmarks/build_cofw"
    )
    print(
        "  300W: "
        f"python tools/landmarks/build_quality_dataset.py --dataset 300w --source-dir {output_root / '300w' / 'extracted'} --output-dir runs/landmarks/build_300w"
    )
    print(
        "  MERL-RAV native: "
        f"combine {output_root / 'merl-rav' / 'extracted'} labels with {output_root / 'aflw' / 'extracted'} images, then build --dataset merl-rav"
    )
    print(
        "  AFLW2000-3D: "
        f"python tools/landmarks/build_quality_dataset.py --dataset aflw2000-3d --source-dir {output_root / 'aflw2000-3d' / 'extracted'} --output-dir runs/landmarks/build_aflw2000_3d"
    )
    print(
        "  Menpo2D: "
        f"python tools/landmarks/build_quality_dataset.py --dataset menpo2d --source-dir {output_root / 'menpo2d' / 'extracted'} --output-dir runs/landmarks/build_menpo2d"
    )
    print(
        "  MultiPIE: "
        f"python tools/landmarks/build_quality_dataset.py --dataset multipie --source-dir {output_root / 'multipie' / 'extracted'} --output-dir runs/landmarks/build_multipie"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/landmarks"))
    parser.add_argument("--dataset", default="all", help="Comma-separated dataset list or 'all'.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded archives after download.")
    parser.add_argument("--force", action="store_true", help="Redownload/re-extract existing files.")
    parser.add_argument("--include-google-drive", action="store_true", help="Download Google Drive assets with gdown when available.")
    parser.add_argument("--include-alternates", action="store_true", help="Include alternate source URLs, currently the official 300W split archive parts.")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip stored SHA256/SHA1 verification.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed download.")
    parser.add_argument("--list", action="store_true", help="List configured sources and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources = _selected_sources(args.dataset, include_alternates=args.include_alternates)

    if args.list:
        for asset in sources:
            location = asset.url or (
                f"gdrive:{asset.google_drive_file_id}" if asset.google_drive_file_id else "manual"
            )
            flags = []
            if asset.alternate:
                flags.append("alternate")
            if asset.sha256:
                flags.append("sha256")
            if asset.sha1:
                flags.append("sha1")
            flag_text = f" [{' '.join(flags)}]" if flags else ""
            print(f"{asset.dataset:12s} {asset.name:32s} {location}{flag_text}")
        return 0

    results = [_process_asset(asset, args) for asset in sources]
    summary = _write_summary(results, Path(args.output_root))
    print(f"\nWrote summary: {summary}")
    _print_build_hints(Path(args.output_root))

    errored = [result for result in results if result.get("status") == "error"]
    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main())
