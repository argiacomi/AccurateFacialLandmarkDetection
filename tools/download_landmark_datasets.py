#!/usr/bin/env python3
"""Download landmark dataset source archives used by local manifest builders."""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.datasets.progress import track
from lib.io_utils import sha1_file, sha256_file

CHUNK_SIZE = 1024 * 1024
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
ALL_DATASETS = (
    "wflw",
    "cofw68",
    "cofw29",
    "helen",
    "lapa",
    "jd-landmark",
    "fll2",
    "fll3",
    "xm2vts",
    "frgc",
    "300vw",
    "wflw-v",
    "300w",
    "aflw",
    "aflw2000-3d",
    "merl-rav",
    "menpo2d",
    "multipie",
)


@dataclass(frozen=True)
class SourceAsset:
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
    alternate_filenames: tuple[str, ...] = field(default_factory=tuple)
    shared_with: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_manual(self) -> bool:
        return self.url is None and self.google_drive_file_id is None

    @property
    def source_kind(self) -> str:
        if self.url is not None:
            return "url"
        if self.google_drive_file_id is not None:
            return "google_drive"
        return "manual"

    @property
    def source_display(self) -> str:
        if self.url is not None:
            return self.url
        if self.google_drive_file_id is not None:
            return f"gdrive:{self.google_drive_file_id}"
        return "manual"

    @property
    def checksum_marker(self) -> str:
        if self.sha256:
            return "sha256"
        if self.sha1:
            return "sha1"
        return "none"

    @property
    def kind_marker(self) -> str:
        if self.is_manual:
            return "manual"
        if self.google_drive_file_id is not None and self.url is None:
            return "gdrive"
        if self.alternate:
            return "alternate"
        return "required" if self.required_for_builder else "optional"


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
        filename="WFLW_images.tar.gz",
        google_drive_file_id="1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC",
        alternate_filenames=("WFLW_images.zip", "WFLW_images.tgz"),
        note="Official WFLW images.",
    ),
    SourceAsset(
        dataset="cofw68",
        name="cofw68 color images",
        filename="COFW_color.zip",
        url="https://data.caltech.edu/records/bc0bf-nc666/files/COFW_color.zip?download=1",
        required_for_builder=False,
        shared_with=("cofw29",),
        note="cofw68 color image archive (shared with cofw29). Pair with cofw68 annotations/JSON for manifest building.",
    ),
    SourceAsset(
        dataset="cofw68",
        name="cofw6868 benchmark annotations",
        filename="cofw68-benchmark-master.zip",
        url="https://github.com/golnazghiasi/cofw68-benchmark/archive/master.zip",
        note="cofw6868 benchmark annotation repository. Convert/organize with cofw68 images before building.",
    ),
    SourceAsset(
        dataset="cofw29",
        name="cofw29 original color images",
        filename="COFW_color.zip",
        url="https://data.caltech.edu/records/bc0bf-nc666/files/COFW_color.zip?download=1",
        shared_with=("cofw68",),
        note="Original COFW 29-point color source (shared with cofw68). Preserve 29-point labels and visibility/occlusion metadata.",
    ),
    SourceAsset(
        dataset="helen",
        name="HELEN dense 194-point annotations",
        filename="annotations.json",
        url="https://s3.amazonaws.com/helen-images/annotations.json",
        extract=False,
        note=(
            "Dense HELEN 194-point annotations. These are an annotation layer over "
            "the existing 300W Helen image cache."
        ),
    ),
    SourceAsset(
        dataset="jd-landmark",
        name="JD-landmark Test_data1",
        filename="Test_data1.zip",
        google_drive_file_id="12wRlDARRKe0u-lzFPRw-klG2MUa_JBQm",
        google_drive_view_url="https://drive.google.com/open?id=12wRlDARRKe0u-lzFPRw-klG2MUa_JBQm",
        required_for_builder=False,
        note="JD-landmark Test Dataset 1 (landmark/picture/rect).",
    ),
    SourceAsset(
        dataset="jd-landmark",
        name="JD-landmark corrected landmarks",
        filename="Corrected_landmark.zip",
        url="https://github.com/facial-landmarks-localization-challenge/facial-landmarks-localization-challenge.github.io/raw/master/Corrected_landmark.zip",
        required_for_builder=False,
        note="Corrected 106-point landmark overrides applied over matching annotation filenames.",
    ),
    SourceAsset(
        dataset="jd-landmark",
        name="JD-landmark training bbox",
        filename="training_dataset_face_detection_bounding_box_v1.zip",
        url="https://github.com/facial-landmarks-localization-challenge/facial-landmarks-localization-challenge.github.io/raw/master/training_dataset_face_detection_bounding_box_v1.zip",
        required_for_builder=False,
        note="Training face-detection bounding boxes attached as bbox metadata when available.",
    ),
    SourceAsset(
        dataset="fll2",
        name="fll2 106-point source",
        filename="fll2.zip",
        google_drive_file_id="16fiVoBaTtOevQa4mH34rWggfkNKNEL2A",
        google_drive_view_url="https://drive.google.com/file/d/16fiVoBaTtOevQa4mH34rWggfkNKNEL2A/view",
        note="fll2 106-point source.",
    ),
    SourceAsset(
        dataset="fll3",
        name="FLL3 106-point source",
        filename="FLL3.zip",
        google_drive_file_id="1F_UnmpRnUnNS3Wk3V6CkJiIUYmG5Wjdr",
        google_drive_view_url="https://drive.google.com/file/d/1F_UnmpRnUnNS3Wk3V6CkJiIUYmG5Wjdr/view",
        note="FLL3 106-point source.",
    ),
    SourceAsset(
        dataset="xm2vts",
        name="XM2VTS source",
        filename="XM2VTS.zip",
        google_drive_file_id="1qdBlQhq9YEt5lzX1OGy5_AyjFL3vWxRs",
        google_drive_view_url="https://drive.google.com/file/d/1qdBlQhq9YEt5lzX1OGy5_AyjFL3vWxRs/view",
        note="Stage with subject/session/capture folders where available; builder preserves those identifiers.",
    ),
    SourceAsset(
        dataset="frgc",
        name="FRGC source",
        filename="FRGC.zip",
        google_drive_file_id="1T2Ux0tjd5CxI9PWZb5sXThuGvWH-oM5p",
        google_drive_view_url="https://drive.google.com/file/d/1T2Ux0tjd5CxI9PWZb5sXThuGvWH-oM5p/view",
        note="Stage with subject/session/capture folders where available; builder preserves those identifiers.",
    ),
    SourceAsset(
        dataset="300vw",
        name="300VW videos",
        filename="300VW_Dataset_2015_12_14.zip",
        url="https://ibug.doc.ic.ac.uk/download/300VW_Dataset_2015_12_14.zip/",
        note="300VW video dataset. Build with frame extraction and video-level split safety.",
    ),
    SourceAsset(
        dataset="wflw-v",
        name="WFLW-V videos",
        filename="WFLW-V.zip",
        google_drive_file_id="1YSJdgIb-vToJIAV04PGh_U7nX6dxVSjt",
        google_drive_view_url="https://drive.google.com/file/d/1YSJdgIb-vToJIAV04PGh_U7nX6dxVSjt/view",
        note="WFLW-V video source. Build with frame extraction and video-level split safety.",
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
        note="Native AFLW package used by MERL-RAV native mode.",
    ),
    SourceAsset(
        dataset="merl-rav",
        name="MERL-RAV labels",
        filename="MERL-RAV_dataset-master.zip",
        url="https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip",
        note="MERL-RAV annotations. Pair with AFLW images by imageNNNNN.",
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
        note="MenpoBenchmark Menpo2D package.",
    ),
    SourceAsset(
        dataset="multipie",
        name="MultiPIE",
        filename="MultiPIE.zip",
        google_drive_file_id="18JFjBTAZqthpORmEf2LuT14IuMYNyD_h",
        google_drive_view_url="https://drive.google.com/file/d/18JFjBTAZqthpORmEf2LuT14IuMYNyD_h/view",
        note="MenpoBenchmark MultiPIE package.",
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
        "cofw68": "cofw68",
        "cofw29": "cofw29",
        "helen": "helen",
        "lapa": "lapa",
        "jd": "jd-landmark",
        "jdlandmark": "jd-landmark",
        "jd-landmark": "jd-landmark",
        "fll2": "fll2",
        "fll3": "fll3",
        "xm2vts": "xm2vts",
        "frgc": "frgc",
        "300vw": "300vw",
        "300-vw": "300vw",
        "wflw-v": "wflw-v",
        "wflwv": "wflw-v",
        "all": "all",
    }
    return aliases.get(key, key)


def normalize_datasets(values: T.Iterable[str]) -> list[str]:
    """Normalize a mix of space- and comma-separated dataset tokens to canonical ids.

    Accepts iterables like ``["wflw-v", "300vw,cofw29"]`` and preserves the
    first-seen order while de-duplicating. ``all`` expands to every dataset id.
    """
    out: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            key = _dataset_key(token)
            if key == "all":
                for dataset in ALL_DATASETS:
                    if dataset not in out:
                        out.append(dataset)
                continue
            if key not in out:
                out.append(key)
    return out


def _resolve_dataset_keys(datasets: T.Sequence[str] | None) -> list[str]:
    selected = list(datasets) if datasets else list(ALL_DATASETS)
    unknown = sorted(set(selected) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return selected


def _selected_sources(
    datasets: T.Sequence[str] | None, *, include_alternates: bool
) -> list[SourceAsset]:
    selected = set(_resolve_dataset_keys(datasets))
    return [
        source
        for source in SOURCES
        if source.dataset in selected and (include_alternates or not source.alternate)
    ]


def _verify(path: Path, *, sha256: str | None, sha1: str | None) -> None:
    if sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != sha256.lower():
            raise ValueError(
                f"sha256 mismatch for {path}: expected {sha256}, got {actual}"
            )
    if sha1 is not None:
        actual = sha1_file(path)
        if actual.lower() != sha1.lower():
            raise ValueError(f"sha1 mismatch for {path}: expected {sha1}, got {actual}")


def _download_url(url: str, destination: Path, *, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return destination
    if force and destination.exists():
        destination.unlink()

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        print(f"Downloading {url} -> {destination}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = (
                int(total_header) if total_header and total_header.isdigit() else None
            )
            bar = track(
                desc=f"Download {destination.name}",
                total=total,
                unit="B",
                unit_scale=True,
            )
            with bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
        if tmp_path.stat().st_size == 0:
            raise OSError(f"download produced an empty file: {tmp_path}")
        os.replace(tmp_path, destination)
        return destination
    except BaseException:
        # BaseException so a Ctrl-C (KeyboardInterrupt) also removes the partial file.
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
    tmp_destination = destination.with_name(f"{destination.name}.part")
    if tmp_destination.exists():
        tmp_destination.unlink()
    try:
        subprocess.run([gdown, file_id, "-O", str(tmp_destination)], check=True)
        if not tmp_destination.is_file() or tmp_destination.stat().st_size == 0:
            raise OSError(
                f"Google Drive download failed or produced empty file: {tmp_destination}"
            )
        os.replace(tmp_destination, destination)
    finally:
        if tmp_destination.exists():
            tmp_destination.unlink()
    return destination


def _extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        members = zf.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(
                    f"blocked zip path traversal member: {member.filename}"
                )
        for member in track(
            members, desc=f"Extract {path.name}", total=len(members), unit="file"
        ):
            zf.extract(member, destination)


def _extract_tar(path: Path, destination: Path) -> None:
    with tarfile.open(path, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"blocked tar link member: {member.name}")
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"blocked tar path traversal member: {member.name}")
        extract_kwargs: dict[str, str] = {"filter": "data"}
        for member in track(
            members, desc=f"Extract {path.name}", total=len(members), unit="file"
        ):
            try:
                tf.extract(member, destination, **extract_kwargs)
            except TypeError:
                # Python without the data filter kwarg; drop it for this and later members.
                extract_kwargs = {}
                tf.extract(member, destination)


def _extract_archive(path: Path, destination: Path, *, force: bool) -> Path:
    if not any(str(path).lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
        return destination
    marker = destination / ".extracted_from.json"
    marker_payload = {
        "archive": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
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
    if zipfile.is_zipfile(path):
        _extract_zip(path, destination)
    elif tarfile.is_tarfile(path):
        _extract_tar(path, destination)
    else:
        raise ValueError(f"unsupported archive format: {path}")
    marker.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _write_manual_steps(asset: SourceAsset, dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / asset.filename
    steps = asset.manual_steps or (
        f"Manually download Google Drive file id {asset.google_drive_file_id}.",
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


def _looks_like_archive(name: T.Any) -> bool:
    lowered = str(name).lower()
    return any(lowered.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _validate_archive(path: Path) -> None:
    """Reject files that carry an archive extension but are not valid archives."""
    if not _looks_like_archive(path):
        return
    if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
        return
    raise ValueError(
        f"{Path(path).name} has an archive extension but is not a valid zip/tar archive; "
        "the source may have returned an HTML error page, login page, or download-denied response."
    )


def _archive_is_usable(path: Path, asset: SourceAsset) -> bool:
    """True if an existing file can be reused (valid archive when it looks like one)."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if _looks_like_archive(path):
        return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)
    return True


def _archive_candidate_names(asset: SourceAsset) -> tuple[str, ...]:
    return tuple(dict.fromkeys((asset.filename, *asset.alternate_filenames)))


def _find_reusable_archive(
    asset: SourceAsset, output_root: Path
) -> tuple[Path, str] | None:
    """Find an existing compatible archive to reuse instead of downloading.

    Checks the dataset's own archive directory (configured and alternate
    filenames), then any datasets that share the same underlying archive.
    Returns ``(path, status)`` with status ``"reused"`` (same dataset) or
    ``"reused_shared"`` (a dataset listed in ``shared_with``). Invalid archives
    (e.g. saved HTML error pages) are skipped so they are never reused.
    """
    output_root = Path(output_root)
    own_dir = output_root / asset.dataset / "archives"
    for name in _archive_candidate_names(asset):
        candidate = own_dir / name
        if _archive_is_usable(candidate, asset):
            return candidate, "reused"
    for other in asset.shared_with:
        other_dir = output_root / other / "archives"
        for name in _archive_candidate_names(asset):
            candidate = other_dir / name
            if _archive_is_usable(candidate, asset):
                return candidate, "reused_shared"
    return None


def _process_asset(asset: SourceAsset, args: argparse.Namespace) -> dict[str, T.Any]:
    dataset_dir = Path(args.output_root) / asset.dataset
    archive_dir = dataset_dir / "archives"
    result: dict[str, T.Any] = {
        "dataset": asset.dataset,
        "name": asset.name,
        "filename": asset.filename,
        "status": "pending",
        "required_for_builder": asset.required_for_builder,
        "alternate": asset.alternate,
        "note": asset.note,
        "source_kind": asset.source_kind,
        "source": asset.source_display,
        "checksum_status": "verified" if (asset.sha256 or asset.sha1) else "none",
    }
    if asset.url is not None:
        result["url"] = asset.url
    if asset.google_drive_file_id is not None:
        result["google_drive_file_id"] = asset.google_drive_file_id
        if asset.google_drive_view_url:
            result["google_drive_view_url"] = asset.google_drive_view_url

    if asset.is_manual:
        path = _write_manual_steps(asset, dataset_dir)
        result.update(status="manual", path=str(path), checksum_status="not_applicable")
        return result

    destination = archive_dir / asset.filename
    try:
        reuse = None if args.force else _find_reusable_archive(asset, args.output_root)
        if reuse is not None:
            path, status = reuse
            print(f"Reusing existing archive {path} for {asset.name}")
            result["reused_from"] = str(path)
            checksum_status = "reused"
        else:
            status = "downloaded"
            if asset.google_drive_file_id:
                path = _download_google_drive(
                    asset.google_drive_file_id, destination, force=args.force
                )
            else:
                assert asset.url is not None
                path = _download_url(asset.url, destination, force=args.force)
            # Reject HTML error/login pages saved with an archive extension before
            # extraction, and never leave them where a later run would reuse them.
            try:
                _validate_archive(path)
            except ValueError:
                if Path(path).exists():
                    Path(path).unlink()
                raise
            if args.skip_checksum:
                checksum_status = "skipped"
            else:
                _verify(path, sha256=asset.sha256, sha1=asset.sha1)
                checksum_status = "verified" if (asset.sha256 or asset.sha1) else "none"
        result.update(
            status=status,
            archive=str(path),
            sha256=sha256_file(path),
            sha1=sha1_file(path),
            checksum_status=checksum_status,
        )
        if args.extract and asset.extract:
            extract_target = dataset_dir / "extracted" / Path(path).name
            extracted = _extract_archive(path, extract_target, force=args.force)
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
    path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


REGISTRY_VERSION = 1
MANUAL_STATUSES = frozenset({"manual", "manual_google_drive"})
REUSED_STATUSES = frozenset({"reused", "reused_shared"})


def registry_path(output_root: Path) -> Path:
    return Path(output_root) / "registry.json"


def _dataset_source_dir(output_root: Path, dataset: str) -> Path:
    """Default extracted source root the builders consume for a dataset."""
    return Path(output_root) / dataset / "extracted"


def build_registry(
    results: list[dict[str, T.Any]], output_root: Path
) -> dict[str, T.Any]:
    output_root = Path(output_root)
    datasets: dict[str, T.Any] = {}
    for result in results:
        dataset = result["dataset"]
        entry = datasets.setdefault(
            dataset,
            {
                "source_dir": str(_dataset_source_dir(output_root, dataset)),
                "assets": [],
            },
        )
        entry["assets"].append(
            {
                "name": result.get("name"),
                "filename": result.get("filename"),
                "status": result.get("status"),
                "archive": result.get("archive"),
                "extracted": result.get("extracted"),
                "source_kind": result.get("source_kind"),
                "source": result.get("source"),
                "checksum_status": result.get("checksum_status"),
                "required_for_builder": result.get("required_for_builder"),
                "alternate": result.get("alternate"),
                "manual": result.get("status") in MANUAL_STATUSES,
                "reused": result.get("status") in REUSED_STATUSES,
                "reused_from": result.get("reused_from"),
            }
        )
    return {
        "version": REGISTRY_VERSION,
        "output_root": str(output_root.resolve()),
        "datasets": datasets,
    }


def _merge_registry(
    existing: dict[str, T.Any], fresh: dict[str, T.Any]
) -> dict[str, T.Any]:
    """Merge a fresh registry into an existing one, replacing only touched datasets."""
    merged = dict(existing) if existing else {}
    merged["version"] = REGISTRY_VERSION
    merged["output_root"] = fresh.get("output_root", merged.get("output_root"))
    datasets = dict(merged.get("datasets") or {})
    datasets.update(fresh.get("datasets") or {})
    merged["datasets"] = datasets
    return merged


def write_registry(results: list[dict[str, T.Any]], output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = registry_path(output_root)
    fresh = build_registry(results, output_root)
    existing = load_registry(output_root)
    payload = _merge_registry(existing, fresh) if existing else fresh
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_registry(output_root: Path) -> dict[str, T.Any] | None:
    path = registry_path(output_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def resolve_source_dir(
    registry: dict[str, T.Any], dataset: str, output_root: Path
) -> Path | None:
    """Resolve the extracted source directory the builder should consume for a dataset."""
    dataset = _dataset_key(dataset)
    entry = (registry.get("datasets") or {}).get(dataset) if registry else None
    candidates: list[Path] = []
    archive_dirs: list[Path] = []
    if entry:
        if entry.get("source_dir"):
            candidates.append(Path(entry["source_dir"]))
        for asset in entry.get("assets") or []:
            if asset.get("extracted"):
                candidates.append(Path(asset["extracted"]))
            if asset.get("archive"):
                archive_dirs.append(Path(asset["archive"]).parent)
    candidates.append(_dataset_source_dir(output_root, dataset))
    # Non-archive single-file assets (e.g. HELEN annotations.json) live in archives/.
    candidates.extend(archive_dirs)
    candidates.append(Path(output_root) / dataset / "archives")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _print_build_hints(output_root: Path) -> None:
    print("\nBuild hints:")
    print(
        "  WFLW: "
        f"python tools/landmarks/build_quality_dataset.py --dataset wflw --source-dir {output_root / 'wflw' / 'extracted'} --output-dir runs/landmarks/build_wflw"
    )
    print(
        "  cofw68: "
        f"python tools/landmarks/build_quality_dataset.py --dataset cofw68 --source-dir {output_root / 'cofw68' / 'extracted'} --output-dir runs/landmarks/build_cofw68"
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
    print(
        "  Issue #8 still-image datasets: "
        "use --dataset helen|lapa|jd-landmark|fll2|fll3|cofw29|xm2vts|frgc; each routes through a dataset-specific parser with native schema validation."
    )
    print(
        "  Issue #8 video datasets: "
        "use --dataset 300vw|wflw-v with --frame-stride/--max-frames-per-video; all frames from a video share split_safe_id=video_id."
    )


def format_list_table(sources: T.Sequence[SourceAsset]) -> str:
    """Render configured sources as a stable, column-aligned table."""
    headers = ("dataset", "asset", "source", "checksum", "kind")
    rows: list[tuple[str, str, str, str, str]] = []
    for asset in sources:
        source = asset.source_display
        if asset.is_manual:
            source = f"manual ({asset.filename})"
        rows.append(
            (
                asset.dataset,
                asset.name,
                source,
                asset.checksum_marker,
                asset.kind_marker,
            )
        )
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        if rows
        else len(headers[col])
        for col in range(len(headers))
    ]

    # Do not pad the final column so trailing whitespace stays out of the output.
    def fmt(values: T.Sequence[str]) -> str:
        cells = [values[col].ljust(widths[col]) for col in range(len(widths) - 1)]
        cells.append(values[-1])
        return "  ".join(cells).rstrip()

    lines = [fmt(headers), fmt(tuple("-" * widths[col] for col in range(len(widths))))]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def download_datasets(
    datasets: T.Sequence[str] | None,
    *,
    output_root: Path,
    extract: bool = True,
    force: bool = False,
    skip_checksum: bool = False,
    include_alternates: bool = False,
    keep_going: bool = True,
) -> tuple[list[dict[str, T.Any]], dict[str, T.Any]]:
    """Programmatic download entry point used by the preparation orchestrator.

    Returns the per-asset results and the persisted registry payload.
    """
    output_root = Path(output_root)
    sources = _selected_sources(datasets, include_alternates=include_alternates)
    args = argparse.Namespace(
        output_root=output_root,
        extract=extract,
        force=force,
        skip_checksum=skip_checksum,
        keep_going=keep_going,
    )
    results = [
        _process_asset(asset, args)
        for asset in track(sources, desc="Datasets", total=len(sources), unit="asset")
    ]
    _write_summary(results, output_root)
    write_registry(results, output_root)
    return results, load_registry(output_root) or build_registry(results, output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/landmarks"))
    parser.add_argument(
        "--dataset",
        default=None,
        help="Comma-separated dataset list or 'all'. Backward-compatible alias for --datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="One or more datasets, space- and/or comma-separated (e.g. --datasets wflw-v 300vw,cofw29). Use 'all' for everything.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded archives after download.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Redownload/re-extract existing files."
    )
    parser.add_argument(
        "--include-alternates",
        action="store_true",
        help="Include alternate source URLs, currently the official 300W split archive parts.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip stored SHA256/SHA1 verification.",
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue after a failed download."
    )
    parser.add_argument(
        "--list", action="store_true", help="List configured sources and exit."
    )
    return parser


def _resolve_requested_datasets(args: argparse.Namespace) -> list[str] | None:
    tokens: list[str] = []
    if args.datasets:
        tokens.extend(args.datasets)
    if args.dataset:
        tokens.append(args.dataset)
    if not tokens:
        return None
    return normalize_datasets(tokens)


def _run(args: argparse.Namespace) -> int:
    requested = _resolve_requested_datasets(args)
    sources = _selected_sources(requested, include_alternates=args.include_alternates)

    if args.list:
        print(format_list_table(sources))
        return 0

    results = [
        _process_asset(asset, args)
        for asset in track(sources, desc="Datasets", total=len(sources), unit="asset")
    ]
    summary = _write_summary(results, Path(args.output_root))
    registry = write_registry(results, Path(args.output_root))
    print(f"\nWrote summary: {summary}")
    print(f"Wrote registry: {registry}")
    _print_build_hints(Path(args.output_root))

    errored = [result for result in results if result.get("status") == "error"]
    return 1 if errored else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user (Ctrl-C); the partial download was cleaned up.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
