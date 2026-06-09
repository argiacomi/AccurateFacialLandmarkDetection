from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lib.core.schema import PROJECTION_MAPS_TO_68, projection_audit_for_schema
from tools import audit_schema_mapping as audit


def _write_npy(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.stack(
        [np.linspace(20, 230, count), np.linspace(30, 210, count)], axis=1
    ).astype(np.float32)
    np.save(path, pts)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((128, 128, 3), 127, dtype=np.uint8))


def _manifest(tmp_path: Path, samples: list[dict]) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest


def _sample(tmp_path: Path, sid: str, schema: str, count: int, *, mapping_audit=None):
    _write_npy(tmp_path / f"{sid}.npy", count)
    _write_image(tmp_path / f"{sid}.jpg")
    entry = {
        "sample_id": sid,
        "source_schema": schema,
        "landmarks": f"{sid}.npy",
        "image": f"{sid}.jpg",
    }
    if mapping_audit is not None:
        entry["mapping_audit"] = mapping_audit
    return entry


def test_report_lists_projection_maps_and_per_sample_map(tmp_path):
    samples = [
        _sample(tmp_path, "a68", "2d_68", 68),
        _sample(tmp_path, "b98", "2d_98", 98),
        _sample(tmp_path, "c106", "2d_106", 106),
        _sample(tmp_path, "d29", "2d_29", 29),
    ]
    out = tmp_path / "out"
    report_path = audit.audit_schema_mapping(_manifest(tmp_path, samples), out)
    report = json.loads(report_path.read_text())

    assert report["projection_maps"] == dict(PROJECTION_MAPS_TO_68)
    assert "map_98_to_68_size" not in report

    by_id = {s["sample_id"]: s for s in report["samples"]}
    assert by_id["a68"]["projection_map"] == "identity"
    assert by_id["a68"]["projected_68_count"] == 68
    assert by_id["b98"]["projection_map"] == "MAP_98_TO_68"
    assert by_id["b98"]["projected_68_count"] == 68
    assert by_id["c106"]["projection_map"] == "MAP_106_TO_68"
    assert by_id["c106"]["projected_68_count"] == 68
    # sparse, non-projectable schema gets no projection map
    assert "projection_map" not in by_id["d29"]
    assert by_id["d29"]["projection_to_68"]["status"] == "not_projectable"


def test_flags_stale_mapping_audit(tmp_path):
    # A 2d_106 sample whose manifest still records the old not_projectable audit.
    stale = {
        "projection_to_68": {
            "status": "not_projectable",
            "source_schema": "2d_106",
            "target_schema": "2d_68",
            "reason": "no audited 68-point overlap map is registered",
        }
    }
    fresh = {"projection_to_68": projection_audit_for_schema("2d_98")}
    samples = [
        _sample(tmp_path, "stale106", "2d_106", 106, mapping_audit=stale),
        _sample(tmp_path, "ok98", "2d_98", 98, mapping_audit=fresh),
    ]
    out = tmp_path / "out"
    report = json.loads(
        audit.audit_schema_mapping(_manifest(tmp_path, samples), out).read_text()
    )

    assert report["mapping_audit_mismatch_count"] == 1
    mism = report["mapping_audit_mismatches"]
    assert len(mism) == 1 and mism[0]["sample_id"] == "stale106"
    assert mism[0]["stored"]["status"] == "not_projectable"
    assert mism[0]["expected"]["status"] == "audited"

    by_id = {s["sample_id"]: s for s in report["samples"]}
    assert by_id["stale106"]["mapping_audit_consistent"] is False
    assert by_id["ok98"]["mapping_audit_consistent"] is True


def test_overlays_are_stratified_per_schema(tmp_path):
    samples = []
    for i in range(3):
        samples.append(_sample(tmp_path, f"w{i}", "2d_98", 98))
    for i in range(3):
        samples.append(_sample(tmp_path, f"h{i}", "2d_106", 106))
    out = tmp_path / "out"
    report = json.loads(
        audit.audit_schema_mapping(
            _manifest(tmp_path, samples),
            out,
            limit_per_schema=2,
            write_overlays=True,
        ).read_text()
    )

    # Each schema gets up to limit_per_schema overlays, not a single global cap.
    assert report["overlay_counts"] == {"2d_98": 2, "2d_106": 2}
    overlays = [s["overlay"] for s in report["samples"] if "overlay" in s]
    assert len(overlays) == 4
    assert all(Path(p).is_file() for p in overlays)
    assert any("/2d_98/" in p for p in overlays)
    assert any("/2d_106/" in p for p in overlays)
