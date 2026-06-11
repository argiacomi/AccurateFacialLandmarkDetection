import json

import cv2
import numpy as np

from tools.stage_prepared_crops import stage_crops


def test_stage_crops_sample_level_geometry_validation(tmp_path):
    img = tmp_path / "native.png"
    assert cv2.imwrite(str(img), np.zeros((300, 300, 3), dtype=np.uint8))

    good = np.stack(
        [np.linspace(20, 200, 106), np.linspace(20, 200, 106)],
        axis=1,
    ).astype(np.float32)
    bad = np.stack(
        [np.linspace(75, 280, 106), np.linspace(344, 2000, 106)],
        axis=1,
    ).astype(np.float32)

    good_path = tmp_path / "good.npy"
    bad_path = tmp_path / "bad.npy"
    np.save(good_path, good)
    np.save(bad_path, bad)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "good",
                        "dataset": "x",
                        "image": str(img),
                        "landmarks": str(good_path),
                    },
                    {
                        "sample_id": "bad",
                        "dataset": "x",
                        "image": str(img),
                        "landmarks": str(bad_path),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out_manifest = tmp_path / "manifest.staged.json"
    stats = stage_crops(
        manifest,
        out_manifest=out_manifest,
        validate_geometry=True,
        drop_invalid_geometry=True,
        workers=1,
    )

    assert len(stats["geometry_issues"]) == 1
    assert stats["geometry_issues"][0]["sample_id"] == "bad"
    assert (
        stats["geometry_issues"][0]["diagnostics"]["reason"]
        == "unreasonable_loader_padding"
    )

    staged = json.loads(out_manifest.read_text(encoding="utf-8"))["samples"]
    by_id = {sample["sample_id"]: sample for sample in staged}
    assert "prepared_image" in by_id["good"]
    assert "bad" not in by_id
