"""Same-image split cleanup keys on global native-file identity.

Different datasets can legitimately reference the same native image (the 300W
cache shared by helen/jd-landmark, MERL-RAV labels over native AFLW frames). The
leakage validator treats a real file path as a global identity, so the cleanup
helper must group such samples globally -- not per dataset -- or it cannot
collapse the cross-dataset same-file split that validation then flags.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.evaluation.split_safe import (
    image_source_ids,
    validate_no_train_test_leakage,
)
from tools import prepare_landmark_dataset as prep

NATIVE = "/data/landmarks/aflw/flickr/0/image04500.jpg"


def test_group_key_for_a_real_path_is_global_and_matches_validator():
    a = {"dataset": "merl-rav", "metadata": {"original_image": NATIVE}}
    b = {"dataset": "aflw2000-3d", "metadata": {"original_image": NATIVE}}

    key_a = prep._sample_image_group_key(a)
    key_b = prep._sample_image_group_key(b)

    # No dataset prefix: the resolved path is the identity for both.
    assert key_a == key_b == str(Path(NATIVE).expanduser())
    # The cleanup key is exactly one of the ids the validator keys leakage on.
    assert key_a in image_source_ids(a)


def test_bare_image_id_stays_dataset_namespaced():
    a = {"dataset": "x", "image_id": "212"}
    b = {"dataset": "y", "image_id": "212"}

    # Bare ids are dataset-local so unrelated "212" ids do not falsely merge.
    assert prep._sample_image_group_key(a) == "x|212"
    assert prep._sample_image_group_key(b) == "y|212"


def test_same_native_file_across_datasets_collapses_to_one_split(tmp_path):
    payload = {
        "samples": [
            {
                "sample_id": "a",
                "dataset": "merl-rav",
                "split": "train",
                "image": "images/merl-rav/a.png",
                "metadata": {"original_image": NATIVE},
            },
            {
                "sample_id": "b",
                "dataset": "aflw2000-3d",
                "split": "test",
                "image": "images/aflw2000-3d/b.png",
                "metadata": {"original_image": NATIVE},
            },
        ]
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    cleaned = prep._clean_same_image_split_leakage(manifest, payload)
    splits = {s["sample_id"]: s["split"] for s in cleaned["samples"]}

    # Test wins so a held-out source image is never pulled into train.
    assert splits == {"a": "test", "b": "test"}

    # And the leakage validator now passes on the cleaned split.
    train = [s for s in cleaned["samples"] if s["split"] == "train"]
    test = [s for s in cleaned["samples"] if s["split"] == "test"]
    validate_no_train_test_leakage(train, test)


def test_distinct_native_files_are_left_untouched(tmp_path):
    payload = {
        "samples": [
            {
                "sample_id": "a",
                "dataset": "helen",
                "split": "train",
                "metadata": {"original_image": "/data/helen/1.jpg"},
            },
            {
                "sample_id": "b",
                "dataset": "jd-landmark",
                "split": "test",
                "metadata": {"original_image": "/data/helen/2.jpg"},
            },
        ]
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    cleaned = prep._clean_same_image_split_leakage(manifest, payload)
    splits = {s["sample_id"]: s["split"] for s in cleaned["samples"]}
    assert splits == {"a": "train", "b": "test"}
