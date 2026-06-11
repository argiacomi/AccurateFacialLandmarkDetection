"""Hard-negative bucket annotation + sampler robustness.

``condition`` is overloaded on prepared manifests: it carries dataset/source
tags (lapa, fll2, 300vw, ...) as well as hard-negative buckets. The training
sampler used to treat any ``condition`` as a bucket, polluting the bucket
dimension. These tests pin the fix from both sides: the prepare step bakes an
authoritative ``metadata.hard_negative_bucket`` into every sample, and the
sampler ignores a ``condition`` that is not a real bucket.
"""

from __future__ import annotations

import json

from lib.datasets.hard_negative_mining import (
    annotate_sample_bucket_in_place,
    resolve_hard_negative_class,
)
from lib.training.domain_balanced_sampler import sample_bucket
from tools import prepare_landmark_dataset as prep


# --- sampler hardening -------------------------------------------------------


def test_dataset_label_in_condition_is_not_a_bucket():
    # No annotation, condition is a dataset label -> must fall back to anchor.
    assert sample_bucket({"dataset": "lapa", "condition": "lapa"}) == "anchor"


def test_conditions_drive_bucket_when_condition_is_a_dataset_label():
    sample = {
        "dataset": "lapa",
        "condition": "lapa",
        "conditions": ["profile", "occlusion"],
    }
    assert sample_bucket(sample) == "profile_occlusion"


def test_real_bucket_in_condition_is_honored():
    assert sample_bucket({"condition": "profile"}) == "profile"
    assert sample_bucket({"condition": "large_yaw_left"}) == "profile"


def test_annotation_metadata_wins_over_condition():
    sample = {
        "condition": "lapa",
        "metadata": {"hard_negative_bucket": "occlusion"},
    }
    assert sample_bucket(sample) == "occlusion"


# --- lib resolver ------------------------------------------------------------


def test_resolve_uses_classifier_then_dataset_default_then_anchor():
    classified = {"dataset": "lapa", "conditions": ["profile"]}
    cls, source = resolve_hard_negative_class(classified)
    assert (cls.bucket, source) == ("profile", "classified_by_label")

    cofw = {"dataset": "cofw68", "conditions": ["clean_face"]}
    cls, source = resolve_hard_negative_class(cofw)
    assert (cls.bucket, source) == ("occlusion", "dataset_default")

    neutral = {"dataset": "lapa", "conditions": ["frontal_unmapped"]}
    cls, source = resolve_hard_negative_class(neutral)
    assert (cls.bucket, source) == ("anchor", "anchor_fallback")


def test_in_place_annotation_is_non_destructive():
    sample = {
        "dataset": "cofw68",
        "condition": "cofw68",
        "conditions": ["cofw68"],
        "metadata": {"image_id": "x"},
    }
    bucket, source = annotate_sample_bucket_in_place(sample)
    assert (bucket, source) == ("occlusion", "dataset_default")
    # Only metadata is written; the overloaded fields are untouched.
    assert sample["condition"] == "cofw68"
    assert sample["conditions"] == ["cofw68"]
    assert sample["metadata"]["hard_negative_bucket"] == "occlusion"
    assert sample["metadata"]["hard_negative_weight"] == 2.0
    assert sample["metadata"]["image_id"] == "x"


# --- prepare step ------------------------------------------------------------


def test_prepare_annotates_buckets_without_touching_condition(tmp_path):
    payload = {
        "samples": [
            {"dataset": "lapa", "condition": "lapa", "conditions": ["lapa"]},
            {"dataset": "cofw68", "condition": "cofw68"},
            {
                "dataset": "wflw",
                "condition": "wflw",
                # already annotated -> must be preserved untouched
                "metadata": {"hard_negative_bucket": "profile_occlusion"},
            },
        ]
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    out = prep._annotate_hard_negative_buckets(manifest, payload)
    samples = out["samples"]

    assert samples[0]["metadata"]["hard_negative_bucket"] == "anchor"
    assert samples[1]["metadata"]["hard_negative_bucket"] == "occlusion"
    assert samples[2]["metadata"]["hard_negative_bucket"] == "profile_occlusion"
    # condition is never rewritten.
    assert [s["condition"] for s in samples] == ["lapa", "cofw68", "wflw"]
    # The sampler now reads real buckets, not dataset labels.
    assert [sample_bucket(s) for s in samples] == [
        "anchor",
        "occlusion",
        "profile_occlusion",
    ]
    # Summary is recorded; the already-annotated sample is not re-counted.
    summary = out["metadata"]["hard_negative_bucket_annotation"]
    assert summary["samples"] == 2
    assert summary["by_bucket"] == {"anchor": 1, "occlusion": 1}
