from __future__ import annotations

import pytest

from lib.training.losses import _schema_head_weight_map


def test_schema_head_weight_map_accepts_valid_weights():
    assert _schema_head_weight_map("landmarks_98=1.5, profile39=0, landmarks_68=2") == {
        "landmarks_98": 1.5,
        "profile39": 0.0,
        "landmarks_68": 2.0,
    }


@pytest.mark.parametrize(
    "spec, message",
    [
        ("landmarks_98", "head=value"),
        ("=1.0", "empty head name"),
        ("landmarks_98=bad", "invalid value"),
        ("landmarks_98=-1", "finite non-negative"),
        ("landmarks_98=nan", "finite non-negative"),
        ("landmarks_98=inf", "finite non-negative"),
    ],
)
def test_schema_head_weight_map_rejects_malformed_weights(spec, message):
    with pytest.raises(ValueError, match=message):
        _schema_head_weight_map(spec)


def test_aggregate_sampler_diagnostics_sums_across_ddp_ranks(monkeypatch):
    import lib.training.heatmap_stage as heatmap_stage

    rank0 = {
        "requested_targets": {
            "bucket": {"profile": 1.0},
            "dataset": {"wflw": 1.0},
            "schema": {"2d_98": 1.0},
        },
        "actual_mix": {
            "bucket": {"profile": 2},
            "dataset": {"wflw": 2},
            "schema": {"2d_98": 2},
        },
        "fallback_counts": {"exact": 2, "exact_to_bucket": 1, "bucket_to_any": 0},
        "missing_targets": {"bucket": [], "dataset": ["missing_a"], "schema": []},
        "rank": 0,
        "world_size": 2,
        "batches_per_rank": 3,
    }
    rank1 = {
        "requested_targets": rank0["requested_targets"],
        "actual_mix": {
            "bucket": {"profile": 1, "occlusion": 2},
            "dataset": {"wflw": 1, "cofw68": 2},
            "schema": {"2d_98": 1, "2d_68": 2},
        },
        "fallback_counts": {"exact": 1, "exact_to_bucket": 0, "bucket_to_any": 2},
        "missing_targets": {
            "bucket": ["profile_occlusion"],
            "dataset": ["missing_b"],
            "schema": ["2d_39"],
        },
        "rank": 1,
        "world_size": 2,
        "batches_per_rank": 3,
    }

    def fake_all_gather_object(output, local):
        output[:] = [rank0, rank1]

    monkeypatch.setattr(heatmap_stage.dist, "is_available", lambda: True)
    monkeypatch.setattr(heatmap_stage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(heatmap_stage.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(heatmap_stage, "distributed_world_size", lambda: 2)
    monkeypatch.setattr(heatmap_stage, "is_rank_zero", lambda: True)

    diagnostics = heatmap_stage._aggregate_sampler_diagnostics(rank0)

    assert diagnostics["rank"] == "all"
    assert diagnostics["world_size"] == 2
    assert diagnostics["batches_per_rank"] == 3
    assert diagnostics["actual_mix"] == {
        "bucket": {"profile": 3, "occlusion": 2},
        "dataset": {"wflw": 3, "cofw68": 2},
        "schema": {"2d_98": 3, "2d_68": 2},
    }
    assert diagnostics["fallback_counts"] == {
        "exact": 3,
        "exact_to_bucket": 1,
        "bucket_to_any": 2,
    }
    assert diagnostics["missing_targets"] == {
        "bucket": ["profile_occlusion"],
        "dataset": ["missing_a", "missing_b"],
        "schema": ["2d_39"],
    }
    assert diagnostics["rank_diagnostics"] == [
        {"rank": 0, "batches_per_rank": 3},
        {"rank": 1, "batches_per_rank": 3},
    ]


def test_aggregate_sampler_diagnostics_returns_local_on_nonzero_rank(monkeypatch):
    import lib.training.heatmap_stage as heatmap_stage

    local = {
        "actual_mix": {"bucket": {"profile": 1}, "dataset": {}, "schema": {}},
        "fallback_counts": {"exact": 1},
        "missing_targets": {"bucket": [], "dataset": [], "schema": []},
        "rank": 1,
        "world_size": 2,
        "batches_per_rank": 1,
    }

    def fake_all_gather_object(output, local_payload):
        output[:] = [{}, local_payload]

    monkeypatch.setattr(heatmap_stage.dist, "is_available", lambda: True)
    monkeypatch.setattr(heatmap_stage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(heatmap_stage.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(heatmap_stage, "distributed_world_size", lambda: 2)
    monkeypatch.setattr(heatmap_stage, "is_rank_zero", lambda: False)

    assert heatmap_stage._aggregate_sampler_diagnostics(local) is local
