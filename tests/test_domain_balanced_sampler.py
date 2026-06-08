import torch

from lib.landmarks.training.data import batch_mix, legacy_domain_balanced_collate
from lib.landmarks.training.domain_balanced_sampler import (
    DomainBalancedBatchSampler,
    parse_target_spec,
    parse_target_spec_for_kind,
    sample_bucket,
    sample_dataset,
)


def _sample(dataset, schema, bucket):
    return {
        "dataset": dataset,
        "source_schema": schema,
        "condition": bucket,
        "metadata": {"hard_negative_bucket": bucket},
    }


def test_parse_target_spec_normalizes_labels():
    assert parse_target_spec("profile-occlusion=2, wflw = 1") == {
        "profile_occlusion": 2.0,
        "wflw": 1.0,
    }


def test_parse_target_spec_rejects_invalid_weights():
    for spec in ("profile=nan", "profile=inf", "profile=-1", "profile=bad", "profile"):
        try:
            parse_target_spec(spec)
        except ValueError as exc:
            assert "profile" in str(exc) or "target item" in str(exc)
        else:
            raise AssertionError(f"expected {spec!r} to be rejected")


def test_bucket_aliases_collapse_to_canonical_targets():
    aliases = {
        "large_yaw": "profile",
        "profile_pose": "profile",
        "yaw_left": "profile",
        "rolled_profile_occlusion": "profile_occlusion",
        "large_yaw_occlusion": "profile_occlusion",
        "occluded": "occlusion",
        "single_eye_visible": "occlusion",
        "mouth_or_jaw_occluded": "occlusion",
        "normal": "anchor",
        "clean": "anchor",
        "frontal": "anchor",
    }
    for alias, canonical in aliases.items():
        assert sample_bucket(_sample("wflw", "2d_98", alias)) == canonical


def test_dataset_target_aliases_reuse_split_safe_normalization():
    assert parse_target_spec_for_kind("300W=1,aflw2000-3d=1,prod=1", kind="dataset") == {
        "w300": 1.0,
        "aflw2000": 1.0,
        "production_validated": 1.0,
    }
    assert sample_dataset(_sample("300W", "2d_68", "anchor")) == "w300"


def test_domain_balanced_sampler_is_reproducible():
    samples = [
        _sample("wflw", "2d_98", "profile"),
        _sample("cofw68", "2d_68", "occlusion"),
        _sample("300w", "2d_68", "anchor"),
        _sample("multipie", "multipie_profile_39", "profile_occlusion"),
    ]
    kwargs = dict(
        samples=samples,
        bucket_targets={"profile": 1, "occlusion": 1, "anchor": 1, "profile_occlusion": 1},
        batch_size=4,
        seed=7,
    )

    first = list(DomainBalancedBatchSampler(**kwargs))
    second = list(DomainBalancedBatchSampler(**kwargs))
    sampler = DomainBalancedBatchSampler(**kwargs)
    list(sampler)

    assert first == second
    assert sorted(first[0]) == [0, 1, 2, 3]
    assert sampler.last_epoch_diagnostics["actual_mix"]["bucket"] == {
        "anchor": 1,
        "occlusion": 1,
        "profile": 1,
        "profile_occlusion": 1,
    }


def test_domain_balanced_sampler_ddp_lengths_match_actual_batches():
    samples = [_sample(f"dataset-{index % 3}", "2d_68", "anchor") for index in range(10)]
    ranks = [
        DomainBalancedBatchSampler(samples, batch_size=2, seed=3, rank=rank, world_size=2)
        for rank in range(2)
    ]

    batches_by_rank = [list(sampler) for sampler in ranks]

    assert [len(sampler) for sampler in ranks] == [3, 3]
    assert [len(batches) for batches in batches_by_rank] == [3, 3]


def test_domain_balanced_sampler_drop_last_drops_uneven_ddp_global_batches():
    samples = [_sample(f"dataset-{index % 3}", "2d_68", "anchor") for index in range(11)]
    ranks = [
        DomainBalancedBatchSampler(
            samples,
            batch_size=2,
            seed=3,
            rank=rank,
            world_size=2,
            drop_last=True,
        )
        for rank in range(2)
    ]

    batches_by_rank = [list(sampler) for sampler in ranks]

    assert [len(sampler) for sampler in ranks] == [2, 2]
    assert [len(batches) for batches in batches_by_rank] == [2, 2]


def test_domain_balanced_sampler_infers_dataset_and_schema_targets():
    samples = [
        _sample("300W", "2d_68", "anchor"),
        _sample("WFLW", "2d_98", "profile"),
        _sample("cofw68", "2d_68", "occlusion"),
    ]
    sampler = DomainBalancedBatchSampler(
        samples,
        bucket_targets={"anchor": 1},
        dataset_targets={},
        schema_targets={},
        batch_size=3,
        seed=5,
    )

    assert sampler.resolved_targets()["dataset"] == {
        "cofw68": 1.0,
        "w300": 1.0,
        "wflw": 1.0,
    }
    assert sampler.resolved_targets()["schema"] == {"2d_68": 1.0, "2d_98": 1.0}


def test_domain_balanced_sampler_falls_back_for_sparse_targets():
    samples = [
        _sample("wflw", "2d_98", "profile"),
        _sample("cofw68", "2d_68", "occlusion"),
    ]
    sampler = DomainBalancedBatchSampler(
        samples,
        bucket_targets={"profile_occlusion": 1},
        dataset_targets={"missing": 1},
        schema_targets={"missing": 1},
        batch_size=4,
        seed=11,
    )

    batch = next(iter(sampler))

    assert len(batch) == 4
    assert {sample_bucket(samples[index]) for index in batch}.issubset({"profile", "occlusion"})
    diagnostics = sampler.last_epoch_diagnostics
    assert diagnostics["missing_targets"]["bucket"] == ["profile_occlusion"]
    assert diagnostics["missing_targets"]["dataset"] == ["missing"]
    assert diagnostics["missing_targets"]["schema"] == ["missing"]
    assert diagnostics["fallback_counts"]["bucket_to_any"] > 0


def test_legacy_domain_balanced_collate_reports_mix():
    image = torch.zeros(3, 4, 4)
    target = torch.zeros(68, 2)
    heatmap = torch.zeros(68, 2, 2)
    weight = torch.tensor(1.0)
    mask = torch.ones(68)
    batch = [
        (
            image,
            target,
            heatmap,
            weight,
            mask,
            {
                "dataset": "300W",
                "source_schema": "2d_68",
                "hard_negative_bucket": "large_yaw",
            },
        ),
        (
            image,
            target,
            heatmap,
            weight,
            mask,
            {
                "dataset": "cofw68",
                "source_schema": "2d_68",
                "hard_negative_bucket": "occluded",
            },
        ),
    ]

    collated = legacy_domain_balanced_collate(batch)

    assert len(collated) == 5
    assert batch_mix(collated) == {
        "bucket": {"profile": 1, "occlusion": 1},
        "dataset": {"w300": 1, "cofw68": 1},
        "schema": {"2d_68": 2},
    }
