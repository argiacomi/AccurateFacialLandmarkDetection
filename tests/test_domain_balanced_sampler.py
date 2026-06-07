from lib.landmarks.training.domain_balanced_sampler import (
    DomainBalancedBatchSampler,
    parse_target_spec,
    sample_bucket,
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


def test_domain_balanced_sampler_is_reproducible():
    samples = [
        _sample("wflw", "2d_98", "profile"),
        _sample("cofw", "2d_68", "occlusion"),
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

    assert first == second
    assert sorted(first[0]) == [0, 1, 2, 3]


def test_domain_balanced_sampler_falls_back_for_sparse_targets():
    samples = [
        _sample("wflw", "2d_98", "profile"),
        _sample("cofw", "2d_68", "occlusion"),
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
