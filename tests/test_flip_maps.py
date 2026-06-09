from __future__ import annotations

import numpy as np
import pytest

from lib.core import schema as S


@pytest.mark.parametrize("schema", ["2d_29", "2d_68", "2d_98", "2d_106", "2d_194"])
def test_flip_map_is_involutive_permutation(schema):
    flip = S.flip_map_for_schema(schema)
    n = S.point_count_for_schema(schema)
    assert flip.shape == (n,)
    assert int(flip.min()) >= 0 and int(flip.max()) < n
    assert len(set(int(x) for x in flip)) == n  # permutation
    assert np.array_equal(flip[flip], np.arange(n))  # involutive
    assert S.has_verified_flip_map(schema)


@pytest.mark.parametrize(
    ("schema", "index", "expected"),
    [
        ("2d_29", 0, 1),
        ("2d_29", 20, 20),  # fixed point
        ("2d_106", 0, 32),
        ("2d_106", 16, 16),  # nose-tip column, fixed
        ("2d_106", 38, 50),
        ("2d_194", 0, 40),
        ("2d_194", 114, 134),  # group swap
        ("2d_194", 20, 20),  # fixed point
    ],
)
def test_flip_map_specific_correspondences(schema, index, expected):
    assert int(S.flip_map_for_schema(schema)[index]) == expected


def test_flip_map_returns_independent_copy():
    a = S.flip_map_for_schema("2d_106")
    a[0] = 999
    assert int(S.flip_map_for_schema("2d_106")[0]) == 32


def test_schema_2d_39_has_no_verified_flip_map():
    assert not S.has_verified_flip_map("2d_39")
    with pytest.raises(ValueError, match="No flip map registered"):
        S.flip_map_for_schema("2d_39")


def test_flipping_landmarks_twice_is_identity():
    flip = S.flip_map_for_schema("2d_106")
    pts = np.random.default_rng(0).random((106, 2)).astype(np.float32)
    once = pts[flip]
    twice = once[flip]
    assert np.array_equal(twice, pts)


def test_validate_flip_map_rejects_bad_maps():
    n = 4
    with pytest.raises(ValueError, match="shape"):
        S._validate_flip_map("x", np.array([0, 1, 2], dtype=np.int64), n)
    with pytest.raises(ValueError, match="out-of-range"):
        S._validate_flip_map("x", np.array([0, 1, 2, 9], dtype=np.int64), n)
    with pytest.raises(ValueError, match="permutation"):
        S._validate_flip_map("x", np.array([0, 1, 1, 2], dtype=np.int64), n)
    with pytest.raises(ValueError, match="involutive"):
        # valid permutation but not its own inverse (3-cycle 0->1->2->0)
        S._validate_flip_map("x", np.array([1, 2, 0, 3], dtype=np.int64), n)
