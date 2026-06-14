from __future__ import annotations

import cv2
import numpy as np
import pytest

from lib.training import augmentation
from lib.training.config import validate_roll_augmentation_probs


def test_roll_rotation_distribution_targets_horizontal_faces():
    assert augmentation.roll_rotation_distribution() == {
        0: pytest.approx(0.5),
        -90: pytest.approx(0.2),
        90: pytest.approx(0.2),
        -45: pytest.approx(0.05),
        45: pytest.approx(0.05),
    }


def test_roll_rotation_precedes_existing_affine_jitter():
    albumentations = pytest.importorskip("albumentations")

    transform = augmentation.GetAugTransform()
    roll_rotation, affine = transform.transforms[:2]

    assert isinstance(roll_rotation, albumentations.OneOf)
    assert [
        (type(child).__name__, getattr(child, "rotate", None), child.p)
        for child in roll_rotation.transforms
    ] == [
        ("NoOp", None, 0.5),
        ("Affine", (-90.0, -90.0), 0.2),
        ("Affine", (90.0, 90.0), 0.2),
        ("Affine", (-45.0, -45.0), 0.05),
        ("Affine", (45.0, 45.0), 0.05),
    ]
    assert affine.rotate == (-20.0, 20.0)


def test_forced_quarter_turn_keeps_keypoints_and_masks_aligned():
    albumentations = pytest.importorskip("albumentations")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (64, 32), 5, 255, -1)
    transform = albumentations.Compose(
        [augmentation._roll_rotation_transform(1.0, 0.0, 1.0)],
        keypoint_params=albumentations.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )
    transform.set_random_seed(0)

    transformed = transform(image=image, keypoints=[(64.0, 32.0)], mask=mask)
    x, y = transformed["keypoints"][0]

    assert transformed["mask"][round(y), round(x)] == 255
    assert (x, y) == pytest.approx((32.0, 191.0))


@pytest.mark.parametrize(
    ("quarter_turn", "diagonal"),
    [(-0.1, 0.1), (0.4, -0.1), (1.1, 0.0), (0.8, 0.3)],
)
def test_roll_rotation_probabilities_reject_invalid_values(quarter_turn, diagonal):
    with pytest.raises(ValueError):
        validate_roll_augmentation_probs(quarter_turn, diagonal)
