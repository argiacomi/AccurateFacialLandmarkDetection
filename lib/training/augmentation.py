# ruff: noqa: E402
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import random
import time

import numpy as np
from PIL import Image

from lib.training.config import (
    DEFAULT_ROLL_DIAGONAL_PROB,
    DEFAULT_ROLL_QUARTER_TURN_PROB,
    validate_roll_augmentation_probs,
)

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None


def _require_albumentations():
    if A is None:
        raise ModuleNotFoundError("albumentations is required for image augmentation")


def roll_rotation_distribution(
    quarter_turn_prob=DEFAULT_ROLL_QUARTER_TURN_PROB,
    diagonal_prob=DEFAULT_ROLL_DIAGONAL_PROB,
    prob_factor=1.0,
):
    factor = float(prob_factor)
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"prob_factor must be between 0 and 1, got {factor}")
    quarter_turn, diagonal = validate_roll_augmentation_probs(
        quarter_turn_prob,
        diagonal_prob,
    )
    quarter_turn *= factor
    diagonal *= factor
    return {
        0: 1.0 - quarter_turn - diagonal,
        -90: quarter_turn / 2.0,
        90: quarter_turn / 2.0,
        -45: diagonal / 2.0,
        45: diagonal / 2.0,
    }


def _roll_rotation_transform(quarter_turn_prob, diagonal_prob, prob_factor):
    distribution = roll_rotation_distribution(
        quarter_turn_prob,
        diagonal_prob,
        prob_factor,
    )
    transforms = []
    for angle, probability in distribution.items():
        if probability <= 0.0:
            continue
        transform = (
            A.NoOp(p=probability)
            if angle == 0
            else A.Affine(rotate=(angle, angle), p=probability)
        )
        transforms.append(transform)
    return A.OneOf(transforms, p=1.0)


def GetAugTransform(
    prob_factor=1.0,
    *,
    roll_quarter_turn_prob=DEFAULT_ROLL_QUARTER_TURN_PROB,
    roll_diagonal_prob=DEFAULT_ROLL_DIAGONAL_PROB,
):
    _require_albumentations()
    roll_rotation = _roll_rotation_transform(
        roll_quarter_turn_prob,
        roll_diagonal_prob,
        prob_factor,
    )
    affine_worker = A.Affine(
        scale={"x": [0.8, 1.2], "y": [0.8, 1.2]},
        translate_px={"x": [-40, 40], "y": [-40, 40]},
        rotate=[-20, 20],
        shear=[-5, 5],
        keep_ratio=False,
        p=0.4 * prob_factor,
    )

    color_jitter = A.ColorJitter(p=0.3 * prob_factor)
    gauss_noise = A.GaussNoise((100 / 255.0, 201 / 255.0), p=0.3 * prob_factor)
    # gauss_noise = A.GaussNoise((100, 201), p=0.3 * prob_factor)
    gauss_blur = A.GaussianBlur((5, 19), p=0.1 * prob_factor)
    gamma_correct = A.RandomGamma(p=0.2 * prob_factor)
    gravel = A.RandomGravel(number_of_patches=2, p=0.1 * prob_factor)
    shadow = A.RandomShadow(p=0.2 * prob_factor)
    rain = A.RandomRain(drop_length=3, blur_value=3, p=0.2 * prob_factor)
    # bright_ness = A.RandomBrightness(p=0.2 * prob_factor)
    bright_contrast = A.RandomBrightnessContrast(p=0.2 * prob_factor)
    gray = A.ToGray(p=0.3 * prob_factor)
    perspective = A.Perspective((0.01, 0.1), p=0.4 * prob_factor)
    # contrast = A.RandomContrast(p=0.2)

    transform = A.Compose(
        [
            roll_rotation,
            affine_worker,
            color_jitter,
            gauss_noise,
            gauss_blur,
            gamma_correct,
            gravel,
            shadow,
            rain,
            # bright_ness,
            bright_contrast,
            # contrast,
            gray,
            perspective,
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    return transform


def GetAugTransform_2(
    prob_factor=1.0,
    *,
    roll_quarter_turn_prob=DEFAULT_ROLL_QUARTER_TURN_PROB,
    roll_diagonal_prob=DEFAULT_ROLL_DIAGONAL_PROB,
):
    _require_albumentations()
    roll_rotation = _roll_rotation_transform(
        roll_quarter_turn_prob,
        roll_diagonal_prob,
        prob_factor,
    )
    affine_worker = A.Affine(
        scale={"x": [0.9, 1.1], "y": [0.9, 1.1]},
        translate_px={"x": [-20, 20], "y": [-20, 20]},
        rotate=[-18, 18],
        # shear=[-5, 5],
        keep_ratio=False,
        p=0.4 * prob_factor,
    )

    color_jitter = A.ColorJitter(p=0.3 * prob_factor)
    # gauss_noise = A.GaussNoise((100, 201), p=0.3 * prob_factor)
    gauss_blur = A.GaussianBlur((5, 19), p=0.1 * prob_factor)
    gamma_correct = A.RandomGamma(p=0.2 * prob_factor)
    gravel = A.RandomGravel(number_of_patches=2, p=0.1 * prob_factor)
    shadow = A.RandomShadow(p=0.2 * prob_factor)
    rain = A.RandomRain(drop_length=3, blur_value=3, p=0.2 * prob_factor)
    bright_ness = A.RandomBrightness(p=0.2 * prob_factor)
    bright_contrast = A.RandomBrightnessContrast(p=0.2 * prob_factor)
    gray = A.ToGray(p=0.3 * prob_factor)
    perspective = A.Perspective((0.01, 0.1), p=0.4 * prob_factor)
    # contrast = A.RandomContrast(p=0.2)

    transform = A.Compose(
        [
            roll_rotation,
            affine_worker,
            color_jitter,
            # gauss_noise,
            gauss_blur,
            gamma_correct,
            gravel,
            shadow,
            rain,
            bright_ness,
            bright_contrast,
            # contrast,
            gray,
            perspective,
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    return transform


if __name__ == "__main__":
    random.seed(int(time.time()))

    img = cv2.imread("/home/mm/Desktop/c.png")[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    affine_worker = A.Affine(
        scale=[0.8, 1.2],
        translate_px={"x": [-80, 80], "y": [-80, 80]},
        rotate=[-20, 20],
        shear=[-10, 10],
        keep_ratio=True,
        p=1,
    )

    color_jitter = A.ColorJitter(p=0.3)
    gauss_noise = A.GaussNoise((200, 201), p=0.3)
    gauss_blur = A.GaussianBlur((5, 19), p=0.3)
    gamma_correct = A.RandomGamma(p=0.3)
    gravel = A.RandomGravel(number_of_patches=2, p=0.1)
    shadow = A.RandomShadow(p=0.3)
    rain = A.RandomRain(drop_length=3, blur_value=3, p=0.3)
    bright_ness = A.RandomBrightness(p=0.3)
    bright_contrast = A.RandomBrightnessContrast(p=0.3)
    gray = A.ToGray(p=0.3)
    contrast = A.RandomContrast(p=0.3)

    transform = A.Compose(
        [
            A.Perspective((0.02, 0.021), p=1),
            # affine_worker,
            # color_jitter,
            # gauss_noise,
            # gauss_blur,
            # gamma_correct,
            # gravel,
            # shadow,
            # rain,
            # bright_ness,
            # bright_contrast,
            # contrast,
            # gray,
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    pos = np.random.rand(10, 2) * 511.0

    transformed = transform(image=img, keypoints=pos)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]

    for i in range(pos.shape[0]):
        p = pos[i]
        x, y = round(float(p[0])), round(float(p[1]))
        cv2.circle(img, (x, y), 2, (0, 225, 255), 2)

    for p in transformed_keypoints:
        x, y = round(float(p[0])), round(float(p[1]))
        cv2.circle(transformed_image, (x, y), 2, (0, 225, 255), 2)

    res = np.concatenate([transformed_image, img], axis=1)

    x = Image.fromarray(res)
    x.show()
