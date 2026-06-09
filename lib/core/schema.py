#!/usr/bin/env python3
"""Canonical landmark schema helpers for CD-ViT manifest tooling."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field

import numpy as np

CANONICAL_SCHEMA = "2d_68"

MAP_98_TO_68 = np.array(
    [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        33,
        34,
        35,
        36,
        37,
        42,
        43,
        44,
        45,
        46,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        63,
        64,
        65,
        67,
        68,
        69,
        71,
        72,
        73,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
    ],
    dtype=np.int64,
)

# Audited 106 -> 68 semantic subsampling for the standard 106-point markup shared
# by LaPa and the JD-landmark Grand Challenge (re-annotated 300W sources), and the
# FLL2/FLL3 derivatives that use the same ``2d_106`` layout. 1-based source indices
# into the 106-point array, emitted in canonical 300W 68-point order. Derived from
# the published LaPa/300W semantic-group correspondence (MDMD supplement).
_LAPA_106_TO_68_1B = (
    # jaw 1-17 (every other point of the 33-pt contour)
    1,
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
    23,
    25,
    27,
    29,
    31,
    33,
    # eyebrows 18-27
    34,
    35,
    36,
    37,
    38,
    43,
    44,
    45,
    46,
    47,
    # nose 28-36
    52,
    53,
    54,
    55,
    58,
    59,
    61,
    63,
    64,
    # eyes 37-48
    67,
    68,
    70,
    71,
    72,
    74,
    76,
    77,
    79,
    80,
    81,
    83,
    # outer mouth 49-60
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    # inner mouth 61-68
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
)
# 0-based indices used for array gathering.
MAP_106_TO_68 = np.asarray(_LAPA_106_TO_68_1B, dtype=np.int64) - 1

PROJECTION_MAPS_TO_68: dict[str, str] = {
    "2d_68": "identity",
    "2d_98": "MAP_98_TO_68",
    "2d_106": "MAP_106_TO_68",
}

SCHEMAS_WITHOUT_VERIFIED_68_PROJECTION = frozenset(
    {
        "2d_29",
        "2d_39",
        "2d_194",
        "menpo2d_profile_39",
        "multipie_profile_39",
    }
)


@dataclass(frozen=True)
class LandmarkSchema:
    """Description of a supported landmark layout."""

    name: str
    points: int
    dimensions: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.points, self.dimensions)


SUPPORTED_SCHEMAS: dict[str, LandmarkSchema] = {
    "2d_4": LandmarkSchema("2d_4", 4, 2),
    "2d_29": LandmarkSchema("2d_29", 29, 2),
    "2d_39": LandmarkSchema("2d_39", 39, 2),
    "2d_51": LandmarkSchema("2d_51", 51, 2),
    "2d_68": LandmarkSchema("2d_68", 68, 2),
    "2d_98": LandmarkSchema("2d_98", 98, 2),
    "2d_106": LandmarkSchema("2d_106", 106, 2),
    "2d_194": LandmarkSchema("2d_194", 194, 2),
    "3d_26": LandmarkSchema("3d_26", 26, 3),
    "menpo2d_profile_39": LandmarkSchema("menpo2d_profile_39", 39, 2),
    "multipie_profile_39": LandmarkSchema("multipie_profile_39", 39, 2),
}

DEFAULT_SCHEMA_HEADS: dict[str, int] = {
    "landmarks_68": 68,
    "landmarks_98": 98,
    "landmarks_106": 106,
    "landmarks_194": 194,
    "profile39": 39,
    "landmarks_29": 29,
}

_SCHEMA_ALIASES = {
    "4": "2d_4",
    "4pt": "2d_4",
    "lm_2d_4": "2d_4",
    "29": "2d_29",
    "29pt": "2d_29",
    "lm_2d_29": "2d_29",
    "39": "2d_39",
    "39pt": "2d_39",
    "profile39": "2d_39",
    "profile_39": "2d_39",
    "lm_2d_39": "2d_39",
    "menpo2d_39": "menpo2d_profile_39",
    "menpo2d_profile39": "menpo2d_profile_39",
    "multipie_39": "multipie_profile_39",
    "multipie_profile39": "multipie_profile_39",
    "51": "2d_51",
    "51pt": "2d_51",
    "lm_2d_51": "2d_51",
    "68": "2d_68",
    "68pt": "2d_68",
    "canonical": "2d_68",
    "lm_2d_68": "2d_68",
    "98": "2d_98",
    "98pt": "2d_98",
    "lm_2d_98": "2d_98",
    "106": "2d_106",
    "106pt": "2d_106",
    "lm_2d_106": "2d_106",
    "194": "2d_194",
    "194pt": "2d_194",
    "lm_2d_194": "2d_194",
    "26": "3d_26",
    "26pt3d": "3d_26",
    "lm_3d_26": "3d_26",
}

WFLW_98_FLIP = np.array(
    [
        32,
        31,
        30,
        29,
        28,
        27,
        26,
        25,
        24,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
        46,
        45,
        44,
        43,
        42,
        50,
        49,
        48,
        47,
        37,
        36,
        35,
        34,
        33,
        41,
        40,
        39,
        38,
        51,
        52,
        53,
        54,
        59,
        58,
        57,
        56,
        55,
        72,
        71,
        70,
        69,
        68,
        75,
        74,
        73,
        64,
        63,
        62,
        61,
        60,
        67,
        66,
        65,
        82,
        81,
        80,
        79,
        78,
        77,
        76,
        87,
        86,
        85,
        84,
        83,
        92,
        91,
        90,
        89,
        88,
        95,
        94,
        93,
        97,
        96,
    ],
    dtype=np.int64,
)

SCHEMAS_WITHOUT_VERIFIED_FLIP_MAPS = frozenset(
    {"2d_39", "menpo2d_profile_39", "multipie_profile_39"}
)

SCHEMA_FLIP_MAPS: dict[str, np.ndarray] = {
    "2d_68": np.array(
        [
            16,
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            26,
            25,
            24,
            23,
            22,
            21,
            20,
            19,
            18,
            17,
            27,
            28,
            29,
            30,
            35,
            34,
            33,
            32,
            31,
            45,
            44,
            43,
            42,
            47,
            46,
            39,
            38,
            37,
            36,
            41,
            40,
            54,
            53,
            52,
            51,
            50,
            49,
            48,
            59,
            58,
            57,
            56,
            55,
            64,
            63,
            62,
            61,
            60,
            67,
            66,
            65,
        ],
        dtype=np.int64,
    ),
    "2d_98": WFLW_98_FLIP,
}


@dataclass(frozen=True, init=False)
class LandmarkPrediction:
    """A single adapter prediction with schema and coordinate metadata."""

    landmarks: np.ndarray
    confidence: np.ndarray | None = None
    model_name: str = ""
    source_landmark_count: int = 68
    coordinate_space: str = "frame"
    metadata: dict[str, T.Any] = field(default_factory=dict)
    schema: str = CANONICAL_SCHEMA

    def __init__(
        self,
        landmarks: T.Sequence[T.Sequence[float]] | np.ndarray | None = None,
        *,
        points: T.Sequence[T.Sequence[float]] | np.ndarray | None = None,
        schema: str | object | None = None,
        confidence: np.ndarray | None = None,
        model_name: str = "",
        source: str | None = None,
        source_landmark_count: int | None = None,
        coordinate_space: str = "frame",
        metadata: dict[str, T.Any] | None = None,
    ) -> None:
        if landmarks is None:
            if points is None:
                raise ValueError("landmarks are required")
            landmarks = points
        elif points is not None:
            raise ValueError("provide either landmarks or points, not both")

        raw = np.asarray(landmarks, dtype="float32")
        schema_name = (
            infer_schema(raw) if schema is None else canonicalize_schema(schema)
        )
        points_array = normalize_landmark_array(raw, schema=schema_name)
        name = model_name if source is None else source
        if source_landmark_count is None:
            source_landmark_count = points_array.shape[0]
        if source_landmark_count <= 0:
            raise ValueError("source_landmark_count must be greater than zero")
        if not coordinate_space.strip():
            raise ValueError("coordinate_space cannot be empty")

        object.__setattr__(self, "landmarks", points_array)
        object.__setattr__(self, "schema", schema_name)
        object.__setattr__(self, "model_name", name)
        object.__setattr__(self, "source_landmark_count", int(source_landmark_count))
        object.__setattr__(self, "coordinate_space", coordinate_space.strip())
        object.__setattr__(self, "metadata", {} if metadata is None else dict(metadata))
        if confidence is None:
            object.__setattr__(self, "confidence", None)
        else:
            conf = np.asarray(confidence, dtype="float32")
            if conf.shape != (points_array.shape[0],):
                raise ValueError(
                    "confidence must be a 1D array with one value per landmark point: "
                    f"expected {(points_array.shape[0],)}, got {conf.shape}"
                )
            if not np.all(np.isfinite(conf)):
                raise ValueError("confidence contains NaN or infinite values")
            object.__setattr__(self, "confidence", conf)

    @property
    def points(self) -> np.ndarray:
        return self.landmarks

    @property
    def source(self) -> str:
        return self.model_name

    def canonical_68(self) -> "LandmarkPrediction":
        points = to_canonical_68(self.landmarks, source_schema=self.schema)
        confidence = None
        if self.confidence is not None and self.schema == CANONICAL_SCHEMA:
            confidence = self.confidence.copy()
        return LandmarkPrediction(
            landmarks=points,
            schema=CANONICAL_SCHEMA,
            confidence=confidence,
            model_name=self.model_name,
            source_landmark_count=self.source_landmark_count,
            coordinate_space=self.coordinate_space,
            metadata=self.metadata,
        )


def canonicalize_schema(schema: str | object) -> str:
    raw = str(schema.name) if hasattr(schema, "name") else str(schema)
    key = raw.strip().lower().replace("-", "_")
    if key in SUPPORTED_SCHEMAS:
        return key
    if key in _SCHEMA_ALIASES:
        return _SCHEMA_ALIASES[key]
    raise ValueError(
        f"Unsupported landmark schema '{schema}'. Supported schemas: {sorted(SUPPORTED_SCHEMAS)}"
    )


def infer_schema(points: np.ndarray) -> str:
    if points.ndim != 2:
        raise ValueError(f"landmarks must be 2D, got shape {points.shape}")
    matches = [
        schema.name
        for schema in SUPPORTED_SCHEMAS.values()
        if points.shape == schema.shape
    ]
    if not matches:
        raise ValueError(f"Cannot infer landmark schema from shape {points.shape}")
    return matches[0]


def normalize_landmark_array(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    *,
    schema: str | object | None = None,
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim == 1:
        if array.size % 2 != 0:
            raise ValueError(
                f"flat landmark arrays must contain x/y pairs, got {array.size} values"
            )
        array = array.reshape((-1, 2))
    if array.ndim != 2:
        raise ValueError(f"landmarks must be a 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("landmarks contain NaN or infinite values")
    if schema is not None:
        schema_name = canonicalize_schema(schema)
        expected = SUPPORTED_SCHEMAS[schema_name].shape
        if array.shape != expected:
            raise ValueError(
                f"landmarks for schema '{schema_name}' must have shape {expected}, got {array.shape}"
            )
    elif array.shape[1] not in (2, 3):
        raise ValueError(f"landmarks must have 2 or 3 dimensions, got {array.shape[1]}")
    return np.ascontiguousarray(array, dtype=dtype)


def flip_map_for_schema(schema: str | object) -> np.ndarray:
    schema_name = canonicalize_schema(schema)
    if schema_name not in SCHEMA_FLIP_MAPS:
        raise ValueError(f"No flip map registered for schema '{schema_name}'")
    return SCHEMA_FLIP_MAPS[schema_name].copy()


def has_verified_flip_map(schema: str | object) -> bool:
    schema_name = canonicalize_schema(schema)
    return schema_name in SCHEMA_FLIP_MAPS


def head_name_for_schema(schema: str | object) -> str:
    schema_name = canonicalize_schema(schema)
    if schema_name == "2d_68":
        return "landmarks_68"
    if schema_name == "2d_98":
        return "landmarks_98"
    if schema_name == "2d_106":
        return "landmarks_106"
    if schema_name == "2d_194":
        return "landmarks_194"
    if schema_name == "2d_29":
        return "landmarks_29"
    if schema_name in {"2d_39", "menpo2d_profile_39", "multipie_profile_39"}:
        return "profile39"
    raise ValueError(
        f"Schema '{schema_name}' is not trainable by the CD-ViT multi-head path"
    )


def point_count_for_schema(schema: str | object) -> int:
    return SUPPORTED_SCHEMAS[canonicalize_schema(schema)].points


def projection_audit_for_schema(
    source_schema: str | object,
    *,
    target_schema: str | object = CANONICAL_SCHEMA,
) -> dict[str, T.Any]:
    """Return the audited projection status for a source schema.

    This intentionally separates native multi-head training support from
    canonical 68-point projection support. New dense/native schemas can train
    on their own heads before a reviewed overlap map exists.
    """

    source = canonicalize_schema(source_schema)
    target = canonicalize_schema(target_schema)
    if source == target:
        return {
            "status": "native",
            "source_schema": source,
            "target_schema": target,
            "map": "identity",
        }
    if target != CANONICAL_SCHEMA:
        return {
            "status": "unsupported_target",
            "source_schema": source,
            "target_schema": target,
            "reason": "projection audit is only defined for canonical 68-point targets",
        }
    if source in PROJECTION_MAPS_TO_68:
        return {
            "status": "audited",
            "source_schema": source,
            "target_schema": target,
            "map": PROJECTION_MAPS_TO_68[source],
        }
    if source in SCHEMAS_WITHOUT_VERIFIED_68_PROJECTION:
        return {
            "status": "not_projectable",
            "source_schema": source,
            "target_schema": target,
            "reason": "no audited 68-point overlap map is registered",
        }
    return {
        "status": "unknown_schema",
        "source_schema": source,
        "target_schema": target,
        "reason": "schema has no projection audit entry",
    }


def to_canonical_68(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    *,
    source_schema: str | object | None = None,
) -> np.ndarray:
    array = normalize_landmark_array(points, schema=source_schema)
    schema = (
        infer_schema(array)
        if source_schema is None
        else canonicalize_schema(source_schema)
    )
    if schema == CANONICAL_SCHEMA:
        return array[:, :2].astype("float32", copy=True)
    if schema == "2d_98":
        return array[MAP_98_TO_68, :2].astype("float32", copy=True)
    if schema == "2d_106":
        return array[MAP_106_TO_68, :2].astype("float32", copy=True)
    raise ValueError(f"Cannot map schema '{schema}' to canonical 68-point landmarks")


def normalize_landmarks(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    *,
    source_schema: str | object | None = None,
    target_schema: str = CANONICAL_SCHEMA,
) -> np.ndarray:
    target = canonicalize_schema(target_schema)
    if target != CANONICAL_SCHEMA:
        raise ValueError(f"Unsupported target landmark schema '{target_schema}'")
    return to_canonical_68(points, source_schema=source_schema)
