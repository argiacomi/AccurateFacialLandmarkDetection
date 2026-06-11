import numpy as np

from lib.datasets.loader_geometry import simulate_loader_geometry


def test_loader_geometry_ok_inside_image():
    pts = np.stack(
        [np.linspace(40, 180, 106), np.linspace(45, 190, 106)],
        axis=1,
    ).astype(np.float32)

    diag = simulate_loader_geometry(pts, (256, 256))

    assert diag["ok"] is True
    assert diag["padding"] == 0.0
    assert diag["landmarks_outside_image"] is False


def test_loader_geometry_flags_unreasonable_padding():
    pts = np.stack(
        [np.linspace(75, 280, 106), np.linspace(344, 1163, 106)],
        axis=1,
    ).astype(np.float32)

    diag = simulate_loader_geometry(pts, (256, 256))

    assert diag["ok"] is False
    assert diag["reason"] == "unreasonable_loader_padding"
    assert diag["landmarks_outside_image"] is True
    assert diag["padded_shape"][0] > 2048 or diag["padded_shape"][1] > 2048
