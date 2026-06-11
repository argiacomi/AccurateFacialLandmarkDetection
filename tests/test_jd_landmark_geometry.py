import cv2
import numpy as np
import pytest

from lib.datasets.build.jd_landmark import _jd_resolve_loader_geometry_path


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _points(x0, x1, y0, y1):
    return np.stack(
        [np.linspace(x0, x1, 106), np.linspace(y0, y1, 106)],
        axis=1,
    ).astype(np.float32)


def test_jd_geometry_path_a_native_fit(tmp_path):
    img = tmp_path / "image.png"
    _image(img)
    pts = _points(30, 200, 40, 220)

    selected_image, selected_points, meta = _jd_resolve_loader_geometry_path(
        output_dir=tmp_path / "out",
        sample_id="native_fit",
        image=img,
        points=pts,
        bbox=None,
        bbox_source="none",
    )

    assert selected_image == img
    assert np.array_equal(selected_points, pts)
    assert meta["loader_geometry_policy"] == "native_fit"


def test_jd_geometry_path_b_bbox_crop_fit(tmp_path):
    img = tmp_path / "image.png"
    _image(img)
    pts = _points(75, 280, 344, 1163)

    selected_image, selected_points, meta = _jd_resolve_loader_geometry_path(
        output_dir=tmp_path / "out",
        sample_id="bbox_crop_fit",
        image=img,
        points=pts,
        bbox=[0, 0, 300, 1200],
        bbox_source="synthetic_bbox",
    )

    assert selected_image != img
    assert selected_image.is_file()
    assert selected_points.shape == pts.shape
    assert meta["loader_geometry_policy"] == "bbox_crop_fit"
    assert meta["loader_geometry_bbox_crop"]["ok"] is True


def test_jd_geometry_path_c_skip_without_valid_fallback(tmp_path):
    img = tmp_path / "image.png"
    _image(img)
    pts = _points(75, 280, 344, 1163)

    with pytest.raises(ValueError, match="invalid loader geometry"):
        _jd_resolve_loader_geometry_path(
            output_dir=tmp_path / "out",
            sample_id="skip",
            image=img,
            points=pts,
            bbox=None,
            bbox_source="none",
        )
