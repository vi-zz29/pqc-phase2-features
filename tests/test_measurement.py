"""Tests for dimension_analysis.measurement — image-based mm recovery."""

import cv2
import numpy as np
import pytest

from dimension_analysis.dxf_parser import CADFeatureSet
from dimension_analysis.feature_matcher import MatchedPair
from dimension_analysis.measurement import (
    _detect_circle_in_roi,
    recover_dimensions,
)
from dimension_analysis.transform_estimator import TransformResult


def _make_hole_image(
    centers: list[tuple[int, int]],
    radius_px: int = 20,
    size: tuple[int, int] = (500, 300),
) -> np.ndarray:
    """Bright part with dark circular holes."""
    img = np.full(size, 220, dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), radius_px, 40, -1)
        cv2.circle(img, (cx, cy), radius_px, 0, 2)
    return img


def _transform_result(scale: float = 10.0) -> TransformResult:
    return TransformResult(
        matrix=np.eye(3, dtype=np.float64),
        scale_px_per_mm=scale,
        translation_px=(0.0, 0.0),
        rotation_deg=0.0,
        residual_error=0.0,
        refined=False,
    )


def _box_cad_features() -> CADFeatureSet:
    return CADFeatureSet(
        part_type="rectangular",
        overall_width=50.0,
        overall_height=80.0,
        hole_positions=[(0.0, 0.0), (30.0, 0.0)],
    )


def test_detect_circle_in_roi_finds_synthetic_hole():
    img = _make_hole_image([(250, 150)], radius_px=18)
    hit = _detect_circle_in_roi(img, 250.0, 150.0, expected_r_px=18.0, total_scale=10.0)
    assert hit is not None
    cx, cy, r_mm = hit
    assert abs(cx - 250.0) < 8.0
    assert abs(cy - 150.0) < 8.0
    assert abs(r_mm - 1.8) < 0.5


def test_hole_spacing_uses_hough_centres_not_cad_projection():
    """Measured spacing must come from image holes, not Stage-2 CAD projection."""
    scale = 10.0
    r_mm = 2.0
    r_px = r_mm * scale

    # Actual holes at 120 and 380 px → 26.0 mm spacing
    img = _make_hole_image([(120, 150), (380, 150)], radius_px=int(r_px))

    # Stage-2 projected centres (would give 30.0 mm if used directly)
    pair_a = MatchedPair(
        feature_type="circle",
        label="#1 hole",
        cad_value_mm=r_mm,
        image_value_px=r_px,
        cad_pos=(0.0, 0.0),
        image_pos_px=(100.0, 150.0),
        scale_px_per_mm=scale,
    )
    pair_b = MatchedPair(
        feature_type="circle",
        label="#2 hole",
        cad_value_mm=r_mm,
        image_value_px=r_px,
        cad_pos=(30.0, 0.0),
        image_pos_px=(400.0, 150.0),
        scale_px_per_mm=scale,
    )

    features = recover_dimensions(
        matched_pairs=[pair_a, pair_b],
        cad_features=_box_cad_features(),
        transform_result=_transform_result(scale),
        real_gray=img,
    )

    spacing = [f for f in features if f.feature_type == "hole_spacing"]
    assert len(spacing) == 1
    sp = spacing[0]
    assert sp.cad_dimension_mm == pytest.approx(30.0, abs=0.01)
    assert sp.measured_dimension_mm == pytest.approx(26.0, abs=1.5)
    assert abs(sp.deviation_mm) > 0.5


def test_hole_spacing_skipped_when_hough_fails():
    """No spacing row when holes cannot be detected in the image."""
    img = np.full((200, 200), 128, dtype=np.uint8)  # uniform — Hough fails
    pair = MatchedPair(
        feature_type="circle",
        label="#1 hole",
        cad_value_mm=2.0,
        image_value_px=20.0,
        cad_pos=(0.0, 0.0),
        image_pos_px=(50.0, 100.0),
        scale_px_per_mm=10.0,
    )
    pair_b = MatchedPair(
        feature_type="circle",
        label="#2 hole",
        cad_value_mm=2.0,
        image_value_px=20.0,
        cad_pos=(30.0, 0.0),
        image_pos_px=(350.0, 100.0),
        scale_px_per_mm=10.0,
    )

    features = recover_dimensions(
        matched_pairs=[pair, pair_b],
        cad_features=_box_cad_features(),
        transform_result=_transform_result(),
        real_gray=img,
    )

    assert not any(f.feature_type == "hole_spacing" for f in features)
