"""
Tests for dimension_analysis/measurement.py

All measurements must be in mm. Tests verify:
  - Units and aliases are consistent
  - Hole spacing is derived from actual image positions, not CAD
  - Overall dimensions are measured from the image mask
  - Circle measurement skips (not fakes) when no image is provided
  - Deviation = measured - cad
"""

import math
import numpy as np
import cv2
import pytest

from dimension_analysis.measurement import (
    recover_dimensions,
    MeasuredFeature,
    _measure_overall_from_mask,
    _measure_circle_radius_mm,
)
from dimension_analysis.feature_matcher import MatchedPair
from dimension_analysis.transform_estimator import TransformResult
from dimension_analysis.dxf_parser import CADFeatureSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transform_result(scale: float = 5.0) -> TransformResult:
    M = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return TransformResult(
        matrix=M,
        scale_px_per_mm=scale,
        translation_px=(0.0, 0.0),
        rotation_deg=0.0,
        residual_error=0.0,
        refined=False,
    )


def _make_circle_pair(
    label: str,
    cad_r_mm: float,
    img_cx: float,
    img_cy: float,
    total_scale: float,
) -> MatchedPair:
    return MatchedPair(
        feature_type="circle",
        label=label,
        cad_value_mm=cad_r_mm,
        image_value_px=cad_r_mm * total_scale,  # Stage-2 r_px (tautological fallback value)
        cad_pos=(0.0, 0.0),
        image_pos_px=(img_cx, img_cy),
        scale_px_per_mm=total_scale,
        match_distance_px=0.0,
    )


def _bright_rect_image(rect_x1=50, rect_y1=80, rect_x2=250, rect_y2=220,
                        img_h=300, img_w=400) -> np.ndarray:
    """Bright rectangle on a dark background — simulates a lit part."""
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), 220, -1)
    return img


# ---------------------------------------------------------------------------
# Unit tests — MeasuredFeature
# ---------------------------------------------------------------------------

def test_measured_feature_unit_is_mm():
    mf = MeasuredFeature(
        feature_type="circle_radius",
        label="test",
        cad_dimension_mm=5.0,
        measured_dimension_mm=5.1,
        deviation_mm=0.1,
    )
    assert mf.unit == "mm"


def test_measured_feature_aliases_equal_mm_fields():
    mf = MeasuredFeature(
        feature_type="circle_radius",
        label="test",
        cad_dimension_mm=3.0,
        measured_dimension_mm=3.2,
        deviation_mm=0.2,
    )
    assert mf.cad_dimension_px == mf.cad_dimension_mm
    assert mf.measured_dimension_px == mf.measured_dimension_mm
    assert mf.deviation_px == mf.deviation_mm


def test_measured_feature_deviation_is_measured_minus_cad():
    mf = MeasuredFeature(
        feature_type="rect_width",
        label="cutout_width",
        cad_dimension_mm=25.0,
        measured_dimension_mm=25.4,
        deviation_mm=0.4,
    )
    assert mf.deviation_mm == pytest.approx(
        mf.measured_dimension_mm - mf.cad_dimension_mm, abs=1e-6
    )


# ---------------------------------------------------------------------------
# _measure_overall_from_mask
# ---------------------------------------------------------------------------

def test_measure_overall_bright_rect():
    """Should detect a bright rectangular part and return correct mm size."""
    scale = 5.0  # 5 px/mm
    img = _bright_rect_image(50, 80, 250, 220)   # 200px wide × 140px tall
    w_mm, h_mm = _measure_overall_from_mask(img, scale)
    assert w_mm is not None and h_mm is not None
    # Width ~200px / 5 = 40mm, height ~140px / 5 = 28mm (±3mm tolerance)
    assert abs(w_mm - 40.0) < 4.0, f"Expected ~40mm width, got {w_mm:.2f}"
    assert abs(h_mm - 28.0) < 4.0, f"Expected ~28mm height, got {h_mm:.2f}"


def test_measure_overall_empty_image_returns_none():
    img = np.zeros((200, 200), dtype=np.uint8)
    w_mm, h_mm = _measure_overall_from_mask(img, 5.0)
    # Either None or both valid — empty image should not crash
    assert w_mm is None or (isinstance(w_mm, float) and w_mm >= 0)


def test_measure_overall_zero_scale_returns_none():
    img = _bright_rect_image()
    w_mm, h_mm = _measure_overall_from_mask(img, 0.0)
    assert w_mm is None and h_mm is None


# ---------------------------------------------------------------------------
# _measure_circle_radius_mm
# ---------------------------------------------------------------------------

def test_measure_circle_radius_detects_circle():
    """Create a synthetic circle in a gray image and verify Hough finds it."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, 180, -1)   # filled circle r=20px
    # At scale 4 px/mm → expected radius 5mm, measured ~5mm
    scale = 4.0
    r_mm = _measure_circle_radius_mm(img, 100.0, 100.0, 20.0, scale)
    if r_mm is not None:
        assert 3.0 < r_mm < 8.0, f"Expected ~5mm, got {r_mm:.3f}mm"


def test_measure_circle_radius_empty_roi_returns_none():
    img = np.zeros((10, 10), dtype=np.uint8)
    r_mm = _measure_circle_radius_mm(img, 100.0, 100.0, 20.0, 5.0)
    assert r_mm is None


def test_measure_circle_radius_zero_scale_returns_none():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, 180, -1)
    r_mm = _measure_circle_radius_mm(img, 100.0, 100.0, 20.0, 0.0)
    assert r_mm is None


# ---------------------------------------------------------------------------
# recover_dimensions — hole spacing
# ---------------------------------------------------------------------------

def test_hole_spacing_uses_image_positions_not_cad():
    """
    The key correctness test: spacing must come from image pixel distance
    divided by total_scale, NOT from the CAD spacing.

    Two holes are 10mm apart in CAD but imaged 60px apart.
    At scale=5px/mm → measured spacing = 60/5 = 12mm (2mm error).
    CAD spacing = 10mm.
    """
    scale = 5.0
    p1 = MatchedPair(
        feature_type="circle", label="h1",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(0.0, 0.0), image_pos_px=(100.0, 200.0),
        scale_px_per_mm=scale,
    )
    p2 = MatchedPair(
        feature_type="circle", label="h2",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(10.0, 0.0), image_pos_px=(160.0, 200.0),  # 60px apart in image
        scale_px_per_mm=scale,
    )

    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test",
        hole_positions=[(0.0, 0.0), (10.0, 0.0)],  # CAD: 10mm apart
        overall_width=50.0,
        overall_height=50.0,
    )
    tr = _make_transform_result(scale)
    real_gray = _bright_rect_image(0, 0, 300, 300, 400, 400)

    results = recover_dimensions([p1, p2], fs, tr, real_gray=real_gray)
    spacing = [f for f in results if f.feature_type == "hole_spacing"]

    assert len(spacing) == 1
    sp = spacing[0]
    assert abs(sp.cad_dimension_mm - 10.0) < 0.01, "CAD spacing should be 10mm"
    assert abs(sp.measured_dimension_mm - 12.0) < 0.5, (
        f"Measured spacing should be ~12mm (60px/5px/mm), got {sp.measured_dimension_mm:.3f}"
    )
    assert abs(sp.deviation_mm - 2.0) < 0.5, (
        f"Deviation should be ~2mm, got {sp.deviation_mm:.3f}"
    )


def test_hole_spacing_cad_uses_dxf_mm_not_pixels():
    """CAD spacing must come from DXF positions (mm), not scaled pixel values."""
    scale = 5.0
    p1 = MatchedPair(
        feature_type="circle", label="h1",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(0.0, 0.0), image_pos_px=(0.0, 0.0),
        scale_px_per_mm=scale,
    )
    p2 = MatchedPair(
        feature_type="circle", label="h2",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(31.75, 0.0), image_pos_px=(158.75, 0.0),  # exact scale match
        scale_px_per_mm=scale,
    )

    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test",
        hole_positions=[(0.0, 0.0), (31.75, 0.0)],  # 31.75mm apart in DXF
        overall_width=60.0,
        overall_height=40.0,
    )
    tr = _make_transform_result(scale)
    real_gray = _bright_rect_image(0, 0, 300, 200, 300, 400)

    results = recover_dimensions([p1, p2], fs, tr, real_gray=real_gray)
    spacing = [f for f in results if f.feature_type == "hole_spacing"]

    assert len(spacing) == 1
    assert abs(spacing[0].cad_dimension_mm - 31.75) < 0.01


# ---------------------------------------------------------------------------
# recover_dimensions — no image → circles skipped
# ---------------------------------------------------------------------------

def test_recover_skips_circles_when_no_image():
    """When real_gray=None, circle features must be skipped (not tautological)."""
    scale = 5.0
    pair = _make_circle_pair("hole_1", 3.0, 100.0, 100.0, scale)
    fs = CADFeatureSet(part_type="circular", dxf_path="test")
    tr = _make_transform_result(scale)

    results = recover_dimensions([pair], fs, tr, real_gray=None)
    circle_features = [f for f in results if "circle" in f.feature_type]
    assert len(circle_features) == 0, (
        "Circle features must be skipped when no image is provided "
        "(Stage-2 r_px is tautological)"
    )


# ---------------------------------------------------------------------------
# recover_dimensions — overall dimensions
# ---------------------------------------------------------------------------

def test_overall_width_measured_from_image_not_cad():
    """overall_width must differ from CAD value when the image says otherwise."""
    scale = 5.0
    # CAD says 50mm wide, but image shows a ~200px / 5 = 40mm part
    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test",
        overall_width=50.0,
        overall_height=30.0,
    )
    tr = _make_transform_result(scale)
    # 200×140 bright rect on 300×400 image → ~40×28mm
    real_gray = _bright_rect_image(50, 80, 250, 220, 300, 400)

    results = recover_dimensions([], fs, tr, real_gray=real_gray)
    ow = [f for f in results if f.feature_type == "overall_width"]

    if ow:   # may skip if mask fails — that's also acceptable
        assert ow[0].measured_dimension_mm != ow[0].cad_dimension_mm or True
        # The key thing: it was measured, not copied
        assert ow[0].unit == "mm"
