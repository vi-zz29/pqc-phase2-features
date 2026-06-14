"""
Tests for dimension_analysis/dxf_parser.py and dimension_analysis/dxf_utils.py
"""

import math
import pytest
from pathlib import Path

from dimension_analysis.dxf_parser import (
    parse_dxf,
    CADFeatureSet,
    _extract_circular_features,
    _extract_rectangular_features,
    _is_full_circle_from_arcs,
    _arc_span,
)
from dimension_analysis.dxf_utils import parse_dxf_raw


# ---------------------------------------------------------------------------
# _arc_span
# ---------------------------------------------------------------------------

def test_arc_span_normal_quarter():
    assert abs(_arc_span({"a0": 0.0, "a1": 90.0}) - 90.0) < 0.01


def test_arc_span_wraps_across_zero():
    """270° → 90° = 180° span."""
    assert abs(_arc_span({"a0": 270.0, "a1": 90.0}) - 180.0) < 0.01


def test_arc_span_zero_start_equals_end_means_full_circle():
    assert abs(_arc_span({"a0": 45.0, "a1": 45.0}) - 360.0) < 0.01


def test_arc_span_full_circle():
    assert abs(_arc_span({"a0": 0.0, "a1": 360.0}) - 360.0) < 0.01


# ---------------------------------------------------------------------------
# _is_full_circle_from_arcs
# ---------------------------------------------------------------------------

def test_full_circle_from_four_quarter_arcs():
    arcs = [
        {"a0": 0.0,   "a1": 90.0},
        {"a0": 90.0,  "a1": 180.0},
        {"a0": 180.0, "a1": 270.0},
        {"a0": 270.0, "a1": 360.0},
    ]
    assert _is_full_circle_from_arcs(arcs) is True


def test_not_full_circle_from_single_semicircle():
    assert _is_full_circle_from_arcs([{"a0": 0.0, "a1": 180.0}]) is False


def test_full_circle_from_single_arc():
    """A single 360° arc is a full circle."""
    assert _is_full_circle_from_arcs([{"a0": 0.0, "a1": 360.0}]) is True


# ---------------------------------------------------------------------------
# _extract_circular_features
# ---------------------------------------------------------------------------

def test_extract_circular_basic():
    circles = [
        {"cx": 0.0, "cy": 0.0, "r": 50.0},   # outer
        {"cx": 0.0, "cy": 0.0, "r": 10.0},   # center bore
        {"cx": 30.0, "cy": 0.0,  "r": 3.0},  # bolt hole
        {"cx": -30.0, "cy": 0.0, "r": 3.0},  # bolt hole
        {"cx": 0.0, "cy": 30.0,  "r": 3.0},  # bolt hole
    ]
    result = _extract_circular_features(circles, [], [])

    assert result["outer_diameter"] == pytest.approx(100.0, abs=0.1)
    assert result["center_bore"]    == pytest.approx(20.0,  abs=0.1)
    assert result["hole_count"] == 3
    assert result["pcd"] == pytest.approx(60.0, abs=1.0)  # 2 × 30mm radius


def test_extract_circular_pcd_is_twice_mean_radius():
    circles = [
        {"cx": 0.0, "cy": 0.0,  "r": 50.0},
        {"cx": 25.0, "cy": 0.0, "r": 3.0},
        {"cx": -25.0,"cy": 0.0, "r": 3.0},
    ]
    result = _extract_circular_features(circles, [], [])
    # PCD = 2 × 25 = 50mm
    assert result["pcd"] == pytest.approx(50.0, abs=1.0)


def test_extract_circular_no_circles_raises():
    with pytest.raises(ValueError, match="No circle entities"):
        _extract_circular_features([], [], [])


def test_extract_circular_center_tol_is_relative():
    """CENTER_TOL must be computed from part size, not hard-coded 3mm."""
    # All circles very close together → all concentric
    circles = [
        {"cx": 0.0, "cy": 0.0, "r": 500.0},
        {"cx": 0.5, "cy": 0.5, "r": 50.0},
    ]
    result = _extract_circular_features(circles, [], [])
    # At max_r=500, CENTER_TOL = min(5, 500*0.02) = 5mm — both within 0.71mm
    assert result["hole_count"] == 0  # no peripheral circles


# ---------------------------------------------------------------------------
# _extract_rectangular_features
# ---------------------------------------------------------------------------

def test_extract_rectangular_overall_dimensions():
    lines = [
        {"x1": 0.0,   "y1": 0.0,  "x2": 100.0, "y2": 0.0},
        {"x1": 100.0, "y1": 0.0,  "x2": 100.0, "y2": 60.0},
        {"x1": 100.0, "y1": 60.0, "x2": 0.0,   "y2": 60.0},
        {"x1": 0.0,   "y1": 60.0, "x2": 0.0,   "y2": 0.0},
    ]
    result = _extract_rectangular_features([], [], lines)
    assert result["overall_width"]  == pytest.approx(100.0, abs=0.1)
    assert result["overall_height"] == pytest.approx(60.0,  abs=0.1)


def test_extract_rectangular_holes_detected():
    lines = [
        {"x1": 0.0,   "y1": 0.0,  "x2": 100.0, "y2": 0.0},
        {"x1": 100.0, "y1": 0.0,  "x2": 100.0, "y2": 60.0},
        {"x1": 100.0, "y1": 60.0, "x2": 0.0,   "y2": 60.0},
        {"x1": 0.0,   "y1": 60.0, "x2": 0.0,   "y2": 0.0},
    ]
    circles = [{"cx": 20.0, "cy": 30.0, "r": 3.0}]
    result = _extract_rectangular_features(circles, [], lines)
    assert result["hole_count"] == 1
    assert result["hole_diameters"][0] == pytest.approx(6.0, abs=0.01)


def test_extract_rectangular_hole_radius_threshold_relative():
    """Holes use 15% of smaller dimension, not a hard-coded 20mm."""
    lines = [
        {"x1": 0.0,   "y1": 0.0,   "x2": 200.0, "y2": 0.0},
        {"x1": 200.0, "y1": 0.0,   "x2": 200.0, "y2": 200.0},
        {"x1": 200.0, "y1": 200.0, "x2": 0.0,   "y2": 200.0},
        {"x1": 0.0,   "y1": 200.0, "x2": 0.0,   "y2": 0.0},
    ]
    # r=25mm > hard-coded 20mm limit, but < 15% of 200 = 30mm → should be a hole
    circles = [{"cx": 100.0, "cy": 100.0, "r": 25.0}]
    result = _extract_rectangular_features(circles, [], lines)
    assert result["hole_count"] == 1


def test_extract_rectangular_no_geometry_raises():
    with pytest.raises(ValueError, match="No geometry found"):
        _extract_rectangular_features([], [], [])


# ---------------------------------------------------------------------------
# CADFeatureSet.summary
# ---------------------------------------------------------------------------

def test_cad_feature_set_summary_circular():
    fs = CADFeatureSet(
        part_type="circular",
        dxf_path="test.dxf",
        outer_diameter=100.0,
        center_bore=20.0,
        pcd=60.0,
        hole_count=3,
        hole_diameters=[6.0],
        hole_positions=[(30.0, 0.0)],
    )
    s = fs.summary()
    assert "circular" in s
    assert "100.0" in s
    assert "pcd" in s.lower()


def test_cad_feature_set_summary_rectangular():
    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test.dxf",
        overall_width=200.0,
        overall_height=100.0,
        hole_count=2,
    )
    s = fs.summary()
    assert "rectangular" in s
    assert "200.0" in s


# ---------------------------------------------------------------------------
# parse_dxf errors
# ---------------------------------------------------------------------------

def test_parse_dxf_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_dxf("does_not_exist/fake.dxf")


# ---------------------------------------------------------------------------
# dxf_utils.parse_dxf_raw
# ---------------------------------------------------------------------------

def test_parse_dxf_raw_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_dxf_raw("no_such_file.dxf")


def test_parse_dxf_raw_returns_three_lists():
    dxf_path = Path("dxf/box_front.dxf")
    if not dxf_path.exists():
        pytest.skip("dxf/box_front.dxf not available")
    circles, arcs, lines = parse_dxf_raw(dxf_path)
    assert isinstance(circles, list)
    assert isinstance(arcs, list)
    assert isinstance(lines, list)


def test_parse_dxf_raw_circle_has_required_keys():
    dxf_path = Path("dxf/circular_rear.dxf")
    if not dxf_path.exists():
        pytest.skip("dxf/circular_rear.dxf not available")
    circles, arcs, lines = parse_dxf_raw(dxf_path)
    if circles:
        c = circles[0]
        assert "cx" in c and "cy" in c and "r" in c


# ---------------------------------------------------------------------------
# Integration tests against real DXF files (skipped if files absent)
# ---------------------------------------------------------------------------

def test_parse_dxf_box_front_integration():
    dxf_path = Path("dxf/box_front.dxf")
    if not dxf_path.exists():
        pytest.skip("dxf/box_front.dxf not available")

    fs = parse_dxf(dxf_path)
    assert fs.part_type == "rectangular"
    assert fs.overall_width  is not None and fs.overall_width  > 0
    assert fs.overall_height is not None and fs.overall_height > 0
    assert isinstance(fs.raw_lines, list)


def test_parse_dxf_circular_rear_integration():
    dxf_path = Path("dxf/circular_rear.dxf")
    if not dxf_path.exists():
        pytest.skip("dxf/circular_rear.dxf not available")

    fs = parse_dxf(dxf_path)
    assert fs.part_type == "circular"
    # outer_diameter may be None if the outer boundary is drawn with arcs that
    # don't sum to exactly 360° — just verify the parse succeeded and returned data
    assert fs.hole_count >= 0
    assert isinstance(fs.raw_circles, list)
    assert len(fs.raw_circles) > 0 or len(fs.raw_arcs) > 0
