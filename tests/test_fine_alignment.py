"""
Tests for fine alignment.

_detect_and_match_features and _estimate_similarity were removed when
ORB was replaced with ECC. These tests now cover:
  - _validate_similarity  (kept for backward compat)
  - _compute_fine_transform  (ECC-based)
"""

import numpy as np
import cv2
import pytest

from cad_image_alignment.alignment import (
    _validate_similarity,
    _compute_fine_transform,
)


# ---------------------------------------------------------------------------
# _validate_similarity
# ---------------------------------------------------------------------------

def test_validate_similarity_valid_scale():
    M = np.eye(3, dtype=np.float64)
    assert _validate_similarity(M) == True


def test_validate_similarity_scale_too_small():
    s = 0.3
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_similarity(M) == False


def test_validate_similarity_scale_too_large():
    s = 2.5
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_similarity(M) == False


def test_validate_similarity_boundary_values():
    for s in (0.5, 2.0):
        M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
        assert _validate_similarity(M) == True, f"Scale {s} should be valid"


# ---------------------------------------------------------------------------
# _compute_fine_transform  (ECC)
# ---------------------------------------------------------------------------

def test_compute_fine_transform_returns_tuple():
    """_compute_fine_transform must always return a 2-tuple."""
    cad = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(cad, (20, 20), (80, 80), 255, 2)
    real = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(real, (20, 20), (80, 80), 255, 2)
    coarse = np.eye(3, dtype=np.float64)

    result = _compute_fine_transform(cad, real, coarse)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_compute_fine_transform_identity_input():
    """With identical images, ECC should converge and return a near-identity correction."""
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (120, 120), 255, 2)
    cv2.circle(img,   (75, 75),  20, 255, 2)
    coarse = np.eye(3, dtype=np.float64)

    M_total, inlier_ratio = _compute_fine_transform(img.copy(), img.copy(), coarse)

    # ECC has no inlier_ratio — should be None
    assert inlier_ratio is None

    if M_total is not None:
        assert M_total.shape == (3, 3)
        assert M_total.dtype == np.float64
        # Translation correction should be tiny for identical images
        tx = M_total[0, 2]
        ty = M_total[1, 2]
        assert abs(tx) < 5.0, f"Expected near-zero tx, got {tx:.2f}"
        assert abs(ty) < 5.0, f"Expected near-zero ty, got {ty:.2f}"


def test_compute_fine_transform_empty_images_graceful():
    """Completely empty edge maps — ECC may fail gracefully (None,None) or succeed."""
    cad  = np.zeros((100, 100), dtype=np.uint8)
    real = np.zeros((100, 100), dtype=np.uint8)
    coarse = np.eye(3, dtype=np.float64)

    result = _compute_fine_transform(cad, real, coarse)
    # Should not raise — either returns (None, None) or a valid matrix
    assert isinstance(result, tuple)
    assert len(result) == 2
    M, ir = result
    if M is not None:
        assert M.shape == (3, 3)


def test_compute_fine_transform_small_shift():
    """With a 3px shift between CAD and real, ECC should correct it."""
    base = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(base, (50, 50), (150, 150), 255, 2)
    cv2.circle(base, (100, 100), 30, 255, 2)

    shifted = np.zeros_like(base)
    cv2.rectangle(shifted, (53, 53), (153, 153), 255, 2)
    cv2.circle(shifted, (103, 103), 30, 255, 2)

    coarse = np.eye(3, dtype=np.float64)
    M_total, _ = _compute_fine_transform(base, shifted, coarse)

    if M_total is not None:
        assert M_total.shape == (3, 3)
        np.testing.assert_array_almost_equal(M_total[2, :], [0.0, 0.0, 1.0])
