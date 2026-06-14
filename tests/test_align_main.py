import pytest
import numpy as np
import logging
from cad_image_alignment import align, AlignmentResult


def create_simple_shape(size=100, offset=(0, 0)):
    img = np.zeros((size, size), dtype=np.uint8)
    x_off, y_off = offset
    img[30+y_off:70+y_off, 30+x_off:70+x_off] = 255
    return img


def test_align_returns_alignment_result():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert isinstance(result, AlignmentResult)


def test_align_result_has_all_fields():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert hasattr(result, 'aligned_image')
    assert hasattr(result, 'transform_matrix')
    assert hasattr(result, 'alignment_score')
    assert hasattr(result, 'strategy')
    assert hasattr(result, 'high_confidence')
    assert hasattr(result, 'inlier_ratio')


def test_align_aligned_image_properties():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert result.aligned_image.shape == real_map.shape
    assert result.aligned_image.dtype == np.uint8


def test_align_transform_matrix_properties():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert result.transform_matrix.shape == (3, 3)
    assert result.transform_matrix.dtype == np.float64


def test_align_alignment_score_bounds():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert 0.0 <= result.alignment_score <= 1.0


def test_align_strategy_values():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert result.strategy in ["ecc_fine", "affine_coarse_only", "identity"]


def test_align_high_confidence_flag():
    from cad_image_alignment.constants import HIGH_CONFIDENCE_THRESHOLD
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    if result.alignment_score >= HIGH_CONFIDENCE_THRESHOLD:
        assert result.high_confidence is True
    else:
        assert result.high_confidence is False


def test_align_identity_fallback_no_contour():
    cad_map = np.zeros((100, 100), dtype=np.uint8)
    cad_map[49:51, 49:51] = 255

    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[49:51, 49:51] = 255

    result = align(cad_map, real_map)

    assert result.strategy == "identity"
    assert np.allclose(result.transform_matrix, np.eye(3))


def test_align_affine_coarse_only_fallback():
    cad_map = create_simple_shape()
    real_map = create_simple_shape(offset=(5, 5))

    result = align(cad_map, real_map)

    assert result.strategy in ["affine_coarse_only", "ecc_fine"]


def test_align_returns_result_even_when_not_high_confidence():
    from cad_image_alignment.constants import HIGH_CONFIDENCE_THRESHOLD
    cad_map = create_simple_shape()
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[10:30, 10:30] = 255

    result = align(cad_map, real_map)

    assert isinstance(result, AlignmentResult)
    if not result.high_confidence:
        assert result.alignment_score < HIGH_CONFIDENCE_THRESHOLD


def test_align_logging_on_fallback(caplog):
    cad_map = np.zeros((100, 100), dtype=np.uint8)
    cad_map[49:51, 49:51] = 255

    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[49:51, 49:51] = 255

    with caplog.at_level(logging.WARNING):
        result = align(cad_map, real_map)

    assert any("fallback" in record.message.lower() or
               "no valid contour" in record.message.lower()
               for record in caplog.records)


def test_align_logging_on_success(caplog):
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    with caplog.at_level(logging.DEBUG):
        result = align(cad_map, real_map)

    debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert any("strategy" in msg.lower() and "score" in msg.lower()
               for msg in debug_messages)


def test_align_logging_on_low_confidence(caplog):
    cad_map = create_simple_shape()
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[10:30, 10:30] = 255

    with caplog.at_level(logging.WARNING):
        result = align(cad_map, real_map)

    if not result.high_confidence:
        assert any("low confidence" in record.message.lower()
                   for record in caplog.records)


def test_align_inlier_ratio_type():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert result.inlier_ratio is None or isinstance(result.inlier_ratio, float)


def test_align_with_resolution_mismatch():
    real_map = create_simple_shape(size=100)
    cad_map = create_simple_shape(size=80)

    result = align(cad_map, real_map)

    assert result.aligned_image.shape == real_map.shape


def test_align_applies_final_transformation():
    cad_map = create_simple_shape()
    real_map = create_simple_shape()

    result = align(cad_map, real_map)

    assert result.aligned_image.shape == real_map.shape
    assert result.aligned_image.dtype == np.uint8
