"""
Transformation Estimator — Stage 5.

Refines the CAD-to-image transformation using matched feature pairs.

The coarse transform (M_cad2img) from Stage 1-2 is already a good
approximation. This stage fits a similarity transform (scale + rotation +
translation) through the matched-pair correspondences to produce a more
accurate CAD→image mapping.

Returns a TransformResult containing:
  - matrix      : refined 3×3 CAD→image transform
  - scale_px_per_mm : pixels per millimetre (used by measurement stage)
  - translation_px  : (tx, ty) residual translation
  - rotation_deg    : rotation angle
  - residual_error  : mean reprojection error in pixels
"""

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from dimension_analysis.feature_matcher import MatchedPair

logger = logging.getLogger(__name__)

# Minimum pairs needed for a reliable fit
MIN_PAIRS_FOR_FIT = 2


@dataclass
class TransformResult:
    matrix: np.ndarray          # 3×3 float64 CAD-mm → image-px
    scale_px_per_mm: float
    translation_px: tuple[float, float]
    rotation_deg: float
    residual_error: float       # mean pixel reprojection error
    refined: bool               # True = fit was computed, False = passthrough


def _extract_point_pairs(
    pairs: list[MatchedPair],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (src_pts, dst_pts) arrays from matched pairs."""
    src, dst = [], []
    for p in pairs:
        src.append([p.cad_pos[0],    p.cad_pos[1]])
        dst.append([p.image_pos_px[0], p.image_pos_px[1]])
    return np.float32(src), np.float32(dst)


def estimate_transform(
    matched_pairs: list[MatchedPair],
    M_initial: np.ndarray,
    scale_initial: float,
) -> TransformResult:
    """
    Estimate a refined CAD→image transformation.

    Parameters
    ----------
    matched_pairs  : output of feature_matcher.match_features()
    M_initial      : initial 3×3 matrix (M_align @ M_cad2edge @ M_dxf2bp)
    scale_initial  : initial scale in px/mm (from blueprint geometry)

    Returns
    -------
    TransformResult
    """
    if len(matched_pairs) < MIN_PAIRS_FOR_FIT:
        logger.warning(
            f"Only {len(matched_pairs)} matched pairs — using initial transform as-is"
        )
        s = scale_initial
        return TransformResult(
            matrix=M_initial,
            scale_px_per_mm=s,
            translation_px=(float(M_initial[0, 2]), float(M_initial[1, 2])),
            rotation_deg=math.degrees(math.atan2(M_initial[1, 0], M_initial[0, 0])),
            residual_error=0.0,
            refined=False,
        )

    src_pts, dst_pts = _extract_point_pairs(matched_pairs)

    # Fit partial affine (similarity: scale + rotation + translation)
    M2x3, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
        maxIters=2000,
        confidence=0.99,
    )

    if M2x3 is None:
        logger.warning("estimateAffinePartial2D failed — using initial transform")
        return TransformResult(
            matrix=M_initial,
            scale_px_per_mm=scale_initial,
            translation_px=(float(M_initial[0, 2]), float(M_initial[1, 2])),
            rotation_deg=math.degrees(math.atan2(M_initial[1, 0], M_initial[0, 0])),
            residual_error=0.0,
            refined=False,
        )

    M3x3 = np.eye(3, dtype=np.float64)
    M3x3[:2, :] = M2x3

    # Decompose
    scale_x = math.sqrt(M3x3[0, 0] ** 2 + M3x3[1, 0] ** 2)
    scale_y = math.sqrt(M3x3[0, 1] ** 2 + M3x3[1, 1] ** 2)
    scale   = (scale_x + scale_y) / 2.0
    angle   = math.degrees(math.atan2(M3x3[1, 0], M3x3[0, 0]))
    tx      = float(M3x3[0, 2])
    ty      = float(M3x3[1, 2])

    # Compute mean reprojection error on inliers
    projected = (M3x3 @ np.vstack([src_pts.T, np.ones((1, len(src_pts)))]))[:2].T
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    if inliers is not None:
        mask = inliers.ravel().astype(bool)
        mean_err = float(errors[mask].mean()) if mask.any() else float(errors.mean())
    else:
        mean_err = float(errors.mean())

    logger.debug(
        f"Refined transform: scale={scale:.4f} px/mm, "
        f"rotation={angle:.2f}°, tx={tx:.1f}, ty={ty:.1f}, "
        f"reprojection_error={mean_err:.2f} px"
    )

    return TransformResult(
        matrix=M3x3,
        scale_px_per_mm=scale,
        translation_px=(tx, ty),
        rotation_deg=angle,
        residual_error=mean_err,
        refined=True,
    )
