import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContourDescriptor:
    contour: np.ndarray
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    bbox_diagonal: float
    pca_angle_deg: float


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    transform_matrix: np.ndarray
    alignment_score: float
    coverage: float
    strategy: str
    high_confidence: bool
    identified: bool
    inlier_ratio: Optional[float]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> np.ndarray:
    if cad_edge_map.dtype != np.uint8:
        raise ValueError(
            f"cad_edge_map has invalid dtype {cad_edge_map.dtype}, expected uint8"
        )
    if real_edge_map.dtype != np.uint8:
        raise ValueError(
            f"real_edge_map has invalid dtype {real_edge_map.dtype}, expected uint8"
        )
    if cad_edge_map.ndim != 2:
        raise ValueError(
            f"cad_edge_map has invalid ndim {cad_edge_map.ndim}, expected 2"
        )
    if real_edge_map.ndim != 2:
        raise ValueError(
            f"real_edge_map has invalid ndim {real_edge_map.ndim}, expected 2"
        )
    if not np.any(cad_edge_map):
        raise ValueError("cad_edge_map is empty (contains no non-zero pixels)")
    if not np.any(real_edge_map):
        raise ValueError("real_edge_map is empty (contains no non-zero pixels)")

    if cad_edge_map.shape != real_edge_map.shape:
        rh, rw = real_edge_map.shape
        ch, cw = cad_edge_map.shape
        logger.debug(
            f"Resolution mismatch: cad {cad_edge_map.shape} → real canvas {real_edge_map.shape}. "
            f"Cropping to part bbox then uniform-scaling to preserve design geometry."
        )
        contours, _ = cv2.findContours(
            cad_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            margin = int(max(bw, bh) * 0.05)
            x1 = max(0, bx - margin)
            y1 = max(0, by - margin)
            x2 = min(cw, bx + bw + margin)
            y2 = min(ch, by + bh + margin)
            cad_edge_map = cad_edge_map[y1:y2, x1:x2]
            logger.debug(
                f"Cropped CAD to part bbox ({bx},{by},{bw},{bh}) + margin → {cad_edge_map.shape}"
            )
        ch2, cw2 = cad_edge_map.shape
        fit_scale = min(rw / cw2, rh / ch2)
        new_w = int(round(cw2 * fit_scale))
        new_h = int(round(ch2 * fit_scale))
        interp = cv2.INTER_AREA if fit_scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(cad_edge_map, (new_w, new_h), interpolation=interp)
        _, scaled = cv2.threshold(scaled, 20, 255, cv2.THRESH_BINARY)
        canvas = np.zeros((rh, rw), dtype=np.uint8)
        y_off = (rh - new_h) // 2
        x_off = (rw - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        cad_edge_map = canvas

    return cad_edge_map


# ---------------------------------------------------------------------------
# PCA angle
# ---------------------------------------------------------------------------

def _compute_pca_angle(contour: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]
    angle_rad = np.arctan2(principal[1], principal[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg


# ---------------------------------------------------------------------------
# Primary contour extraction
# ---------------------------------------------------------------------------

def _extract_primary_contour(edge_map: np.ndarray) -> Optional[ContourDescriptor]:
    from .constants import MIN_CONTOUR_AREA_FRACTION

    contours, _ = cv2.findContours(
        edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = edge_map.shape[0] * edge_map.shape[1]
    min_area = image_area * MIN_CONTOUR_AREA_FRACTION
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        logger.warning(
            f"No contour found with area >= {MIN_CONTOUR_AREA_FRACTION * 100}% "
            f"of image area ({min_area:.1f} pixels)"
        )
        return None

    all_pts = np.vstack([c.reshape(-1, 2) for c in valid_contours])
    hull = cv2.convexHull(all_pts)

    M_hull = cv2.moments(hull)
    if M_hull["m00"] == 0:
        primary = max(valid_contours, key=cv2.contourArea)
        M_hull = cv2.moments(primary)
        hull = primary

    centroid_x = M_hull["m10"] / M_hull["m00"]
    centroid_y = M_hull["m01"] / M_hull["m00"]
    centroid = (centroid_x, centroid_y)

    x, y, w, h = cv2.boundingRect(hull)
    bbox = (x, y, w, h)
    bbox_diagonal = np.sqrt(w ** 2 + h ** 2)

    primary_contour = max(valid_contours, key=cv2.contourArea)
    pca_angle_deg = _compute_pca_angle(primary_contour)

    return ContourDescriptor(
        contour=hull.reshape(-1, 1, 2).astype(np.int32),
        centroid=centroid,
        bbox=bbox,
        bbox_diagonal=bbox_diagonal,
        pca_angle_deg=pca_angle_deg,
    )


# ---------------------------------------------------------------------------
# Affine matrix builder
# ---------------------------------------------------------------------------

def _build_affine_matrix(
    scale: float,
    angle_deg: float,
    src_centroid: tuple[float, float],
    dst_centroid: tuple[float, float],
) -> np.ndarray:
    cx_src, cy_src = src_centroid
    cx_dst, cy_dst = dst_centroid
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    s_cos = scale * cos_a
    s_sin = scale * sin_a
    tx = cx_dst - cx_src
    ty = cy_dst - cy_src
    M = np.array([
        [s_cos, -s_sin, cx_src * (1 - s_cos) + cy_src * s_sin + tx],
        [s_sin,  s_cos, cy_src * (1 - s_cos) - cx_src * s_sin + ty],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)
    return M


# ---------------------------------------------------------------------------
# Silhouette fill — with open-contour guard
# ---------------------------------------------------------------------------

def _fill_silhouette(edge_map: np.ndarray) -> np.ndarray:
    h, w = edge_map.shape

    closed = cv2.morphologyEx(
        edge_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    canvas[1:h + 1, 1:w + 1] = closed

    inv = cv2.bitwise_not(canvas)
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(inv, ff_mask, (0, 0), 0)

    filled = inv[1:h + 1, 1:w + 1]

    # Guard: if the filled region is >80% of the image, the flood fill escaped
    # an open contour and filled the background. Fall back to closed edges.
    filled_fraction = float(np.count_nonzero(filled)) / float(h * w)
    if filled_fraction > 0.80:
        logger.debug(
            "_fill_silhouette: flood-fill escaped open contour "
            f"(filled {filled_fraction:.1%}), using morphologically closed edges"
        )
        return closed

    return filled


# ---------------------------------------------------------------------------
# Alignment score (IoU on filled silhouettes)
# ---------------------------------------------------------------------------

def _compute_alignment_score(
    aligned_image: np.ndarray,
    reference: np.ndarray,
) -> float:
    cad_filled  = _fill_silhouette(aligned_image)
    real_filled = _fill_silhouette(reference)
    intersection = np.logical_and(cad_filled > 0, real_filled > 0).sum()
    union        = np.logical_or(cad_filled  > 0, real_filled > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# ---------------------------------------------------------------------------
# Coarse transform (grid search over angle / scale)
# ---------------------------------------------------------------------------

def _compute_coarse_transform(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> Optional[np.ndarray]:
    COARSE_SCALE = 0.5
    h_full, w_full = real_edge_map.shape
    h_small = max(1, int(h_full * COARSE_SCALE))
    w_small = max(1, int(w_full * COARSE_SCALE))

    cad_small  = cv2.resize(cad_edge_map,  (w_small, h_small), interpolation=cv2.INTER_AREA)
    real_small = cv2.resize(real_edge_map, (w_small, h_small), interpolation=cv2.INTER_AREA)
    _, cad_small  = cv2.threshold(cad_small,  20, 255, cv2.THRESH_BINARY)
    _, real_small = cv2.threshold(real_small, 20, 255, cv2.THRESH_BINARY)

    cad_descriptor = _extract_primary_contour(cad_small)
    if cad_descriptor is None:
        logger.warning("No valid contour found in CAD edge map, cannot compute coarse transform")
        return None

    real_descriptor = _extract_primary_contour(real_small)
    if real_descriptor is None:
        logger.warning("No valid contour found in real edge map, cannot compute coarse transform")
        return None

    cad_area  = cv2.contourArea(cad_descriptor.contour)
    real_area = cv2.contourArea(real_descriptor.contour)
    if cad_area <= 0:
        logger.warning("CAD contour has zero area, falling back to diagonal scale")
        base_scale = real_descriptor.bbox_diagonal / cad_descriptor.bbox_diagonal
    else:
        base_scale = np.sqrt(real_area / cad_area)

    logger.debug(
        f"Coarse alignment: area_scale={base_scale:.3f}, "
        f"cad_pca={cad_descriptor.pca_angle_deg:.1f}°, "
        f"real_pca={real_descriptor.pca_angle_deg:.1f}°"
    )

    scale_band = [0.90, 0.95, 1.00, 1.05, 1.10]
    pca_diff   = real_descriptor.pca_angle_deg - cad_descriptor.pca_angle_deg
    coarse_step = 10
    coarse_angles = list(set(
        list(range(0, 360, coarse_step)) + [
            round(pca_diff) % 360,
            round(pca_diff + 180) % 360,
        ]
    ))

    EARLY_EXIT_SCORE = 0.88
    GRID_SCALE = 0.5
    h_grid = max(1, int(h_small * GRID_SCALE))
    w_grid = max(1, int(w_small * GRID_SCALE))
    cad_grid  = cv2.resize(cad_small,  (w_grid, h_grid), interpolation=cv2.INTER_AREA)
    real_grid = cv2.resize(real_small, (w_grid, h_grid), interpolation=cv2.INTER_AREA)
    _, cad_grid  = cv2.threshold(cad_grid,  20, 255, cv2.THRESH_BINARY)
    _, real_grid = cv2.threshold(real_grid, 20, 255, cv2.THRESH_BINARY)

    grid_desc_cad  = _extract_primary_contour(cad_grid)
    grid_desc_real = _extract_primary_contour(real_grid)
    if grid_desc_cad is None or grid_desc_real is None:
        grid_desc_cad  = cad_descriptor
        grid_desc_real = real_descriptor
        cad_grid  = cad_small
        real_grid = real_small

    real_grid_f = real_grid.astype(np.bool_)

    def _fast_iou(warped_bin: np.ndarray) -> float:
        w_bool = warped_bin.astype(np.bool_)
        inter = int(np.logical_and(w_bool, real_grid_f).sum())
        union = int(np.logical_or(w_bool,  real_grid_f).sum())
        return float(inter) / float(union) if union > 0 else 0.0

    best_score_grid = -1.0
    best_angle_coarse, best_sf_coarse = coarse_angles[0], scale_band[0]
    top_candidates: list[tuple[float, int, float]] = []

    outer_done = False
    for sf in scale_band:
        if outer_done:
            break
        s = base_scale * sf
        for angle in coarse_angles:
            M = _build_affine_matrix(
                scale=s, angle_deg=angle,
                src_centroid=grid_desc_cad.centroid,
                dst_centroid=grid_desc_real.centroid,
            )
            warped = apply_transform(cad_grid, M, output_shape=real_grid.shape)
            score  = _fast_iou(warped)
            if score > best_score_grid:
                best_score_grid = score
                best_angle_coarse, best_sf_coarse = angle, sf
            top_candidates.append((score, angle, sf))
            if best_score_grid >= EARLY_EXIT_SCORE:
                outer_done = True
                break

    top_candidates.sort(key=lambda x: x[0], reverse=True)
    top_n      = min(20, len(top_candidates))
    top_angles = list(set(int(c[1]) for c in top_candidates[:top_n]))
    top_sfs    = list(set(c[2]      for c in top_candidates[:top_n]))

    real_grid_filled = _fill_silhouette(real_grid)
    best_score_verify = -1.0
    best_angle_coarse, best_sf_coarse = coarse_angles[0], scale_band[0]

    for sf in top_sfs:
        s = base_scale * sf
        for angle in top_angles:
            M = _build_affine_matrix(
                scale=s, angle_deg=angle,
                src_centroid=grid_desc_cad.centroid,
                dst_centroid=grid_desc_real.centroid,
            )
            warped = apply_transform(cad_grid, M, output_shape=real_grid.shape)
            cad_filled = _fill_silhouette(warped)
            intersection = np.logical_and(cad_filled > 0, real_grid_filled > 0).sum()
            union        = np.logical_or(cad_filled  > 0, real_grid_filled > 0).sum()
            score = float(intersection) / float(union) if union > 0 else 0.0
            if score > best_score_verify:
                best_score_verify = score
                best_angle_coarse, best_sf_coarse = angle, sf

    fine_angles  = list(range(best_angle_coarse - coarse_step,
                               best_angle_coarse + coarse_step + 1))
    fine_angles += list(range(int(pca_diff) - coarse_step,
                               int(pca_diff) + coarse_step + 1))
    fine_angles += list(range(int(pca_diff + 180) - coarse_step,
                               int(pca_diff + 180) + coarse_step + 1))
    fine_angles  = list(set(fine_angles))

    best_fine_angle, best_fine_sf = best_angle_coarse, best_sf_coarse
    best_score_fine = -1.0

    for sf in [best_sf_coarse - 0.03, best_sf_coarse, best_sf_coarse + 0.03]:
        s = base_scale * sf
        for angle in fine_angles:
            M = _build_affine_matrix(
                scale=s, angle_deg=angle,
                src_centroid=grid_desc_cad.centroid,
                dst_centroid=grid_desc_real.centroid,
            )
            warped = apply_transform(cad_grid, M, output_shape=real_grid.shape)
            cad_filled = _fill_silhouette(warped)
            intersection = np.logical_and(cad_filled > 0, real_grid_filled > 0).sum()
            union        = np.logical_or(cad_filled  > 0, real_grid_filled > 0).sum()
            score = float(intersection) / float(union) if union > 0 else 0.0
            if score > best_score_fine:
                best_score_fine = score
                best_fine_angle, best_fine_sf = angle, sf

    real_filled = _fill_silhouette(real_small)
    best_M   = np.eye(3, dtype=np.float64)
    best_score = -1.0

    for sf in [best_fine_sf - 0.03, best_fine_sf, best_fine_sf + 0.03]:
        s = base_scale * sf
        for angle in [best_fine_angle - 1, best_fine_angle, best_fine_angle + 1]:
            M = _build_affine_matrix(
                scale=s, angle_deg=angle,
                src_centroid=cad_descriptor.centroid,
                dst_centroid=real_descriptor.centroid,
            )
            warped = apply_transform(cad_small, M, output_shape=real_small.shape)
            cad_filled = _fill_silhouette(warped)
            intersection = np.logical_and(cad_filled > 0, real_filled > 0).sum()
            union        = np.logical_or(cad_filled  > 0, real_filled > 0).sum()
            score = float(intersection) / float(union) if union > 0 else 0.0
            if score > best_score:
                best_score, best_M = score, M

    logger.debug(f"Coarse best score={best_score:.4f}")

    S_down = np.array([[COARSE_SCALE, 0, 0],
                        [0, COARSE_SCALE, 0],
                        [0, 0,            1]], dtype=np.float64)
    S_up   = np.array([[1 / COARSE_SCALE, 0, 0],
                        [0, 1 / COARSE_SCALE, 0],
                        [0, 0,               1]], dtype=np.float64)
    return S_up @ best_M @ S_down


# ---------------------------------------------------------------------------
# ECC fine alignment  (replaces ORB-based approach)
# ---------------------------------------------------------------------------

def _compute_fine_transform(
    coarsely_aligned_cad: np.ndarray,
    real_edge_map: np.ndarray,
    coarse_matrix: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Refine the coarse transform using ECC (Enhanced Correlation Coefficient).

    Uses MOTION_EUCLIDEAN (rotation + translation only).  ECC is seeded with
    identity because the coarse_matrix has already been applied to produce
    coarsely_aligned_cad — so the residual correction should be near-identity.

    Returns (M_total_3x3, None).  No inlier_ratio concept for ECC.
    """
    from .constants import ECC_MAX_ITERATIONS, ECC_EPSILON, ECC_WARP_MODE

    # Dilate edges so the gradient field has overlap to work with
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cad_d  = cv2.dilate(coarsely_aligned_cad, kernel, iterations=2)
    real_d = cv2.dilate(real_edge_map,         kernel, iterations=2)

    # ECC needs float32
    src_f = cad_d.astype(np.float32)
    dst_f = real_d.astype(np.float32)

    # CRITICAL: seed ECC with IDENTITY, not the coarse matrix.
    # coarsely_aligned_cad has already been warped by coarse_matrix.
    # ECC only needs to find the small residual correction (near-identity).
    # Seeding with the full coarse matrix (which encodes scale) causes
    # ECC to diverge because MOTION_EUCLIDEAN cannot represent scale.
    warp_init = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERATIONS,
        ECC_EPSILON,
    )

    # gaussFiltSize must be a positive odd integer.
    # 5 works for most images; if images are very high-res this can be increased.
    try:
        _, warp_ecc = cv2.findTransformECC(
            dst_f, src_f,   # (template=real, input=cad)
            warp_init,
            ECC_WARP_MODE,
            criteria,
            None,           # inputMask
            5,              # gaussFiltSize — must be positive odd integer
        )
    except cv2.error as exc:
        logger.warning(f"ECC fine alignment failed: {exc}")
        return None, None

    # Build 3×3 from 2×3 result
    M_ecc = np.eye(3, dtype=np.float64)
    M_ecc[:2, :] = warp_ecc.astype(np.float64)

    # Validate: ECC corrects only small residuals — scale must be near 1
    # (MOTION_EUCLIDEAN preserves scale so this is a sanity check on convergence)
    scale = float(np.sqrt(M_ecc[0, 0] ** 2 + M_ecc[1, 0] ** 2))
    if not (0.92 <= scale <= 1.08):
        logger.warning(
            f"ECC fine alignment rejected: scale={scale:.3f} outside [0.92, 1.08] "
            f"(indicates divergence)"
        )
        return None, None

    # M_ecc is the residual correction applied ON TOP of coarse_matrix
    M_total = M_ecc @ coarse_matrix
    logger.debug(f"ECC fine alignment succeeded: residual_scale={scale:.4f}")
    return M_total, None


# ---------------------------------------------------------------------------
# Kept for backward compatibility with tests that import _validate_similarity
# ---------------------------------------------------------------------------

def _validate_similarity(M: np.ndarray) -> bool:
    """
    Check that the scale encoded in a similarity matrix is within valid range.
    Kept for test backward-compatibility.
    """
    from .constants import SCALE_MIN, SCALE_MAX
    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    is_valid = SCALE_MIN <= scale <= SCALE_MAX
    if not is_valid:
        logger.debug(
            f"Similarity scale {scale:.3f} outside valid range [{SCALE_MIN}, {SCALE_MAX}]"
        )
    return is_valid


# ---------------------------------------------------------------------------
# Main align function
# ---------------------------------------------------------------------------

def align(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> AlignmentResult:
    from .constants import HIGH_CONFIDENCE_THRESHOLD, COVERAGE_THRESHOLD

    cad_edge_map = _validate_inputs(cad_edge_map, real_edge_map)

    final_matrix: np.ndarray
    strategy: str
    inlier_ratio: Optional[float] = None

    M_coarse = _compute_coarse_transform(cad_edge_map, real_edge_map)

    if M_coarse is None:
        logger.warning(
            "Coarse alignment failed: no valid contour found. "
            "Falling back to identity transform."
        )
        final_matrix = np.eye(3, dtype=np.float64)
        strategy = "identity"
    else:
        coarsely_aligned_cad = apply_transform(
            cad_edge_map, M_coarse, output_shape=real_edge_map.shape
        )

        M_fine, fine_inlier_ratio = _compute_fine_transform(
            coarsely_aligned_cad, real_edge_map, M_coarse
        )

        if M_fine is not None:
            logger.debug("ECC fine alignment succeeded.")
            final_matrix = M_fine
            strategy = "ecc_fine"
            inlier_ratio = fine_inlier_ratio   # None for ECC
        else:
            logger.warning(
                "Fine alignment failed. Falling back to coarse affine transform only."
            )
            final_matrix = M_coarse
            strategy = "affine_coarse_only"

    aligned_image = apply_transform(
        cad_edge_map, final_matrix, output_shape=real_edge_map.shape
    )

    cad_filled  = _fill_silhouette(aligned_image)
    real_filled = _fill_silhouette(real_edge_map)

    intersection    = np.logical_and(cad_filled > 0, real_filled > 0).sum()
    union           = np.logical_or(cad_filled  > 0, real_filled > 0).sum()
    alignment_score = float(intersection) / float(union) if union > 0 else 0.0

    real_area = int((real_filled > 0).sum())
    covered   = int(intersection)
    coverage  = float(covered) / float(real_area) if real_area > 0 else 0.0

    high_confidence = alignment_score >= HIGH_CONFIDENCE_THRESHOLD
    identified      = coverage >= COVERAGE_THRESHOLD

    if not high_confidence:
        logger.warning(
            f"Low confidence alignment: score={alignment_score:.4f} < {HIGH_CONFIDENCE_THRESHOLD}"
        )

    logger.debug(
        f"Alignment complete: strategy={strategy}, score={alignment_score:.4f}"
    )

    return AlignmentResult(
        aligned_image=aligned_image,
        transform_matrix=final_matrix,
        alignment_score=alignment_score,
        coverage=coverage,
        strategy=strategy,
        high_confidence=high_confidence,
        identified=identified,
        inlier_ratio=inlier_ratio,
    )


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def apply_transform(
    edge_map: np.ndarray,
    matrix: np.ndarray,
    output_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    if output_shape is None:
        output_shape = edge_map.shape
    return cv2.warpPerspective(
        edge_map,
        matrix,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# Template matching helper
# ---------------------------------------------------------------------------

@dataclass
class TemplateMatch:
    name: str
    result: AlignmentResult
    rank: int


def match_best_template(
    templates: list[tuple[str, np.ndarray]],
    real_edge_map: np.ndarray,
) -> list[TemplateMatch]:
    if not templates:
        raise ValueError(
            "templates list is empty — provide at least one (name, cad_edge_map) pair"
        )
    results = []
    for name, cad_edge_map in templates:
        logger.debug(f"Aligning template '{name}'...")
        result = align(cad_edge_map, real_edge_map)
        results.append((name, result))
        logger.debug(
            f"Template '{name}': coverage={result.coverage:.4f}, "
            f"iou={result.alignment_score:.4f}, strategy={result.strategy}"
        )
    results.sort(key=lambda x: x[1].coverage, reverse=True)
    return [
        TemplateMatch(name=name, result=result, rank=i + 1)
        for i, (name, result) in enumerate(results)
    ]
