"""
Feature Matcher — Stage 4.

Matches CAD features (from CADFeatureSet) to image-detected features.

Fixes applied:
  - Bug 1/Scale: image_value_px is now the raw image-pixel measurement.
    When built from existing_verification, we store r_px directly and
    carry total_scale so measurement.py divides by the correct scale.
  - Bug 2/Radius-diameter: CAD circle matching uses both position AND
    radius proximity to avoid selecting the wrong concentric circle.
  - Bug 3/Spacing: box parts have no raw_circles; CAD position is taken
    directly from the BOX_FEATURES dict geometry, not a circle lookup.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from dimension_analysis.dxf_parser import CADFeatureSet

logger = logging.getLogger(__name__)

# Minimum number of matched pairs to proceed
MIN_MATCHED_PAIRS = 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageCircle:
    """A circle detected in the image (pixel coordinates)."""
    cx_px: float
    cy_px: float
    radius_px: float
    confidence: float = 1.0
    label: str = ""


@dataclass
class ImageRect:
    """An axis-aligned rectangle detected in the image (pixel coordinates)."""
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    label: str = ""

    @property
    def cx_px(self) -> float:
        return (self.x1_px + self.x2_px) / 2.0

    @property
    def cy_px(self) -> float:
        return (self.y1_px + self.y2_px) / 2.0

    @property
    def width_px(self) -> float:
        return abs(self.x2_px - self.x1_px)

    @property
    def height_px(self) -> float:
        return abs(self.y2_px - self.y1_px)


@dataclass
class MatchedPair:
    """
    A confirmed match between one CAD feature and one image feature.

    Fields
    ------
    feature_type   : 'circle' | 'rect'
    label          : human-readable name
    cad_value_mm   : CAD nominal (radius mm for circles, (w,h) mm for rects)
    image_value_px : measured size in final-image pixels
                     (radius px for circles, (w,h) px for rects)
    cad_pos        : CAD centre (x, y) in mm
    image_pos_px   : detected centre (cx, cy) in final-image pixels
    scale_px_per_mm: the correct DXF-mm → final-image-px scale to use when
                     converting image_value_px back to mm in measurement.py
    match_distance_px : pixel distance between projected and detected centre
    """
    feature_type: str
    label: str
    cad_value_mm: float | tuple
    image_value_px: float | tuple
    cad_pos: tuple[float, float]
    image_pos_px: tuple[float, float]
    scale_px_per_mm: float = 0.0       # ← carries the correct scale
    match_distance_px: float = 0.0


# ---------------------------------------------------------------------------
# Image feature detection
# ---------------------------------------------------------------------------

def _detect_circles_in_image(
    gray: np.ndarray,
    min_radius_px: int = 3,
    max_radius_px: int = 200,
) -> list[ImageCircle]:
    """Detect circles using HoughCircles on the grayscale image."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles_raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min_radius_px * 2),
        param1=60,
        param2=25,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    result: list[ImageCircle] = []
    if circles_raw is not None:
        for i, (cx, cy, r) in enumerate(circles_raw[0]):
            result.append(ImageCircle(
                cx_px=float(cx), cy_px=float(cy), radius_px=float(r),
                label=f"img_circle_{i+1}"
            ))
    logger.debug(f"HoughCircles found {len(result)} circles")
    return result


def _detect_rects_in_image(
    gray: np.ndarray,
    min_area_px: int = 100,
) -> list[ImageRect]:
    """Detect axis-aligned rectangles via contour approximation."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result: list[ImageRect] = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            xs = [p[0][0] for p in approx]
            ys = [p[0][1] for p in approx]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                continue
            result.append(ImageRect(
                x1_px=float(x1), y1_px=float(y1),
                x2_px=float(x2), y2_px=float(y2),
                label=f"img_rect_{i+1}"
            ))
    logger.debug(f"Contour rect detection found {len(result)} rect candidates")
    return result


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def _project_cad_to_image(
    cad_x: float, cad_y: float,
    M: np.ndarray,
) -> tuple[float, float]:
    """Apply a 3×3 homogeneous matrix to a CAD (mm) point → image (px)."""
    pt = np.array([cad_x, cad_y, 1.0], dtype=np.float64)
    projected = M @ pt
    if abs(projected[2]) > 1e-9:
        projected = projected / projected[2]
    return float(projected[0]), float(projected[1])


# ---------------------------------------------------------------------------
# Hungarian-assignment matching (fresh detection path)
# ---------------------------------------------------------------------------

def _match_circles(
    cad_circles: list[dict],
    img_circles: list[ImageCircle],
    M_cad2img: np.ndarray,
    scale_px_per_mm: float,
    pos_tol_px: float = 40.0,
    radius_ratio_tol: float = 0.40,
) -> list[MatchedPair]:
    if not cad_circles or not img_circles:
        return []
    n_cad = len(cad_circles)
    n_img = len(img_circles)
    INF = 1e9
    cost = np.full((n_cad, n_img), INF)

    for i, cc in enumerate(cad_circles):
        px_c, py_c = _project_cad_to_image(cc["cx"], cc["cy"], M_cad2img)
        r_cad_px = cc["r"] * scale_px_per_mm
        for j, ic in enumerate(img_circles):
            dist = math.hypot(px_c - ic.cx_px, py_c - ic.cy_px)
            if dist > pos_tol_px:
                continue
            ratio = ic.radius_px / r_cad_px if r_cad_px > 0 else INF
            if ratio < (1.0 - radius_ratio_tol) or ratio > (1.0 + radius_ratio_tol):
                continue
            cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: list[MatchedPair] = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= INF:
            continue
        cc = cad_circles[i]
        ic = img_circles[j]
        pairs.append(MatchedPair(
            feature_type="circle",
            label=f"circle_{i+1}",
            cad_value_mm=cc["r"],
            image_value_px=ic.radius_px,
            cad_pos=(cc["cx"], cc["cy"]),
            image_pos_px=(ic.cx_px, ic.cy_px),
            scale_px_per_mm=scale_px_per_mm,
            match_distance_px=cost[i, j],
        ))
    logger.debug(f"Circle matching: {len(pairs)}/{n_cad} CAD circles matched")
    return pairs


def _match_rects(
    cad_rects: list[dict],
    img_rects: list[ImageRect],
    M_cad2img: np.ndarray,
    scale_px_per_mm: float,
    pos_tol_px: float = 40.0,
    size_ratio_tol: float = 0.40,
) -> list[MatchedPair]:
    if not cad_rects or not img_rects:
        return []
    n_cad = len(cad_rects)
    n_img = len(img_rects)
    INF = 1e9
    cost = np.full((n_cad, n_img), INF)

    for i, cr in enumerate(cad_rects):
        cad_cx = (cr["x1"] + cr["x2"]) / 2.0
        cad_cy = (cr["y1"] + cr["y2"]) / 2.0
        px_c, py_c = _project_cad_to_image(cad_cx, cad_cy, M_cad2img)
        cad_w_px = abs(cr["x2"] - cr["x1"]) * scale_px_per_mm
        cad_h_px = abs(cr["y2"] - cr["y1"]) * scale_px_per_mm
        for j, ir in enumerate(img_rects):
            dist = math.hypot(px_c - ir.cx_px, py_c - ir.cy_px)
            if dist > pos_tol_px:
                continue
            rw = ir.width_px  / cad_w_px if cad_w_px > 0 else INF
            rh = ir.height_px / cad_h_px if cad_h_px > 0 else INF
            if (rw < 1.0 - size_ratio_tol or rw > 1.0 + size_ratio_tol or
                    rh < 1.0 - size_ratio_tol or rh > 1.0 + size_ratio_tol):
                continue
            cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: list[MatchedPair] = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= INF:
            continue
        cr = cad_rects[i]
        ir = img_rects[j]
        pairs.append(MatchedPair(
            feature_type="rect",
            label=f"rect_{i+1}",
            cad_value_mm=(abs(cr["x2"] - cr["x1"]), abs(cr["y2"] - cr["y1"])),
            image_value_px=(ir.width_px, ir.height_px),
            cad_pos=((cr["x1"]+cr["x2"])/2.0, (cr["y1"]+cr["y2"])/2.0),
            image_pos_px=(ir.cx_px, ir.cy_px),
            scale_px_per_mm=scale_px_per_mm,
            match_distance_px=cost[i, j],
        ))
    logger.debug(f"Rect matching: {len(pairs)}/{n_cad} CAD rects matched")
    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolution_pos_tol(gray: np.ndarray) -> float:
    """
    Compute a position tolerance (px) that scales with image resolution.
    Keeps tolerance proportional to image diagonal: ~4% of diagonal,
    clamped to [20, 80] px.
    """
    h, w = gray.shape[:2]
    diag = math.hypot(h, w)
    return max(20.0, min(80.0, diag * 0.04))


def match_features(
    cad_features: CADFeatureSet,
    real_gray: np.ndarray,
    M_cad2img: np.ndarray,
    scale_px_per_mm: float,
    existing_verification: Optional[list[dict]] = None,
    total_scale: float = 0.0,
) -> list[MatchedPair]:
    """
    Match CAD features to image features.

    Parameters
    ----------
    cad_features          : parsed CAD feature set
    real_gray             : grayscale image (uint8, H×W)
    M_cad2img             : 3×3 matrix mapping CAD mm → image px
    scale_px_per_mm       : blueprint scale (from compute_M_dxf2bp)
    existing_verification : Stage-2 verification results
                            {kind, cx_px, cy_px, r_px, x1_px, y1_px,
                             x2_px, y2_px, found, label, ...}
    total_scale           : DXF-mm → final-image-px combined scale.
                            Used to populate MatchedPair.scale_px_per_mm
                            so that measurement.py converts correctly.
                            Falls back to scale_px_per_mm if 0.

    Returns
    -------
    list[MatchedPair]
    """
    # Determine the correct pixel-to-mm conversion scale.
    # total_scale = scale_px_per_mm * cad2edge_sc * align_sc
    # r_px in existing_verification was computed as feat_r_mm * total_scale,
    # so to recover mm we need to divide by total_scale, not by scale_px_per_mm.
    px_scale = total_scale if total_scale > 0 else scale_px_per_mm

    pairs: list[MatchedPair] = []

    # Resolution-aware position tolerance (used in existing_verification path too)
    img_diag = math.hypot(real_gray.shape[0], real_gray.shape[1])
    pos_tol_px = max(20.0, min(80.0, img_diag * 0.04))
    logger.debug(f"match_features: pos_tol_px={pos_tol_px:.1f} (img_diag={img_diag:.1f})")

    # ── Fast path: reuse existing Stage-2 verification results ───────────
    if existing_verification:
        logger.debug(f"Reusing {len(existing_verification)} existing verification results")

        for i, feat in enumerate(existing_verification):
            kind = feat.get("kind")

            if kind == "circle":
                r_px  = float(feat.get("r_px", 1))
                cx_px = float(feat.get("cx_px", 0))
                cy_px = float(feat.get("cy_px", 0))

                # ── Bug 2 fix: match CAD circle by position AND radius ────
                # r_px = cad_r_mm * total_scale  →  expected_r_mm = r_px / px_scale
                expected_r_mm = r_px / px_scale if px_scale > 0 else 0.0
                cad_r_mm = expected_r_mm
                cad_c    = (0.0, 0.0)
                best_score = 1e9

                for cc in cad_features.raw_circles:
                    px_p, py_p = _project_cad_to_image(cc["cx"], cc["cy"], M_cad2img)
                    pos_dist = math.hypot(px_p - cx_px, py_p - cy_px)
                    # radius proximity in mm
                    r_dist   = abs(cc["r"] - expected_r_mm)
                    # Combined score: position (px) + 10× radius mismatch (mm)
                    score = pos_dist + 10.0 * r_dist
                    if score < best_score:
                        best_score = score
                        cad_c    = (cc["cx"], cc["cy"])
                        cad_r_mm = cc["r"]

                # ── Bug 3 fix: if no raw_circles (box parts), derive CAD pos
                # from the BOX_FEATURES geometry via the combined transform ─
                if not cad_features.raw_circles:
                    # Back-project the image point to CAD coordinates using
                    # the inverse of M_cad2img.
                    M_inv = np.linalg.inv(M_cad2img)
                    pt_img = np.array([cx_px, cy_px, 1.0], dtype=np.float64)
                    pt_cad = M_inv @ pt_img
                    if abs(pt_cad[2]) > 1e-9:
                        pt_cad /= pt_cad[2]
                    cad_c    = (float(pt_cad[0]), float(pt_cad[1]))
                    cad_r_mm = expected_r_mm   # no better info available

                pairs.append(MatchedPair(
                    feature_type="circle",
                    label=feat.get("label", f"feature_{i+1}"),
                    cad_value_mm=cad_r_mm,
                    image_value_px=r_px,
                    cad_pos=cad_c,
                    image_pos_px=(cx_px, cy_px),
                    scale_px_per_mm=px_scale,   # ← correct inversion scale
                    match_distance_px=best_score,
                ))

            elif kind == "rect":
                x1_px   = float(feat.get("x1_px", 0))
                y1_px   = float(feat.get("y1_px", 0))
                x2_px   = float(feat.get("x2_px", 0))
                y2_px   = float(feat.get("y2_px", 0))
                w_px    = abs(x2_px - x1_px)
                h_px    = abs(y2_px - y1_px)
                cx_px_c = (x1_px + x2_px) / 2.0
                cy_px_c = (y1_px + y2_px) / 2.0

                # ── Bug 3 fix: find CAD rect by projected centre proximity ─
                cad_c  = (0.0, 0.0)
                cad_wh: tuple[float, float] = (w_px / px_scale, h_px / px_scale)
                best_dist = 1e9

                for cr in cad_features.rects:
                    px_p, py_p = _project_cad_to_image(cr.cx, cr.cy, M_cad2img)
                    d = math.hypot(px_p - cx_px_c, py_p - cy_px_c)
                    if d < best_dist:
                        best_dist = d
                        cad_c  = (cr.cx, cr.cy)
                        cad_wh = (cr.width, cr.height)

                # If no structured rects found, back-project image centre
                if not cad_features.rects:
                    M_inv = np.linalg.inv(M_cad2img)
                    pt_img = np.array([cx_px_c, cy_px_c, 1.0], dtype=np.float64)
                    pt_cad = M_inv @ pt_img
                    if abs(pt_cad[2]) > 1e-9:
                        pt_cad /= pt_cad[2]
                    cad_c  = (float(pt_cad[0]), float(pt_cad[1]))
                    cad_wh = (w_px / px_scale, h_px / px_scale)

                pairs.append(MatchedPair(
                    feature_type="rect",
                    label=feat.get("label", f"feature_{i+1}"),
                    cad_value_mm=cad_wh,
                    image_value_px=(w_px, h_px),
                    cad_pos=cad_c,
                    image_pos_px=(cx_px_c, cy_px_c),
                    scale_px_per_mm=px_scale,   # ← correct inversion scale
                    match_distance_px=best_dist,
                ))

        if pairs:
            logger.debug(f"Matched {len(pairs)} features from existing verification")
            return pairs

    # ── Fallback: fresh detection ─────────────────────────────────────────
    logger.debug("Running fresh feature detection on image")

    img_circles = _detect_circles_in_image(real_gray)
    img_rects   = _detect_rects_in_image(real_gray)

    raw_rects = [
        {"x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2}
        for r in cad_features.rects
    ]

    # Resolution-aware position tolerance
    img_diag = math.hypot(real_gray.shape[0], real_gray.shape[1])
    pos_tol_px = max(20.0, min(80.0, img_diag * 0.04))
    logger.debug(f"Resolution-aware pos_tol_px={pos_tol_px:.1f} (img_diag={img_diag:.1f})")

    circle_pairs = _match_circles(
        cad_features.raw_circles, img_circles,
        M_cad2img, scale_px_per_mm,
        pos_tol_px=pos_tol_px,
    )
    rect_pairs = _match_rects(
        raw_rects, img_rects,
        M_cad2img, scale_px_per_mm,
        pos_tol_px=pos_tol_px,
    )

    pairs = circle_pairs + rect_pairs

    if len(pairs) < MIN_MATCHED_PAIRS:
        raise ValueError(
            f"Feature matching failed: only {len(pairs)} pairs found "
            f"(minimum {MIN_MATCHED_PAIRS} required)"
        )

    logger.debug(f"Total matched pairs: {len(pairs)}")
    return pairs
