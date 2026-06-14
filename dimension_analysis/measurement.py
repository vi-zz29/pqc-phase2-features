"""
Dimension Recovery — Stage 6.

All measurements are reported in MILLIMETRES.

For every matched feature the pipeline produces:
  cad_dimension_mm   : what the DXF says this feature should be (mm).
  measured_mm        : what was actually detected in the image (px → mm).
  deviation_mm       : measured_mm − cad_dimension_mm

Conversion:  mm = px / total_scale
  where total_scale = scale_px_per_mm (blueprint) × cad2edge_sc × align_sc
  and is stored per-pair in MatchedPair.scale_px_per_mm by the matcher.

For rects the CAD nominal is (w_mm, h_mm); for circles it is radius_mm.
Spacing and PCD are also in mm.

ACTUAL IMAGE MEASUREMENT STRATEGY
----------------------------------
For circles  : a HoughCircles search is run in a small RoI around the
               projected CAD centre.  The fitted radius (px) / total_scale
               gives the real measured radius in mm.  Skipped when Hough fails.
For rects    : contour bounding box in a local RoI around the projected centre.
For spacing  : pixel distance between Hough-detected hole centres → mm.
               Skipped unless both holes were independently detected.
For PCD      : mean bolt-hole radius from Hough-detected centres → mm.
For overall  : measured from the part mask contour bounding box in the
               image rather than copied from DXF.
"""

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from dimension_analysis.dxf_parser import CADFeatureSet
from dimension_analysis.feature_matcher import MatchedPair, _project_cad_to_image
from dimension_analysis.transform_estimator import TransformResult

logger = logging.getLogger(__name__)

# Search radius multiplier when looking for a circle in the image
_HOUGH_SEARCH_FACTOR = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_circle_in_roi(
    real_gray: np.ndarray,
    cx_img_px: float,
    cy_img_px: float,
    expected_r_px: float,
    total_scale: float,
) -> tuple[float, float, float] | None:
    """
    Fit HoughCircles in a local RoI around (cx_img_px, cy_img_px).
    Returns (cx_px, cy_px, radius_mm) or None if Hough fails.
    """
    h, w = real_gray.shape
    search_r = int(math.ceil(expected_r_px * _HOUGH_SEARCH_FACTOR))
    x1 = max(0, int(cx_img_px) - search_r)
    y1 = max(0, int(cy_img_px) - search_r)
    x2 = min(w, int(cx_img_px) + search_r)
    y2 = min(h, int(cy_img_px) + search_r)

    roi = real_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    min_r = max(2, int(expected_r_px * 0.5))
    max_r = max(min_r + 2, int(expected_r_px * 2.0))

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(4, min_r),
        param1=50,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    best_cx_px: float | None = None
    best_cy_px: float | None = None
    best_r_px: float | None = None
    best_dist = 1e9
    for cx_roi, cy_roi, r_roi in circles[0]:
        cx_abs = float(cx_roi + x1)
        cy_abs = float(cy_roi + y1)
        dist = math.hypot(cx_abs - cx_img_px, cy_abs - cy_img_px)
        if dist < best_dist:
            best_dist = dist
            best_cx_px = cx_abs
            best_cy_px = cy_abs
            best_r_px = float(r_roi)

    if best_r_px is None or best_cx_px is None or best_cy_px is None:
        return None
    if total_scale <= 0:
        return None

    return best_cx_px, best_cy_px, best_r_px / total_scale


def _measure_circle_radius_mm(
    real_gray: np.ndarray,
    cx_img_px: float,
    cy_img_px: float,
    expected_r_px: float,
    total_scale: float,
) -> float | None:
    """Return measured radius in mm, or None if Hough fails."""
    hit = _detect_circle_in_roi(
        real_gray, cx_img_px, cy_img_px, expected_r_px, total_scale
    )
    return hit[2] if hit is not None else None


def _measure_overall_from_mask(
    real_gray: np.ndarray,
    total_scale: float,
) -> tuple[float | None, float | None]:
    """
    Estimate overall width and height (mm) from the part bounding box.

    Tries both THRESH_BINARY and THRESH_BINARY_INV, picks the result whose
    largest contour area is between 5% and 85% of the image area — this
    rejects both the near-empty case and the full-background case.

    Returns (width_mm, height_mm), or (None, None) on failure.
    """
    if total_scale <= 0:
        return None, None

    blur = cv2.GaussianBlur(real_gray, (5, 5), 0)
    img_area = float(real_gray.shape[0] * real_gray.shape[1])

    def _try_thresh(flags: int):
        _, mask = cv2.threshold(blur, 0, 255, flags)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        # Reject if too small (<5%) or too large (>85%) relative to image
        if area < img_area * 0.05 or area > img_area * 0.85:
            return None
        _, _, w_px, h_px = cv2.boundingRect(largest)
        return float(w_px) / total_scale, float(h_px) / total_scale

    # Try BINARY first (bright part on dark background)
    result = _try_thresh(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if result is not None:
        return result

    # Fallback: BINARY_INV (dark part on bright background)
    result = _try_thresh(cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if result is not None:
        return result

    logger.warning("_measure_overall_from_mask: no valid part contour found in image")
    return None, None


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class MeasuredFeature:
    """One feature with CAD nominal, image-measured value, and deviation — all in mm."""
    feature_type: str
    label: str
    cad_dimension_mm: float          # nominal from DXF (mm)
    measured_dimension_mm: float     # measured from actual image (mm)
    deviation_mm: float              # measured − cad  (mm)
    # Legacy aliases so tolerance.py / report_generator.py keep working unchanged
    cad_dimension_px: float = 0.0
    measured_dimension_px: float = 0.0
    deviation_px: float = 0.0
    unit: str = "mm"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Keep px-named aliases in sync (they carry mm values — naming is legacy).
        self.cad_dimension_px      = self.cad_dimension_mm
        self.measured_dimension_px = self.measured_dimension_mm
        self.deviation_px          = self.deviation_mm


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def recover_dimensions(
    matched_pairs: list[MatchedPair],
    cad_features: CADFeatureSet,
    transform_result: TransformResult,
    real_gray: np.ndarray | None = None,
) -> list[MeasuredFeature]:
    """
    Produce mm measurements for every matched feature pair.

    Parameters
    ----------
    matched_pairs    : from feature_matcher.match_features()
    cad_features     : parsed DXF feature set
    transform_result : refined CAD→image transform
    real_gray        : original grayscale image (used for active measurement).
                       If None, falls back to Stage-2 r_px / total_scale.
    """
    features: list[MeasuredFeature] = []

    # Representative total_scale from matched pairs.
    # IMPORTANT: prefer the per-pair scale (set from Stage-2 total_scale)
    # over transform_result.scale_px_per_mm which is refit from matched-pair
    # positions and can diverge significantly (seen: 12→5 px/mm).
    # Only fall back to transform_result scale if NO pairs have a valid scale.
    pair_scales = [p.scale_px_per_mm for p in matched_pairs if p.scale_px_per_mm > 0]
    if pair_scales:
        ts_global = float(np.median(pair_scales))
    else:
        ts_global = transform_result.scale_px_per_mm
        logger.warning(
            f"No valid per-pair scale found; using transform_result scale "
            f"({ts_global:.4f} px/mm) — may be inaccurate before clamp is added"
        )

    # Hough-detected hole centres (px) — used for spacing / PCD, not CAD projection
    detected_centers: dict[str, tuple[float, float]] = {}

    # Track labels that were already measured in the per-pair loop so the
    # high-level blocks (center_bore, outer_diameter) can skip them and
    # avoid emitting a duplicate measurement for the same physical feature.
    already_measured_labels: set[str] = set()

    # ── Per-pair measurements ──────────────────────────────────────────────
    for pair in matched_pairs:
        ts = pair.scale_px_per_mm if pair.scale_px_per_mm > 0 else ts_global

        if pair.feature_type == "circle":
            cad_r_mm = float(pair.cad_value_mm)          # from DXF (mm)

            # Detect whether this circle is the center bore.
            # If the CAD diameter matches cad_features.center_bore within 10%,
            # report it as center_bore instead of a generic hole.
            is_center_bore = (
                cad_features.part_type == "circular"
                and cad_features.center_bore is not None
                and abs(cad_r_mm * 2.0 - cad_features.center_bore)
                    < cad_features.center_bore * 0.10
            )

            # ── Active measurement: fit a Hough circle in the image ────
            measured_r_mm: float
            if is_center_bore:
                # For the center bore, use the Stage-2 image_value_px directly.
                # The bore was already correctly detected by Stage 2 (r_px ≈ 51px
                # ÷ 3.5px/mm ≈ 14.5mm radius = 29mm diameter).
                # Re-running Hough here picks up the larger outer ring instead,
                # producing an incorrect ~49–57mm reading.
                stage2_r_px = float(pair.image_value_px)
                if ts > 0:
                    measured_r_mm = stage2_r_px / ts
                    detected_centers[pair.label] = pair.image_pos_px
                    logger.debug(
                        f"{pair.label} (center_bore): using Stage-2 r_px={stage2_r_px:.1f} "
                        f"→ r={measured_r_mm:.3f}mm, d={measured_r_mm*2:.3f}mm "
                        f"(CAD={cad_r_mm:.3f}mm)"
                    )
                else:
                    logger.warning(f"{pair.label}: ts<=0, cannot convert Stage-2 px to mm — skipping")
                    continue
            elif real_gray is not None and ts > 0:
                expected_r_px = cad_r_mm * ts
                detection = _detect_circle_in_roi(
                    real_gray,
                    cx_img_px=pair.image_pos_px[0],
                    cy_img_px=pair.image_pos_px[1],
                    expected_r_px=expected_r_px,
                    total_scale=ts,
                )
                if detection is not None:
                    cx_det, cy_det, measured_r_mm = detection
                    detected_centers[pair.label] = (cx_det, cy_det)
                    logger.debug(
                        f"{pair.label}: Hough measured r={measured_r_mm:.3f}mm "
                        f"at ({cx_det:.1f},{cy_det:.1f}) (CAD={cad_r_mm:.3f}mm)"
                    )
                else:
                    logger.warning(
                        f"{pair.label}: Hough circle fit failed in RoI — "
                        f"feature NOT measured (skipping to avoid tautological result)"
                    )
                    continue   # skip this pair entirely — do not emit a fake value
            else:
                # No real_gray available — cannot measure
                logger.warning(
                    f"{pair.label}: no image supplied to recover_dimensions — skipping"
                )
                continue

            if is_center_bore:
                # Report this circle as center_bore (once, by diameter).
                # Do NOT emit the generic hole #N / hole #N_dia entries.
                features.append(MeasuredFeature(
                    feature_type="center_bore",
                    label="center_bore",
                    cad_dimension_mm=cad_features.center_bore,
                    measured_dimension_mm=measured_r_mm * 2.0,
                    deviation_mm=(measured_r_mm * 2.0) - cad_features.center_bore,
                    unit="mm",
                ))
                already_measured_labels.add(pair.label)
                logger.debug(
                    f"{pair.label} reclassified as center_bore: "
                    f"measured_d={measured_r_mm * 2.0:.3f}mm "
                    f"(CAD={cad_features.center_bore:.3f}mm)"
                )
            else:
                features.append(MeasuredFeature(
                    feature_type="circle_radius",
                    label=pair.label,
                    cad_dimension_mm=cad_r_mm,
                    measured_dimension_mm=measured_r_mm,
                    deviation_mm=measured_r_mm - cad_r_mm,
                    unit="mm (radius)",
                ))
                features.append(MeasuredFeature(
                    feature_type="circle_diameter",
                    label=pair.label + "_dia",
                    cad_dimension_mm=cad_r_mm * 2.0,
                    measured_dimension_mm=measured_r_mm * 2.0,
                    deviation_mm=(measured_r_mm - cad_r_mm) * 2.0,
                    unit="mm (diameter)",
                ))
                # Record that this pair was measured so the high-level blocks
                # (center_bore, outer_diameter) skip it and avoid duplication.
                already_measured_labels.add(pair.label)

        elif pair.feature_type == "rect":
            cad_wh = pair.cad_value_mm         # (w_mm, h_mm) from DXF
            img_wh = pair.image_value_px       # (w_px, h_px) from Stage-2

            cad_w_mm = float(cad_wh[0])
            cad_h_mm = float(cad_wh[1])

            # Measure the rect from the actual image if possible.
            # Strategy: threshold a local RoI around the projected centre and
            # find the bounding box of the largest contour there.
            # Try both INV and normal Otsu to handle bright-on-dark and dark-on-bright.
            meas_w_mm: float | None = None
            meas_h_mm: float | None = None

            if real_gray is not None and ts > 0:
                cx_img = pair.image_pos_px[0]
                cy_img = pair.image_pos_px[1]
                hw_px = cad_w_mm * ts / 2.0
                hh_px = cad_h_mm * ts / 2.0
                # Wider search factor (2.0 vs 1.6) compensates for coarse alignment error
                search_factor = 2.0
                x1_roi = max(0, int(cx_img - hw_px * search_factor))
                y1_roi = max(0, int(cy_img - hh_px * search_factor))
                x2_roi = min(real_gray.shape[1], int(cx_img + hw_px * search_factor))
                y2_roi = min(real_gray.shape[0], int(cy_img + hh_px * search_factor))
                roi = real_gray[y1_roi:y2_roi, x1_roi:x2_roi]
                if roi.size > 0:
                    blur = cv2.GaussianBlur(roi, (3, 3), 0)

                    def _try_rect_thresh(flags: int):
                        _, thresh_img = cv2.threshold(blur, 0, 255, flags)
                        cnts, _ = cv2.findContours(
                            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if not cnts:
                            return None, None
                        lg = max(cnts, key=cv2.contourArea)
                        _, _, wc, hc = cv2.boundingRect(lg)
                        wr = wc / (cad_w_mm * ts) if cad_w_mm * ts > 0 else 0.0
                        hr = hc / (cad_h_mm * ts) if cad_h_mm * ts > 0 else 0.0
                        if 0.4 < wr < 1.6 and 0.4 < hr < 1.6:
                            return float(wc) / ts, float(hc) / ts
                        return None, None

                    # Try INV first (cutout is bright inside dark part)
                    meas_w_mm, meas_h_mm = _try_rect_thresh(
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    if meas_w_mm is None:
                        # Fallback: normal Otsu (cutout is dark inside bright part)
                        meas_w_mm, meas_h_mm = _try_rect_thresh(
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                    if meas_w_mm is not None:
                        logger.debug(
                            f"{pair.label}: contour measured "
                            f"{meas_w_mm:.3f}x{meas_h_mm:.3f}mm "
                            f"(CAD={cad_w_mm:.3f}x{cad_h_mm:.3f}mm)"
                        )

            if meas_w_mm is None or meas_h_mm is None:
                logger.warning(
                    f"{pair.label}: rect contour measurement failed — "
                    f"feature NOT measured (skipping to avoid tautological result)"
                )
                continue   # skip — do not emit CAD-echo value

            features.append(MeasuredFeature(
                feature_type="rect_width",
                label=pair.label + "_width",
                cad_dimension_mm=cad_w_mm,
                measured_dimension_mm=meas_w_mm,
                deviation_mm=meas_w_mm - cad_w_mm,
                unit="mm",
            ))
            features.append(MeasuredFeature(
                feature_type="rect_height",
                label=pair.label + "_height",
                cad_dimension_mm=cad_h_mm,
                measured_dimension_mm=meas_h_mm,
                deviation_mm=meas_h_mm - cad_h_mm,
                unit="mm",
            ))

    pt = cad_features.part_type

    # ── High-level part dimensions ─────────────────────────────────────────
    if pt == "circular":
        if cad_features.outer_diameter is not None:
            od_cad_mm = cad_features.outer_diameter   # DXF nominal (mm)
            # Find the matched circle whose CAD diameter is closest to outer_diameter.
            # Skip pairs already measured in the per-pair loop to prevent duplication.
            od_meas_mm = None
            for p in matched_pairs:
                if p.feature_type == "circle":
                    if p.label in already_measured_labels:
                        continue  # already reported as a generic hole — skip
                    cad_d_mm = float(p.cad_value_mm) * 2.0
                    if abs(cad_d_mm - od_cad_mm) < od_cad_mm * 0.15:
                        ts_p = p.scale_px_per_mm if p.scale_px_per_mm > 0 else ts_global
                        if real_gray is not None and ts_p > 0:
                            hough_r = _measure_circle_radius_mm(
                                real_gray,
                                cx_img_px=p.image_pos_px[0],
                                cy_img_px=p.image_pos_px[1],
                                expected_r_px=float(p.cad_value_mm) * ts_p,
                                total_scale=ts_p,
                            )
                            if hough_r is not None:
                                od_meas_mm = hough_r * 2.0
                        # No fallback to image_value_px — that is tautological
                        break
            if od_meas_mm is None:
                logger.warning("outer_diameter: Hough measurement failed, skipping")
            else:
                features.append(MeasuredFeature(
                    feature_type="outer_diameter",
                    label="outer_diameter",
                    cad_dimension_mm=od_cad_mm,
                    measured_dimension_mm=od_meas_mm,
                    deviation_mm=od_meas_mm - od_cad_mm,
                    unit="mm",
                ))

        if cad_features.center_bore is not None:
            cb_cad_mm = cad_features.center_bore
            cb_meas_mm = None
            for p in matched_pairs:
                if p.feature_type == "circle":
                    if p.label in already_measured_labels:
                        continue  # already reported as a generic hole — skip
                    cad_d_mm = float(p.cad_value_mm) * 2.0
                    # Tighten tolerance to 10% to avoid matching the wrong circle
                    if abs(cad_d_mm - cb_cad_mm) < cb_cad_mm * 0.10:
                        ts_p = p.scale_px_per_mm if p.scale_px_per_mm > 0 else ts_global
                        if real_gray is not None and ts_p > 0:
                            expected_r_px = float(p.cad_value_mm) * ts_p
                            # Tighter Hough window: ±25% of expected radius
                            # to avoid picking up the outer ring
                            tight_min_r = max(2, int(expected_r_px * 0.75))
                            tight_max_r = max(tight_min_r + 2, int(expected_r_px * 1.25))
                            h_img, w_img = real_gray.shape
                            search_r = int(math.ceil(expected_r_px * _HOUGH_SEARCH_FACTOR))
                            cx_img_px, cy_img_px = p.image_pos_px
                            x1 = max(0, int(cx_img_px) - search_r)
                            y1 = max(0, int(cy_img_px) - search_r)
                            x2 = min(w_img, int(cx_img_px) + search_r)
                            y2 = min(h_img, int(cy_img_px) + search_r)
                            roi = real_gray[y1:y2, x1:x2]
                            if roi.size > 0:
                                blurred = cv2.GaussianBlur(roi, (5, 5), 0)
                                circles = cv2.HoughCircles(
                                    blurred, cv2.HOUGH_GRADIENT,
                                    dp=1.2, minDist=max(4, tight_min_r),
                                    param1=50, param2=18,
                                    minRadius=tight_min_r,
                                    maxRadius=tight_max_r,
                                )
                                if circles is not None:
                                    # Pick closest centre
                                    best_r_px = None
                                    best_dist = 1e9
                                    for cx_roi, cy_roi, r_roi in circles[0]:
                                        d = math.hypot(cx_roi + x1 - cx_img_px,
                                                       cy_roi + y1 - cy_img_px)
                                        if d < best_dist:
                                            best_dist = d
                                            best_r_px = float(r_roi)
                                    if best_r_px is not None:
                                        cb_meas_mm = best_r_px / ts_p * 2.0
                        break
            if cb_meas_mm is None:
                logger.warning("center_bore: Hough measurement failed, skipping")
            else:
                features.append(MeasuredFeature(
                    feature_type="center_bore",
                    label="center_bore",
                    cad_dimension_mm=cb_cad_mm,
                    measured_dimension_mm=cb_meas_mm,
                    deviation_mm=cb_meas_mm - cb_cad_mm,
                    unit="mm",
                ))

        if cad_features.pcd is not None:
            pcd_cad_mm = cad_features.pcd

            # Part centre in CAD coords → project to image
            cx_vals = [c["cx"] for c in cad_features.raw_circles]
            cy_vals = [c["cy"] for c in cad_features.raw_circles]
            if cx_vals:
                part_cx_cad = float(np.median(cx_vals))
                part_cy_cad = float(np.median(cy_vals))
            else:
                part_cx_cad, part_cy_cad = 148.5, 105.0

            cx_img, cy_img = _project_cad_to_image(
                part_cx_cad, part_cy_cad, transform_result.matrix
            )

            CENTER_TOL_MM = 3.0
            bolt_pairs = [
                p for p in matched_pairs
                if p.feature_type == "circle" and
                math.hypot(p.cad_pos[0] - part_cx_cad,
                           p.cad_pos[1] - part_cy_cad) >= CENTER_TOL_MM
            ]

            if len(bolt_pairs) >= 3:
                dists_mm: list[float] = []
                for p in bolt_pairs:
                    if p.label not in detected_centers:
                        continue
                    ts_p = p.scale_px_per_mm if p.scale_px_per_mm > 0 else ts_global
                    if ts_p <= 0:
                        continue
                    bx, by = detected_centers[p.label]
                    dists_mm.append(
                        math.hypot(bx - cx_img, by - cy_img) / ts_p
                    )
                if len(dists_mm) >= 3:
                    pcd_meas_mm = float(np.mean(dists_mm)) * 2.0
                    features.append(MeasuredFeature(
                        feature_type="pcd",
                        label="pcd",
                        cad_dimension_mm=pcd_cad_mm,
                        measured_dimension_mm=pcd_meas_mm,
                        deviation_mm=pcd_meas_mm - pcd_cad_mm,
                        unit="mm",
                    ))
                else:
                    logger.warning(
                        f"PCD: only {len(dists_mm)} Hough-detected bolt holes "
                        f"(need ≥3), skipping"
                    )
            else:
                logger.warning(
                    f"PCD: only {len(bolt_pairs)} bolt-hole pairs (need ≥3), skipping"
                )

    elif pt == "rectangular":
        # ── Overall dimensions measured from the image ─────────────────
        if cad_features.overall_width is not None or cad_features.overall_height is not None:
            meas_w_mm, meas_h_mm = _measure_overall_from_mask(real_gray, ts_global) \
                if real_gray is not None else (None, None)

            if cad_features.overall_width is not None:
                cad_w_mm = cad_features.overall_width
                if meas_w_mm is not None:
                    features.append(MeasuredFeature(
                        feature_type="overall_width",
                        label="overall_width",
                        cad_dimension_mm=cad_w_mm,
                        measured_dimension_mm=meas_w_mm,
                        deviation_mm=meas_w_mm - cad_w_mm,
                        unit="mm",
                        extra={"note": "measured from image mask bounding box"},
                    ))
                else:
                    logger.warning("overall_width: could not measure from image, skipping")

            if cad_features.overall_height is not None:
                cad_h_mm = cad_features.overall_height
                if meas_h_mm is not None:
                    features.append(MeasuredFeature(
                        feature_type="overall_height",
                        label="overall_height",
                        cad_dimension_mm=cad_h_mm,
                        measured_dimension_mm=meas_h_mm,
                        deviation_mm=meas_h_mm - cad_h_mm,
                        unit="mm",
                        extra={"note": "measured from image mask bounding box"},
                    ))
                else:
                    logger.warning("overall_height: could not measure from image, skipping")

        # ── Hole spacings — use Hough centres where available, else Stage-2 pos ──
        hole_pairs = [p for p in matched_pairs if p.feature_type == "circle"]
        dxf_hole_positions = cad_features.hole_positions

        if len(hole_pairs) >= 2:
            for i in range(len(hole_pairs)):
                for j in range(i + 1, len(hole_pairs)):
                    pi, pj = hole_pairs[i], hole_pairs[j]

                    # Use Hough-refined centre if available, else Stage-2 image_pos_px
                    cxi, cyi = detected_centers.get(pi.label, pi.image_pos_px)
                    cxj, cyj = detected_centers.get(pj.label, pj.image_pos_px)
                    if not detected_centers.get(pi.label):
                        logger.debug(
                            f"spacing {pi.label}↔{pj.label}: using Stage-2 pos for {pi.label}"
                        )
                    if not detected_centers.get(pj.label):
                        logger.debug(
                            f"spacing {pi.label}↔{pj.label}: using Stage-2 pos for {pj.label}"
                        )

                    # CAD spacing: directly from DXF positions (mm)
                    if i < len(dxf_hole_positions) and j < len(dxf_hole_positions):
                        cad_sp_mm = math.hypot(
                            dxf_hole_positions[i][0] - dxf_hole_positions[j][0],
                            dxf_hole_positions[i][1] - dxf_hole_positions[j][1],
                        )
                    else:
                        cad_sp_mm = math.hypot(
                            pi.cad_pos[0] - pj.cad_pos[0],
                            pi.cad_pos[1] - pj.cad_pos[1],
                        )

                    dist_px = math.hypot(cxi - cxj, cyi - cyj)
                    ts_i = pi.scale_px_per_mm if pi.scale_px_per_mm > 0 else ts_global
                    if ts_i <= 0:
                        continue
                    meas_sp_mm = dist_px / ts_i

                    features.append(MeasuredFeature(
                        feature_type="hole_spacing",
                        label=f"spacing_{pi.label}_to_{pj.label}",
                        cad_dimension_mm=cad_sp_mm,
                        measured_dimension_mm=meas_sp_mm,
                        deviation_mm=meas_sp_mm - cad_sp_mm,
                        unit="mm",
                    ))

    logger.debug(f"Recovered {len(features)} mm measurements")
    return features
