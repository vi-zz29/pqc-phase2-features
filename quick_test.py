"""
quick_test.py  —  Single entry point for the full inspection pipeline.

Pipeline (strict gating — any FAIL stops execution for that image):
  Stage 1: Identification
  Stage 2: Feature Extraction
  Stage 3: DXF Parsing
  Stage 4: CAD-Image Feature Matching
  Stage 5: Transformation Estimation
  Stage 6: Dimension Recovery
  Stage 7: Tolerance Verification
  Stage 8: Inspection Report Generation

Usage:
    python quick_test.py
"""

import logging
import math
import sys
import traceback

import cv2
import numpy as np
from pathlib import Path

from cad_image_alignment import align, match_best_template

# ── Dimension-analysis pipeline (Stages 3-8) ──────────────────────────────
from dimension_analysis import (
    parse_dxf,
    match_features,
    estimate_transform,
    recover_dimensions,
    verify_tolerances,
    generate_reports,
)
from dimension_analysis.dxf_utils import parse_dxf_raw

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUTS_DIR     = Path("inputs")
BLUEPRINTS_DIR = Path("blueprints")
DXF_DIR        = Path("dxf")
OUTPUTS_DIR    = Path("outputs")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

DXF_CENTER_CX  = 148.5
DXF_CENTER_CY  = 105.0
DXF_CENTER_TOL = 2.0


# ===========================================================================
# ── EXISTING FUNCTIONS — DO NOT MODIFY ─────────────────────────────────────
# ===========================================================================

def preprocess_cad(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load blueprint: {path}")
    inv   = cv2.bitwise_not(img)
    blur  = cv2.GaussianBlur(inv, (3, 3), 0)
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    return edges


def preprocess_real(img: np.ndarray) -> tuple:
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    real_masked = cv2.bitwise_and(img, img, mask=mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(real_masked, cv2.MORPH_GRADIENT, k3)
    _, internal = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
    internal = cv2.morphologyEx(
        internal, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    outer = cv2.Canny(mask, 50, 150)
    real_edges = cv2.bitwise_or(internal, outer)
    return real_edges, mask


def parse_dxf_circles(path: Path) -> list[dict]:
    """Parse CIRCLE entities from a DXF file. Delegates to shared dxf_utils."""
    circles, _, _ = parse_dxf_raw(path)
    return circles


def get_holes_for_view(circles: list[dict], view: str) -> list[dict]:
    """
    Return bolt holes and the centre hole from the DXF circle list.
    All circles whose centre is within DXF_CENTER_TOL of the part centre are
    excluded from the bolt-hole list — they are concentric rings (outer diameter,
    grooves, center bore). Previously only r≈14.5 was excluded, which caused
    center-bore circles to appear as bolt holes AND be re-measured as center_bore.
    """
    holes = []
    center_hole = None
    for c in circles:
        dist = math.hypot(c["cx"] - DXF_CENTER_CX, c["cy"] - DXF_CENTER_CY)
        if dist < DXF_CENTER_TOL:
            # ALL concentric circles are excluded from bolt holes.
            # Keep only the known center-through-bore for verification.
            if abs(c["r"] - 14.5) < 0.1:
                center_hole = c
            # Other concentric circles (outer ring, grooves) are silently skipped.
            continue
        if view in ("front", "top"):
            if abs(c["cx"] - 152.16) < 0.5 and abs(c["cy"] - 63.16) < 0.5:
                continue
        elif view == "rear":
            if (abs(c["cx"] - 152.16) < 0.5 and abs(c["cy"] - 146.84) < 0.5
                    and abs(c["r"] - 2.0) < 0.1):
                continue
        if c["r"] <= 10.0:
            holes.append(c)
    if center_hole is not None:
        holes.append(center_hole)
    return holes


BOX_FEATURES: dict[str, list[dict]] = {
    "box_front": [
        {"kind": "circle", "cx": 165.168750, "cy":  95.475000, "r": 3.048, "label": "#1 hole"},
        {"kind": "circle", "cx": 165.168750, "cy": 127.225000, "r": 3.048, "label": "#2 hole"},
        {"kind": "rect",
         "x1": 133.291750, "y1": 92.173000,
         "x2": 158.945750, "y2": 117.827000,
         "label": "#3 rect cutout"},
    ],
    "box_rear": [
        {"kind": "rect",
         "x1": 138.054250, "y1": 92.173000,
         "x2": 163.708250, "y2": 117.827000,
         "label": "#1 rect cutout"},
    ],
}


def compute_M_dxf2bp(blueprint_path: Path) -> tuple[np.ndarray, float]:
    img = cv2.imread(str(blueprint_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(blueprint_path))

    dark = img < 200
    row_idx = np.where(np.any(dark, axis=1))[0]
    col_idx = np.where(np.any(dark, axis=0))[0]
    if len(row_idx) == 0:
        raise ValueError(f"No content in {blueprint_path}")

    rmin, rmax = int(row_idx[0]),  int(row_idx[-1])
    cmin, cmax = int(col_idx[0]),  int(col_idx[-1])
    bp_cx = (cmin + cmax) / 2.0
    bp_cy = (rmin + rmax) / 2.0
    bp_w  = cmax - cmin
    bp_h  = rmax - rmin

    stem = blueprint_path.stem.lower()
    if "box" in stem:
        dxf_cx = 148.5
        dxf_cy = 105.0
        dxf_span_x = 173.106 - 123.894
        dxf_span_y = 145.163 -  64.838
        scale = min(bp_w / dxf_span_x, bp_h / dxf_span_y)
    else:
        dxf_cx = DXF_CENTER_CX
        dxf_cy = DXF_CENTER_CY
        scale  = min(bp_w, bp_h) / 106.0

    M = np.array([
        [ scale,      0,  bp_cx - dxf_cx * scale],
        [     0, -scale,  bp_cy + dxf_cy * scale],
        [     0,      0,  1.0                    ],
    ], dtype=np.float64)
    return M, scale


def compute_M_cad2edge(cad_edge_map: np.ndarray,
                       real_shape: tuple[int, int]) -> np.ndarray:
    rh, rw = real_shape
    ch, cw = cad_edge_map.shape

    contours, _ = cv2.findContours(
        cad_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, cw_crop, ch_crop = 0, 0, cw, ch
    if contours:
        largest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)
        margin = int(max(bw, bh) * 0.05)
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(cw, bx + bw + margin)
        y2 = min(ch, by + bh + margin)
        cw_crop = x2 - x1
        ch_crop = y2 - y1

    fit_scale = min(rw / cw_crop, rh / ch_crop)
    new_w = int(round(cw_crop * fit_scale))
    new_h = int(round(ch_crop * fit_scale))
    x_off = (rw - new_w) // 2
    y_off = (rh - new_h) // 2

    M = np.array([
        [fit_scale, 0,          -x1 * fit_scale + x_off],
        [0,         fit_scale,  -y1 * fit_scale + y_off],
        [0,         0,          1.0                     ],
    ], dtype=np.float64)
    return M


def check_hole_in_image(real_gray: np.ndarray,
                        cx_px: int, cy_px: int, r_px: int,
                        search_margin: int = 6) -> tuple[bool, float]:
    """
    Check whether a hole exists at (cx_px, cy_px) with radius r_px.

    Strategy: try the projected centre first, then jitter ±search_offset
    to handle small projection errors from coarse-only alignment.
    Returns (found, best_ratio).
    """
    h, w = real_gray.shape

    r_inner = max(2, r_px)
    # Scale annulus margin with hole size to avoid bleeding into adjacent features
    adaptive_margin = max(4, int(r_inner * 0.4))
    r_outer = r_inner + adaptive_margin

    def _ratio_at(cx: int, cy: int) -> float:
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return 1.0
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        inner_mask   = dist_sq <= r_inner ** 2
        annulus_mask = (dist_sq > r_inner ** 2) & (dist_sq <= r_outer ** 2)
        inner_px   = real_gray[inner_mask]
        annulus_px = real_gray[annulus_mask]
        if inner_px.size == 0 or annulus_px.size == 0:
            return 1.0
        mean_annulus = float(annulus_px.mean())
        if mean_annulus < 1.0:
            return 1.0
        return float(inner_px.mean()) / mean_annulus

    # Neighbourhood search: jitter up to ±search_offset pixels around projection
    # This compensates for positional error in coarse-only alignment.
    search_offset = max(4, r_inner // 2)
    best_ratio = _ratio_at(cx_px, cy_px)
    for dx in range(-search_offset, search_offset + 1, max(1, search_offset // 2)):
        for dy in range(-search_offset, search_offset + 1, max(1, search_offset // 2)):
            if dx == 0 and dy == 0:
                continue
            r = _ratio_at(cx_px + dx, cy_px + dy)
            # Keep the ratio that is furthest from 1.0 (strongest hole signal)
            if abs(r - 1.0) > abs(best_ratio - 1.0):
                best_ratio = r

    # Decision: bright hole (reflection) or dark hole (bore)
    return (best_ratio > 1.15 or best_ratio < 0.88), best_ratio


def check_rect_in_image(real_gray: np.ndarray,
                        corners_px: list[tuple[int, int]],
                        border_px: int = 8) -> tuple[bool, float]:
    h, w = real_gray.shape

    xs = [c[0] for c in corners_px]
    ys = [c[1] for c in corners_px]
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))

    if x2 <= x1 or y2 <= y1:
        return False, 1.0

    shrink = max(2, (x2 - x1) // 8)
    ix1 = min(x1 + shrink, x2 - shrink)
    ix2 = max(x1 + shrink, x2 - shrink)
    iy1 = min(y1 + shrink, y2 - shrink)
    iy2 = max(y1 + shrink, y2 - shrink)

    if ix2 <= ix1 or iy2 <= iy1:
        return False, 1.0

    inner = real_gray[iy1:iy2, ix1:ix2]

    bx1 = max(0, x1 - border_px)
    bx2 = min(w, x2 + border_px)
    by1 = max(0, y1 - border_px)
    by2 = min(h, y2 + border_px)

    outer_region = real_gray[by1:by2, bx1:bx2].copy()
    rel_ix1 = ix1 - bx1; rel_ix2 = ix2 - bx1
    rel_iy1 = iy1 - by1; rel_iy2 = iy2 - by1
    outer_region[rel_iy1:rel_iy2, rel_ix1:rel_ix2] = 0
    border_px_vals = outer_region[outer_region > 0]

    if inner.size == 0 or border_px_vals.size == 0:
        return False, 1.0

    mean_inner  = float(inner.mean())
    mean_border = float(border_px_vals.mean())

    if mean_border < 1.0:
        return False, 1.0

    ratio = mean_inner / mean_border
    return (ratio > 1.20 or ratio < 0.80), ratio


def map_dxf_point(dxf_x: float, dxf_y: float,
                  M_combined: np.ndarray) -> tuple[int, int]:
    pt = np.array([dxf_x, dxf_y, 1.0], dtype=np.float64)
    mapped = M_combined @ pt
    if abs(mapped[2]) > 1e-9:
        mapped /= mapped[2]
    return int(round(mapped[0])), int(round(mapped[1]))


def verify_box_features(real_gray: np.ndarray,
                        features: list[dict],
                        M_dxf2bp: np.ndarray,
                        M_cad2edge: np.ndarray,
                        M_align: np.ndarray,
                        scale_px_per_mm: float) -> list[dict]:
    M_combined  = M_align @ M_cad2edge @ M_dxf2bp
    align_scale = math.sqrt(M_align[0, 0] ** 2 + M_align[1, 0] ** 2)
    cad2edge_sc = math.sqrt(M_cad2edge[0, 0] ** 2 + M_cad2edge[1, 0] ** 2)
    total_scale = scale_px_per_mm * cad2edge_sc * align_scale

    results = []
    for i, feat in enumerate(features, start=1):
        if feat["kind"] == "circle":
            cx_px, cy_px = map_dxf_point(feat["cx"], feat["cy"], M_combined)
            r_px = max(3, int(round(feat["r"] * total_scale)))
            found, ratio = check_hole_in_image(real_gray, cx_px, cy_px, r_px)
            results.append({
                "idx":    i,
                "kind":   "circle",
                "label":  feat["label"],
                "cx_px":  cx_px,
                "cy_px":  cy_px,
                "r_px":   r_px,
                "found":  found,
                "ratio":  ratio,
            })

        elif feat["kind"] == "rect":
            corners = [
                map_dxf_point(feat["x1"], feat["y1"], M_combined),
                map_dxf_point(feat["x2"], feat["y1"], M_combined),
                map_dxf_point(feat["x2"], feat["y2"], M_combined),
                map_dxf_point(feat["x1"], feat["y2"], M_combined),
            ]
            found, ratio = check_rect_in_image(real_gray, corners)
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            results.append({
                "idx":    i,
                "kind":   "rect",
                "label":  feat["label"],
                "corners": corners,
                "x1_px":  min(xs), "y1_px": min(ys),
                "x2_px":  max(xs), "y2_px": max(ys),
                "found":  found,
                "ratio":  ratio,
            })

    return results


def verify_holes(real_gray: np.ndarray,
                 holes_dxf: list[dict],
                 M_dxf2bp: np.ndarray,
                 M_cad2edge: np.ndarray,
                 M_align: np.ndarray,
                 scale_px_per_mm: float) -> list[dict]:
    M_combined  = M_align @ M_cad2edge @ M_dxf2bp
    align_scale = math.sqrt(M_align[0, 0] ** 2 + M_align[1, 0] ** 2)
    cad2edge_sc = math.sqrt(M_cad2edge[0, 0] ** 2 + M_cad2edge[1, 0] ** 2)
    total_scale = scale_px_per_mm * cad2edge_sc * align_scale

    results = []
    for i, hole in enumerate(holes_dxf, start=1):
        pt     = np.array([hole["cx"], hole["cy"], 1.0], dtype=np.float64)
        mapped = M_combined @ pt
        if abs(mapped[2]) > 1e-9:
            mapped /= mapped[2]
        cx_px = int(round(mapped[0]))
        cy_px = int(round(mapped[1]))
        r_px  = max(3, int(round(hole["r"] * total_scale)))

        found, ratio = check_hole_in_image(real_gray, cx_px, cy_px, r_px)
        results.append({
            "idx":    i,
            "kind":   "circle",
            "cx_dxf": hole["cx"],
            "cy_dxf": hole["cy"],
            "r_dxf":  hole["r"],
            "cx_px":  cx_px,
            "cy_px":  cy_px,
            "r_px":   r_px,
            "found":  found,
            "ratio":  ratio,
        })
    return results


def draw_feature_verification(real_gray: np.ndarray,
                              results: list[dict],
                              title: str) -> np.ndarray:
    vis = cv2.cvtColor(real_gray, cv2.COLOR_GRAY2BGR)
    FOUND   = (0, 220, 80)
    MISSING = (0, 60, 255)
    WHITE   = (255, 255, 255)

    for r in results:
        color = FOUND if r["found"] else MISSING
        label = r.get("label") or f"#{r.get('idx', '?')}"

        if r["kind"] == "circle":
            cx, cy, rp = r["cx_px"], r["cy_px"], r["r_px"]
            cv2.circle(vis, (cx, cy), rp,     color, 2, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), rp + 5, color, 1, cv2.LINE_AA)
            overlay = vis.copy()
            cv2.circle(overlay, (cx, cy), rp, color, -1)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            lx = max(2, min(vis.shape[1] - tw - 2, cx - tw // 2))
            ly = max(12, min(vis.shape[0] - 2, cy - rp - 6))
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, WHITE, 1, cv2.LINE_AA)

        elif r["kind"] == "rect":
            p1 = (r["x1_px"], r["y1_px"])
            p2 = (r["x2_px"], r["y2_px"])
            overlay = vis.copy()
            cv2.rectangle(overlay, p1, p2, color, -1)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            cv2.rectangle(vis, p1, p2, color, 2, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            lx = max(2, p1[0])
            ly = max(12, p1[1] - 6)
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, WHITE, 1, cv2.LINE_AA)

    found_n   = sum(1 for r in results if r["found"])
    missing_n = len(results) - found_n
    cv2.putText(vis, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Found: {found_n}  Missing: {missing_n}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                FOUND if missing_n == 0 else MISSING, 1, cv2.LINE_AA)
    return vis


def draw_hole_verification(real_gray, hole_results, title):
    return draw_feature_verification(real_gray, hole_results, title)


def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def print_result(name: str, result, rank: int = None):
    prefix = f"  {name}" if rank == 1 else f"  {rank}. {name}" if rank else f"   {name}"
    print(f"\n{prefix}")
    print(f"   Coverage:        {result.coverage:.4f}  ({result.coverage:.1%})")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Strategy:        {result.strategy}")
    if result.inlier_ratio:
        print(f"   Inlier Ratio:    {result.inlier_ratio:.2%}")


def save_outputs(out_dir: Path, blueprint_stem: str,
                 real_edges: np.ndarray, mask: np.ndarray,
                 result, is_best: bool = False,
                 debug_written: bool = False) -> bool:
    prefix = f"{blueprint_stem}_"
    cv2.imwrite(str(out_dir / f"{prefix}aligned.png"), result.aligned_image)
    overlay = np.zeros((*real_edges.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = result.aligned_image
    overlay[:, :, 1] = real_edges
    cv2.imwrite(str(out_dir / f"{prefix}overlay.png"), overlay)
    if not debug_written:
        cv2.imwrite(str(out_dir / "debug_mask.png"),       mask)
        cv2.imwrite(str(out_dir / "debug_real_edges.png"), real_edges)
    if is_best:
        cv2.imwrite(str(out_dir / "best_aligned.png"), result.aligned_image)
        cv2.imwrite(str(out_dir / "best_overlay.png"),  overlay)
    return True


# ===========================================================================
# ── NEW: Dimension-analysis pipeline helper ─────────────────────────────────
# ===========================================================================

def _run_dimension_pipeline(
    inp: Path,
    real: np.ndarray,
    real_edges: np.ndarray,
    best_name: str,
    best_result,
    M_dxf2bp: np.ndarray,
    M_cad2edge: np.ndarray,
    scale_px_per_mm: float,
    feat_results_stage2: list[dict],
    total_scale: float = 0.0,
) -> None:
    """
    Runs Stages 3-8 for one input image.

    Parameters
    ----------
    inp                  : Path to the input image file
    real                 : grayscale image array (uint8)
    real_edges           : edge map from preprocess_real()
    best_name            : identified blueprint stem
    best_result          : AlignmentResult from match_best_template()
    M_dxf2bp             : DXF-mm → blueprint-px matrix
    M_cad2edge           : blueprint-px → edge-map-px matrix
    scale_px_per_mm      : px per mm from compute_M_dxf2bp()  (blueprint scale)
    feat_results_stage2  : existing verification results from Stage 2
                           (list of dicts with kind/cx_px/cy_px/r_px/found/…)
    total_scale          : the full DXF-mm → final-image-px scale used when
                           r_px was computed in Stage 2 (= scale_px_per_mm
                           * cad2edge_sc * align_scale).  Passed in so that
                           measurement.py can convert r_px → mm correctly.

    Any failure prints [FAIL] <Stage> Reason: … and returns immediately.
    """
    image_stem = inp.stem
    PASS = lambda s: print(f"   [PASS] {s}")
    FAIL = lambda s, r: print(f"   [FAIL] {s}  Reason: {r}")

    # Combined transform: DXF-mm → real-image-px
    M_align   = best_result.transform_matrix
    M_cad2img = M_align @ M_cad2edge @ M_dxf2bp

    # ── Stage 3: DXF Parsing ────────────────────────────────────────────────
    dxf_path = DXF_DIR / f"{best_name}.dxf"
    try:
        cad_features = parse_dxf(dxf_path)
        PASS(f"DXF Parsing  ({len(cad_features.raw_circles)} circles, "
             f"{len(cad_features.raw_arcs)} arcs, "
             f"{len(cad_features.raw_lines)} lines)")
    except FileNotFoundError as exc:
        FAIL("DXF Parsing", str(exc))
        return
    except Exception as exc:
        FAIL("DXF Parsing", f"{type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        return

    # ── Stage 4: CAD-Image Feature Matching ─────────────────────────────────
    try:
        matched_pairs = match_features(
            cad_features=cad_features,
            real_gray=real,
            M_cad2img=M_cad2img,
            scale_px_per_mm=scale_px_per_mm,
            existing_verification=feat_results_stage2 if feat_results_stage2 else None,
            total_scale=total_scale,
        )
        if not matched_pairs:
            FAIL("CAD-Image Feature Matching", "No feature pairs could be matched")
            return
        PASS(f"CAD-Image Feature Matching  ({len(matched_pairs)} pairs matched)")
    except ValueError as exc:
        FAIL("CAD-Image Feature Matching", str(exc))
        return
    except Exception as exc:
        FAIL("CAD-Image Feature Matching", f"{type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        return

    # ── Stage 5: Transformation Estimation ──────────────────────────────────
    try:
        transform_result = estimate_transform(
            matched_pairs=matched_pairs,
            M_initial=M_cad2img,
            # Pass total_scale (DXF-mm → image-px) not just blueprint scale.
            # This makes the scale-mismatch warning meaningful: if Stage 5
            # refits to something very different from total_scale, the warning
            # correctly flags a problem rather than always firing due to unit mismatch.
            scale_initial=total_scale if total_scale > 0 else scale_px_per_mm,
        )
        PASS(
            f"Transformation Estimation  "
            f"(scale={transform_result.scale_px_per_mm:.4f} px/mm  "
            f"rotation={transform_result.rotation_deg:.2f}°  "
            f"{'refined' if transform_result.refined else 'passthrough'})"
        )
    except Exception as exc:
        FAIL("Transformation Estimation", f"{type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        return

    # ── Stage 6: Dimension Recovery ─────────────────────────────────────────
    try:
        measured_features = recover_dimensions(
            matched_pairs=matched_pairs,
            cad_features=cad_features,
            transform_result=transform_result,
            real_gray=real,
        )
        if not measured_features:
            FAIL("Dimension Recovery", "No dimensions could be recovered")
            return
        PASS(f"Dimension Recovery  ({len(measured_features)} dimensions recovered)")
    except ValueError as exc:
        FAIL("Dimension Recovery", str(exc))
        return
    except Exception as exc:
        FAIL("Dimension Recovery", f"{type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        return

    # ── Stage 7: Tolerance Verification ─────────────────────────────────────
    try:
        tolerance_results = verify_tolerances(measured_features)
        n_pass = sum(1 for t in tolerance_results if t.status == "PASS")
        n_fail = sum(1 for t in tolerance_results if t.status == "FAIL")
        PASS(
            f"Tolerance Verification  "
            f"({n_pass} PASS  /  {n_fail} FAIL  out of {len(tolerance_results)})"
        )
    except ValueError as exc:
        FAIL("Tolerance Verification", str(exc))
        return
    except Exception as exc:
        FAIL("Tolerance Verification", f"{type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        return

    # ── Stage 8: Inspection Report Generation ───────────────────────────────
    try:
        generate_reports(
            image_stem=image_stem,
            identified_as=best_name,
            tolerance_results=tolerance_results,
            matched_pairs=matched_pairs,
            scale_px_per_mm=transform_result.scale_px_per_mm,
            alignment_score=best_result.alignment_score,
            coverage=best_result.coverage,
            strategy=best_result.strategy,
            real_gray=real,
        )
        PASS("Report Generated")
    except Exception as exc:
        FAIL("Report Generation", f"{type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        return


# ===========================================================================
# ── MAIN ────────────────────────────────────────────────────────────────────
# ===========================================================================

def main():
    print("=" * 70)
    print("Inspection Pipeline  --  Batch Mode")
    print("=" * 70)

    # ── Pre-flight checks ───────────────────────────────────────────────────
    if not INPUTS_DIR.exists():
        print(f"\n[FAIL] Inputs folder not found: {INPUTS_DIR.resolve()}")
        return
    input_images = collect_images(INPUTS_DIR)
    if not input_images:
        print(f"\n[FAIL] No images found in {INPUTS_DIR.resolve()}")
        return

    if not BLUEPRINTS_DIR.exists():
        print(f"\n[FAIL] Blueprints folder not found: {BLUEPRINTS_DIR.resolve()}")
        return
    blueprint_images = collect_images(BLUEPRINTS_DIR)
    if not blueprint_images:
        print(f"\n[FAIL] No images found in {BLUEPRINTS_DIR.resolve()}")
        return

    print(f"\nFound {len(input_images)} input image(s)  in '{INPUTS_DIR}'")
    print(f"Found {len(blueprint_images)} blueprint(s)  in '{BLUEPRINTS_DIR}'")

    print(f"\nLoading blueprints...")
    templates: list[tuple[str, np.ndarray]] = []
    for bp in blueprint_images:
        try:
            edges = preprocess_cad(bp)
            templates.append((bp.stem, edges))
            print(f"   [OK] {bp.name}  ({np.count_nonzero(edges)} edge px)")
        except FileNotFoundError as e:
            print(f"   [FAIL] {e}")

    if not templates:
        print("\n[FAIL] No valid blueprints loaded.")
        return

    # ── Per-image loop ───────────────────────────────────────────────────────
    for idx, inp in enumerate(input_images, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(input_images)}]  Input: {inp.name}")
        print(f"{'=' * 70}")

        # ---------------------------------------------------------------
        # Helper: print a consistent FAIL banner and note that the
        # pipeline is stopping for this image.  Does NOT call sys.exit —
        # we simply skip to the next image via the pipeline_ok flag.
        # ---------------------------------------------------------------
        def stage_fail(stage: str, reason: str) -> None:
            print(f"\n   [FAIL] {stage}")
            print(f"          Reason: {reason}")
            print(f"   STOPPING PIPELINE for {inp.name}")

        # ── Load image ───────────────────────────────────────────────────
        real = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
        if real is None:
            stage_fail("Image Load", "cv2.imread returned None")
            continue
        print(f"   Shape: {real.shape}")

        # ==============================================================
        # STAGE 1: IDENTIFICATION
        # ==============================================================
        print(f"\n   Stage 1: Identification")
        print(f"   Preprocessing...")
        real_edges, mask = preprocess_real(real)
        print(f"   Real edges: {np.count_nonzero(real_edges)} pixels")

        out_dir = OUTPUTS_DIR / inp.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output folder: {out_dir.resolve()}")

        print(f"   Aligning against {len(templates)} blueprint(s)...")
        matches = match_best_template(templates, real_edges)

        print(f"\n   Results -- ranked by coverage:")
        for m in matches:
            print_result(m.name, m.result, rank=m.rank)

        best = matches[0]

        if not best.result.identified:
            stage_fail(
                "Stage 1: Identification",
                f"best match '{best.name}' coverage {best.result.coverage:.1%} "
                f"is below threshold"
            )
            continue  # ← GATE: nothing below this runs

        print(f"\n   [PASS] Stage 1: Identification")
        print(f"          Matched: '{best.name}'  coverage {best.result.coverage:.1%}")
        best_name = best.name

        # Save alignment overlays (informational — not a pipeline gate)
        print(f"\n   Saving alignment outputs...")
        debug_written = False
        for m in matches:
            is_best = (m.rank == 1)
            debug_written = save_outputs(
                out_dir=out_dir, blueprint_stem=m.name,
                real_edges=real_edges, mask=mask,
                result=m.result, is_best=is_best,
                debug_written=debug_written,
            )
            tag = " <- best" if is_best else ""
            print(f"   [OK] {m.name}_aligned.png  |  {m.name}_overlay.png{tag}")

        # ==============================================================
        # STAGE 2: FEATURE EXTRACTION
        # ==============================================================
        print(f"\n   Stage 2: Feature Extraction")

        # --- Determine view, DXF path, and blueprint PNG ----------------
        name_lower = best_name.lower()
        if "front" in name_lower or "top" in name_lower:
            view = "front"
        elif "rear" in name_lower or "back" in name_lower:
            view = "rear"
        else:
            view = None

        if view is None:
            stage_fail(
                "Stage 2: Feature Extraction",
                f"cannot determine view (front/rear) from blueprint name '{best_name}'"
            )
            continue  # ← GATE

        dxf_path = DXF_DIR / f"{best_name}.dxf"
        if not dxf_path.exists():
            stage_fail(
                "Stage 2: Feature Extraction",
                f"no DXF file found at {dxf_path}"
            )
            continue  # ← GATE

        blueprint_path = None
        for ext in IMAGE_EXTS:
            candidate = BLUEPRINTS_DIR / f"{best_name}{ext}"
            if candidate.exists():
                blueprint_path = candidate
                break
        if blueprint_path is None:
            stage_fail(
                "Stage 2: Feature Extraction",
                f"no blueprint PNG found for '{best_name}' in {BLUEPRINTS_DIR}"
            )
            continue  # ← GATE

        # --- Compute transforms needed for feature verification ---------
        try:
            M_dxf2bp, scale_px_mm = compute_M_dxf2bp(blueprint_path)
            cad_edges_for_bp      = preprocess_cad(blueprint_path)
            M_cad2edge            = compute_M_cad2edge(cad_edges_for_bp,
                                                       real_edges.shape)
        except Exception as exc:
            stage_fail(
                "Stage 2: Feature Extraction",
                f"transform setup failed: {type(exc).__name__}: {exc}"
            )
            continue  # ← GATE

        M_align = best.result.transform_matrix
        print(f"   Scale: {scale_px_mm:.3f} px/mm (blueprint)")

        # --- Run verification (box vs circular) -------------------------
        feat_results_stage2: list[dict] = []
        is_box = "box" in best_name.lower()

        if is_box:
            features = BOX_FEATURES.get(best_name, [])
            if not features:
                stage_fail(
                    "Stage 2: Feature Extraction",
                    f"no feature definition in BOX_FEATURES for '{best_name}'"
                )
                continue  # ← GATE

            feat_results_stage2 = verify_box_features(
                real_gray=real,
                features=features,
                M_dxf2bp=M_dxf2bp,
                M_cad2edge=M_cad2edge,
                M_align=M_align,
                scale_px_per_mm=scale_px_mm,
            )
            found_count   = sum(1 for r in feat_results_stage2 if r["found"])
            missing_count = len(feat_results_stage2) - found_count

            print(f"   Features to verify: {len(features)}")
            print(f"   Results: {found_count}/{len(feat_results_stage2)} features found")
            for r in feat_results_stage2:
                tag = "[OK]  " if r["found"] else "[MISS]"
                if r["kind"] == "circle":
                    print(f"     {tag} {r['label']}  "
                          f"px=({r['cx_px']},{r['cy_px']})  "
                          f"r={r['r_px']}px  ratio={r['ratio']:.2f}")
                else:
                    print(f"     {tag} {r['label']}  "
                          f"px=({r['x1_px']},{r['y1_px']})-"
                          f"({r['x2_px']},{r['y2_px']})  ratio={r['ratio']:.2f}")

            vis = draw_feature_verification(
                real, feat_results_stage2,
                title=f"{inp.stem} -> {best_name} | "
                      f"{found_count}/{len(feat_results_stage2)} features"
            )
            cv2.imwrite(str(out_dir / "hole_verification.png"), vis)
            print(f"   [OK] hole_verification.png saved")

            if missing_count > 0:
                missing_labels = [r["label"] for r in feat_results_stage2
                                  if not r["found"]]
                stage_fail(
                    "Stage 2: Feature Extraction",
                    f"{missing_count} feature(s) missing: {missing_labels}"
                )
                continue  # ← GATE: dimension analysis MUST NOT run

        else:  # circular part
            circles_dxf = parse_dxf_circles(dxf_path)
            holes       = get_holes_for_view(circles_dxf, view)
            print(f"   DXF holes loaded: {len(holes)}")

            if not holes:
                stage_fail(
                    "Stage 2: Feature Extraction",
                    f"DXF '{dxf_path.name}' contains no holes for view='{view}'"
                )
                continue  # ← GATE

            hole_results = verify_holes(
                real_gray=real,
                holes_dxf=holes,
                M_dxf2bp=M_dxf2bp,
                M_cad2edge=M_cad2edge,
                M_align=M_align,
                scale_px_per_mm=scale_px_mm,
            )
            found_count   = sum(1 for r in hole_results if r["found"])
            missing_count = len(hole_results) - found_count

            print(f"   Results: {found_count}/{len(hole_results)} holes found")
            for r in hole_results:
                tag = "[OK]  " if r["found"] else "[MISS]"
                print(f"     {tag} #{r['idx']:2d}  "
                      f"dxf=({r['cx_dxf']:.1f},{r['cy_dxf']:.1f})  "
                      f"px=({r['cx_px']},{r['cy_px']})  "
                      f"r={r['r_px']}px  ratio={r['ratio']:.2f}")

            vis = draw_hole_verification(
                real, hole_results,
                title=f"{inp.stem} -> {best_name} | "
                      f"holes {found_count}/{len(hole_results)}"
            )
            cv2.imwrite(str(out_dir / "hole_verification.png"), vis)
            print(f"   [OK] hole_verification.png saved")

            # Normalise for Stage 4 reuse
            feat_results_stage2 = []
            for r in hole_results:
                entry = dict(r)
                entry["label"] = f"hole #{r['idx']}"
                feat_results_stage2.append(entry)

            if missing_count > 0:
                missing_ids = [r["idx"] for r in hole_results if not r["found"]]
                stage_fail(
                    "Stage 2: Feature Extraction",
                    f"{missing_count} hole(s) missing: {missing_ids}"
                )
                continue  # ← GATE: dimension analysis MUST NOT run

        # All features present — Stage 2 passed
        print(f"\n   [PASS] Stage 2: Feature Extraction")
        print(f"          All {len(feat_results_stage2)} features verified")

        # ==============================================================
        # STAGES 3-8: DIMENSION ANALYSIS
        # Only reached when BOTH Stage 1 AND Stage 2 have passed.
        # ==============================================================
        print(f"\n   ── Dimension Analysis (Stages 3-8) ──")
        _align_sc    = math.sqrt(best.result.transform_matrix[0, 0] ** 2 +
                                  best.result.transform_matrix[1, 0] ** 2)
        _cad2edge_sc = math.sqrt(M_cad2edge[0, 0] ** 2 + M_cad2edge[1, 0] ** 2)
        _total_scale = scale_px_mm * _cad2edge_sc * _align_sc

        _run_dimension_pipeline(
            inp=inp,
            real=real,
            real_edges=real_edges,
            best_name=best_name,
            best_result=best.result,
            M_dxf2bp=M_dxf2bp,
            M_cad2edge=M_cad2edge,
            scale_px_per_mm=scale_px_mm,
            feat_results_stage2=feat_results_stage2,
            total_scale=_total_scale,
        )

    print(f"\n{'=' * 70}")
    print(f"[OK] Done!  All outputs saved under '{OUTPUTS_DIR.resolve()}'")
    print(f"[OK] Reports saved under 'reports/'")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
