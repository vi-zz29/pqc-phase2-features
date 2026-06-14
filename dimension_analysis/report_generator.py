"""
Inspection Report Generator — Stage 8.

All dimensions are in MILLIMETRES.

Output artefacts:
  reports/<stem>_report.csv   — tabular, mm columns
  reports/<stem>_report.json  — structured, mm columns
  reports/<stem>_report.png   — full composite inspection image

The PNG layout:
  ┌─────────────────────────────────────────────────────┐
  │  HEADER BAR  (title · score · coverage · status)    │
  ├──────────────────────┬──────────────────────────────┤
  │                      │  LEGEND PANEL                │
  │   ANNOTATED IMAGE    │  ── per-feature table ──     │
  │   (real photo +      │  idx  feature  CAD  Meas  d  │
  │    feature overlays) │  ...                         │
  │                      │  ── SUMMARY ──               │
  │                      │  PASS / FAIL  overall badge  │
  └──────────────────────┴──────────────────────────────┘

  Each feature on the image:
    • Filled semi-transparent shape (green=PASS, red=FAIL)
    • Solid border ring
    • Numbered callout dot → leader line → callout box with
      CAD px / Meas px / deviation px
"""

import csv
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dimension_analysis.feature_matcher import MatchedPair
from dimension_analysis.tolerance import ToleranceResult

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")

# ── Palette ──────────────────────────────────────────────────────────────────
_C_PASS    = (40,  200,  60)   # green
_C_FAIL    = (30,   50, 220)   # red (BGR)
_C_WARN    = (20,  160, 240)   # amber
_C_TEXT    = (255, 255, 255)
_C_DARK    = (18,   18,  18)
_C_PANEL   = (28,   28,  28)
_C_GRID    = (55,   55,  55)
_C_CALLOUT = (220, 220, 220)   # light grey callout box text

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONT2 = cv2.FONT_HERSHEY_DUPLEX
_FS    = 0.36   # small
_FM    = 0.44   # medium
_FL    = 0.58   # large
_TH    = 1
_TH2   = 2

# Target width for the annotated image panel (px).
# Taller images are scaled down to this width.
_IMG_TARGET_W = 900


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _build_record(tr: ToleranceResult, present: bool) -> dict:
    return {
        "feature_name":        tr.label,
        "feature_present":     "YES" if present else "NO",
        "cad_size_mm":         round(tr.cad_dimension_mm,      4),
        "measured_size_mm":    round(tr.measured_dimension_mm, 4),
        "deviation_mm":        round(tr.deviation_mm,          4),
        "tolerance_mm":        round(tr.tolerance_mm,          4),
        "status":              tr.status,
        "unit":                tr.unit,
    }


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_json(report: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put(img, text, x, y, color=_C_TEXT, fs=_FM, th=_TH, font=_FONT):
    cv2.putText(img, text, (int(x), int(y)), font, fs, color, th, cv2.LINE_AA)


def _bg_text(img, text, x, y, fg=_C_TEXT, bg=_C_DARK, fs=_FS, th=_TH):
    (tw, th_), bl = cv2.getTextSize(text, _FONT, fs, th)
    pad = 3
    cv2.rectangle(img,
                  (int(x) - pad, int(y) - th_ - bl - pad),
                  (int(x) + tw + pad, int(y) + bl + pad),
                  bg, -1)
    cv2.putText(img, text, (int(x), int(y)), _FONT, fs, fg, th, cv2.LINE_AA)


def _callout_box(img, lines: list[tuple[str, tuple]], cx, cy, color):
    """
    Draw a multi-line callout box near (cx, cy).
    lines = [(text, text_color), ...]
    """
    fs, th_val = _FS, _TH
    pad = 5
    line_h = 16
    widths = []
    for txt, _ in lines:
        (tw, _), _ = cv2.getTextSize(txt, _FONT, fs, th_val)
        widths.append(tw)
    box_w = max(widths) + pad * 2 + 2
    box_h = len(lines) * line_h + pad * 2

    H, W = img.shape[:2]
    # Place box: try right of feature, else left
    bx = min(cx + 10, W - box_w - 2)
    by = max(2, min(cy - box_h // 2, H - box_h - 2))

    # Draw connecting line from feature edge to box
    cv2.line(img, (cx, cy), (bx, by + box_h // 2), color, 1, cv2.LINE_AA)

    # Box background
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), _C_DARK, -1)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), color, 1, cv2.LINE_AA)

    for i, (txt, tcol) in enumerate(lines):
        cv2.putText(img, txt,
                    (bx + pad, by + pad + line_h * (i + 1) - 2),
                    _FONT, fs, tcol, th_val, cv2.LINE_AA)


def _draw_index_dot(img, idx: int, cx: int, cy: int, color):
    """Small numbered dot at feature centre."""
    r = 9
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r, _C_DARK, 1, cv2.LINE_AA)
    label = str(idx)
    (tw, th_), _ = cv2.getTextSize(label, _FONT, 0.32, 1)
    cv2.putText(img, label,
                (cx - tw // 2, cy + th_ // 2),
                _FONT, 0.32, _C_DARK, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main visual builder
# ---------------------------------------------------------------------------

def _build_visual_report(
    real_gray: np.ndarray,
    matched_pairs: list[MatchedPair],
    tolerance_results: list[ToleranceResult],
    image_stem: str,
    identified_as: str,
    alignment_score: float,
    coverage: float,
    scale_px_per_mm: float,
    pass_count: int,
    fail_count: int,
) -> np.ndarray:

    # ── 1. Prepare the photo panel ────────────────────────────────────────
    # Convert to BGR and scale to target width
    photo = cv2.cvtColor(real_gray, cv2.COLOR_GRAY2BGR)
    H0, W0 = photo.shape[:2]
    scale_vis = _IMG_TARGET_W / W0
    new_w = _IMG_TARGET_W
    new_h = int(H0 * scale_vis)
    photo = cv2.resize(photo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Build label→pair lookup, skipping duplicate _dia / _width / _height entries
    # so we only draw one overlay per physical feature location.
    # We draw from the "base" pair (circle or rect), not the derived diameter entry.
    drawn_locations: set[str] = set()   # key = "cx,cy"
    pair_by_label: dict[str, MatchedPair] = {p.label: p for p in matched_pairs}

    # Map each ToleranceResult to its pair (base label only — no suffixes)
    def _find_pair(tr: ToleranceResult):
        p = pair_by_label.get(tr.label)
        if p:
            return p
        base = tr.label.rsplit("_", 1)[0]
        return pair_by_label.get(base)

    # Assign a sequential index to each unique feature location
    feature_index: dict[str, int] = {}   # label → display index
    idx_counter = 1
    for p in matched_pairs:
        feature_index[p.label] = idx_counter
        idx_counter += 1

    # Group tolerance results by base pair label so we collect all rows
    # (radius + diameter, or width + height) for one callout box
    tol_by_pair: dict[str, list[ToleranceResult]] = defaultdict(list)
    for tr in tolerance_results:
        p = _find_pair(tr)
        if p:
            tol_by_pair[p.label].append(tr)

    # ── 2. Draw overlays ──────────────────────────────────────────────────
    for pair in matched_pairs:
        trs = tol_by_pair.get(pair.label, [])
        if not trs:
            continue

        # Overall status for this feature = FAIL if any row fails
        overall_ok = all(t.status == "PASS" for t in trs)
        color = _C_PASS if overall_ok else _C_FAIL

        # Scale coordinates to the resized image
        cx = int(round(pair.image_pos_px[0] * scale_vis))
        cy = int(round(pair.image_pos_px[1] * scale_vis))
        loc_key = f"{cx},{cy}"
        if loc_key in drawn_locations:
            continue
        drawn_locations.add(loc_key)

        idx = feature_index[pair.label]

        if pair.feature_type == "circle":
            r = max(3, int(round(float(pair.image_value_px) * scale_vis)))

            # Semi-transparent fill
            ov = photo.copy()
            cv2.circle(ov, (cx, cy), r, color, -1)
            cv2.addWeighted(ov, 0.22, photo, 0.78, 0, photo)

            # Solid ring
            cv2.circle(photo, (cx, cy), r,     color, 2, cv2.LINE_AA)
            cv2.circle(photo, (cx, cy), r + 4, color, 1, cv2.LINE_AA)

            # Dashed CAD circle (expected size)
            cad_r = max(3, int(round(
                float(pair.cad_value_mm) * pair.scale_px_per_mm * scale_vis
            )))
            # Draw dashed circle as 36 short arcs
            for seg in range(0, 360, 20):
                cv2.ellipse(photo, (cx, cy), (cad_r, cad_r),
                            0, seg, seg + 10,
                            (180, 180, 60), 1, cv2.LINE_AA)

            # Callout box
            meas_r  = float(pair.image_value_px) / pair.scale_px_per_mm if pair.scale_px_per_mm > 0 else 0.0
            cad_r_  = float(pair.cad_value_mm)
            dev_r   = meas_r - cad_r_
            lines = [
                (f"#{idx} {pair.label}", _C_CALLOUT),
                (f"CAD  r={cad_r_:.3f}mm", (160, 160, 160)),
                (f"Meas r={meas_r:.3f}mm", _C_TEXT),
                (f"Dev  {dev_r:+.3f}mm", color),
            ]
            _callout_box(photo, lines, cx, cy - r - 4, color)
            _draw_index_dot(photo, idx, cx, cy, color)

        elif pair.feature_type == "rect":
            img_wh = pair.image_value_px
            hw = int(round(img_wh[0] * scale_vis / 2))
            hh = int(round(img_wh[1] * scale_vis / 2))
            x1, y1 = cx - hw, cy - hh
            x2, y2 = cx + hw, cy + hh

            ov = photo.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.22, photo, 0.78, 0, photo)
            cv2.rectangle(photo, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Dashed CAD rect
            cad_hw = int(round(float(pair.cad_value_mm[0])
                               * pair.scale_px_per_mm * scale_vis / 2))
            cad_hh = int(round(float(pair.cad_value_mm[1])
                               * pair.scale_px_per_mm * scale_vis / 2))
            for pts in [
                [(cx-cad_hw, cy-cad_hh), (cx+cad_hw, cy-cad_hh)],
                [(cx+cad_hw, cy-cad_hh), (cx+cad_hw, cy+cad_hh)],
                [(cx+cad_hw, cy+cad_hh), (cx-cad_hw, cy+cad_hh)],
                [(cx-cad_hw, cy+cad_hh), (cx-cad_hw, cy-cad_hh)],
            ]:
                # Dashed line
                p1, p2 = pts
                dist = int(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                for d in range(0, dist, 10):
                    t0, t1 = d / dist, min((d + 5) / dist, 1.0)
                    lx1 = int(p1[0] + (p2[0]-p1[0]) * t0)
                    ly1 = int(p1[1] + (p2[1]-p1[1]) * t0)
                    lx2 = int(p1[0] + (p2[0]-p1[0]) * t1)
                    ly2 = int(p1[1] + (p2[1]-p1[1]) * t1)
                    cv2.line(photo, (lx1, ly1), (lx2, ly2),
                             (180, 180, 60), 1, cv2.LINE_AA)

            # Callout
            mw = float(img_wh[0]) / pair.scale_px_per_mm if pair.scale_px_per_mm > 0 else 0.0
            mh = float(img_wh[1]) / pair.scale_px_per_mm if pair.scale_px_per_mm > 0 else 0.0
            cw = float(pair.cad_value_mm[0])
            ch = float(pair.cad_value_mm[1])
            lines = [
                (f"#{idx} {pair.label}", _C_CALLOUT),
                (f"CAD  {cw:.2f}x{ch:.2f}mm", (160, 160, 160)),
                (f"Meas {mw:.2f}x{mh:.2f}mm", _C_TEXT),
                (f"dW={mw-cw:+.2f}  dH={mh-ch:+.2f}mm", color),
            ]
            _callout_box(photo, lines, cx + hw + 4, cy, color)
            _draw_index_dot(photo, idx, cx, cy, color)

    # ── 3. Header bar ─────────────────────────────────────────────────────
    overall_str = "PASS" if fail_count == 0 else "FAIL"
    ov_col = _C_PASS if overall_str == "PASS" else _C_FAIL
    hdr_h = 52
    hdr = np.full((hdr_h, new_w, 3), _C_DARK, dtype=np.uint8)
    cv2.line(hdr, (0, hdr_h-1), (new_w, hdr_h-1), _C_GRID, 1)

    _put(hdr, f"INSPECTION  {image_stem.upper()}",         8, 20, _C_TEXT,  _FL, _TH2)
    _put(hdr, f"Identified as: {identified_as}",           8, 40, (180,180,180), _FS)
    _put(hdr, f"Score:{alignment_score:.3f}",            380, 20, (180,180,180), _FM)
    _put(hdr, f"Cov:{coverage:.1%}",                     510, 20, (180,180,180), _FM)
    _put(hdr, f"Scale:{scale_px_per_mm:.2f}px/mm",       620, 20, (180,180,180), _FM)
    _put(hdr, f"Units: mm",                              750, 20, (180,180,180), _FS)
    _put(hdr, f"{overall_str}  {pass_count}P/{fail_count}F",
         380, 40, ov_col, _FM, _TH2)

    # ── 4. Legend panel ───────────────────────────────────────────────────
    LEG_W   = 460
    ROW_H   = 20
    PAD     = 10
    HDR_ROW = 34

    # Column widths
    C_IDX  = 28
    C_NAME = 170
    C_CAD  = 70
    C_MEAS = 70
    C_DEV  = 62
    C_TOL  = 55
    C_ST   = 40

    # Collect unique rows — one per tolerance result
    rows = []
    seen_locs: set = set()
    for tr in tolerance_results:
        rows.append(tr)

    n_data_rows = len(rows)
    # Extra rows: header + separator + summary
    n_total_rows = n_data_rows + 4
    leg_h_content = HDR_ROW + n_total_rows * ROW_H + PAD * 3
    leg_h = max(new_h + hdr_h, leg_h_content)

    leg = np.full((leg_h, LEG_W, 3), _C_PANEL, dtype=np.uint8)
    cv2.line(leg, (0, 0), (0, leg_h), _C_GRID, 2)

    # Panel title
    y = PAD + 16
    _put(leg, "DIMENSION REPORT  (mm)", PAD, y, _C_TEXT, _FM, _TH2)
    y += 6
    cv2.line(leg, (PAD, y), (LEG_W - PAD, y), _C_GRID, 1)
    y += ROW_H

    # Column headers
    x0 = PAD
    _put(leg, "#",     x0,           y, (140,140,140), _FS)
    _put(leg, "Feature",x0+C_IDX,    y, (140,140,140), _FS)
    _put(leg, "CAD",   x0+C_IDX+C_NAME, y, (140,140,140), _FS)
    _put(leg, "Meas",  x0+C_IDX+C_NAME+C_CAD, y, (140,140,140), _FS)
    _put(leg, "Dev",   x0+C_IDX+C_NAME+C_CAD+C_MEAS, y, (140,140,140), _FS)
    _put(leg, "Tol",   x0+C_IDX+C_NAME+C_CAD+C_MEAS+C_DEV, y, (140,140,140), _FS)
    _put(leg, "St",    x0+C_IDX+C_NAME+C_CAD+C_MEAS+C_DEV+C_TOL, y, (140,140,140), _FS)
    y += 4
    cv2.line(leg, (PAD, y), (LEG_W - PAD, y), _C_GRID, 1)
    y += ROW_H - 4

    # Data rows — alternating row background
    for row_i, tr in enumerate(rows):
        color = _C_PASS if tr.status == "PASS" else _C_FAIL
        tick  = "OK" if tr.status == "PASS" else "FAIL"

        # Alternating bg
        if row_i % 2 == 0:
            cv2.rectangle(leg, (0, y - ROW_H + 4), (LEG_W, y + 4),
                          (35, 35, 35), -1)

        # Find the pair index for this label
        p = pair_by_label.get(tr.label)
        if p is None:
            base = tr.label.rsplit("_", 1)[0]
            p = pair_by_label.get(base)
        idx_str = str(feature_index.get(p.label, "-")) if p else "-"

        name = tr.label[:22]
        cad  = f"{tr.cad_dimension_mm:.3f}"
        meas = f"{tr.measured_dimension_mm:.3f}"
        dev  = f"{tr.deviation_mm:+.3f}"
        tol  = f"+-{tr.tolerance_mm:.2f}"

        _put(leg, idx_str,  x0,                             y, (160,160,160), _FS)
        _put(leg, name,     x0+C_IDX,                       y, _C_TEXT,       _FS)
        _put(leg, cad,      x0+C_IDX+C_NAME,                y, (160,160,160), _FS)
        _put(leg, meas,     x0+C_IDX+C_NAME+C_CAD,          y, _C_TEXT,       _FS)
        _put(leg, dev,      x0+C_IDX+C_NAME+C_CAD+C_MEAS,   y, color,         _FS)
        _put(leg, tol,      x0+C_IDX+C_NAME+C_CAD+C_MEAS+C_DEV, y, (120,120,120), _FS)
        _put(leg, tick,     x0+C_IDX+C_NAME+C_CAD+C_MEAS+C_DEV+C_TOL, y, color, _FS)

        y += ROW_H

    # Summary
    y += 6
    cv2.line(leg, (PAD, y), (LEG_W - PAD, y), _C_GRID, 1)
    y += ROW_H
    total = len(rows)
    _put(leg, f"Total: {total}   Pass: {pass_count}   Fail: {fail_count}",
         PAD, y, _C_TEXT, _FM)
    y += ROW_H + 4

    # Big overall badge
    badge_text = f"  OVERALL: {overall_str}  "
    badge_color = _C_PASS if overall_str == "PASS" else _C_FAIL
    (bw, bh_), _ = cv2.getTextSize(badge_text, _FONT2, _FL, _TH2)
    bx, by = PAD, y
    cv2.rectangle(leg, (bx - 4, by - bh_ - 4), (bx + bw + 4, by + 6),
                  badge_color, -1)
    cv2.putText(leg, badge_text, (bx, by),
                _FONT2, _FL, _C_DARK, _TH2, cv2.LINE_AA)

    # ── 5. Assemble final image ───────────────────────────────────────────
    # image panel = header + photo
    img_panel = np.vstack([hdr, photo])

    # Pad to same height
    total_h = max(img_panel.shape[0], leg_h)
    if img_panel.shape[0] < total_h:
        pad = np.full((total_h - img_panel.shape[0], new_w, 3),
                      _C_DARK, dtype=np.uint8)
        img_panel = np.vstack([img_panel, pad])
    if leg.shape[0] < total_h:
        pad = np.full((total_h - leg.shape[0], LEG_W, 3),
                      _C_PANEL, dtype=np.uint8)
        leg = np.vstack([leg, pad])

    return np.hstack([img_panel, leg])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reports(
    image_stem: str,
    identified_as: str,
    tolerance_results: list[ToleranceResult],
    matched_pairs: list[MatchedPair],
    scale_px_per_mm: float,
    alignment_score: float,
    coverage: float,
    strategy: str,
    real_gray=None,
    extra_metadata: dict | None = None,
) -> dict:
    out_dir = _ensure_reports_dir()

    matched_labels = {p.label for p in matched_pairs}
    records = []
    for tr in tolerance_results:
        # Feature is present only if it was genuinely matched — never fall back
        # to deviation==0 as a proxy (that was a tautological artefact).
        present = tr.label in matched_labels or any(
            tr.label.startswith(p.label) for p in matched_pairs
        )
        records.append(_build_record(tr, present))

    pass_count = sum(1 for r in records if r["status"] == "PASS")
    fail_count = sum(1 for r in records if r["status"] == "FAIL")
    total      = len(records)
    overall    = "PASS" if fail_count == 0 else "FAIL"
    timestamp  = datetime.now().isoformat(timespec="seconds")

    report = {
        "report_metadata": {
            "timestamp":       timestamp,
            "image":           image_stem,
            "identified_as":   identified_as,
            "alignment_score": round(alignment_score, 4),
            "coverage":        round(coverage, 4),
            "strategy":        strategy,
            "scale_px_per_mm": round(scale_px_per_mm, 4),
            "total_features":  total,
            "pass_count":      pass_count,
            "fail_count":      fail_count,
            "overall_status":  overall,
            "units":           "mm",
        },
        "features": records,
    }
    if extra_metadata:
        report["extra"] = extra_metadata

    csv_path  = out_dir / f"{image_stem}_report.csv"
    json_path = out_dir / f"{image_stem}_report.json"
    png_path  = out_dir / f"{image_stem}_report.png"

    _write_csv(records, csv_path)
    _write_json(report, json_path)

    if real_gray is not None:
        try:
            vis = _build_visual_report(
                real_gray, matched_pairs, tolerance_results,
                image_stem, identified_as, alignment_score, coverage,
                scale_px_per_mm, pass_count, fail_count,
            )
            cv2.imwrite(str(png_path), vis)
        except Exception as exc:
            logger.warning(f"PNG failed: {exc}")
            png_path = None

    # Console — ASCII only (Windows cmd safe)
    SEP = "-" * 76
    print(f"\n{SEP}")
    print(f"  INSPECTION REPORT (mm)  --  {image_stem}  ->  {identified_as}")
    print(SEP)
    print(f"  Timestamp : {timestamp}")
    print(f"  Score     : {alignment_score:.4f}   Coverage: {coverage:.1%}")
    print(f"  Scale ref : {scale_px_per_mm:.4f} px/mm")
    print(f"  Features  : {total}   PASS: {pass_count}   FAIL: {fail_count}")
    print(SEP)
    print(f"  {'Feature':<28} {'Present':<8} {'CAD mm':>8} {'Meas mm':>8} "
          f"{'Dev mm':>8} {'Tol mm':>7}  Status")
    print(f"  {'-'*72}")

    for r in records:
        tag = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        print(
            f"  {r['feature_name']:<28} "
            f"{r['feature_present']:<8} "
            f"{r['cad_size_mm']:>8.3f} "
            f"{r['measured_size_mm']:>8.3f} "
            f"{r['deviation_mm']:>+8.3f} "
            f"{r['tolerance_mm']:>7.3f}  "
            f"{tag}"
        )

    print(SEP)
    overall_tag = "[PASS]" if overall == "PASS" else "[FAIL]"
    print(f"  OVERALL: {overall_tag}   ({pass_count}/{total} pass)")
    print(SEP)
    png_name = png_path.name if png_path else "(none)"
    print(f"  Saved: {csv_path.name}  |  {json_path.name}  |  {png_name}")
    print(f"{SEP}\n")

    return report
