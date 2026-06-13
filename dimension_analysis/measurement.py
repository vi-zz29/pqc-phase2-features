"""
Dimension Recovery — Stage 6.

Reports everything in PIXELS.

No mm conversion. No scale dependency.

For every matched feature:
  - cad_size_px   : what the DXF says this feature should be, projected to
                    final-image pixels using total_scale (DXF-mm → px).
  - measured_px   : what was actually detected in the image, in pixels.
  - deviation_px  : measured_px − cad_size_px

For rects the CAD size is (w_px, h_px); for circles it is radius_px.
Spacing is in pixels between detected image centres.
PCD is in pixels from detected image centre to projected part centre.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from dimension_analysis.dxf_parser import CADFeatureSet
from dimension_analysis.feature_matcher import MatchedPair, _project_cad_to_image
from dimension_analysis.transform_estimator import TransformResult

logger = logging.getLogger(__name__)


@dataclass
class MeasuredFeature:
    """One feature with its CAD projected size, detected size, and deviation — all in pixels."""
    feature_type: str
    label: str
    cad_dimension_px: float          # expected size in pixels (from DXF × total_scale)
    measured_dimension_px: float     # detected size in pixels
    deviation_px: float              # measured − cad
    # Keep these for report display compatibility; both equal to px values
    cad_dimension_mm: float = 0.0    # alias — always equals cad_dimension_px
    measured_dimension_mm: float = 0.0  # alias — always equals measured_dimension_px
    deviation_mm: float = 0.0        # alias — always equals deviation_px
    unit: str = "px"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Keep the mm aliases in sync so the rest of the pipeline
        # (tolerance, report) that reads these fields still works.
        self.cad_dimension_mm     = self.cad_dimension_px
        self.measured_dimension_mm = self.measured_dimension_px
        self.deviation_mm          = self.deviation_px


def recover_dimensions(
    matched_pairs: list[MatchedPair],
    cad_features: CADFeatureSet,
    transform_result: TransformResult,
) -> list[MeasuredFeature]:
    """
    Produce pixel-domain measurements for every matched feature pair.

    Parameters
    ----------
    matched_pairs    : from feature_matcher.match_features()
                       Each pair carries:
                         cad_value_mm    — nominal CAD size in mm
                         image_value_px  — detected size in final-image pixels
                         scale_px_per_mm — total_scale (DXF-mm → final-image-px)
    cad_features     : parsed DXF feature set
    transform_result : refined transform (used only for projecting the part
                       centre when computing PCD)

    Returns
    -------
    list[MeasuredFeature] — all values in pixels.
    """
    features: list[MeasuredFeature] = []

    # ── Per-pair measurements ─────────────────────────────────────────────
    for pair in matched_pairs:
        # total_scale stored per-pair by the matcher
        ts = pair.scale_px_per_mm   # DXF-mm → final-image-px

        if pair.feature_type == "circle":
            cad_r_mm  = float(pair.cad_value_mm)
            img_r_px  = float(pair.image_value_px)   # already in pixels
            cad_r_px  = cad_r_mm * ts                # expected radius in px

            features.append(MeasuredFeature(
                feature_type="circle_radius",
                label=pair.label,
                cad_dimension_px=cad_r_px,
                measured_dimension_px=img_r_px,
                deviation_px=img_r_px - cad_r_px,
                unit="px (radius)",
            ))
            features.append(MeasuredFeature(
                feature_type="circle_diameter",
                label=pair.label + "_dia",
                cad_dimension_px=cad_r_px * 2.0,
                measured_dimension_px=img_r_px * 2.0,
                deviation_px=(img_r_px - cad_r_px) * 2.0,
                unit="px (diameter)",
            ))

        elif pair.feature_type == "rect":
            cad_wh  = pair.cad_value_mm       # (w_mm, h_mm)
            img_wh  = pair.image_value_px     # (w_px, h_px)
            cad_w_px = float(cad_wh[0]) * ts
            cad_h_px = float(cad_wh[1]) * ts

            features.append(MeasuredFeature(
                feature_type="rect_width",
                label=pair.label + "_width",
                cad_dimension_px=cad_w_px,
                measured_dimension_px=float(img_wh[0]),
                deviation_px=float(img_wh[0]) - cad_w_px,
                unit="px",
            ))
            features.append(MeasuredFeature(
                feature_type="rect_height",
                label=pair.label + "_height",
                cad_dimension_px=cad_h_px,
                measured_dimension_px=float(img_wh[1]),
                deviation_px=float(img_wh[1]) - cad_h_px,
                unit="px",
            ))

    # ── High-level part dimensions in pixels ─────────────────────────────
    pt = cad_features.part_type

    # Pick a representative scale from the first pair with a valid scale
    ts_global = next(
        (p.scale_px_per_mm for p in matched_pairs if p.scale_px_per_mm > 0),
        transform_result.scale_px_per_mm,
    )

    if pt == "circular":
        if cad_features.outer_diameter is not None:
            od_cad_px = cad_features.outer_diameter * ts_global
            # Best measured estimate: the largest matched circle diameter
            od_meas_px = od_cad_px
            for p in matched_pairs:
                if p.feature_type == "circle":
                    candidate = float(p.image_value_px) * 2.0
                    cad_d_px  = float(p.cad_value_mm) * p.scale_px_per_mm * 2.0
                    if abs(cad_d_px - od_cad_px) < od_cad_px * 0.15:
                        od_meas_px = candidate
                        break
            features.append(MeasuredFeature(
                feature_type="outer_diameter",
                label="outer_diameter",
                cad_dimension_px=od_cad_px,
                measured_dimension_px=od_meas_px,
                deviation_px=od_meas_px - od_cad_px,
                unit="px",
            ))

        if cad_features.center_bore is not None:
            cb_cad_px  = cad_features.center_bore * ts_global
            cb_meas_px = cb_cad_px
            for p in matched_pairs:
                if p.feature_type == "circle":
                    cad_d_px = float(p.cad_value_mm) * p.scale_px_per_mm * 2.0
                    if abs(cad_d_px - cb_cad_px) < cb_cad_px * 0.15:
                        cb_meas_px = float(p.image_value_px) * 2.0
                        break
            features.append(MeasuredFeature(
                feature_type="center_bore",
                label="center_bore",
                cad_dimension_px=cb_cad_px,
                measured_dimension_px=cb_meas_px,
                deviation_px=cb_meas_px - cb_cad_px,
                unit="px",
            ))

        if cad_features.pcd is not None:
            pcd_cad_px = cad_features.pcd * ts_global

            # Part centre in CAD coords
            cx_vals = [c["cx"] for c in cad_features.raw_circles]
            cy_vals = [c["cy"] for c in cad_features.raw_circles]
            CENTER_TOL_MM = 3.0
            if cx_vals:
                part_cx_cad = float(np.median(cx_vals))
                part_cy_cad = float(np.median(cy_vals))
            else:
                part_cx_cad, part_cy_cad = 148.5, 105.0

            # Project part centre to image
            cx_img, cy_img = _project_cad_to_image(
                part_cx_cad, part_cy_cad, transform_result.matrix
            )

            # Bolt-hole pairs only
            bolt_pairs = [
                p for p in matched_pairs
                if p.feature_type == "circle" and
                math.hypot(p.cad_pos[0] - part_cx_cad,
                           p.cad_pos[1] - part_cy_cad) >= CENTER_TOL_MM
            ]

            if len(bolt_pairs) >= 3:
                dists_px = [
                    math.hypot(p.image_pos_px[0] - cx_img,
                               p.image_pos_px[1] - cy_img)
                    for p in bolt_pairs
                ]
                pcd_meas_px = float(np.mean(dists_px)) * 2.0
            else:
                pcd_meas_px = pcd_cad_px

            features.append(MeasuredFeature(
                feature_type="pcd",
                label="pcd",
                cad_dimension_px=pcd_cad_px,
                measured_dimension_px=pcd_meas_px,
                deviation_px=pcd_meas_px - pcd_cad_px,
                unit="px",
            ))

    elif pt == "rectangular":
        # Overall dimensions from DXF boundary × scale
        if cad_features.overall_width is not None:
            w_px = cad_features.overall_width * ts_global
            features.append(MeasuredFeature(
                feature_type="overall_width",
                label="overall_width",
                cad_dimension_px=w_px,
                measured_dimension_px=w_px,
                deviation_px=0.0,
                unit="px",
                extra={"note": "derived from DXF boundary"},
            ))
        if cad_features.overall_height is not None:
            h_px = cad_features.overall_height * ts_global
            features.append(MeasuredFeature(
                feature_type="overall_height",
                label="overall_height",
                cad_dimension_px=h_px,
                measured_dimension_px=h_px,
                deviation_px=0.0,
                unit="px",
                extra={"note": "derived from DXF boundary"},
            ))

        # Hole spacings in pixels
        hole_pairs = [p for p in matched_pairs if p.feature_type == "circle"]
        dxf_hole_positions = cad_features.hole_positions

        if len(hole_pairs) >= 2:
            for i in range(len(hole_pairs)):
                for j in range(i + 1, len(hole_pairs)):
                    pi, pj = hole_pairs[i], hole_pairs[j]

                    # CAD spacing in pixels
                    if len(dxf_hole_positions) > i and len(dxf_hole_positions) > j:
                        cad_sp_mm = math.hypot(
                            dxf_hole_positions[i][0] - dxf_hole_positions[j][0],
                            dxf_hole_positions[i][1] - dxf_hole_positions[j][1],
                        )
                        ts_i = pi.scale_px_per_mm if pi.scale_px_per_mm > 0 else ts_global
                        cad_sp_px = cad_sp_mm * ts_i
                    else:
                        cad_sp_px = math.hypot(
                            pi.cad_pos[0] - pj.cad_pos[0],
                            pi.cad_pos[1] - pj.cad_pos[1],
                        )

                    # Measured spacing in pixels — direct image pixel distance
                    meas_sp_px = math.hypot(
                        pi.image_pos_px[0] - pj.image_pos_px[0],
                        pi.image_pos_px[1] - pj.image_pos_px[1],
                    )

                    features.append(MeasuredFeature(
                        feature_type="hole_spacing",
                        label=f"spacing_{pi.label}_to_{pj.label}",
                        cad_dimension_px=cad_sp_px,
                        measured_dimension_px=meas_sp_px,
                        deviation_px=meas_sp_px - cad_sp_px,
                        unit="px",
                    ))

    logger.debug(f"Recovered {len(features)} pixel measurements")
    return features
