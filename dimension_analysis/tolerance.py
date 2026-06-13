"""
Tolerance Verification — Stage 7.

All thresholds are in PIXELS.

Default tolerances (±px):
  circle_radius    : ±2 px
  circle_diameter  : ±4 px
  rect_width       : ±5 px
  rect_height      : ±5 px
  hole_spacing     : ±5 px
  outer_diameter   : ±8 px
  center_bore      : ±5 px
  pcd              : ±8 px
  overall_width    : ±10 px
  overall_height   : ±10 px
  _default         : ±5 px
"""

import logging
from dataclasses import dataclass

from dimension_analysis.measurement import MeasuredFeature

logger = logging.getLogger(__name__)

# Default tolerances per feature_type — all in PIXELS
DEFAULT_TOLERANCES: dict[str, float] = {
    "circle_radius":   2.0,
    "circle_diameter": 4.0,
    "rect_width":      5.0,
    "rect_height":     5.0,
    "hole_spacing":    5.0,
    "outer_diameter":  8.0,
    "center_bore":     5.0,
    "pcd":             8.0,
    "overall_width":  10.0,
    "overall_height": 10.0,
    "_default":        5.0,
}


@dataclass
class ToleranceResult:
    feature_type: str
    label: str
    cad_dimension_mm: float       # field name kept for pipeline compat; value is in px
    measured_dimension_mm: float  # field name kept for pipeline compat; value is in px
    deviation_mm: float           # field name kept for pipeline compat; value is in px
    tolerance_mm: float           # field name kept for pipeline compat; value is in px
    status: str                   # "PASS" | "FAIL"
    unit: str = "px"


def verify_tolerances(
    measured_features: list[MeasuredFeature],
    tolerances: dict[str, float] | None = None,
) -> list[ToleranceResult]:
    """
    Apply pixel tolerance rules to every measured feature.

    Parameters
    ----------
    measured_features : output of measurement.recover_dimensions()
                        All values are in pixels.
    tolerances        : optional override dict mapping feature_type → ±px

    Returns
    -------
    list[ToleranceResult]
    """
    if not measured_features:
        raise ValueError(
            "Tolerance verification failed: no measured features supplied"
        )

    tol_map = {**DEFAULT_TOLERANCES}
    if tolerances:
        tol_map.update(tolerances)

    results: list[ToleranceResult] = []
    pass_count = 0
    fail_count = 0

    for mf in measured_features:
        tol    = tol_map.get(mf.feature_type, tol_map["_default"])
        passed = abs(mf.deviation_px) <= tol
        status = "PASS" if passed else "FAIL"

        if passed:
            pass_count += 1
        else:
            fail_count += 1
            logger.warning(
                f"FAIL  {mf.label}: "
                f"CAD={mf.cad_dimension_px:.1f}px  "
                f"measured={mf.measured_dimension_px:.1f}px  "
                f"deviation={mf.deviation_px:+.1f}px  "
                f"tolerance=±{tol:.0f}px"
            )

        results.append(ToleranceResult(
            feature_type=mf.feature_type,
            label=mf.label,
            cad_dimension_mm=mf.cad_dimension_px,
            measured_dimension_mm=mf.measured_dimension_px,
            deviation_mm=mf.deviation_px,
            tolerance_mm=tol,
            status=status,
            unit=mf.unit,
        ))

    logger.debug(f"Tolerance check: {pass_count} PASS, {fail_count} FAIL")
    return results
