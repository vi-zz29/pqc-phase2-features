"""
Tolerance Verification — Stage 7.

All thresholds are in MILLIMETRES.

Default tolerances (±mm):
  circle_radius    : ±0.10 mm
  circle_diameter  : ±0.20 mm
  rect_width       : ±0.30 mm
  rect_height      : ±0.30 mm
  hole_spacing     : ±0.30 mm
  outer_diameter   : ±0.50 mm
  center_bore      : ±0.30 mm
  pcd              : ±0.50 mm
  overall_width    : ±1.00 mm
  overall_height   : ±1.00 mm
  _default         : ±0.30 mm
"""

import logging
from dataclasses import dataclass

from dimension_analysis.measurement import MeasuredFeature

logger = logging.getLogger(__name__)

# Default tolerances per feature_type — all in MILLIMETRES
DEFAULT_TOLERANCES: dict[str, float] = {
    "circle_radius":   0.10,
    "circle_diameter": 0.20,
    "rect_width":      0.30,
    "rect_height":     0.30,
    "hole_spacing":    0.30,
    "outer_diameter":  0.50,
    "center_bore":     0.30,
    "pcd":             0.50,
    "overall_width":   1.00,
    "overall_height":  1.00,
    "_default":        0.30,
}


@dataclass
class ToleranceResult:
    feature_type: str
    label: str
    cad_dimension_mm: float       # nominal from DXF (mm)
    measured_dimension_mm: float  # measured from image (mm)
    deviation_mm: float           # measured − nominal (mm)
    tolerance_mm: float           # ±tolerance (mm)
    status: str                   # "PASS" | "FAIL"
    unit: str = "mm"


def verify_tolerances(
    measured_features: list[MeasuredFeature],
    tolerances: dict[str, float] | None = None,
) -> list[ToleranceResult]:
    """
    Apply mm tolerance rules to every measured feature.

    Parameters
    ----------
    measured_features : output of measurement.recover_dimensions()
                        All values are in mm.
    tolerances        : optional override dict mapping feature_type → ±mm

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
        passed = abs(mf.deviation_mm) <= tol
        status = "PASS" if passed else "FAIL"

        if passed:
            pass_count += 1
        else:
            fail_count += 1
            logger.warning(
                f"FAIL  {mf.label}: "
                f"CAD={mf.cad_dimension_mm:.3f}mm  "
                f"measured={mf.measured_dimension_mm:.3f}mm  "
                f"deviation={mf.deviation_mm:+.3f}mm  "
                f"tolerance=±{tol:.2f}mm"
            )

        results.append(ToleranceResult(
            feature_type=mf.feature_type,
            label=mf.label,
            cad_dimension_mm=mf.cad_dimension_mm,
            measured_dimension_mm=mf.measured_dimension_mm,
            deviation_mm=mf.deviation_mm,
            tolerance_mm=tol,
            status=status,
            unit=mf.unit,
        ))

    logger.debug(f"Tolerance check: {pass_count} PASS, {fail_count} FAIL")
    return results
