"""
Dimension analysis pipeline.

Stages (run only after identification + feature extraction succeed):
  3. DXF Parsing
  4. CAD-Image Feature Matching
  5. Transformation Estimation
  6. Dimension Recovery
  7. Tolerance Verification
  8. Inspection Report Generation
"""

from dimension_analysis.dxf_parser import parse_dxf, CADFeatureSet
from dimension_analysis.feature_matcher import match_features, MatchedPair
from dimension_analysis.transform_estimator import estimate_transform, TransformResult
from dimension_analysis.measurement import recover_dimensions, MeasuredFeature
from dimension_analysis.tolerance import verify_tolerances, ToleranceResult
from dimension_analysis.report_generator import generate_reports

__all__ = [
    "parse_dxf",
    "CADFeatureSet",
    "match_features",
    "MatchedPair",
    "estimate_transform",
    "TransformResult",
    "recover_dimensions",
    "MeasuredFeature",
    "verify_tolerances",
    "ToleranceResult",
    "generate_reports",
]
