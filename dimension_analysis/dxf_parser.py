"""
DXF Parser — Stage 3.

Parses DXF files using ezdxf and returns a structured CADFeatureSet.

For circular parts (circular_rear, circular_top) extracts:
  - outer_diameter, center_bore, pcd, hole_diameters, hole_count,
    hole_positions, groove_diameters, slot_width

For rectangular parts (box_front, box_rear) extracts:
  - overall_width, overall_height, hole_diameters, hole_spacings,
    cutout_sizes, cutout_positions, hole_edge_distances, corner_radii

All coordinates are in DXF units (mm).
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CADCircle:
    """A full circle in CAD space."""
    cx: float
    cy: float
    radius: float
    diameter: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.diameter = self.radius * 2.0


@dataclass
class CADRect:
    """An axis-aligned rectangle derived from lines / arcs in CAD space."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class CADFeatureSet:
    """
    All features extracted from one DXF file in a standardised format.

    `part_type` is either 'circular' or 'rectangular'.
    `raw_circles` and `raw_lines` hold every entity for downstream use.
    """
    part_type: str                          # 'circular' | 'rectangular'
    dxf_path: str

    # --- generic containers ---
    circles: list[CADCircle] = field(default_factory=list)
    rects:   list[CADRect]   = field(default_factory=list)

    # --- circular-part specifics ---
    outer_diameter:    Optional[float] = None
    center_bore:       Optional[float] = None   # diameter
    pcd:               Optional[float] = None   # pitch-circle diameter
    hole_diameters:    list[float]     = field(default_factory=list)
    hole_count:        int             = 0
    hole_positions:    list[tuple[float, float]] = field(default_factory=list)
    groove_diameters:  list[float]     = field(default_factory=list)
    slot_width:        Optional[float] = None

    # --- rectangular-part specifics ---
    overall_width:      Optional[float] = None
    overall_height:     Optional[float] = None
    cutout_sizes:       list[tuple[float, float]] = field(default_factory=list)   # (w, h)
    cutout_positions:   list[tuple[float, float]] = field(default_factory=list)   # (cx, cy)
    hole_spacings:      list[float] = field(default_factory=list)
    hole_edge_distances: list[float] = field(default_factory=list)
    corner_radii:       list[float] = field(default_factory=list)

    # --- raw geometry (for feature matcher) ---
    raw_circles: list[dict] = field(default_factory=list)   # {cx,cy,r}
    raw_lines:   list[dict] = field(default_factory=list)   # {x1,y1,x2,y2}
    raw_arcs:    list[dict] = field(default_factory=list)   # {cx,cy,r,a0,a1}

    def summary(self) -> str:
        lines = [f"CADFeatureSet  part_type={self.part_type}  src={self.dxf_path}"]
        if self.part_type == "circular":
            lines.append(f"  outer_diameter    : {self.outer_diameter}")
            lines.append(f"  center_bore (dia) : {self.center_bore}")
            lines.append(f"  PCD               : {self.pcd}")
            lines.append(f"  hole_count        : {self.hole_count}")
            lines.append(f"  hole_diameters    : {[round(d,3) for d in self.hole_diameters]}")
            lines.append(f"  hole_positions    : {[(round(x,2),round(y,2)) for x,y in self.hole_positions]}")
            lines.append(f"  groove_diameters  : {[round(d,3) for d in self.groove_diameters]}")
        else:
            lines.append(f"  overall_width     : {self.overall_width}")
            lines.append(f"  overall_height    : {self.overall_height}")
            lines.append(f"  cutout_sizes      : {[(round(w,2),round(h,2)) for w,h in self.cutout_sizes]}")
            lines.append(f"  cutout_positions  : {[(round(x,2),round(y,2)) for x,y in self.cutout_positions]}")
            lines.append(f"  hole_count        : {self.hole_count}")
            lines.append(f"  hole_diameters    : {[round(d,3) for d in self.hole_diameters]}")
            lines.append(f"  hole_spacings     : {[round(s,3) for s in self.hole_spacings]}")
            lines.append(f"  corner_radii      : {[round(r,3) for r in self.corner_radii]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw_entities(dxf_path: Path) -> tuple[list, list, list]:
    """Return (circles, arcs, lines) as plain dicts using ezdxf."""
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    circles, arcs, lines = [], [], []

    for entity in msp:
        t = entity.dxftype()
        try:
            if t == "CIRCLE":
                cx, cy = float(entity.dxf.center.x), float(entity.dxf.center.y)
                r = float(entity.dxf.radius)
                circles.append({"cx": cx, "cy": cy, "r": r})

            elif t == "ARC":
                cx, cy = float(entity.dxf.center.x), float(entity.dxf.center.y)
                r  = float(entity.dxf.radius)
                a0 = float(entity.dxf.start_angle)
                a1 = float(entity.dxf.end_angle)
                arcs.append({"cx": cx, "cy": cy, "r": r, "a0": a0, "a1": a1})

            elif t == "LINE":
                x1, y1 = float(entity.dxf.start.x), float(entity.dxf.start.y)
                x2, y2 = float(entity.dxf.end.x),   float(entity.dxf.end.y)
                lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

            elif t == "LWPOLYLINE":
                pts = list(entity.get_points())
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i][0], pts[i][1]
                    x2, y2 = pts[i+1][0], pts[i+1][1]
                    lines.append({"x1": float(x1), "y1": float(y1),
                                  "x2": float(x2), "y2": float(y2)})
                if entity.is_closed and len(pts) >= 2:
                    x1, y1 = pts[-1][0], pts[-1][1]
                    x2, y2 = pts[0][0],  pts[0][1]
                    lines.append({"x1": float(x1), "y1": float(y1),
                                  "x2": float(x2), "y2": float(y2)})

        except Exception as exc:
            logger.debug(f"Skipping entity {t}: {exc}")

    logger.debug(
        f"Loaded {len(circles)} circles, {len(arcs)} arcs, {len(lines)} lines "
        f"from {dxf_path.name}"
    )
    return circles, arcs, lines


def _bounding_box_from_lines(lines: list[dict]) -> Optional[tuple[float, float, float, float]]:
    """Return (min_x, min_y, max_x, max_y) of all line endpoints, or None."""
    if not lines:
        return None
    xs = [l["x1"] for l in lines] + [l["x2"] for l in lines]
    ys = [l["y1"] for l in lines] + [l["y2"] for l in lines]
    return min(xs), min(ys), max(xs), max(ys)


def _group_arcs_by_center(arcs: list[dict], tol: float = 0.5) -> dict:
    """Group arcs by their centre point (rounded to `tol`)."""
    groups: dict[tuple, list] = {}
    for a in arcs:
        key = (round(a["cx"] / tol) * tol, round(a["cy"] / tol) * tol)
        groups.setdefault(key, []).append(a)
    return groups


def _arc_span(arc: dict) -> float:
    span = (arc["a1"] - arc["a0"]) % 360.0
    if span == 0:
        span = 360.0
    return span


def _is_full_circle_from_arcs(arc_list: list[dict], tol: float = 5.0) -> bool:
    """True if arcs at the same centre sum to ≈360°."""
    total = sum(_arc_span(a) for a in arc_list)
    return abs(total - 360.0) < tol


# ---------------------------------------------------------------------------
# Circular-part feature extraction
# ---------------------------------------------------------------------------

def _extract_circular_features(
    circles: list[dict],
    arcs: list[dict],
    lines: list[dict],
) -> dict:
    """
    Returns a dict with keys matching CADFeatureSet circular fields.
    Centre is assumed at the largest-radius full circle (or the most
    concentric group).
    """
    # Classify circles by radius
    # Largest = outer boundary, smallest full circles = holes / bore
    sorted_c = sorted(circles, key=lambda c: c["r"], reverse=True)

    # Also reconstruct full circles from arcs grouped by centre
    arc_groups = _group_arcs_by_center(arcs)
    arc_full_circles = []
    for (cx, cy), grp in arc_groups.items():
        if _is_full_circle_from_arcs(grp):
            # average radius
            r_avg = float(np.mean([a["r"] for a in grp]))
            arc_full_circles.append({"cx": cx, "cy": cy, "r": r_avg})

    all_circles = circles + arc_full_circles

    # Find the centremost cluster: the circle(s) whose centres cluster tightly
    if not all_circles:
        raise ValueError("No circle entities found for circular part")

    # Use median centre as part centre
    cx_vals = [c["cx"] for c in all_circles]
    cy_vals = [c["cy"] for c in all_circles]
    part_cx = float(np.median(cx_vals))
    part_cy = float(np.median(cy_vals))

    CENTER_TOL = 3.0  # mm

    concentric = [c for c in all_circles
                  if math.hypot(c["cx"] - part_cx, c["cy"] - part_cy) < CENTER_TOL]
    peripheral = [c for c in all_circles
                  if math.hypot(c["cx"] - part_cx, c["cy"] - part_cy) >= CENTER_TOL]

    concentric_sorted = sorted(concentric, key=lambda c: c["r"], reverse=True)

    outer_diameter = None
    center_bore    = None
    groove_diameters = []

    if concentric_sorted:
        outer_diameter = concentric_sorted[0]["r"] * 2.0
        if len(concentric_sorted) >= 2:
            center_bore = concentric_sorted[-1]["r"] * 2.0
            # anything between smallest and largest = groove
            for c in concentric_sorted[1:-1]:
                groove_diameters.append(c["r"] * 2.0)

    # Peripheral circles = bolt-hole pattern
    hole_diameters  = sorted(set(round(c["r"] * 2.0, 4) for c in peripheral))
    hole_positions  = [(c["cx"], c["cy"]) for c in peripheral]
    hole_count      = len(peripheral)

    # PCD = 2 × average distance of peripheral circles from centre
    pcd = None
    if peripheral:
        dists = [math.hypot(c["cx"] - part_cx, c["cy"] - part_cy) for c in peripheral]
        pcd = float(np.mean(dists)) * 2.0

    return {
        "outer_diameter":   outer_diameter,
        "center_bore":      center_bore,
        "pcd":              pcd,
        "hole_diameters":   hole_diameters,
        "hole_count":       hole_count,
        "hole_positions":   hole_positions,
        "groove_diameters": groove_diameters,
        "slot_width":       None,   # extend here if slots are detected
    }


# ---------------------------------------------------------------------------
# Rectangular-part feature extraction
# ---------------------------------------------------------------------------

def _extract_rectangular_features(
    circles: list[dict],
    arcs: list[dict],
    lines: list[dict],
) -> dict:
    """
    Returns a dict with keys matching CADFeatureSet rectangular fields.
    Outer boundary is inferred from the axis-aligned extent of all geometry.
    Holes come from CIRCLE entities.
    Rectangular cutouts are detected from closed rectangular line loops.
    Corner radii come from arcs at corners.
    """
    # Overall bounding box from lines + circles
    all_x, all_y = [], []
    for l in lines:
        all_x += [l["x1"], l["x2"]]
        all_y += [l["y1"], l["y2"]]
    for c in circles:
        all_x += [c["cx"] - c["r"], c["cx"] + c["r"]]
        all_y += [c["cy"] - c["r"], c["cy"] + c["r"]]
    for a in arcs:
        all_x += [a["cx"] - a["r"], a["cx"] + a["r"]]
        all_y += [a["cy"] - a["r"], a["cy"] + a["r"]]

    if not all_x:
        raise ValueError("No geometry found for rectangular part")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    overall_width  = max_x - min_x
    overall_height = max_y - min_y

    # Holes = CIRCLE entities (small radius)
    hole_circles = [c for c in circles if c["r"] < 20.0]
    hole_diameters = sorted(set(round(c["r"] * 2.0, 4) for c in hole_circles))
    hole_positions_list = [(c["cx"], c["cy"]) for c in hole_circles]
    hole_count = len(hole_circles)

    # Detect rectangular cutouts from closed line loops
    # A rectangular cutout is 4 lines forming an axis-aligned closed rectangle
    cutout_sizes: list[tuple[float, float]] = []
    cutout_positions: list[tuple[float, float]] = []

    h_lines = [(l["y1"], l["x1"], l["x2"]) for l in lines
               if abs(l["y1"] - l["y2"]) < 0.5]   # horizontal
    v_lines = [(l["x1"], l["y1"], l["y2"]) for l in lines
               if abs(l["x1"] - l["x2"]) < 0.5]   # vertical

    MATCH_TOL = 1.0
    used_h, used_v = set(), set()

    for ih1, (y1, xa1, xb1) in enumerate(h_lines):
        if ih1 in used_h:
            continue
        for ih2, (y2, xa2, xb2) in enumerate(h_lines):
            if ih2 <= ih1 or ih2 in used_h:
                continue
            # two horizontal lines at different y
            if abs(y1 - y2) < MATCH_TOL:
                continue
            # x-ranges must overlap significantly
            x_lo = max(min(xa1, xb1), min(xa2, xb2))
            x_hi = min(max(xa1, xb1), max(xa2, xb2))
            if x_hi - x_lo < 1.0:
                continue
            # look for two vertical lines connecting them
            rect_y1, rect_y2 = min(y1, y2), max(y1, y2)
            rect_x_vals = []
            for iv, (vx, vy1, vy2) in enumerate(v_lines):
                if iv in used_v:
                    continue
                vy_lo, vy_hi = min(vy1, vy2), max(vy1, vy2)
                if (abs(vy_lo - rect_y1) < MATCH_TOL and
                        abs(vy_hi - rect_y2) < MATCH_TOL):
                    rect_x_vals.append((iv, vx))
            if len(rect_x_vals) >= 2:
                x_sorted = sorted(rect_x_vals, key=lambda t: t[1])
                iv_left,  vx_left  = x_sorted[0]
                iv_right, vx_right = x_sorted[-1]
                w = abs(vx_right - vx_left)
                h = abs(rect_y2 - rect_y1)
                if w > 1.0 and h > 1.0:
                    cutout_sizes.append((round(w, 4), round(h, 4)))
                    cutout_positions.append((
                        round((vx_left + vx_right) / 2.0, 4),
                        round((rect_y1 + rect_y2) / 2.0, 4)
                    ))
                    used_h.add(ih1); used_h.add(ih2)
                    used_v.add(iv_left); used_v.add(iv_right)

    # Hole spacings: pairwise distances between hole centres
    hole_spacings: list[float] = []
    for i, (x1, y1) in enumerate(hole_positions_list):
        for x2, y2 in hole_positions_list[i+1:]:
            hole_spacings.append(round(math.hypot(x2 - x1, y2 - y1), 4))
    hole_spacings.sort()

    # Hole-edge distances (min distance from each hole centre to part boundary)
    hole_edge_distances: list[float] = []
    for cx, cy in hole_positions_list:
        d = min(cx - min_x, max_x - cx, cy - min_y, max_y - cy)
        hole_edge_distances.append(round(d, 4))

    # Corner radii: arcs at the corner positions of the bounding box
    corner_radii: list[float] = []
    CORNER_TOL = 20.0  # mm — look within this band of the bounding box corners
    for a in arcs:
        near_corner = (
            (abs(a["cx"] - min_x) < CORNER_TOL or abs(a["cx"] - max_x) < CORNER_TOL) and
            (abs(a["cy"] - min_y) < CORNER_TOL or abs(a["cy"] - max_y) < CORNER_TOL)
        )
        if near_corner:
            corner_radii.append(round(a["r"], 4))
    corner_radii = sorted(set(corner_radii))

    return {
        "overall_width":       overall_width,
        "overall_height":      overall_height,
        "hole_diameters":      hole_diameters,
        "hole_count":          hole_count,
        "hole_positions":      hole_positions_list,
        "hole_spacings":       hole_spacings,
        "cutout_sizes":        cutout_sizes,
        "cutout_positions":    cutout_positions,
        "hole_edge_distances": hole_edge_distances,
        "corner_radii":        corner_radii,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Map blueprint stem → part type
_PART_TYPE_MAP = {
    "box_front":    "rectangular",
    "box_rear":     "rectangular",
    "circular_rear": "circular",
    "circular_top":  "circular",
}


def parse_dxf(dxf_path: str | Path) -> CADFeatureSet:
    """
    Parse a DXF file and return a fully populated CADFeatureSet.

    Raises:
        FileNotFoundError  – DXF file does not exist
        ValueError         – unsupported part type or no geometry found
    """
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF not found: {dxf_path}")

    stem = dxf_path.stem.lower()
    part_type = _PART_TYPE_MAP.get(stem)
    if part_type is None:
        # Fallback heuristic: if the name contains 'box' → rectangular
        if "box" in stem:
            part_type = "rectangular"
        else:
            part_type = "circular"
        logger.warning(
            f"Unknown part name '{stem}' — guessing part_type='{part_type}'"
        )

    logger.debug(f"Parsing DXF '{dxf_path.name}'  part_type={part_type}")

    circles, arcs, lines = _load_raw_entities(dxf_path)

    if part_type == "circular":
        specific = _extract_circular_features(circles, arcs, lines)
    else:
        specific = _extract_rectangular_features(circles, arcs, lines)

    # Build structured circles list for generic access
    structured_circles = [
        CADCircle(cx=c["cx"], cy=c["cy"], radius=c["r"],
                  label=f"circle_{i+1}")
        for i, c in enumerate(circles)
    ]

    # Build structured rects from detected cutouts
    structured_rects = []
    for i, ((w, h), (cx, cy)) in enumerate(
            zip(specific.get("cutout_sizes", []),
                specific.get("cutout_positions", []))):
        structured_rects.append(CADRect(
            x1=cx - w / 2, y1=cy - h / 2,
            x2=cx + w / 2, y2=cy + h / 2,
            label=f"cutout_{i+1}"
        ))

    fs = CADFeatureSet(
        part_type=part_type,
        dxf_path=str(dxf_path),
        circles=structured_circles,
        rects=structured_rects,
        raw_circles=circles,
        raw_lines=lines,
        raw_arcs=arcs,
        # circular specifics
        outer_diameter=specific.get("outer_diameter"),
        center_bore=specific.get("center_bore"),
        pcd=specific.get("pcd"),
        hole_diameters=specific.get("hole_diameters", []),
        hole_count=specific.get("hole_count", 0),
        hole_positions=specific.get("hole_positions", []),
        groove_diameters=specific.get("groove_diameters", []),
        slot_width=specific.get("slot_width"),
        # rectangular specifics
        overall_width=specific.get("overall_width"),
        overall_height=specific.get("overall_height"),
        cutout_sizes=specific.get("cutout_sizes", []),
        cutout_positions=specific.get("cutout_positions", []),
        hole_spacings=specific.get("hole_spacings", []),
        hole_edge_distances=specific.get("hole_edge_distances", []),
        corner_radii=specific.get("corner_radii", []),
    )

    logger.debug(f"DXF parse complete:\n{fs.summary()}")
    return fs
