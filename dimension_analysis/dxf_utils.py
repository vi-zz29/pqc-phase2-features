"""
dxf_utils.py — Shared raw DXF entity parser.

Used by quick_test.py and the utility scripts (inspect_box_dxf.py,
show_box_features.py, show_box_holes.py, visualize_holes.py) to read DXF
entities without the full ezdxf-based CADFeatureSet pipeline.

This is the single source of truth for the hand-rolled text parser — all
five previous duplicate copies should delegate here.
"""

from pathlib import Path


def parse_dxf_raw(
    path: str | Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse a DXF file and return (circles, arcs, lines) as plain dicts.

    circles : [{"cx": float, "cy": float, "r": float}, ...]
    arcs    : [{"cx": float, "cy": float, "r": float,
                "a0": float, "a1": float}, ...]
    lines   : [{"x1": float, "y1": float,
                "x2": float, "y2": float}, ...]

    Uses a direct text parser — no ezdxf dependency required.
    """
    circles: list[dict] = []
    arcs:    list[dict] = []
    lines:   list[dict] = []

    with open(str(path), "r") as f:
        raw = f.readlines()

    # Build list of (group_code, value) pairs
    pairs: list[tuple[int, str]] = []
    i = 0
    while i + 1 < len(raw):
        try:
            pairs.append((int(raw[i].strip()), raw[i + 1].strip()))
        except ValueError:
            pass
        i += 2

    # Find the ENTITIES section
    ent_start = ent_end = None
    for idx, (code, val) in enumerate(pairs):
        if code == 2 and val == "ENTITIES":
            ent_start = idx + 1
        if ent_start and code == 0 and val == "ENDSEC":
            ent_end = idx
            break

    if ent_start is None:
        return circles, arcs, lines

    # Split into entity blocks
    blocks: list[dict] = []
    current: dict | None = None
    for code, val in pairs[ent_start:ent_end]:
        if code == 0:
            if current is not None:
                blocks.append(current)
            current = {"type": val, "data": {}}
        elif current is not None:
            current["data"][code] = val
    if current is not None:
        blocks.append(current)

    # Extract geometry from each block
    for block in blocks:
        d = block["data"]
        try:
            if block["type"] == "CIRCLE":
                circles.append({
                    "cx": float(d[10]),
                    "cy": float(d[20]),
                    "r":  float(d[40]),
                })
            elif block["type"] == "ARC":
                arcs.append({
                    "cx": float(d[10]),
                    "cy": float(d[20]),
                    "r":  float(d[40]),
                    "a0": float(d.get(50, 0)),
                    "a1": float(d.get(51, 360)),
                })
            elif block["type"] == "LINE":
                lines.append({
                    "x1": float(d[10]),
                    "y1": float(d[20]),
                    "x2": float(d[11]),
                    "y2": float(d[21]),
                })
        except (KeyError, ValueError, TypeError):
            pass

    return circles, arcs, lines
