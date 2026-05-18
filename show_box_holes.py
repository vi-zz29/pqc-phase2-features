"""
Renders the box DXF files and highlights ONLY the CIRCLE entities
(which are the actual holes in engineering drawings).
For box parts, holes are full circles — arcs are structural outlines.
Saves to outputs/box_inspection/ for review before any code is written.
"""
import math, cv2, numpy as np
from pathlib import Path
from collections import defaultdict

OUT = Path("outputs/box_inspection")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 900
PAD  = 60


def parse_dxf_all(path):
    circles, arcs, lines = [], [], []
    with open(path) as f:
        raw = f.readlines()
    pairs = []
    i = 0
    while i + 1 < len(raw):
        try:
            pairs.append((int(raw[i].strip()), raw[i+1].strip()))
        except:
            pass
        i += 2
    ent_start = ent_end = None
    for idx, (code, val) in enumerate(pairs):
        if code == 2 and val == "ENTITIES":
            ent_start = idx + 1
        if ent_start and code == 0 and val == "ENDSEC":
            ent_end = idx
            break
    if not ent_start:
        return circles, arcs, lines
    blocks = []
    current = None
    for code, val in pairs[ent_start:ent_end]:
        if code == 0:
            if current:
                blocks.append(current)
            current = {"type": val, "data": {}}
        elif current:
            current["data"][code] = val
    if current:
        blocks.append(current)
    for b in blocks:
        d = b["data"]
        try:
            if b["type"] == "CIRCLE":
                circles.append({"cx": float(d[10]), "cy": float(d[20]), "r": float(d[40])})
            elif b["type"] == "ARC":
                arcs.append({
                    "cx": float(d[10]), "cy": float(d[20]), "r": float(d[40]),
                    "a0": float(d.get(50, 0)), "a1": float(d.get(51, 360))
                })
            elif b["type"] == "LINE":
                lines.append({
                    "x1": float(d[10]), "y1": float(d[20]),
                    "x2": float(d[11]), "y2": float(d[21])
                })
        except:
            pass
    return circles, arcs, lines


def make_transform(circles, arcs, lines):
    all_x, all_y = [], []
    for c in circles:
        all_x += [c["cx"]-c["r"], c["cx"]+c["r"]]
        all_y += [c["cy"]-c["r"], c["cy"]+c["r"]]
    for a in arcs:
        all_x += [a["cx"]-a["r"], a["cx"]+a["r"]]
        all_y += [a["cy"]-a["r"], a["cy"]+a["r"]]
    for l in lines:
        all_x += [l["x1"], l["x2"]]
        all_y += [l["y1"], l["y2"]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span = max(max_x - min_x, max_y - min_y) or 1
    scale = (SIZE - 2*PAD) / span
    cx_off = PAD + (SIZE - 2*PAD - (max_x - min_x)*scale) / 2
    cy_off = PAD + (SIZE - 2*PAD - (max_y - min_y)*scale) / 2
    return min_x, min_y, scale, cx_off, cy_off


def to_px(x, y, min_x, min_y, scale, cx_off, cy_off):
    return (int(round((x - min_x)*scale + cx_off)),
            int(round(SIZE - ((y - min_y)*scale + cy_off))))


def r_px(r, scale):
    return max(1, int(round(r * scale)))


def render(circles, arcs, lines, hole_circles, title, out_path):
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 30
    T = make_transform(circles, arcs, lines)
    min_x, min_y, scale, cx_off, cy_off = T

    def px(x, y): return to_px(x, y, min_x, min_y, scale, cx_off, cy_off)
    def rp(r):    return r_px(r, scale)

    # Draw structural geometry in white
    for l in lines:
        cv2.line(img, px(l["x1"], l["y1"]), px(l["x2"], l["y2"]),
                 (200, 200, 200), 1, cv2.LINE_AA)
    for a in arcs:
        cx, cy = px(a["cx"], a["cy"])
        cv2.ellipse(img, (cx, cy), (rp(a["r"]), rp(a["r"])),
                    0, -a["a1"], -a["a0"], (200, 200, 200), 1, cv2.LINE_AA)
    for c in circles:
        cx, cy = px(c["cx"], c["cy"])
        cv2.circle(img, (cx, cy), rp(c["r"]), (200, 200, 200), 1, cv2.LINE_AA)

    # Highlight hole circles in green with number labels
    for i, h in enumerate(hole_circles, start=1):
        cx, cy = px(h["cx"], h["cy"])
        rr = rp(h["r"])

        # Filled highlight
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), rr, (0, 255, 128), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        cv2.circle(img, (cx, cy), rr,     (0, 255, 128), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rr + 8, (0, 80, 255),  1, cv2.LINE_AA)

        label = f"#{i}  r={h['r']:.2f}mm"
        cv2.putText(img, label,
                    (cx - 10, cy - rr - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.putText(img, title, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{len(hole_circles)} hole(s) shown in green",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}")


# ── Process each box DXF ──────────────────────────────────────────────────────
for name in ["box_front", "box_rear"]:
    circles, arcs, lines = parse_dxf_all(f"dxf/{name}.dxf")
    print(f"\n{name}:")
    print(f"  CIRCLE entities: {len(circles)}")
    print(f"  ARC entities:    {len(arcs)}")
    print(f"  LINE entities:   {len(lines)}")

    if circles:
        print("  Circles (potential holes):")
        for i, c in enumerate(circles, 1):
            print(f"    #{i}  cx={c['cx']:.3f}  cy={c['cy']:.3f}  r={c['r']:.4f}mm")
    else:
        print("  No CIRCLE entities found — box parts use ARCs for outlines only.")
        print("  Holes in this part need to be identified from the input image or")
        print("  confirmed manually from the DXF arc geometry.")

    # For now show ALL circles as candidate holes
    render(circles, arcs, lines, circles,
           f"{name} — CIRCLE entities (candidate holes)",
           OUT / f"{name}_candidate_holes.png")

print("\nDone. Check outputs/box_inspection/")
print("Please confirm which holes are correct before proceeding.")
