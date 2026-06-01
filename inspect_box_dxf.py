import math, cv2, numpy as np
from pathlib import Path

OUT = Path("outputs/box_inspection")
OUT.mkdir(parents=True, exist_ok=True)

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


def render_dxf(circles, arcs, lines, title, out_path, highlight_arcs=None):
    SIZE = 900
    PAD  = 40

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

    if not all_x:
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span = max(max_x - min_x, max_y - min_y) or 1
    scale = (SIZE - 2*PAD) / span
    cx_off = PAD + (SIZE - 2*PAD - (max_x - min_x)*scale) / 2
    cy_off = PAD + (SIZE - 2*PAD - (max_y - min_y)*scale) / 2

    def to_px(x, y):
        return int(round((x - min_x)*scale + cx_off)), int(round(SIZE - ((y - min_y)*scale + cy_off)))

    def r_px(r):
        return max(1, int(round(r * scale)))

    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 30

    for l in lines:
        cv2.line(img, to_px(l["x1"], l["y1"]), to_px(l["x2"], l["y2"]), (200,200,200), 1, cv2.LINE_AA)

    for i, a in enumerate(arcs):
        cx, cy = to_px(a["cx"], a["cy"])
        rp = r_px(a["r"])
        color = (200, 200, 200)
        if highlight_arcs and i in highlight_arcs:
            color = (0, 255, 128)
        cv2.ellipse(img, (cx, cy), (rp, rp), 0, -a["a1"], -a["a0"], color, 2, cv2.LINE_AA)

    for c in circles:
        cx, cy = to_px(c["cx"], c["cy"])
        cv2.circle(img, (cx, cy), r_px(c["r"]), (200,200,200), 1, cv2.LINE_AA)

    cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}")


for name in ["box_front", "box_rear"]:
    circles, arcs, lines = parse_dxf_all(f"dxf/{name}.dxf")
    print(f"\n{name}: {len(circles)} circles, {len(arcs)} arcs, {len(lines)} lines")

    from collections import defaultdict
    center_groups = defaultdict(list)
    for i, a in enumerate(arcs):
        key = (round(a["cx"], 2), round(a["cy"], 2))
        center_groups[key].append((i, a))

    print("  Arc groups by center:")
    hole_arc_indices = []
    for (cx, cy), group in sorted(center_groups.items()):
        radii = [a["r"] for _, a in group]
        span_deg = [(a["a1"] - a["a0"]) % 360 for _, a in group]
        print(f"    center=({cx},{cy})  radii={[round(r,3) for r in radii]}  spans={[round(s,1) for s in span_deg]}")
        if max(radii) < 15.0:
            for idx, _ in group:
                hole_arc_indices.append(idx)

    render_dxf(circles, arcs, lines, name, OUT / f"{name}_all.png")
    render_dxf(circles, arcs, lines, name + " (green=hole candidates)",
               OUT / f"{name}_holes.png", highlight_arcs=set(hole_arc_indices))

print("\nDone. Check outputs/box_inspection/")
