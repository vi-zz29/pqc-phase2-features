import math
import cv2
import numpy as np
from pathlib import Path

OUT_DIR = Path("outputs/hole_visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_dxf(path: str):
    circles, arcs, lines = [], [], []

    with open(path, "r") as f:
        raw = f.readlines()

    pairs = []
    i = 0
    while i + 1 < len(raw):
        code  = raw[i].strip()
        value = raw[i + 1].strip()
        try:
            pairs.append((int(code), value))
        except ValueError:
            pass
        i += 2

    ent_start = ent_end = None
    for idx, (code, val) in enumerate(pairs):
        if code == 2 and val == "ENTITIES":
            ent_start = idx + 1
        if ent_start and code == 0 and val == "ENDSEC":
            ent_end = idx
            break

    if ent_start is None:
        return circles, arcs, lines

    entity_pairs = pairs[ent_start:ent_end]

    blocks = []
    current = None
    for code, val in entity_pairs:
        if code == 0:
            if current is not None:
                blocks.append(current)
            current = {"type": val, "data": {}}
        elif current is not None:
            current["data"][code] = val
            if code in (10, 20, 11, 21, 30, 31):
                current["data"].setdefault(f"_{code}_list", []).append(val)

    if current is not None:
        blocks.append(current)

    for block in blocks:
        t = block["type"]
        d = block["data"]

        def fget(code, default=None):
            v = d.get(code)
            if v is None:
                return default
            try:
                return float(v)
            except ValueError:
                return default

        if t == "CIRCLE":
            cx = fget(10)
            cy = fget(20)
            r  = fget(40)
            if cx is not None and cy is not None and r is not None:
                circles.append({"cx": cx, "cy": cy, "r": r})

        elif t == "ARC":
            cx      = fget(10)
            cy      = fget(20)
            r       = fget(40)
            a_start = fget(50)
            a_end   = fget(51)
            if cx is not None and r is not None:
                arcs.append({"cx": cx, "cy": cy, "r": r,
                             "a_start": a_start, "a_end": a_end})

        elif t == "LINE":
            xs = d.get("_10_list", [])
            ys = d.get("_20_list", [])
            x2s = d.get("_11_list", [])
            y2s = d.get("_21_list", [])
            x1 = float(xs[0]) if xs else None
            y1 = float(ys[0]) if ys else None
            x2 = float(x2s[0]) if x2s else None
            y2 = float(y2s[0]) if y2s else None
            if x1 is not None and x2 is not None:
                lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return circles, arcs, lines


def classify_holes_front(circles, max_hole_radius=10.0,
                         center_cx=148.5, center_cy=105.0, center_tol=2.0):
    holes, rings = [], []
    center_hole = None

    for c in circles:
        dist = math.hypot(c["cx"] - center_cx, c["cy"] - center_cy)

        if dist < center_tol:
            if abs(c["r"] - 14.5) < 0.1:
                center_hole = c
            else:
                rings.append(c)
            continue

        if abs(c["cx"] - 152.16) < 0.5 and abs(c["cy"] - 63.16) < 0.5:
            rings.append(c)
            continue

        if c["r"] <= max_hole_radius:
            holes.append(c)
        else:
            rings.append(c)

    if center_hole is not None:
        holes.append(center_hole)

    return holes, rings


def classify_holes_rear(circles, max_hole_radius=10.0,
                        center_cx=148.5, center_cy=105.0, center_tol=2.0):
    holes, rings = [], []
    center_hole = None

    for c in circles:
        dist = math.hypot(c["cx"] - center_cx, c["cy"] - center_cy)

        if dist < center_tol:
            if abs(c["r"] - 14.5) < 0.1:
                center_hole = c
            else:
                rings.append(c)
            continue

        if abs(c["cx"] - 152.16) < 0.5 and abs(c["cy"] - 146.84) < 0.5 and abs(c["r"] - 2.0) < 0.1:
            rings.append(c)
            continue

        if c["r"] <= max_hole_radius:
            holes.append(c)
        else:
            rings.append(c)

    if center_hole is not None:
        holes.append(center_hole)

    return holes, rings


PADDING   = 40
IMG_SIZE  = 900

def make_transform(all_x, all_y, img_size, padding):
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    w = max_x - min_x or 1
    h = max_y - min_y or 1
    scale = (img_size - 2 * padding) / max(w, h)
    tx = padding - min_x * scale + (img_size - 2 * padding - w * scale) / 2
    ty = padding - min_y * scale + (img_size - 2 * padding - h * scale) / 2
    return scale, tx, ty


def dxf_to_px(x, y, scale, tx, ty, img_h):
    px = int(round(x * scale + tx))
    py = int(round(img_h - (y * scale + ty)))
    return px, py


def draw_dxf(circles, arcs, lines, holes, rings, title, out_path):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 30

    all_x, all_y = [], []
    for c in circles:
        all_x += [c["cx"] - c["r"], c["cx"] + c["r"]]
        all_y += [c["cy"] - c["r"], c["cy"] + c["r"]]
    for a in arcs:
        all_x += [a["cx"] - a["r"], a["cx"] + a["r"]]
        all_y += [a["cy"] - a["r"], a["cy"] + a["r"]]
    for l in lines:
        all_x += [l["x1"], l["x2"]]
        all_y += [l["y1"], l["y2"]]

    scale, tx, ty = make_transform(all_x, all_y, IMG_SIZE, PADDING)

    def to_px(x, y):
        return dxf_to_px(x, y, scale, tx, ty, IMG_SIZE)

    def r_px(r):
        return max(1, int(round(r * scale)))

    for c in rings:
        cx, cy = to_px(c["cx"], c["cy"])
        cv2.circle(img, (cx, cy), r_px(c["r"]), (200, 200, 200), 1, cv2.LINE_AA)

    for a in arcs:
        cx, cy = to_px(a["cx"], a["cy"])
        rp = r_px(a["r"])
        start_a = -a["a_end"]
        end_a   = -a["a_start"]
        cv2.ellipse(img, (cx, cy), (rp, rp), 0,
                    start_a, end_a, (200, 200, 200), 1, cv2.LINE_AA)

    for l in lines:
        p1 = to_px(l["x1"], l["y1"])
        p2 = to_px(l["x2"], l["y2"])
        cv2.line(img, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    HOLE_COLOR    = (0, 255, 128)
    LABEL_COLOR   = (0, 220, 255)
    MARKER_COLOR  = (0, 80, 255)

    for idx, h in enumerate(holes, start=1):
        cx, cy = to_px(h["cx"], h["cy"])
        rp     = r_px(h["r"])

        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), rp, HOLE_COLOR, -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        cv2.circle(img, (cx, cy), rp,     HOLE_COLOR,   2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rp + 6, MARKER_COLOR, 1, cv2.LINE_AA)

        label = str(idx)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.45
        thick  = 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
        lx = cx - tw // 2
        ly = cy - rp - 10
        lx = max(4, min(IMG_SIZE - tw - 4, lx))
        ly = max(th + 4, min(IMG_SIZE - 4, ly))
        cv2.putText(img, label, (lx, ly), font, fscale, LABEL_COLOR, thick, cv2.LINE_AA)

    cv2.putText(img, title, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{len(holes)} holes highlighted (green)",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HOLE_COLOR, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}")


def draw_legend(holes, title, out_path):
    row_h  = 28
    margin = 16
    cols   = 2
    rows   = math.ceil(len(holes) / cols)
    w      = 520
    h      = margin * 2 + row_h * rows + 50

    img = np.ones((h, w, 3), dtype=np.uint8) * 30
    cv2.putText(img, title + " — Hole Legend",
                (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for idx, hole in enumerate(holes, start=1):
        col = (idx - 1) % cols
        row = (idx - 1) // cols
        x   = margin + col * (w // cols)
        y   = 50 + row * row_h

        text = f"#{idx:2d}  cx={hole['cx']:.1f}  cy={hole['cy']:.1f}  r={hole['r']:.2f}"
        cv2.putText(img, text, (x, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}")


def process(dxf_path, label, out_prefix):
    print(f"\n{'-' * 50}")
    print(f"Processing: {dxf_path}  ({label})")
    circles, arcs, lines = parse_dxf(dxf_path)
    print(f"  Total circles: {len(circles)}  |  arcs: {len(arcs)}  |  lines: {len(lines)}")

    if "front" in out_prefix:
        holes, rings = classify_holes_front(circles)
    else:
        holes, rings = classify_holes_rear(circles)

    print(f"  Holes: {len(holes)}  |  Structural rings: {len(rings)}")
    for i, h in enumerate(holes, 1):
        print(f"    #{i:2d}  cx={h['cx']:.2f}  cy={h['cy']:.2f}  r={h['r']:.3f}")

    draw_dxf(circles, arcs, lines, holes, rings,
             label, OUT_DIR / f"{out_prefix}_holes.png")
    draw_legend(holes, label, OUT_DIR / f"{out_prefix}_legend.png")


if __name__ == "__main__":
    process("dxf/circular_front.dxf", "Front View",  "front")
    process("dxf/circular_rear.dxf",  "Rear View",   "rear")
    print(f"\nDone. Check: {OUT_DIR.resolve()}")
