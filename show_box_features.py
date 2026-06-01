import cv2, numpy as np
from pathlib import Path

OUT = Path("outputs/box_inspection")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 900
PAD  = 80

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


def make_T(circles, arcs, lines):
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
    ox = PAD + (SIZE - 2*PAD - (max_x - min_x)*scale) / 2
    oy = PAD + (SIZE - 2*PAD - (max_y - min_y)*scale) / 2
    return min_x, min_y, scale, ox, oy


def px(x, y, T):
    min_x, min_y, scale, ox, oy = T
    return (int(round((x - min_x)*scale + ox)),
            int(round(SIZE - ((y - min_y)*scale + oy))))

def rp(r, T):
    return max(1, int(round(r * T[2])))


def render(circles, arcs, lines, features, title, out_path):
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 245

    T = make_T(circles, arcs, lines)

    for l in lines:
        cv2.line(img, px(l["x1"], l["y1"], T), px(l["x2"], l["y2"], T),
                 (80, 80, 80), 1, cv2.LINE_AA)
    for a in arcs:
        cx, cy = px(a["cx"], a["cy"], T)
        cv2.ellipse(img, (cx, cy), (rp(a["r"], T), rp(a["r"], T)),
                    0, -a["a1"], -a["a0"], (80, 80, 80), 1, cv2.LINE_AA)
    for c in circles:
        cxp, cyp = px(c["cx"], c["cy"], T)
        cv2.circle(img, (cxp, cyp), rp(c["r"], T), (80, 80, 80), 1, cv2.LINE_AA)

    CIRCLE_COLOR = (0, 160, 50)
    RECT_COLOR   = (200, 80, 0)
    LABEL_BG     = (30, 30, 30)

    for f in features:
        if f["type"] == "circle":
            cxp, cyp = px(f["cx"], f["cy"], T)
            r = rp(f["r"], T)
            overlay = img.copy()
            cv2.circle(overlay, (cxp, cyp), r, CIRCLE_COLOR, -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.circle(img, (cxp, cyp), r,     CIRCLE_COLOR, 2, cv2.LINE_AA)
            cv2.circle(img, (cxp, cyp), r + 7, CIRCLE_COLOR, 1, cv2.LINE_AA)
            label = f["label"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            lx = cxp - tw // 2
            ly = cyp - r - 12
            cv2.rectangle(img, (lx-2, ly-th-2), (lx+tw+2, ly+2), (220,220,220), -1)
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, CIRCLE_COLOR, 1, cv2.LINE_AA)

        elif f["type"] == "rect":
            p1 = px(f["x1"], f["y1"], T)
            p2 = px(f["x2"], f["y2"], T)
            overlay = img.copy()
            cv2.rectangle(overlay, p1, p2, RECT_COLOR, -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.rectangle(img, p1, p2, RECT_COLOR, 2, cv2.LINE_AA)
            label = f["label"]
            lx = min(p1[0], p2[0])
            ly = min(p1[1], p2[1]) - 10
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (lx-2, ly-th-2), (lx+tw+2, ly+2), (220,220,220), -1)
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, RECT_COLOR, 1, cv2.LINE_AA)

    cv2.putText(img, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "Green = circular holes   Orange = rectangular cutout",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")


circles_f, arcs_f, lines_f = parse_dxf_all("dxf/box_front.dxf")

front_features = [
    {"type": "circle", "cx": 165.168750, "cy":  95.475000, "r": 3.048, "label": "#1 hole"},
    {"type": "circle", "cx": 165.168750, "cy": 127.225000, "r": 3.048, "label": "#2 hole"},
    {"type": "rect",
     "x1": 133.291750, "y1": 92.173000,
     "x2": 158.945750, "y2": 117.827000,
     "label": "#3 rect cutout"},
]

render(circles_f, arcs_f, lines_f, front_features,
       "box_front — features to verify (please confirm)",
       OUT / "box_front_features.png")

print("\nbox_front features:")
for f in front_features:
    if f["type"] == "circle":
        print(f"  {f['label']}: circle at ({f['cx']:.2f}, {f['cy']:.2f})  r={f['r']:.3f}mm")
    else:
        print(f"  {f['label']}: rect ({f['x1']:.2f},{f['y1']:.2f}) -> ({f['x2']:.2f},{f['y2']:.2f})")


circles_r, arcs_r, lines_r = parse_dxf_all("dxf/box_rear.dxf")

rear_features = [
    {"type": "rect",
     "x1": 138.054250, "y1": 92.173000,
     "x2": 163.708250, "y2": 117.827000,
     "label": "#1 rect cutout"},
]

render(circles_r, arcs_r, lines_r, rear_features,
       "box_rear — features to verify (please confirm)",
       OUT / "box_rear_features.png")

print("\nbox_rear features:")
for f in rear_features:
    print(f"  {f['label']}: rect ({f['x1']:.2f},{f['y1']:.2f}) -> ({f['x2']:.2f},{f['y2']:.2f})")

print("\nCheck outputs/box_inspection/box_front_features.png")
print("Check outputs/box_inspection/box_rear_features.png")
print("Confirm these are correct before proceeding.")
