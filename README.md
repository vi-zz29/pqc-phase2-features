# CAD-to-Image Alignment & Hole Verification

A computer vision pipeline that takes real photos of mechanical parts, identifies which part it is by matching against CAD blueprints, and verifies that all expected holes are physically present.

---

## What It Does

1. **Identifies the part** — compares the input photo against all blueprints and picks the best match
2. **Aligns the CAD** — warps the blueprint to fit the real image
3. **Verifies holes** — maps every hole from the DXF drawing onto the real image and checks if it's actually there

---

## Project Structure

```
alignment/
├── inputs/               # Real photos to inspect
├── blueprints/           # CAD blueprint PNGs (circular_top.png, circular_rear.png)
├── dxf/                  # DXF drawings (circular_top.dxf, circular_rear.dxf)
├── outputs/              # All results saved here (auto-created)
│   └── <input_name>/
│       ├── hole_verification.png   ← main output
│       ├── best_aligned.png
│       ├── best_overlay.png
│       └── ...
├── cad_image_alignment/  # Core alignment library
│   ├── alignment.py
│   └── constants.py
├── quick_test.py         # Main entry point
└── visualize_holes.py    # DXF hole visualization utility
```

---

## How to Run

```bash
py quick_test.py
```

Place input images in `inputs/`. Results appear in `outputs/<image_name>/`.

---

## Pipeline Overview

### Step 1 — Preprocessing

The real image and blueprint are both converted to edge maps before alignment.

- **Real image**: Otsu thresholding segments the part from background → morphological cleanup → Canny edges + gradient edges combined
- **Blueprint PNG**: Inverted, blurred, Canny edges extracted

### Step 2 — Coarse Alignment

Finds the approximate rotation and scale to match the CAD to the real image.

- **PCA** extracts the principal orientation axis of each contour
- A brute-force search tries all angles (0–360°, 10° steps) across a scale band
- Each candidate is scored using **IoU (Intersection over Union)** on filled silhouettes
- Runs on a **50% downsampled** image for speed (~4x faster)
- Early exit if score exceeds 0.88

### Step 3 — Fine Alignment

Refines the coarse result using feature matching.

- **ORB** detects keypoints and computes binary descriptors on the coarsely aligned edges
- **Brute-force Hamming matching** finds corresponding points
- **RANSAC** robustly estimates a similarity transform, rejecting outliers
- Result: a precise 3×3 transform matrix mapping CAD → real image

### Step 4 — Identification

The best-matching blueprint is selected by **coverage score** (what fraction of the real part's area is covered by the aligned CAD). A coverage ≥ 85% counts as identified.

### Step 5 — Hole Verification

After identification, the DXF file for the matched blueprint is loaded and holes are verified.

**Coordinate chain (no approximations):**
```
DXF mm
  → blueprint PNG pixels      (content bbox scale)
  → resized edge-map pixels   (replicates _validate_inputs exactly)
  → real image pixels         (alignment transform matrix)
```

This chain is fully dynamic — it works correctly for any image size, zoom level, or rotation because the alignment transform already encodes all of that.

**Hole detection**: at each mapped position, the mean pixel intensity inside the hole area is compared to the surrounding annulus. A ratio > 1.20 (bright hole) or < 0.80 (dark hole) counts as present.

---

## Hole Definitions

Holes were defined from the DXF files and agreed manually:

| View | Holes | Notes |
|------|-------|-------|
| Front / Top | 19 | 12 bolt holes (r=2.6mm), 3 mounting holes (r=1.65mm), 3 small holes (r=2.25mm), 1 detail hole (r=2.0mm), center bore (r=14.5mm) |
| Rear | 5 | 3 small holes (r=2.25–3.25mm), center bore (r=14.5mm) |

Center rings (r=27–42.75mm) are structural geometry, not holes.

---

## Output Files

| File | Description |
|------|-------------|
| `hole_verification.png` | Real image annotated with green (found) / red (missing) circles |
| `best_aligned.png` | Aligned CAD edge map |
| `best_overlay.png` | Red=CAD, Green=Real, Yellow=overlap |
| `debug_mask.png` | Segmentation mask |
| `debug_real_edges.png` | Extracted edge map |

---

## Algorithms Used

| Stage | Algorithm |
|-------|-----------|
| Segmentation | Otsu thresholding |
| Edge extraction | Canny, Morphological Gradient |
| Orientation estimation | PCA on contour points |
| Coarse alignment | Brute-force rotation/scale search + IoU scoring |
| Silhouette filling | Flood fill |
| Fine alignment | ORB keypoints + Hamming BF matching + RANSAC |
| Transform estimation | `estimateAffinePartial2D` (similarity: 4 DOF) |
| Hole mapping | 3-stage matrix chain (DXF → blueprint → edge-map → image) |
| Hole detection | Intensity contrast ratio (inner vs annulus) |

---

## Performance Notes

- Coarse search runs on 50% downsampled images (~4x speedup)
- Blueprints are preprocessed once and reused for all input images
- Debug images (gradient, masked) are skipped to reduce I/O
- Early exit in coarse search when a high-quality match is found

---

## Requirements

```
opencv-python
numpy
```

Install with:
```bash
pip install -r requirements.txt
```
