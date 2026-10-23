"""
Preprocessing script for the Gehler-Shi 568 dataset (Canon 1D + Canon 5D PNGs).

Pipeline per image:
1. Load 12-bit linear PNG (uint16, BGR)
2. Subtract camera black level (Canon 1D = 0, Canon 5D = 129)
3. White-balance using the ground-truth illumination from real_illum_568..mat
4. Normalise to [0, 1] float32
5. Load the hand-labelled colorchecker quad from groundtruth_568/coordinates/
6. Compute a grey-pixel mask  (saturation < GREY_THRESH after white balancing)
7. Zero-out the colorchecker region from the grey mask

Illumination rows in real_rgb follow the official dataset order: the first 86
entries match sorted 8D5U* (Canon 1D) filenames; the remaining 482 match sorted
IMG_* (Canon 5D) filenames. Do not index illuminations from a single global sort
of both prefixes without this per-camera split.

Outputs (under OUT_DIR/):
    wb/       *.npy  – white-balanced float32 images  (H x W x 3, RGB)
    wb_png/   *.png  – gamma-corrected 8-bit PNGs for visual inspection
    masks/    *.png  – grey-pixel masks, colorchecker removed  (0 / 255)
    cc_masks/ *.png  – colorchecker region masks for reference (0 / 255)
"""

import numpy as np
import scipy.io
import cv2
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
IMG_DIR = Path("png_canon1d/cs/chroma/data/canon_dataset/568_dataset/png")
GT_FILE = Path("groundtruth_568/real_illum_568..mat")
CC_DIR  = Path("groundtruth_568/coordinates")
OUT_DIR = Path("preprocessed_canon1d")

# ── Camera constants (Gehler-Shi re-processed PNGs) ──────────────────────────
CANON1D_COUNT   = 86    # real_rgb rows 0:86
BLACK_LEVEL_1D  = 0     # Canon 1D Mark II
BLACK_LEVEL_5D  = 129   # Canon 5D (see groundtruth_568/description.txt)

# ── Grey-pixel thresholds ────────────────────────────────────────────────────
GREY_THRESH       = 0.10   # HSV saturation (max-min)/max < this → grey
BRIGHTNESS_THRESH = 0.02   # max(R,G,B) must exceed this (exclude near-black)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_illuminations(path: Path) -> np.ndarray:
    """Return (568, 3) float64 ground-truth RGB illuminations."""
    return scipy.io.loadmat(str(path))["real_rgb"]


def build_illumination_map(illuminations: np.ndarray, stems: list[str]) -> dict[str, np.ndarray]:
    """
    Map each image stem to its ground-truth illumination vector.

    real_rgb is stored in per-camera filename order (sorted 8D5U*, then sorted IMG_*),
    not by a blind global enumeration that can drift when coordinate files exist for
    missing PNGs.
    """
    canon1d = sorted(s for s in stems if s.startswith("8D5U"))
    canon5d = sorted(s for s in stems if s.startswith("IMG_"))
    other = [s for s in stems if s not in canon1d and s not in canon5d]

    if other:
        raise ValueError(f"Unrecognised filename prefixes: {other[:5]}")
    if len(canon1d) != CANON1D_COUNT:
        raise ValueError(
            f"Expected {CANON1D_COUNT} Canon 1D (8D5U*) images, found {len(canon1d)}"
        )
    expected_5d = len(illuminations) - CANON1D_COUNT
    if len(canon5d) != expected_5d:
        raise ValueError(
            f"Expected {expected_5d} Canon 5D (IMG_*) images, found {len(canon5d)}"
        )

    illum_map: dict[str, np.ndarray] = {}
    for i, stem in enumerate(canon1d):
        illum_map[stem] = illuminations[i]
    for i, stem in enumerate(canon5d):
        illum_map[stem] = illuminations[CANON1D_COUNT + i]
    return illum_map


def black_level_for(stem: str) -> int:
    if stem.startswith("IMG_"):
        return BLACK_LEVEL_5D
    return BLACK_LEVEL_1D


def white_balance(img_rgb: np.ndarray, illum: np.ndarray) -> np.ndarray:
    """
    Divide each channel by illumination (green-normalised), scale to [0, 1].
    img_rgb : (H, W, 3) float32 RGB
    illum   : (3,) float64
    """
    scale = illum / (illum[1] + 1e-8)
    wb = img_rgb / (scale[np.newaxis, np.newaxis, :] + 1e-8)
    wb /= (wb.max() + 1e-8)
    return wb.astype(np.float32)


def compute_grey_mask(wb_img: np.ndarray) -> np.ndarray:
    """
    Binary mask: 255 = achromatic pixel (low saturation, sufficient brightness).
    """
    r, g, b = wb_img[:, :, 0], wb_img[:, :, 1], wb_img[:, :, 2]
    pix_max = np.maximum.reduce([r, g, b])
    pix_min = np.minimum.reduce([r, g, b])
    saturation = (pix_max - pix_min) / (pix_max + 1e-8)
    is_grey = (saturation < GREY_THRESH) & (pix_max > BRIGHTNESS_THRESH)
    return (is_grey * 255).astype(np.uint8)


def load_colorchecker_mask(stem: str, img_h: int, img_w: int) -> np.ndarray:
    """
    Build a filled polygon mask for the colorchecker region.

    The *_macbeth.txt coordinate files have this layout:
      Line 1 : orig_width  orig_height   (reference image dimensions)
      Lines 2-5  : x y  for the 4 outer corners of the whole chart
      Lines 6-101: x y  for the 4 corners of each of the 24 patches (4×24)

    Coordinates are in the original full-resolution RAW space; they are
    scaled to the half-resolution PNG dimensions before use.

    Returns (H, W) uint8: 255 = colorchecker, 0 = scene.
    """
    coord_file = CC_DIR / f"{stem}_macbeth.txt"
    if not coord_file.exists():
        return np.zeros((img_h, img_w), dtype=np.uint8)

    lines = coord_file.read_text().split()
    # Line 1: reference dimensions
    orig_w, orig_h = float(lines[0]), float(lines[1])
    scale_x = img_w / orig_w
    scale_y = img_h / orig_h

    # Lines 2-5: 4 outer corners (each is "x y" = 2 values)
    corners = []
    offset = 2   # start after the two dimension values
    for _ in range(4):
        x = float(lines[offset]) * scale_x
        y = float(lines[offset + 1]) * scale_y
        corners.append([x, y])
        offset += 2

    quad = np.array(corners, dtype=np.float32)
    # convexHull ensures a proper winding order regardless of corner order in file
    hull = cv2.convexHull(quad).astype(np.int32)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    return mask


# ── Main ─────────────────────────────────────────────────────────────────────

def process_dataset():
    wb_dir     = OUT_DIR / "wb"
    wb_png_dir = OUT_DIR / "wb_png"
    mask_dir   = OUT_DIR / "masks"
    cc_dir     = OUT_DIR / "cc_masks"
    for d in (wb_dir, wb_png_dir, mask_dir, cc_dir):
        d.mkdir(parents=True, exist_ok=True)

    illuminations = load_illuminations(GT_FILE)   # (568, 3)

    images = sorted(IMG_DIR.glob("*.png"))
    n = len(images)
    print(f"Found {n} images (Canon 1D + Canon 5D)")

    illum_map = build_illumination_map(illuminations, [p.stem for p in images])
    process_order = sorted(illum_map.keys())

    print(f"Output: {OUT_DIR.resolve()}\n")
    print(f"{'#':>4}  {'Image':<20}  {'Illumination (R,G,B)':>28}  {'Grey%':>6}  {'Grey-CC%':>9}")
    print("-" * 73)

    for idx, stem in enumerate(process_order):
        img_path = IMG_DIR / f"{stem}.png"
        raw_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw_bgr is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping.")
            continue

        img_rgb = raw_bgr[:, :, ::-1].astype(np.float32)
        img_rgb = np.clip(img_rgb - black_level_for(stem), 0, None)

        illum = illum_map[stem]
        wb    = white_balance(img_rgb, illum)

        H, W  = wb.shape[:2]
        grey  = compute_grey_mask(wb)
        cc    = load_colorchecker_mask(stem, H, W)

        grey_clean = grey.copy()
        grey_clean[cc > 0] = 0

        np.save(wb_dir / f"{stem}_wb.npy", wb)

        wb_gamma = np.power(np.clip(wb, 0, 1), 1 / 2.2)
        cv2.imwrite(str(wb_png_dir / f"{stem}_wb.png"),
                    (wb_gamma[:, :, ::-1] * 255).astype(np.uint8))

        cv2.imwrite(str(mask_dir / f"{stem}_grey_mask.png"), grey_clean)
        cv2.imwrite(str(cc_dir   / f"{stem}_cc_mask.png"),   cc)

        grey_pct  = grey.mean() / 255 * 100
        clean_pct = grey_clean.mean() / 255 * 100
        print(f"{idx+1:>4}  {stem + '.png':<20}  "
              f"[{illum[0]:6.0f},{illum[1]:6.0f},{illum[2]:6.0f}]  "
              f"{grey_pct:5.1f}%  {clean_pct:8.1f}%")

    print("\nDone.")
    print(f"  White-balanced .npy  ->  {wb_dir}")
    print(f"  White-balanced .png  ->  {wb_png_dir}")
    print(f"  Grey masks (clean)   ->  {mask_dir}")
    print(f"  CC region masks      ->  {cc_dir}")


if __name__ == "__main__":
    process_dataset()
