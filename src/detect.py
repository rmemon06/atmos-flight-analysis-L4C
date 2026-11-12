"""Step 3: find the centre of every ring pattern.

Rings alternate bright and dark, so simply looking for bright spots breaks
each pattern into pieces. Instead:

1. contrast = flattened - 1   (0 where nothing's there)
2. square it, so bright and dark fringes both count as positive
3. blur it, so each whole ring pattern merges into one smooth hill
4. divide by what noise alone would give, so the threshold means the same
   thing in dim, noisy areas and in bright, clean ones
5. keep the hilltops that stand well above the noise
6. move each hilltop to the pattern's true centre: a far-away particle's
   hill has a flat top, so its highest point can be 60+ px off
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from flatten import Flattened, flatten, pixel_noise
from raw_io import load_raw, parse_filename

PRESMOOTH = 1.5      # px. Light blur first: cuts pixel noise, but fringes near
                     # a ring's centre are many pixels wide, so they survive
ENERGY_BLUR = 28.0   # px. Has to be wider than the first ring: a ring's exact
                     # centre has almost no contrast, so a smaller blur makes a
                     # doughnut with two hilltops instead of one hill
MIN_SEPARATION = 60  # px. Only the highest hilltop within this distance is kept
THRESHOLD = 6.0      # a hilltop must be this many times above the noise
NOISE_BLOCK = 64     # px. Noise is measured in blocks of this size
REFINE_RADIUS = 160  # px. Size of the patch used to find a pattern's true centre
MERGE_DISTANCE = 20  # px. Detections this close after refining are the same ring


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    strength: float   # how many times stronger than noise alone


def contrast_map(flat: np.ndarray) -> np.ndarray:
    """How far each pixel is from 'nothing there' (0), lightly smoothed to cut noise."""
    return cv2.GaussianBlur((flat - 1).astype(np.float32), (0, 0), PRESMOOTH)


def noise_energy(flat: np.ndarray, beam: np.ndarray) -> np.ndarray:
    """The energy you'd expect at each pixel if there were only noise there."""
    h, w = flat.shape
    by, bx = h // NOISE_BLOCK, w // NOISE_BLOCK
    # pixel_noise gives one value per block, row by row, so it reshapes into a grid
    noise = pixel_noise(flat, beam, NOISE_BLOCK)[:, 1].reshape(by, bx)
    # A block covered by a big fibre can look noisier than it is; its neighbours outvote it
    noise = ndimage.median_filter(noise, size=3, mode="nearest")
    variance = cv2.resize((noise ** 2).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # The pre-smoothing averages ~28 neighbouring pixels, which shrinks random
    # noise. For a Gaussian blur of width s, the variance left is 1 / (4 pi s^2).
    return variance / (4 * np.pi * PRESMOOTH ** 2)


def strength_map(fl: Flattened) -> np.ndarray:
    """Ring energy at every pixel, as a multiple of what noise alone would give."""
    contrast = contrast_map(fl.flat)
    energy = cv2.GaussianBlur(contrast * contrast, (0, 0), ENERGY_BLUR)
    return energy / noise_energy(fl.flat, fl.beam)


def hilltops(strength: np.ndarray, usable: np.ndarray,
             threshold: float = THRESHOLD, min_separation: int = MIN_SEPARATION) -> list[Detection]:
    """Points that are the highest within min_separation and above the threshold."""
    size = 2 * min_separation + 1
    highest_nearby = ndimage.maximum_filter(strength, size=size, mode="nearest")
    is_top = (strength == highest_nearby) & (strength > threshold) & usable
    ys, xs = np.nonzero(is_top)
    found = [Detection(int(x), int(y), float(strength[y, x])) for x, y in zip(xs, ys)]
    return sorted(found, key=lambda d: d.strength, reverse=True)


def refine_centre(contrast: np.ndarray, x: int, y: int,
                  radius: int = REFINE_RADIUS, rounds: int = 2) -> tuple[int, int]:
    """Move (x, y) to the true centre of the ring pattern around it.

    A ring pattern looks the same rotated 180 degrees about its true centre.
    Convolving a patch with itself measures how well it matches its own
    rotated copy at every possible centre; the best match is the peak. If
    the true centre is d pixels from the patch's centre, the peak sits 2d
    from the middle, so we halve it. Done with FFTs, so it's fast.
    """
    h, w = contrast.shape
    # A smooth window fades the patch out towards its edges, so other
    # particles near the edge count for less
    window = np.outer(np.hanning(2 * radius + 1), np.hanning(2 * radius + 1))
    size = 2 * (2 * radius + 1)   # zero-padding so the convolution doesn't wrap around
    for _ in range(rounds):       # second round: the window is now centred on the ring
        if not (radius <= x < w - radius and radius <= y < h - radius):
            break                 # too close to the edge to cut a full patch
        patch = contrast[y - radius:y + radius + 1, x - radius:x + radius + 1]
        patch = (patch - patch.mean()) * window
        spectrum = np.fft.rfft2(patch, s=(size, size))
        conv = np.fft.irfft2(spectrum * spectrum, s=(size, size))
        peak_y, peak_x = np.unravel_index(np.argmax(conv), conv.shape)
        x = int(round(x + (peak_x - 2 * radius) / 2))
        y = int(round(y + (peak_y - 2 * radius) / 2))
    return x, y


def detect(fl: Flattened) -> list[Detection]:
    """All ring centres in a flattened frame, strongest first."""
    contrast = contrast_map(fl.flat)
    found: list[Detection] = []
    for hill in hilltops(strength_map(fl), fl.usable):        # strongest first
        x, y = refine_centre(contrast, hill.x, hill.y)
        if not fl.usable[y, x]:
            continue
        # Two hilltops on one wide ring both refine to the same centre: keep the first
        if any(np.hypot(x - d.x, y - d.y) < MERGE_DISTANCE for d in found):
            continue
        found.append(Detection(x, y, hill.strength))
    return found


def save_preview(fl: Flattened, found: list[Detection], path: Path) -> None:
    """The whole frame and a full-resolution zoom, with every detection circled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    lo, hi = np.percentile(fl.flat[fl.usable], [0.5, 99.5])
    view = np.clip((fl.flat - lo) / (hi - lo), 0, 1)
    h, w = view.shape
    y0, y1, x0, x1 = h - 1100, h - 100, 100, 1300   # zoom: busy bottom-left corner

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100,
                             gridspec_kw=dict(width_ratios=[1.33, 1.2]))
    axes[0].imshow(view[::4, ::4], cmap="gray")
    axes[0].add_patch(plt.Rectangle((x0 / 4, y0 / 4), (x1 - x0) / 4, (y1 - y0) / 4,
                                    fill=False, ec="#2a78d6", lw=1.5, ls="--"))
    axes[1].imshow(view[y0:y1, x0:x1], cmap="gray")
    for d in found:
        axes[0].add_patch(Circle((d.x / 4, d.y / 4), 12, fill=False, ec="#eb6834", lw=1.5))
        if x0 <= d.x < x1 and y0 <= d.y < y1:
            axes[1].add_patch(Circle((d.x - x0, d.y - y0), 40, fill=False, ec="#eb6834", lw=2))
    axes[0].set_title(f"{len(found)} detections (blue box = zoom)")
    axes[1].set_title("Zoom, full resolution")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    # Usage: python src/detect.py path/to/frame.raw
    raw_path = Path(sys.argv[1])
    info = parse_filename(raw_path)
    fl = flatten(load_raw(raw_path))
    found = detect(fl)

    strengths = np.array([d.strength for d in found])
    print(f"frame {info.camera_id}: {len(found)} detections")
    if len(found):
        print(f"strength: median {np.median(strengths):.0f}x noise, strongest {strengths.max():.0f}x")
        print("strongest five (x, y, strength):")
        for d in found[:5]:
            print(f"  {d.x:5d} {d.y:5d} {d.strength:8.0f}")

    out = Path("figures") / f"detect_frame{info.camera_id:03d}.png"
    save_preview(fl, found, out)
    print(f"saved {out}")
