"""Step 3: find every particle by refocusing the hologram.

A particle's rings are its own light, spread out over the few millimetres it
travelled to the sensor. Running the light backwards ("refocusing", see
optics.Refocuser) gathers all of those rings back into one small dark dot where
the particle is. That's why this finds faint particles: rings too faint to see
one by one still add up, in step, into a dot that stands out from the noise.

(The first version of this step squared the rings and blurred them into one
hill per particle. Squaring throws away whether each fringe is dark or bright,
so faint rings drowned in the noise, and next to big particles they made no
hill of their own. In the busy part of flight frame 1 it found 8 of the ~25
particles that refocusing shows.)

1. Refocus the evened-out frame to a stack of depths, 2.5-26 mm from the sensor.
2. At each depth, score how much darker each point is than its surroundings,
   in multiples of the noise there (its "dot score").
3. Keep points that are the darkest within 15 px and score over 8.
4. Refocus more finely around the busiest depth. In flight almost every
   particle sits on one surface 4.4 mm from the sensor, and small particles
   are only sharp within about 0.2 mm of their own depth.
5. A particle shows up at several neighbouring depths and is sharpest at its
   own. Keep each one once, where it scores highest: that depth is its
   distance from the sensor.
6. Measure how stretched its dot is (1 = round). describe.py uses that to
   tell particles from fibres and streaks, which are lines when in focus.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from flatten import Flattened, flatten
from optics import PIXEL, WAVELENGTH, Refocuser
from raw_io import load_raw, parse_filename

PRESMOOTH = 1.5      # px. Light blur for contrast_map(), used by describe.py's dark check

# Depths searched, metres: 0.5 mm steps near the sensor, where most particles
# are, then 2 mm steps. Something sharpest at the first or last depth is really
# in focus outside this range (or is a ring, not a dot), so it isn't kept
COARSE_DEPTHS = np.round(np.r_[np.arange(2.5, 9.01, 0.5), np.arange(10.0, 16.01, 1.0), np.arange(18.0, 26.01, 2.0)], 2) * 1e-3
FINE_REACH = 0.4e-3  # m. Finer depths either side of the busiest coarse depth...
FINE_STEP = 0.1e-3   # ...this far apart

DOT_BLUR = 1.5       # px. Light blur: keeps dots, cuts pixel noise
SURROUND_BLUR = 12.0 # px. Heavy blur: what the area around a dot looks like
NOISE_BLOCK = 128    # px. Noise is measured in blocks this size
THRESHOLD = 8.0      # a dot must score this many times the noise. In pure noise,
                     # at every depth, nothing reaches it (tested)
MIN_SEPARATION = 15  # px. Only the darkest point within this distance counts
BORDER = 24          # px. Dots this close to the masked-off area aren't trusted
MERGE_DISTANCE = 20  # px. The same particle at different depths lands within this
RING_COPY = 0.45     # a dot under 0.45x a particle's score at another depth, on its
RING_COPY_RINGS = 4  # first 4 rings, is one of its rings (see keep_sharpest). On fakes,
RING_COPY_MAX = 30   # ring copies scored up to 0.41x; real neighbours 0.43x or more
SHAPE_RADIUS = 15    # px. Patch used to measure how stretched a dot is


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    z: float          # distance from the sensor where its dot is sharpest, m
    strength: float   # dot score: how many times darker than the noise
    stretch: float    # 1 = a round dot; a line (fibre, streak) is much higher


def contrast_map(flat: np.ndarray) -> np.ndarray:
    """How far each pixel is from 'nothing there' (0), lightly smoothed to cut noise."""
    return cv2.GaussianBlur((flat - 1).astype(np.float32), (0, 0), PRESMOOTH)


def darkness(brightness: np.ndarray) -> np.ndarray:
    """How much darker each point is than its surroundings (> 0 = darker)."""
    return (cv2.GaussianBlur(brightness, (0, 0), SURROUND_BLUR)
            - cv2.GaussianBlur(brightness, (0, 0), DOT_BLUR))


def block_noise(dark: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """The noise in darkness() at every pixel, from the spread in 128 px blocks.

    Uses the median absolute deviation, so the few real dots in a block don't
    count as noise. Only usable pixels count: the masked-off area is plain and
    would make a block look quieter than it is. Every other pixel is enough.
    """
    b = NOISE_BLOCK
    hb, wb = dark.shape[0] // b, dark.shape[1] // b

    def blocks(a):
        return a[:hb * b:2, :wb * b:2].reshape(hb, b // 2, wb, b // 2).transpose(0, 2, 1, 3).reshape(hb, wb, -1)

    values = np.where(blocks(usable), blocks(dark), np.nan)
    enough = np.mean(np.isfinite(values), axis=2) > 0.25
    centre = np.nanmedian(np.where(enough[..., None], values, 0), axis=2)
    spread = 1.4826 * np.nanmedian(np.where(enough[..., None], np.abs(values - centre[..., None]), 0), axis=2)
    spread[~enough] = np.median(spread[enough]) if enough.any() else 1.0
    spread = ndimage.median_filter(spread, size=3, mode="nearest")   # neighbours outvote odd blocks
    return cv2.resize(spread.astype(np.float32), (dark.shape[1], dark.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def dot_score(brightness: np.ndarray, usable: np.ndarray):
    """Dot score at every pixel (darkness in multiples of the noise), plus darkness and noise."""
    dark = darkness(brightness)
    noise = np.maximum(block_noise(dark, usable), 1e-9)
    score = dark / noise
    score[~usable] = 0
    return score, dark, noise


def dot_shape(dark: np.ndarray, noise: np.ndarray, x: int, y: int) -> tuple[float, float, float]:
    """The centre of the dark dot at (x, y), and how stretched it is (1 = round).

    Uses the dot's core: the connected patch at least half as dark as its
    darkest point, so a neighbouring dot isn't joined on. Its centre is the
    middle of that patch, weighted by darkness; inside a big dot the single
    darkest pixel is set by noise. Stretch compares the patch's length with
    its width, from the spread of its pixels along its two main directions.
    """
    r = SHAPE_RADIUS
    h, w = dark.shape
    y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
    level = max(3 * noise[y, x], 0.5 * dark[y, x])
    excess = dark[y0:y1, x0:x1] - level
    labels, _ = ndimage.label(excess > 0)
    own = labels == labels[y - y0, x - x0]
    if labels[y - y0, x - x0] == 0 or own.sum() < 3:
        return float(x), float(y), 1.0
    weight = excess * own
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = weight.sum()
    my, mx = (weight * yy).sum() / total, (weight * xx).sum() / total
    cyy = (weight * (yy - my) ** 2).sum() / total + 0.25   # +0.25: a pixel's own width
    cxx = (weight * (xx - mx) ** 2).sum() / total + 0.25
    cxy = (weight * (yy - my) * (xx - mx)).sum() / total
    half_gap = np.sqrt(((cyy - cxx) / 2) ** 2 + cxy ** 2)
    longest, shortest = (cyy + cxx) / 2 + half_gap, (cyy + cxx) / 2 - half_gap
    return float(mx), float(my), float(np.sqrt(longest / max(shortest, 1e-6)))


def dots_at(refocus: Refocuser, z: float, usable: np.ndarray) -> list[Detection]:
    """Every dark dot scoring over the threshold when refocused to depth z."""
    score, dark, noise = dot_score(refocus.brightness(z), usable)
    candidates = score > THRESHOLD
    if not candidates.any():
        return []
    darkest_nearby = ndimage.maximum_filter(score, size=2 * MIN_SEPARATION + 1, mode="nearest")
    found = []
    for x, y in zip(*np.nonzero(candidates & (score == darkest_nearby))[::-1]):
        cx, cy, s = dot_shape(dark, noise, x, y)
        found.append(Detection(int(round(cx)), int(round(cy)), z, float(score[y, x]), s))
    return found


def ring_number(dist, dz):
    """Which of a particle's rings passes dist px away, refocused dz metres from it.

    The k-th ring is at radius sqrt(2 k wavelength dz), so k = r^2 / (2 wavelength dz).
    """
    return (dist * PIXEL) ** 2 / (2 * WAVELENGTH * np.maximum(dz, 1e-9))


def keep_sharpest(found: list[Detection]) -> list[Detection]:
    """One detection per particle, where it scores highest. Strongest first.

    Two kinds of copy are dropped:
    - the same particle at a neighbouring depth: within 20 px of a stronger one;
    - a strong particle's rings, refocused to a depth well away from its own,
      can look like faint dots. On its first 4 rings such a dot scores under
      0.45x the particle; further out the rings are fainter, and so is the
      limit (0.45 x 4 / ring number). A dot under that is one of its rings.
    """
    found = sorted(found, key=lambda d: d.strength, reverse=True)
    kx, ky, kz, ks = (np.empty(len(found)) for _ in range(4))
    kept: list[Detection] = []
    for d in found:
        n = len(kept)
        dist = np.hypot(kx[:n] - d.x, ky[:n] - d.y)
        dz = np.abs(kz[:n] - d.z)
        same = dist < MERGE_DISTANCE
        ring = ring_number(dist, dz)
        limit = RING_COPY * np.minimum(1, RING_COPY_RINGS / np.maximum(ring, 1e-9))
        ring_copy = (dz > FINE_REACH) & (ring <= RING_COPY_MAX) & (d.strength < limit * ks[:n])
        if not (same | ring_copy).any():
            kx[n], ky[n], kz[n], ks[n] = d.x, d.y, d.z, d.strength
            kept.append(d)
    return kept


def detect(fl: Flattened, workers: int = -1) -> list[Detection]:
    """Every particle in a flattened frame, strongest first. workers: threads for the FFTs."""
    refocus = Refocuser(np.where(fl.usable, fl.flat, 1.0), workers=workers)
    # Where the usable area meets the masked-off part, refocusing makes ripples
    # off the boundary, so dots right next to it aren't trusted
    usable = ndimage.binary_erosion(fl.usable, iterations=BORDER)
    near, far = COARSE_DEPTHS[0], COARSE_DEPTHS[-1]

    def inside(dots):   # sharpest at the first or last depth = really in focus outside the range
        return [d for d in dots if near < d.z < far]

    found: list[Detection] = []
    for z in COARSE_DEPTHS:
        found += dots_at(refocus, z, usable)
    # Refocus finely around the depth that holds the most particles
    depths, counts = np.unique([d.z for d in inside(keep_sharpest(found))] or [near], return_counts=True)
    busiest = depths[np.argmax(counts)]
    for z in busiest + np.arange(-FINE_REACH, FINE_REACH + 1e-9, FINE_STEP):
        new = min(abs(z - c) for c in COARSE_DEPTHS) > FINE_STEP / 2   # not already done
        if new and near < z < far:
            found += dots_at(refocus, float(z), usable)
    return inside(keep_sharpest(found))


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

    print(f"frame {info.camera_id}: {len(found)} detections")
    if len(found):
        depths = np.array([d.z for d in found]) * 1e3
        print(f"depth: median {np.median(depths):.2f} mm, "
              f"{np.mean(np.abs(depths - np.median(depths)) < 0.5) * 100:.0f}% within 0.5 mm of it")
        print("strongest five (x, y, depth mm, score):")
        for d in found[:5]:
            print(f"  {d.x:5d} {d.y:5d} {d.z * 1e3:6.2f} {d.strength:8.0f}")

    out = Path("figures") / f"detect_frame{info.camera_id:03d}.png"
    save_preview(fl, found, out)
    print(f"saved {out}")
