"""Step 4: describe each detection.

Two measurements decide whether a detection is a particle:

- Roundness. A particle makes circular rings; fibres, streaks, and points
  that aren't on any ring centre don't. Average the patch around the
  detection in circles (its "rotational average"), which is what it would
  look like if it were perfectly round, then measure how closely the real
  patch matches that. 1.0 means perfectly round.
- Dark fraction: how much of the patch is solid black. A particle's shadow
  is small. A big fibre can fill the whole patch, and from inside a big dark
  blob things can look fairly round, so roundness alone misses it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detect import Detection, contrast_map, detect
from flatten import Flattened, flatten
from raw_io import load_raw, parse_filename

ROUND_RADIUS = 80   # px. Patch radius. Fits inside the usable area's 80 px edge margin
ROUND_MIN = 0.6     # chosen on fake data: keeps 87% of fake particles, rejects every fake fibre
BLACK = -0.4        # contrast below this counts as solid black (60%+ of the light blocked)
DARK_MAX = 0.15     # a particle's patch is under 15% solid black. Real rings in frame 1
                    # reach 5% at most; the big fibres are 23-53%


@dataclass(frozen=True)
class Described:
    x: int
    y: int
    strength: float
    roundness: float
    dark_fraction: float

    @property
    def is_particle(self) -> bool:
        return self.roundness >= ROUND_MIN and self.dark_fraction < DARK_MAX


def rotational_average(patch: np.ndarray) -> np.ndarray:
    """What the patch would look like if it were perfectly round about its centre."""
    r = patch.shape[0] // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ring = np.sqrt(xx ** 2 + yy ** 2).round().astype(int)   # which 1 px ring each pixel is in
    totals = np.bincount(ring.ravel(), weights=patch.ravel())
    counts = np.bincount(ring.ravel())
    ring_means = totals / np.maximum(counts, 1)
    return ring_means[ring]                                  # paint each pixel with its ring's mean


def roundness(contrast: np.ndarray, x: int, y: int, radius: int = ROUND_RADIUS) -> float:
    """How closely the pattern around (x, y) matches a perfectly round version of itself."""
    h, w = contrast.shape
    if not (radius <= x < w - radius and radius <= y < h - radius):
        return float("nan")   # too close to the edge to cut a full patch
    patch = contrast[y - radius:y + radius + 1, x - radius:x + radius + 1].astype(np.float64)
    round_version = rotational_average(patch)

    # Weighted correlation: pixels near the centre count most, fading to zero
    # at the edge, so a neighbouring particle at the edge of the patch matters less
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt(xx ** 2 + yy ** 2)
    weight = np.where(dist <= radius, 0.5 * (1 + np.cos(np.pi * dist / radius)), 0.0)

    def centred(a):
        return a - (weight * a).sum() / weight.sum()

    a, b = centred(patch), centred(round_version)
    return float((weight * a * b).sum() / np.sqrt((weight * a * a).sum() * (weight * b * b).sum()))


def dark_fraction(contrast: np.ndarray, x: int, y: int, radius: int = ROUND_RADIUS) -> float:
    """Fraction of the round patch around (x, y) that is solid black."""
    h, w = contrast.shape
    if not (radius <= x < w - radius and radius <= y < h - radius):
        return float("nan")
    patch = contrast[y - radius:y + radius + 1, x - radius:x + radius + 1]
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    inside = xx ** 2 + yy ** 2 <= radius ** 2
    return float((patch[inside] < BLACK).mean())


def describe(fl: Flattened, found: list[Detection]) -> list[Described]:
    contrast = contrast_map(fl.flat)
    return [Described(d.x, d.y, d.strength, roundness(contrast, d.x, d.y),
                      dark_fraction(contrast, d.x, d.y)) for d in found]


def save_preview(fl: Flattened, described: list[Described], path: Path) -> None:
    """The 8 roundest and 8 least round detections, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contrast = contrast_map(fl.flat)
    ranked = sorted((d for d in described if not np.isnan(d.roundness)), key=lambda d: d.roundness)
    rows = [("Least round", ranked[:8]), ("Roundest", ranked[::-1][:8])]

    fig, axes = plt.subplots(2, 8, figsize=(18, 5.4), dpi=100)
    r = ROUND_RADIUS
    for row, (label, picks) in enumerate(rows):
        for col in range(8):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(picks):
                ax.axis("off")
                continue
            d = picks[col]
            patch = contrast[d.y - r:d.y + r + 1, d.x - r:d.x + r + 1]
            lim = np.percentile(np.abs(patch), 99)
            ax.imshow(patch, cmap="gray", vmin=-lim, vmax=lim)
            ax.set_title(f"{d.roundness:.2f}", fontsize=11)
        axes[row, 0].set_ylabel(label, fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    # Usage: python src/describe.py path/to/frame.raw
    raw_path = Path(sys.argv[1])
    info = parse_filename(raw_path)
    fl = flatten(load_raw(raw_path))
    described = describe(fl, detect(fl))

    n_particles = sum(d.is_particle for d in described)
    print(f"frame {info.camera_id}: {len(described)} detections: "
          f"{n_particles} particles, {len(described) - n_particles} other")
    big_dark = [d for d in described if d.dark_fraction >= DARK_MAX]
    print(f"  {len(big_dark)} rejected for being mostly solid black (big fibres)")
    out = Path("figures") / f"describe_frame{info.camera_id:03d}.png"
    save_preview(fl, described, out)
    print(f"saved {out}")
