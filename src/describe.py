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

It also repairs one kind of detection mistake. Two similar rings side by
side look the same rotated 180 degrees about their midpoint, so the
detector's centre-finding can settle between them. A point between two
rings isn't round, so whenever a detection isn't round, we search around
it for round centres and keep those instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from detect import Detection, contrast_map, detect, noise_energy
from flatten import Flattened, flatten
from raw_io import load_raw, parse_filename

ROUND_RADIUS = 80   # px. Patch radius. Fits inside the usable area's 80 px edge margin
ROUND_MIN = 0.6     # chosen on fake data: keeps 87% of fake particles, rejects every fake fibre
BLACK = -0.4        # contrast below this counts as solid black (60%+ of the light blocked)
DARK_MAX = 0.15     # a particle's patch is under 15% solid black. Real rings in frame 1
                    # reach 5% at most; the big fibres are 23-53%

SEARCH_REACH = 90   # px. How far around a non-round detection to look for round centres
SEARCH_STEP = 8     # px between the points tried in that first, coarse search
SEARCH_RADIUS = 50  # px. A smaller patch while searching: 2.5x faster, and less
                    # confused by the neighbour we're trying to separate from
MERGE_DISTANCE = 35 # px. Centres found this close together are the same ring
SIGNAL_MIN = 0.5    # a patch needs at least half as much real pattern as noise to be judged


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


@lru_cache(maxsize=4)
def _geometry(radius: int):
    """For a patch of this radius: which 1 px ring each pixel is in, and its weight.

    Worked out once per radius and reused, because roundness gets called
    thousands of times per frame during the search.
    """
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt(xx ** 2 + yy ** 2)
    ring = dist.round().astype(int).ravel()
    # Pixels near the centre count most, fading to zero at the edge, so a
    # neighbouring particle at the edge of the patch matters less
    weight = np.where(dist <= radius, 0.5 * (1 + np.cos(np.pi * dist / radius)), 0.0).ravel()
    return ring, weight, np.bincount(ring)


def rotational_average(patch: np.ndarray) -> np.ndarray:
    """What the patch would look like if it were perfectly round about its centre."""
    ring, _, counts = _geometry(patch.shape[0] // 2)
    ring_means = np.bincount(ring, weights=patch.ravel()) / np.maximum(counts, 1)
    return ring_means[ring].reshape(patch.shape)   # paint each pixel with its ring's mean


def roundness(contrast: np.ndarray, x: int, y: int, radius: int = ROUND_RADIUS,
              noise_var: float = 0.0) -> float:
    """How closely the pattern around (x, y) matches a perfectly round version of itself.

    noise_var is how much of the patch's variation we expect from noise alone
    (from detect.noise_energy). Noise makes even a perfect ring look less
    round, so in dim, noisy areas faint rings would be rejected unfairly.
    Allowing for it asks: how round is the part of the patch that isn't noise?
    """
    h, w = contrast.shape
    if not (radius <= x < w - radius and radius <= y < h - radius):
        return float("nan")   # too close to the edge to cut a full patch
    patch = contrast[y - radius:y + radius + 1, x - radius:x + radius + 1].astype(np.float64)
    _, weight, _ = _geometry(radius)
    weight = weight / weight.sum()
    a = patch.ravel()
    b = rotational_average(patch).ravel()   # its rotational average has almost no noise left

    # Weighted correlation between the patch and its round version...
    a = a - (weight * a).sum()
    b = b - (weight * b).sum()
    covariance = (weight * a * b).sum()
    # ...except the patch's own variation has the expected noise taken out first
    signal = (weight * a * a).sum() - noise_var
    if signal < SIGNAL_MIN * noise_var:
        return 0.0   # almost all noise: nothing there to call round
    return float(np.clip(covariance / np.sqrt(signal * (weight * b * b).sum()), -1.0, 1.0))


def round_centres_near(contrast: np.ndarray, usable: np.ndarray, noise: np.ndarray,
                       x0: int, y0: int) -> list[tuple[int, int]]:
    """Centres of round patterns near (x0, y0), for a detection that isn't round itself."""
    h, w = contrast.shape

    def score(x, y):
        if not (0 <= x < w and 0 <= y < h and usable[y, x]):
            return -1.0
        r = roundness(contrast, x, y, SEARCH_RADIUS, noise[y, x])
        return -1.0 if np.isnan(r) else r

    # 1. Coarse search: roundness on a grid of points around the detection
    offsets = range(-SEARCH_REACH, SEARCH_REACH + 1, SEARCH_STEP)
    grid = np.array([[score(x0 + dx, y0 + dy) for dx in offsets] for dy in offsets])

    centres = []
    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            # Keep grid points that are fairly round and rounder than all their neighbours
            neighbours = grid[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if grid[i, j] < ROUND_MIN - 0.15 or grid[i, j] < neighbours.max():
                continue
            # 2. Fine search: step towards rounder points, in smaller and smaller steps
            x, y, best = x0 + dx, y0 + dy, grid[i, j]
            for step in (4, 2, 1):
                moved = True
                while moved:
                    moved = False
                    for sx, sy in ((step, 0), (-step, 0), (0, step), (0, -step),
                                   (step, step), (step, -step), (-step, step), (-step, -step)):
                        candidate = score(x + sx, y + sy)
                        if candidate > best:
                            x, y, best, moved = x + sx, y + sy, candidate, True
            centres.append((x, y))
    return centres


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
    """Roundness and dark fraction for every detection, splitting misplaced ones."""
    contrast = contrast_map(fl.flat)
    noise = noise_energy(fl.flat, fl.beam)   # expected noise at every pixel
    kept: list[Described] = []

    def round_at(x: int, y: int) -> float:
        return roundness(contrast, x, y, noise_var=noise[y, x])

    def keep(x: int, y: int, strength: float) -> None:
        if any(np.hypot(x - k.x, y - k.y) < MERGE_DISTANCE for k in kept):
            return   # already have this ring
        kept.append(Described(x, y, strength, round_at(x, y), dark_fraction(contrast, x, y)))

    for d in sorted(found, key=lambda d: d.strength, reverse=True):
        r, dark = round_at(d.x, d.y), dark_fraction(contrast, d.x, d.y)
        if r >= ROUND_MIN or dark >= DARK_MAX or np.isnan(r):
            keep(d.x, d.y, d.strength)   # round already, or a big fibre, or at the edge
            continue
        # Not round: maybe it landed between particles. Look for round centres nearby.
        centres = [(x, y) for x, y in round_centres_near(contrast, fl.usable, noise, d.x, d.y)
                   if round_at(x, y) >= ROUND_MIN]
        for x, y in centres:
            keep(x, y, d.strength)
        if not centres:
            keep(d.x, d.y, d.strength)   # genuinely not round: stays as "other"
    return kept


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
