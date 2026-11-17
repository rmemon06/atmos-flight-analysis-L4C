"""Step 4: decide which detections are particles.

Every detection is a dark dot in the refocused frame (see detect.py). Two
checks decide whether it's a particle:

- Stretch: in focus, a particle is a compact dot. A fibre or a streak is a
  line, so its dot is stretched. detect.py measures it: 1 = round.
- Dark fraction: how much of the hologram around it is solid black. A big
  fibre close to the sensor blocks so much light that the area around it is
  mostly black, and it breaks into several dots along its edge. A particle's
  shadow is small.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detect import Detection, contrast_map, detect
from flatten import Flattened, flatten
from raw_io import load_raw, parse_filename

STRETCH_MAX = 4.0   # a particle's dot is at most this stretched. Short rod-shaped particles in
                    # frame 1 reach about 3.5; long thin fibres run off the patch and score more
DARK_RADIUS = 80    # px. Patch radius for the dark check. Fits inside the usable area's edge margin
BLACK = -0.4        # contrast below this counts as solid black (60%+ of the light blocked)
DARK_MAX = 0.15     # a particle's patch is under 15% solid black. Real rings in frame 1
                    # reach 5% at most; the big fibres are 23-53%


@dataclass(frozen=True)
class Described:
    x: int
    y: int
    z: float            # distance from the sensor, m
    strength: float
    stretch: float
    dark_fraction: float

    @property
    def is_particle(self) -> bool:
        return self.stretch <= STRETCH_MAX and self.dark_fraction < DARK_MAX


def dark_fraction(contrast: np.ndarray, x: int, y: int, radius: int = DARK_RADIUS) -> float:
    """Fraction of the round patch around (x, y) that is solid black."""
    h, w = contrast.shape
    if not (radius <= x < w - radius and radius <= y < h - radius):
        return float("nan")
    patch = contrast[y - radius:y + radius + 1, x - radius:x + radius + 1]
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    inside = xx ** 2 + yy ** 2 <= radius ** 2
    return float((patch[inside] < BLACK).mean())


def describe(fl: Flattened, found: list[Detection]) -> list[Described]:
    """Stretch and dark fraction for every detection."""
    contrast = contrast_map(fl.flat)
    return [Described(d.x, d.y, d.z, d.strength, d.stretch, dark_fraction(contrast, d.x, d.y))
            for d in found]


if __name__ == "__main__":
    # Usage: python src/describe.py path/to/frame.raw
    raw_path = Path(sys.argv[1])
    info = parse_filename(raw_path)
    fl = flatten(load_raw(raw_path))
    described = describe(fl, detect(fl))

    n_particles = sum(d.is_particle for d in described)
    print(f"frame {info.camera_id}: {len(described)} detections: "
          f"{n_particles} particles, {len(described) - n_particles} other")
    print(f"  {sum(d.stretch > STRETCH_MAX for d in described)} too stretched (fibres, streaks)")
    print(f"  {sum(d.dark_fraction >= DARK_MAX for d in described)} mostly solid black (big fibres)")
