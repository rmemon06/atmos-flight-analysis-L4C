"""Step 2: flatten a raw frame.

A particle's rings change the light by a percentage, but the laser beam is
about 70 times brighter in one corner than another. Dividing each frame by a
smooth map of the beam turns every pixel into "fraction of normal brightness"
(1.0 = nothing there), so the same particle gives the same signal wherever it
sits. Parts of the sensor that are too dim to trust, or too close to the
edge, are masked out.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from raw_io import load_raw, parse_filename

# Camera reading with no light at all. Measured from flight frame 133, taken
# after the laser had switched off: mean 27.9, noise 1.3, flat across the
# sensor. It matters most in the dim areas: the top of frame 1 reads about 40,
# but only about 12 of that is laser light.
BLACK_LEVEL = 28.0

# Beam map settings. The median window has to be much wider than a ring
# pattern, or the beam map starts tracing the rings instead of the beam.
SHRINK = 16          # work on a 16x smaller copy: the beam is smooth, so nothing is lost
MEDIAN_SIZE = 15     # median window in shrunk pixels (15 x 16 = 240 px at full size)
SMOOTH_SIGMA = 4.0   # light blur afterwards, in shrunk pixels, to remove the median's blockiness

# Usable-area settings
MIN_BEAM = 4         # laser light (camera zero already removed) below which there's
                     # almost nothing to measure. Dim areas don't need masking beyond
                     # this: the detector judges every spot against the noise measured
                     # in its own area. Tested: pure noise lit with just 4 units gives
                     # no false detections
EDGE = 80            # pixels ignored at each edge: the beam map is least reliable
                     # there, and later steps cut out patches around each ring


@dataclass
class Flattened:
    flat: np.ndarray    # reading / beam: 1.0 means nothing there
    beam: np.ndarray    # smooth beam brightness at every pixel
    usable: np.ndarray  # True where the pixel can be trusted


def smooth_shape(img: np.ndarray, shrink: int = SHRINK, median_size: int = MEDIAN_SIZE,
                 smooth_sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """One pass: shrink, median filter, light blur, enlarge back to full size."""
    h, w = img.shape
    # INTER_AREA averages each 16x16 block, so the small copy is also less noisy
    small = cv2.resize(img, (w // shrink, h // shrink), interpolation=cv2.INTER_AREA)

    # Near the edges the filter window hangs off the image and needs made-up
    # values. Copying the edge pixel would pretend the beam goes flat there.
    # An "odd" reflection continues the slope instead.
    pad = median_size + int(3 * smooth_sigma)
    big = np.pad(small, pad, mode="reflect", reflect_type="odd")

    # A median ignores outliers like the big fibres; a plain blur would be dragged by them
    big = ndimage.median_filter(big, size=median_size)
    big = cv2.GaussianBlur(big, (0, 0), smooth_sigma)
    small = big[pad:-pad, pad:-pad]
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def beam_map(img: np.ndarray) -> np.ndarray:
    """Estimate the beam's smooth shape, ignoring the particles on top of it."""
    first = smooth_shape(img)
    # The median and blur shave about 2% off the top of the bright hill. Dividing
    # by the first guess leaves what it got wrong (ideally 1.0 everywhere);
    # smoothing that the same way and multiplying it back in corrects most of it.
    correction = smooth_shape(img / np.maximum(first, 1.0))
    return first * correction


def usable_mask(beam: np.ndarray, min_beam: float = MIN_BEAM, edge: int = EDGE) -> np.ndarray:
    """True where the beam is bright enough and the pixel isn't too close to the edge."""
    usable = beam > min_beam
    usable[:edge, :] = False
    usable[-edge:, :] = False
    usable[:, :edge] = False
    usable[:, -edge:] = False
    return usable


def flatten(raw: np.ndarray, black_level: float = BLACK_LEVEL) -> Flattened:
    """Divide a frame by its beam map, and work out which pixels are usable."""
    img = raw.astype(np.float32) - black_level
    beam = beam_map(img)
    # np.maximum stops a divide-by-zero if the beam estimate ever reaches 0
    flat = img / np.maximum(beam, 1.0)
    return Flattened(flat=flat, beam=beam, usable=usable_mask(beam))


def pixel_noise(flat: np.ndarray, beam: np.ndarray, block: int = 64) -> np.ndarray:
    """For each block of pixels: (beam brightness, noise as a fraction of brightness).

    Uses differences between neighbouring pixels. Rings change slowly from one
    pixel to the next and noise doesn't, so the differences are almost pure
    noise. The median absolute deviation (MAD) ignores the occasional big jump
    at the edge of a particle.
    """
    h, w = flat.shape
    rows = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            d = np.diff(flat[y:y + block, x:x + block], axis=1).ravel()
            mad = np.median(np.abs(d - np.median(d)))
            # 1.4826 turns a MAD into a standard deviation (for normal noise).
            # The difference of two noisy pixels has sqrt(2) times the noise of one.
            noise = 1.4826 * mad / np.sqrt(2)
            rows.append((beam[y:y + block, x:x + block].mean(), noise))
    return np.array(rows)


def save_preview(raw: np.ndarray, result: Flattened, path: Path) -> None:
    """Raw frame, beam map and flattened frame side by side, to check by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def small(a):
        return cv2.resize(a.astype(np.float32), (a.shape[1] // 4, a.shape[0] // 4),
                          interpolation=cv2.INTER_AREA)

    def stretch(a):
        lo, hi = np.percentile(a, [0.5, 99.5])
        return np.clip((a - lo) / (hi - lo), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=100)
    axes[0].imshow(stretch(small(raw)), cmap="gray")
    axes[0].set_title("1. Raw frame")
    im = axes[1].imshow(small(result.beam), cmap="inferno")
    axes[1].set_title("2. Beam map")
    fig.colorbar(im, ax=axes[1], fraction=0.035, label="camera reading")
    flat_small = small(result.flat)
    usable_small = small(result.usable) > 0.5
    # Set the contrast from usable pixels only, so the noisy masked edges don't wash it out
    lo, hi = np.percentile(flat_small[usable_small], [0.5, 99.5])
    axes[2].imshow(np.clip((flat_small - lo) / (hi - lo), 0, 1), cmap="gray")
    overlay = np.zeros(flat_small.shape + (4,))
    overlay[~usable_small] = (0.92, 0.41, 0.2, 0.45)   # orange, semi-transparent
    axes[2].imshow(overlay)
    axes[2].set_title("3. Flattened (orange = masked)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    # Usage: python src/flatten.py path/to/frame.raw
    raw_path = Path(sys.argv[1])
    info = parse_filename(raw_path)
    raw = load_raw(raw_path)
    result = flatten(raw)

    usable_pct = 100 * result.usable.mean()
    noise = pixel_noise(result.flat, result.beam)
    noise = noise[noise[:, 0] > MIN_BEAM]               # only blocks in usable brightness
    order = np.argsort(noise[:, 0])
    tenth = max(1, len(order) // 10)
    dim_noise = 100 * np.median(noise[order[:tenth], 1])
    bright_noise = 100 * np.median(noise[order[-tenth:], 1])

    print(f"frame {info.camera_id} at {info.captured_utc.isoformat(timespec='milliseconds')}")
    print(f"laser light: {result.beam.min():.0f} to {result.beam.max():.0f} "
          f"(after removing the camera's zero of {BLACK_LEVEL:.0f})")
    print(f"usable: {usable_pct:.0f}% of the sensor")
    print(f"noise: {dim_noise:.1f}% in the dimmest usable areas, {bright_noise:.1f}% in the brightest")

    out = Path("figures") / f"flatten_frame{info.camera_id:03d}.png"
    save_preview(raw, result, out)
    print(f"saved {out}")
