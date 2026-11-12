"""Step 5: measure each particle's size and its distance from the sensor.

A particle's rings carry both numbers:

- Their spacing gives the distance z. The rings go as cos(pi r^2 / (lambda z)),
  so plotted against r^2 they repeat every 2 lambda z.
- How fast they fade out gives the diameter d. Bigger particles' rings fade
  sooner; the first fade-out is at r = 1.22 lambda z / d.

For each particle:

1. Average the frame in thin circles around its centre (its "radial profile").
   Only the particle's own rings are centred there, so neighbours average out.
   The centre is nudged in quarter-pixel steps until the rings are clearest.
2. Read z off the ring spacing (the "ruler").
3. Fit the exact Fresnel pattern of a round, dark particle to the profile,
   adjusting d and z together. A quick far-field grid search gives the
   starting values.

Tested on fake particles made on a large field and cut to size like a real
frame: 36 of 40 sizes within 15% and 37 of 40 distances within 5%, for
5-80 um particles 2-20 mm away. It struggles with 5 um particles far away
and 80 um particles close to the sensor. It assumes a parallel laser beam.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, special

from optics import PIXEL, WAVELENGTH

Z_MIN, Z_MAX = 0.5e-3, 30e-3   # m. Range of distances searched
CENTRE_RADIUS = 150            # px. Profile radius used to find the centre and read the ruler
FIT_RADIUS_MAX = 600           # px. Largest profile radius used in the fit
FIT_R2_MIN = 0.3               # a good fit explains at least 30% of the ring variation
FIT_RADIUS_MIN = 150           # ...and, unless the particle is so close that its rings get finer
                               # than the pixels sooner, had 150 px of rings: a fit cut short by
                               # the sensor's edge can match noise
QUADRATURE = 112               # points for the integral across the particle in the Fresnel formula
_GX, _GW = np.polynomial.legendre.leggauss(QUADRATURE)


@dataclass(frozen=True)
class Sized:
    x: float            # refined centre, px (sub-pixel)
    y: float
    z: float            # distance from the sensor, m
    diameter: float     # m
    diameter_sd: float  # 1-sigma uncertainty from the fit, m
    fit_r2: float       # share of the ring variation the formula explains
    fit_radius: int     # px of rings used
    edge_limited: bool  # True if the sensor's edge cut the rings short

    @property
    def is_good(self) -> bool:
        return (self.fit_r2 >= FIT_R2_MIN and not self.edge_limited
                and self.diameter_sd < 0.25 * self.diameter + 1e-6)


def radial_profile(img: np.ndarray, cx: float, cy: float, radius: int, du: float):
    """Mean brightness in thin rings of equal r^2 width du around (cx, cy).

    Returns ring radii (px), mean brightness and pixel counts. NaN pixels are skipped.
    """
    h, w = img.shape
    x0, x1 = int(max(0, np.floor(cx - radius - 1))), int(min(w, np.ceil(cx + radius + 2)))
    y0, y1 = int(max(0, np.floor(cy - radius - 1))), int(min(h, np.ceil(cy + radius + 2)))
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = (u <= radius * radius) & np.isfinite(sub)
    idx = (u[inside] / du).astype(int)
    n = int(radius * radius / du) + 1
    counts = np.bincount(idx, minlength=n)[:n]
    sums = np.bincount(idx, weights=sub[inside], minlength=n)[:n]
    has = counts > 0
    return np.sqrt((np.arange(n)[has] + 0.5) * du), sums[has] / counts[has], counts[has]


def ring_distance(r: np.ndarray, profile: np.ndarray, pad: int = 8) -> tuple[float, float]:
    """Distance (m) from the ring spacing, and how clearly the rings repeat.

    Against r^2 the rings are a steady wave with frequency f = 1 / (2 lambda z),
    so the strongest frequency in the profile gives z.
    """
    du = 2.0
    u = np.arange((r * r).min(), (r * r).max(), du)
    q = np.interp(u, r * r, profile)
    q = q - np.polyval(np.polyfit(u, q, 3), u)          # drop the slow dip in the middle
    nfft = pad * len(u)
    power = np.abs(np.fft.rfft(q * np.hanning(len(u)), nfft))
    f = np.fft.rfftfreq(nfft, du)                        # cycles per px^2
    z = np.where(f > 0, PIXEL ** 2 / (2 * WAVELENGTH * np.maximum(f, 1e-30)), np.inf)
    k = int(np.argmax(np.where((z >= Z_MIN) & (z <= Z_MAX), power, 0)))
    return float(z[k]), float(power[k])


def find_centre(img: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Nudge (x, y) to where the rings are clearest: up to 3 px in 1 px steps, then 0.25 px."""
    best = (-1.0, x, y)
    for step, span in ((1.0, 3), (0.25, 2)):
        bx, by = best[1], best[2]
        for dy in np.arange(-span, span + 1) * step:
            for dx in np.arange(-span, span + 1) * step:
                r, p, _ = radial_profile(img, bx + dx, by + dy, CENTRE_RADIUS, 2.0)
                clarity = ring_distance(r, p, pad=4)[1]
                if clarity > best[0]:
                    best = (clarity, bx + dx, by + dy)
    return best[1], best[2]


def _pixel_blur(r: np.ndarray, z: float) -> np.ndarray:
    """Each pixel averages the rings over its width, which dims the finest ones."""
    return np.sinc(r * PIXEL / (WAVELENGTH * z))


def ring_pattern(r_px: np.ndarray, z: float, radius: float, gain: float, offset: float) -> np.ndarray:
    """Radial profile of a round, dark particle (exact Fresnel diffraction, parallel beam).

    gain scales the ring contrast (real rings are fainter: partial coherence, other
    particles); offset is the background level.
    """
    r = r_px * PIXEL
    rho = radius * (_GX + 1) / 2
    w = _GW * radius / 2
    lz = WAVELENGTH * z
    across = (special.j0(2 * np.pi * np.outer(r, rho) / lz) * (np.exp(1j * np.pi * rho ** 2 / lz) * rho * w)).sum(1)
    field = 1 - (2 * np.pi / (1j * lz)) * np.exp(1j * np.pi * r ** 2 / lz) * across
    return offset + gain * (np.abs(field) ** 2 - 1) * _pixel_blur(r, z)


def _jinc(v: np.ndarray) -> np.ndarray:
    out = np.ones_like(v)
    nz = np.abs(v) > 1e-9
    out[nz] = 2 * special.j1(v[nz]) / v[nz]
    return out


def _starting_points(r_px, profile, weights, z0, n=2):
    """Quick grid search with the far-field formula; returns the n best (z, radius, gain) with distinct sizes."""
    r = r_px * PIXEL
    found = []
    for z in z0 * np.geomspace(0.6, 1.25, 33):
        theta = np.pi * r * r / (WAVELENGTH * z)
        blur = _pixel_blur(r, z)
        for radius in np.geomspace(1e-6, 90e-6, 48):
            env = _jinc(2 * np.pi * radius * r / (WAVELENGTH * z))
            X = np.stack([np.ones_like(r), np.sin(theta) * env * blur, np.cos(theta) * env * blur, env ** 2], 1)
            coef, *_ = np.linalg.lstsq(X * weights[:, None], profile * weights, rcond=None)
            cost = float(np.sum(((X @ coef) - profile) ** 2 * weights ** 2))
            strength = np.pi * radius ** 2 / (WAVELENGTH * z)          # far-field ring amplitude / 2
            found.append((cost, z, radius, float(np.hypot(coef[1], coef[2]) / (2 * strength))))
    found.sort()
    picks, sizes = [], []
    for _, z, radius, gain in found:
        if all(abs(np.log(radius / s)) > 0.4 for s in sizes):
            picks.append((z, radius, gain)); sizes.append(radius)
        if len(picks) == n:
            break
    return picks


def size_particle(img: np.ndarray, x: float, y: float) -> Sized:
    """Size the particle whose rings are centred near (x, y). img: flattened frame, NaN = unusable."""
    cx, cy = find_centre(img, x, y)
    r, p, _ = radial_profile(img, cx, cy, CENTRE_RADIUS, 2.0)
    z0 = ring_distance(r, p)[0]
    # Beyond this radius the rings are finer than 2 px and can't be trusted
    finest = WAVELENGTH * z0 / (2 * PIXEL ** 2)
    h, w = img.shape
    room = min(cx - 2, cy - 2, w - cx - 3, h - cy - 3)
    radius_px = int(max(40, min(FIT_RADIUS_MAX, 0.9 * finest, room)))
    edge_limited = room < min(FIT_RADIUS_MIN, 0.9 * finest)
    period = 2 * WAVELENGTH * z0 / PIXEL ** 2                       # ring period in r^2, px^2
    r, p, counts = radial_profile(img, cx, cy, radius_px, max(1.0, period / 12))
    weights = np.sqrt(counts / counts.mean())
    # The fit works in mm and um so all four numbers are of a similar size for the optimiser
    def model(v):
        return ring_pattern(r, v[0] * 1e-3, v[1] * 1e-6, v[2], v[3])

    z0_mm = z0 * 1e3
    best = None
    for z_s, a_s, g_s in _starting_points(r, p, weights, z0):
        a_um = a_s * 1e6
        start = [z_s * 1e3, a_um, float(np.clip(g_s, 0.05, 1.9)), float(np.median(p))]
        res = optimize.least_squares(
            lambda v: (model(v) - p) * weights, start,
            bounds=([0.5 * z0_mm, 0.5, 0.0, 0.5], [1.4 * z0_mm, 150.0, 2.0, 1.5]),
            x_scale=[0.05 * z0_mm, max(1.0, 0.2 * a_um), 0.1, 0.01], max_nfev=80)
        if best is None or res.cost < best.cost:
            best = res
    z_mm, radius_um, _, _ = best.x
    resid = model(best.x) - p
    r2 = 1 - np.sum(resid ** 2 * counts) / np.sum((p - np.average(p, weights=counts)) ** 2 * counts)
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * (2 * best.cost / max(1, len(p) - 4))
        radius_sd_um = float(np.sqrt(max(cov[1, 1], 0)))
    except np.linalg.LinAlgError:
        radius_sd_um = float("nan")
    return Sized(cx, cy, z_mm * 1e-3, 2 * radius_um * 1e-6, 2 * radius_sd_um * 1e-6, float(r2), radius_px,
                 bool(edge_limited))


# ---------------------------------------------------------------- command line

COLUMNS = ["x", "y", "strength", "is_particle", "centre_x", "centre_y", "z_mm", "diameter_um",
           "diameter_sd_um", "fit_r2", "fit_radius_px", "good_fit"]
_IMG = None


def _init(npy_path: str) -> None:
    global _IMG
    _IMG = np.load(npy_path)


def _one(row: dict) -> dict:
    s = size_particle(_IMG, float(row["x"]), float(row["y"]))
    return dict(x=row["x"], y=row["y"], strength=row.get("strength", ""), is_particle=row.get("is_particle", ""),
                centre_x=round(s.x, 2), centre_y=round(s.y, 2), z_mm=round(s.z * 1e3, 3),
                diameter_um=round(s.diameter * 1e6, 2), diameter_sd_um=round(s.diameter_sd * 1e6, 2),
                fit_r2=round(s.fit_r2, 3), fit_radius_px=s.fit_radius, good_fit=s.is_good)


def main() -> None:
    parser = argparse.ArgumentParser(description="Size the particles in one raw frame")
    parser.add_argument("raw", type=Path, help="raw frame")
    parser.add_argument("--detections", type=Path,
                        help="that frame's CSV from run.py (skips detection); default: detect again")
    parser.add_argument("--out", type=Path, help="CSV to write (default results/sizes_frameXXX.csv)")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()

    from flatten import flatten
    from raw_io import load_raw, parse_filename
    info = parse_filename(args.raw)
    fl = flatten(load_raw(args.raw))
    if args.detections:
        with args.detections.open() as f:
            f.readline()                                    # summary line
            rows = list(csv.DictReader(f))
    else:
        from describe import describe
        from detect import detect
        rows = [dict(x=d.x, y=d.y, strength=round(d.strength, 2), is_particle=d.is_particle)
                for d in describe(fl, detect(fl))]
    out = args.out or Path("results") / f"sizes_frame{info.camera_id:03d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Workers read the frame from a temporary file rather than each being sent a copy
    tmp = out.with_suffix(".frame.npy")
    np.save(tmp, np.where(fl.usable, fl.flat, np.nan).astype(np.float64))
    # One maths thread per worker: several workers each starting a full thread pool
    # fight over the cores and run many times slower
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    start = time.time()
    try:
        with out.open("w", newline="") as f, ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init, initargs=(str(tmp),),
                mp_context=multiprocessing.get_context("spawn")) as pool:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            good = []
            for i, result in enumerate(pool.map(_one, rows, chunksize=2), 1):
                writer.writerow(result)
                if result["good_fit"] and str(result["is_particle"]) == "True":
                    good.append(result["diameter_um"])
                if i % 25 == 0:
                    print(f"  {i}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"frame {info.camera_id}: {len(rows)} sized, {len(good)} particles with a good fit, "
          f"median {np.median(good):.1f} um -> {out}" if good else f"no good fits -> {out}")


if __name__ == "__main__":
    sys.exit(main())
