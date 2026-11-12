"""The physics of the holography setup, shared by the tests and later steps.

Used to make fake holograms where we know the right answer: particles at
known positions and distances from the sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WAVELENGTH = 520e-9   # laser wavelength, metres (green, 520 nm)
PIXEL = 1.85e-6       # camera pixel size, metres


@dataclass(frozen=True)
class Particle:
    x: int            # centre on the sensor, pixels
    y: int
    z: float          # distance from the sensor, metres
    diameter: float   # metres


def propagate(field: np.ndarray, z: float) -> np.ndarray:
    """Work out what a light field looks like after travelling a distance z.

    Angular-spectrum method: split the light into plane waves heading in
    slightly different directions (a Fourier transform), delay each one by
    how far it has to travel to cover distance z, then add them back up.
    """
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, PIXEL)     # spatial frequencies, cycles per metre
    fy = np.fft.fftfreq(ny, PIXEL)
    FX, FY = np.meshgrid(fx, fy)
    # How steeply each plane wave travels; >0 means it really travels forward
    along = 1 - (WAVELENGTH * FX) ** 2 - (WAVELENGTH * FY) ** 2
    # Delay relative to the straight-through beam, k z (sqrt(along) - 1). The -1 matters:
    # the unblocked beam in hologram() isn't delayed, so leaving it out puts a spurious
    # exp(ikz) between the two, which shifts every ring and spins round once per 0.5 um
    # of z. Written as -(1 - along) / (1 + sqrt(along)) so no precision is lost.
    root = np.sqrt(np.maximum(along, 0))
    phase_delay = np.exp(-1j * 2 * np.pi / WAVELENGTH * z * (1 - np.maximum(along, 0)) / (1 + root))
    phase_delay[along < 0] = 0         # waves that can't travel just die out
    return np.fft.ifft2(np.fft.fft2(field) * phase_delay)


def hologram(shape: tuple[int, int], particles: list[Particle], window: int = 1024) -> np.ndarray:
    """Brightness on the sensor from a set of small opaque particles (1.0 = no particle).

    Each particle blocks a small disc of light. What reaches the sensor is
    the unblocked beam minus the light that particle took out, after that
    missing light has spread out over distance z. Where the two overlap they
    interfere, which makes the rings.

    Each particle is worked out in its own window, so particles must sit at
    least window/2 pixels from the edge.
    """
    half = window // 2
    yy, xx = np.mgrid[-half:half, -half:half]
    disc_area = (xx * PIXEL) ** 2 + (yy * PIXEL) ** 2

    blocked = np.zeros(shape, dtype=complex)
    for p in particles:
        disc = (disc_area <= (p.diameter / 2) ** 2).astype(complex)
        blocked[p.y - half:p.y + half, p.x - half:p.x + half] += propagate(disc, p.z)
    return np.abs(1 - blocked) ** 2
