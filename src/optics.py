"""The physics of the holography setup, shared by the tests and later steps.

Used to make fake holograms where we know the right answer (particles at
known positions and distances from the sensor), and to refocus real ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

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


class Refocuser:
    """Refocus one hologram to many depths, quickly.

    Refocusing runs the light backwards from the sensor: the same maths as
    propagate(), with the distance reversed. Each particle's rings then gather
    back into a small dark spot at the depth where the particle is.

    propagate() does a forward and an inverse FFT every time. Here the forward
    FFT is done once, so each extra depth costs one inverse FFT.
    """

    def __init__(self, light: np.ndarray, pad: int = 256, workers: int = -1):
        """light: an evened-out frame (1.0 = nothing there). NaN counts as 1.0."""
        h, w = light.shape
        self.h, self.w, self.pad, self.workers = h, w, pad, workers
        # Padding with plain, unblocked light stops the FFT wrapping rings round
        # from one edge to the other. Sizes the FFT handles fastest
        H, W = sfft.next_fast_len(h + 2 * pad), sfft.next_fast_len(w + 2 * pad)
        field = np.ones((H, W), np.complex64)
        field[pad:pad + h, pad:pad + w] = np.sqrt(np.clip(np.nan_to_num(light, nan=1.0), 0, None))
        self.spectrum = sfft.fft2(field, workers=workers)
        fx = sfft.fftfreq(W, PIXEL).astype(np.float32)
        fy = sfft.fftfreq(H, PIXEL).astype(np.float32)
        along = 1 - (WAVELENGTH * fx[None, :]) ** 2 - (WAVELENGTH * fy[:, None]) ** 2
        root = np.sqrt(np.maximum(along, 0))
        # Phase delay per metre travelled, as in propagate(); waves that can't travel are dropped
        self.delay = (2 * np.pi / WAVELENGTH * (1 - np.maximum(along, 0)) / (1 + root)).astype(np.float32)
        self.travels = along > 0

    def brightness(self, z: float) -> np.ndarray:
        """What the frame looks like refocused z metres back from the sensor (1.0 = nothing there)."""
        kernel = np.exp(1j * (np.float32(z) * self.delay)).astype(np.complex64)
        kernel[~self.travels] = 0
        field = sfft.ifft2(self.spectrum * kernel, workers=self.workers)
        p = self.pad
        return (np.abs(field[p:p + self.h, p:p + self.w]) ** 2).astype(np.float32)
