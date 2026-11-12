"""Tests for size.py, on fake particles of known size and distance.

The fakes are made on a 2048 px field, so the rings spread out naturally
and aren't wrapped round, then sized from the middle like a real frame.
"""

import numpy as np
import pytest

from optics import PIXEL, WAVELENGTH, Particle, hologram
from size import radial_profile, ring_distance, size_particle

FIELD = 2048
MID = FIELD // 2


def fake(diameter, z, seed=0):
    """A flattened fake frame (1.0 = nothing there) with one particle in the middle."""
    light = hologram((FIELD, FIELD), [Particle(MID, MID, z, diameter)], window=FIELD)
    rng = np.random.default_rng(seed)
    return rng.poisson(1000 * light) / 1000          # about 3% noise


def test_ruler_reads_distance_from_ring_spacing():
    z = 7e-3
    yy, xx = np.mgrid[-200:201, -200:201] * PIXEL
    rings = 1 + 0.1 * np.cos(np.pi * (xx ** 2 + yy ** 2) / (WAVELENGTH * z))
    r, p, _ = radial_profile(rings, 200, 200, 150, 2.0)
    assert ring_distance(r, p)[0] == pytest.approx(z, rel=0.02)


@pytest.mark.parametrize("diameter, z", [(10e-6, 5e-3), (20e-6, 2e-3), (40e-6, 8e-3)])
def test_finds_size_and_distance(diameter, z):
    s = size_particle(fake(diameter, z), MID - 3, MID + 2)   # start a few px off, like a detection
    assert s.diameter == pytest.approx(diameter, rel=0.1)
    assert s.z == pytest.approx(z, rel=0.03)
    assert s.is_good


def test_noise_alone_is_not_a_good_fit():
    rng = np.random.default_rng(1)
    noise = rng.poisson(1000 * np.ones((FIELD, FIELD))) / 1000
    assert not size_particle(noise, MID, MID).is_good
