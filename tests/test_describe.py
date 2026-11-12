"""Tests for describe.py: round particles pass, fibres and off-centre points don't."""

import numpy as np

from describe import describe, rotational_average, roundness
from detect import Detection, contrast_map
from flatten import flatten
from optics import PIXEL, Particle, hologram, propagate

SHAPE = (1400, 1400)
CENTRE = 700


def fake_flattened(light, seed=0):
    rng = np.random.default_rng(seed)
    return flatten(rng.poisson(1000 * light).astype(np.float32))


def fibre(length_um, width_um, z, angle=0.6):
    """Light reaching the sensor from a thin opaque rod (a fake fibre) at the centre."""
    half = 512
    yy, xx = np.mgrid[-half:half, -half:half] * PIXEL
    along = xx * np.cos(angle) + yy * np.sin(angle)
    across = -xx * np.sin(angle) + yy * np.cos(angle)
    rod = (np.abs(along) <= length_um * 1e-6 / 2) & (np.abs(across) <= width_um * 1e-6 / 2)
    blocked = np.zeros(SHAPE, complex)
    blocked[CENTRE - half:CENTRE + half, CENTRE - half:CENTRE + half] = propagate(rod.astype(complex), z)
    return np.abs(1 - blocked) ** 2


def test_perfectly_round_pattern_scores_about_one():
    yy, xx = np.mgrid[-150:151, -150:151]
    rings = np.cos(np.hypot(xx, yy) / 4.0)          # perfect concentric rings
    assert roundness(rings, 150, 150) > 0.98
    # ...and its rotational average is (almost) itself
    patch = rings[70:231, 70:231]
    assert np.corrcoef(rotational_average(patch).ravel(), patch.ravel())[0, 1] > 0.98


def test_particle_is_a_particle():
    fl = fake_flattened(hologram(SHAPE, [Particle(CENTRE, CENTRE, 8e-3, 25e-6)]))
    d = describe(fl, [Detection(CENTRE, CENTRE, 100.0)])[0]
    assert d.roundness > 0.9
    assert d.is_particle


def test_fibre_is_not_a_particle():
    fl = fake_flattened(fibre(length_um=400, width_um=12, z=6e-3))
    d = describe(fl, [Detection(CENTRE, CENTRE, 100.0)])[0]
    assert not d.is_particle


def test_big_dark_fibre_is_rejected_by_dark_fraction():
    # Close to the sensor and thick, like the big curly fibres in frame 1
    fl = fake_flattened(fibre(length_um=800, width_um=120, z=1e-3))
    d = describe(fl, [Detection(CENTRE, CENTRE, 100.0)])[0]
    assert d.dark_fraction > 0.15
    assert not d.is_particle


def test_off_centre_point_is_much_less_round():
    fl = fake_flattened(hologram(SHAPE, [Particle(CENTRE, CENTRE, 8e-3, 25e-6)]))
    c = contrast_map(fl.flat)
    assert roundness(c, CENTRE + 40, CENTRE) < roundness(c, CENTRE, CENTRE) - 0.3


def test_detection_between_two_particles_becomes_two_particles():
    # The detector can land halfway between two similar rings, because the
    # pair looks the same rotated 180 degrees about their midpoint
    left, right = Particle(CENTRE - 55, CENTRE, 8e-3, 25e-6), Particle(CENTRE + 55, CENTRE, 10e-3, 22e-6)
    fl = fake_flattened(hologram(SHAPE, [left, right]))
    described = describe(fl, [Detection(CENTRE, CENTRE, 100.0)])   # one detection, in the middle
    particles = [d for d in described if d.is_particle]
    assert len(particles) == 2
    for p in (left, right):
        assert min(np.hypot(d.x - p.x, d.y - p.y) for d in particles) <= 3
