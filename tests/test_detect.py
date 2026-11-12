"""Tests for detect.py, using fake holograms with particles at known places."""

import numpy as np
import pytest

from detect import detect, strength_map
from flatten import BLACK_LEVEL, flatten
from optics import Particle, hologram

SHAPE = (2200, 2900)

# Eight particles at different distances (4-24 mm) and sizes (15-28 um),
# spread across bright and dim parts of the beam
PLANTED = [
    Particle(560, 600, 4e-3, 18e-6), Particle(1150, 600, 12e-3, 25e-6),
    Particle(1750, 600, 20e-3, 28e-6), Particle(2340, 600, 8e-3, 22e-6),
    Particle(560, 1600, 16e-3, 20e-6), Particle(1150, 1600, 6e-3, 15e-6),
    Particle(1750, 1600, 24e-3, 24e-6), Particle(2340, 1600, 14e-3, 18e-6),
]


def fake_beam(shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return 80 + 1400 * np.exp(-(((xx - 300) / 2000) ** 2 + ((yy - 2100) / 1700) ** 2))


def fake_frame(particles, shape=SHAPE, seed=0):
    """Like a real camera frame: beam x hologram, photon (Poisson) noise, plus the camera's zero."""
    rng = np.random.default_rng(seed)
    light = fake_beam(shape) * hologram(shape, particles)
    return rng.poisson(light).astype(np.float32) + BLACK_LEVEL


@pytest.fixture(scope="module")
def planted_detections():
    # Built once and shared by the tests below, because making it takes a few seconds
    return detect(flatten(fake_frame(PLANTED)))


def nearest(found, x, y):
    return min(np.hypot(d.x - x, d.y - y) for d in found)


def test_finds_every_planted_particle_in_the_right_place(planted_detections):
    for p in PLANTED:
        assert nearest(planted_detections, p.x, p.y) <= 2, f"missed particle at z={p.z * 1e3:.0f} mm"


def test_no_false_alarms(planted_detections):
    for d in planted_detections:
        assert min(np.hypot(d.x - p.x, d.y - p.y) for p in PLANTED) <= 15


def test_pure_noise_has_no_detections():
    assert detect(flatten(fake_frame([]))) == []


def test_noise_model_is_right():
    # With only noise, "strength" should be about 1 by definition.
    # If this drifts, the maths in noise_energy is wrong.
    fl = flatten(fake_frame([]))
    assert np.median(strength_map(fl)[fl.usable]) == pytest.approx(1.0, abs=0.25)


def test_far_particle_centre_is_found_and_counted_once():
    # A far particle's energy hill has a flat top, so its highest point can be
    # 30+ px off-centre, and it can have two hilltops. Refining fixes both.
    shape = (2200, 2200)
    found = detect(flatten(fake_frame([Particle(1100, 1100, 20e-3, 28e-6)], shape)))
    assert len(found) == 1
    assert nearest(found, 1100, 1100) <= 2


@pytest.mark.xfail(reason="known limitation: a very far, very small particle has rings "
                          "500+ px wide, and its outer rings form hills of their own")
def test_very_far_small_particle_counted_once():
    shape = (2200, 2200)
    found = detect(flatten(fake_frame([Particle(1100, 1100, 24e-3, 15e-6)], shape)))
    assert len(found) == 1


def test_crowded_frame_benchmark():
    # 60 particles of mixed strength, some only 70 px apart. This is a measured
    # benchmark, not a pass/fail on perfection. The first version found 33 of
    # 60; this one finds 44. Rings within ~100 px of a much stronger neighbour
    # are the ones it still misses. Raise the bar here whenever detection improves.
    rng = np.random.default_rng(5)
    particles = []
    while len(particles) < 60:
        x, y = int(rng.uniform(520, 2380)), int(rng.uniform(520, 1680))
        if all(np.hypot(x - p.x, y - p.y) > 70 for p in particles):
            particles.append(Particle(x, y, rng.uniform(3, 16) * 1e-3, rng.uniform(10, 40) * 1e-6))
    light = fake_beam(SHAPE) * hologram(SHAPE, particles)
    found = detect(flatten(rng.poisson(light).astype(np.float32) + BLACK_LEVEL))

    hits = sum(nearest(found, p.x, p.y) <= 5 for p in particles)
    wrong = sum(min(np.hypot(d.x - p.x, d.y - p.y) for p in particles) > 8 for d in found)
    assert hits >= 42
    assert wrong <= 3
