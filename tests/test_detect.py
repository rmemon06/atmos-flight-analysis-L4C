"""Tests for detect.py, using fake holograms with particles at known places and depths."""

import numpy as np
import pytest

from describe import describe
from detect import detect, dot_score
from flatten import BLACK_LEVEL, flatten
from optics import Particle, Refocuser, hologram

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


def fake_frame(particles, shape=SHAPE, seed=0, beam=None):
    """Like a real camera frame: beam x hologram, photon (Poisson) noise, plus the camera's zero."""
    rng = np.random.default_rng(seed)
    light = (fake_beam(shape) if beam is None else beam) * hologram(shape, particles)
    return rng.poisson(light).astype(np.float32) + BLACK_LEVEL


def particles(fl):
    """Detect, then keep only what describe() decides are particles."""
    return [d for d in describe(fl, detect(fl)) if d.is_particle]


@pytest.fixture(scope="module")
def planted_flattened():
    # Built once and shared by the tests below, because making it takes a few seconds
    return flatten(fake_frame(PLANTED))


@pytest.fixture(scope="module")
def planted_detections(planted_flattened):
    return detect(planted_flattened)


def nearest(found, x, y):
    return min(found, key=lambda d: np.hypot(d.x - x, d.y - y))


def test_finds_every_planted_particle_in_the_right_place_and_depth(planted_detections):
    for p in PLANTED:
        d = nearest(planted_detections, p.x, p.y)
        assert np.hypot(d.x - p.x, d.y - p.y) <= 2, f"missed particle at z={p.z * 1e3:.0f} mm"
        # Its sharpest depth is within one step of the depths searched (0.5 mm
        # steps out to 9 mm, then 2 mm). size.py measures distance precisely
        assert abs(d.z - p.z) <= (0.6e-3 if p.z < 9e-3 else 2.1e-3),             f"particle at {p.z * 1e3:.0f} mm found at {d.z * 1e3:.1f} mm"


def test_no_false_particles(planted_flattened):
    found = particles(planted_flattened)
    assert len(found) == len(PLANTED)
    for d in found:
        assert min(np.hypot(d.x - p.x, d.y - p.y) for p in PLANTED) <= 3


def test_pure_noise_has_no_detections():
    assert detect(flatten(fake_frame([]))) == []


def test_noise_model_is_right():
    # With only noise, the dot score's spread should be about 1 by definition.
    # If this drifts, the noise measurement in block_noise is wrong
    fl = flatten(fake_frame([]))
    score, _, _ = dot_score(Refocuser(np.where(fl.usable, fl.flat, 1.0)).brightness(5e-3), fl.usable)
    s = score[fl.usable]
    spread = 1.4826 * np.median(np.abs(s - np.median(s)))
    assert spread == pytest.approx(1.0, abs=0.15)


def test_far_particle_is_found_once():
    shape = (2200, 2200)
    found = particles(flatten(fake_frame([Particle(1100, 1100, 20e-3, 28e-6)], shape)))
    assert len(found) == 1
    assert np.hypot(found[0].x - 1100, found[0].y - 1100) <= 2


def test_very_far_small_particle_is_found_once():
    # Its rings are 500+ px wide and very faint
    shape = (2200, 2200)
    found = particles(flatten(fake_frame([Particle(1100, 1100, 24e-3, 15e-6)], shape)))
    assert len(found) == 1


def test_faint_particle_next_to_a_bright_one():
    # Why detection refocuses: the first version (energy hills) missed faint
    # particles near bright ones, because the bright one's rings drowned them.
    # Here an 8 um particle sits 90 px from a 40 um one, both 4.4 mm away, in dim light
    shape = (1600, 1600)
    bright, faint = Particle(760, 800, 4.4e-3, 40e-6), Particle(850, 800, 4.4e-3, 8e-6)
    found = particles(flatten(fake_frame([bright, faint], shape, beam=np.full(shape, 150.0))))
    assert len(found) == 2
    for p in (bright, faint):
        d = nearest(found, p.x, p.y)
        assert np.hypot(d.x - p.x, d.y - p.y) <= 3


def test_crowded_frame_benchmark():
    # 60 particles of mixed strength, some only 70 px apart. This is a measured
    # benchmark, not a pass/fail on perfection. The energy-hill version found 44
    # of 60; refocusing finds 57 with 1 false one (the misses are small particles
    # between the depths searched). Raise the bar here whenever detection improves.
    rng = np.random.default_rng(5)
    planted = []
    while len(planted) < 60:
        x, y = int(rng.uniform(520, 2380)), int(rng.uniform(520, 1680))
        if all(np.hypot(x - p.x, y - p.y) > 70 for p in planted):
            planted.append(Particle(x, y, rng.uniform(3, 16) * 1e-3, rng.uniform(10, 40) * 1e-6))
    light = fake_beam(SHAPE) * hologram(SHAPE, planted)
    found = particles(flatten(rng.poisson(light).astype(np.float32) + BLACK_LEVEL))

    hits = sum(min(np.hypot(d.x - p.x, d.y - p.y) for d in found) <= 5 for p in planted)
    wrong = sum(min(np.hypot(d.x - p.x, d.y - p.y) for p in planted) > 8 for d in found)
    print(f"crowded benchmark: {hits} of 60 found, {wrong} wrong")
    assert hits >= 56
    assert wrong <= 3
