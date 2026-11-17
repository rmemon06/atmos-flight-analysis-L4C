"""Tests for describe.py: particles pass, fibres don't."""

import numpy as np

from describe import describe
from detect import Detection, detect
from flatten import BLACK_LEVEL, flatten
from optics import PIXEL, Particle, hologram, propagate

SHAPE = (1400, 1400)
CENTRE = 700


def fake_flattened(light, seed=0, level=1000):
    rng = np.random.default_rng(seed)
    return flatten(rng.poisson(level * light).astype(np.float32) + BLACK_LEVEL)


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


def test_particle_is_a_particle():
    fl = fake_flattened(hologram(SHAPE, [Particle(CENTRE, CENTRE, 8e-3, 25e-6)]))
    found = [d for d in describe(fl, detect(fl)) if d.is_particle]
    assert len(found) == 1
    assert found[0].stretch < 1.5


def test_fibre_is_not_a_particle():
    # In focus, a fibre is a line: every dot found along it is stretched
    fl = fake_flattened(fibre(length_um=400, width_um=12, z=6e-3))
    described = describe(fl, detect(fl))
    assert described                                   # it is seen...
    assert not any(d.is_particle for d in described)   # ...but not counted as a particle


def test_big_dark_fibre_is_rejected_by_dark_fraction():
    # Close to the sensor and thick, like the big curly fibres in frame 1
    fl = fake_flattened(fibre(length_um=800, width_um=120, z=1e-3))
    d = describe(fl, [Detection(CENTRE, CENTRE, 3.5e-3, 100.0, 1.0)])[0]
    assert d.dark_fraction > 0.15
    assert not d.is_particle


def test_faint_particle_in_dim_light_is_still_a_particle():
    light = hologram(SHAPE, [Particle(CENTRE, CENTRE, 8e-3, 25e-6)])
    dim = fake_flattened(light, seed=1, level=20)       # 20 units of light, very noisy
    found = [d for d in describe(dim, detect(dim)) if d.is_particle]
    assert len(found) == 1
    assert np.hypot(found[0].x - CENTRE, found[0].y - CENTRE) <= 3


def test_pure_noise_in_dim_light_has_no_particles():
    for seed, light in enumerate((4, 12, 40)):
        rng = np.random.default_rng(seed)
        fl = flatten(rng.poisson(np.full((1500, 2000), float(light))).astype(np.float32) + BLACK_LEVEL)
        assert not any(d.is_particle for d in describe(fl, detect(fl)))
