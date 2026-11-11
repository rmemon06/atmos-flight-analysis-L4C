"""Tests for flatten.py, using fake frames where we know the right answer."""

import numpy as np
import pytest

from flatten import flatten, pixel_noise, usable_mask

H, W = 1024, 1536
BRIGHT = (350, 800)   # (x, y) near the beam's centre
DIM = (1400, 150)     # (x, y) out in the dim part of the beam


def fake_beam() -> np.ndarray:
    """A smooth beam, bright in the lower left like the real one."""
    yy, xx = np.mgrid[0:H, 0:W]
    return 80 + 1500 * np.exp(-(((xx - 300) / 900) ** 2 + ((yy - 900) / 800) ** 2))


def particle(x: int, y: int, depth: float = 0.10, size: float = 12) -> np.ndarray:
    """A fake particle that dims the light by `depth` (10%) at its centre."""
    yy, xx = np.mgrid[0:H, 0:W]
    return -depth * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * size ** 2))


def test_empty_frame_flattens_to_one():
    result = flatten(fake_beam())
    inside = result.flat[result.usable]
    assert abs(inside.mean() - 1) < 0.01
    assert np.abs(inside - 1).max() < 0.03


def test_same_particle_gives_same_signal_in_bright_and_dim_areas():
    beam = fake_beam()
    # Check the setup really is bright vs dim
    assert beam[BRIGHT[1], BRIGHT[0]] > 5 * beam[DIM[1], DIM[0]]

    raw = beam * (1 + particle(*BRIGHT) + particle(*DIM))
    flat = flatten(raw).flat
    assert 1 - flat[BRIGHT[1], BRIGHT[0]] == pytest.approx(0.10, abs=0.01)
    assert 1 - flat[DIM[1], DIM[0]] == pytest.approx(0.10, abs=0.01)


def test_black_level_has_to_be_removed():
    offset = 20.0
    raw = fake_beam() * (1 + particle(*DIM)) + offset

    ignored = 1 - flatten(raw).flat[DIM[1], DIM[0]]
    removed = 1 - flatten(raw, black_level=offset).flat[DIM[1], DIM[0]]

    assert removed == pytest.approx(0.10, abs=0.01)
    assert ignored < removed   # forgetting the offset makes the particle look weaker


def test_mask_drops_dim_areas_and_edges():
    beam = np.full((H, W), 1000.0)
    beam[:200] = 30   # a strip too dim to use
    usable = usable_mask(beam, min_beam=60, edge=80)

    assert not usable[:200].any()
    assert not usable[-80:].any()
    assert not usable[:, :80].any()
    assert not usable[:, -80:].any()
    assert usable[200:-80, 80:-80].all()


def test_pixel_noise_measures_known_noise():
    rng = np.random.default_rng(0)
    beam = fake_beam()
    raw = beam * (1 + rng.normal(0, 0.03, beam.shape))   # 3% noise everywhere
    result = flatten(raw)
    noise = pixel_noise(result.flat, result.beam)
    assert np.median(noise[:, 1]) == pytest.approx(0.03, rel=0.1)
