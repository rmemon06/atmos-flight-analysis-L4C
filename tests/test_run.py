"""Tests for run.py: dark frames are skipped, results are saved and combined."""

import numpy as np

from flatten import BLACK_LEVEL
from raw_io import HEIGHT, WIDTH
from run import combine, frame_file, is_laser_on, process_frame, read_frame_file


def write_raw(folder, camera_id, value):
    """A fake raw file with a valid name, every pixel set to `value`."""
    name = f"frame_{camera_id:08d}_id_{camera_id}_20260810T070642_000000000Z.raw"
    path = folder / name
    np.full((HEIGHT, WIDTH), value, dtype="<u2").tofile(path)
    return path


def test_laser_on_check():
    assert not is_laser_on(np.full((10, 10), BLACK_LEVEL))         # camera's zero only
    assert is_laser_on(np.full((10, 10), BLACK_LEVEL + 150))       # like a lit flight frame


def test_dark_frame_is_skipped_and_saved(tmp_path):
    (tmp_path / "out" / "frames").mkdir(parents=True)
    raw = write_raw(tmp_path, 133, int(BLACK_LEVEL))
    summary = process_frame(raw, tmp_path / "out")
    assert summary["laser_on"] is False
    assert summary["particles"] == 0

    saved, rows = read_frame_file(frame_file(tmp_path / "out", 133))
    assert saved["laser_on"] == "False"
    assert rows == []


def test_combine_joins_every_frame(tmp_path):
    out = tmp_path / "out"
    (out / "frames").mkdir(parents=True)
    for camera_id in (98, 133):
        process_frame(write_raw(tmp_path, camera_id, int(BLACK_LEVEL)), out)
    frames, detections = combine(out)
    assert frames == 2
    assert detections == 0
    assert (out / "summary.csv").read_text().count("\n") == 3   # header + 2 frames
