"""Load raw frames from the ATMOS holography camera.

Each .raw file is a headerless dump from the camera: 4024 x 3036 pixels,
one 12-bit brightness reading per pixel, stored as a 16-bit little-endian
integer (2 bytes per pixel).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WIDTH = 4024
HEIGHT = 3036
BYTES_PER_PIXEL = 2
MAX_12_BIT = 4095

# Example: frame_00000001_id_1_20260810T070642_484624135Z.raw
FILENAME_PATTERN = re.compile(
    r"frame_(?P<index>\d+)_id_(?P<camera_id>\d+)_(?P<time>\d{8}T\d{6})_(?P<nanos>\d{9})Z"
)


@dataclass(frozen=True)
class FrameInfo:
    index: int              # position in the recording session's file sequence
    camera_id: int          # frame ID reported by the camera
    captured_utc: datetime  # when the frame was taken


def parse_filename(path: str | Path) -> FrameInfo:
    """Read the frame number and capture time out of a raw file's name."""
    name = Path(path).name
    match = FILENAME_PATTERN.search(name)
    if match is None:
        raise ValueError(f"not an ATMOS frame filename: {name}")

    whole_seconds = datetime.strptime(match["time"], "%Y%m%dT%H%M%S")
    micros = int(match["nanos"]) // 1000  # datetime only goes down to microseconds
    captured = whole_seconds.replace(microsecond=micros, tzinfo=timezone.utc)

    return FrameInfo(
        index=int(match["index"]),
        camera_id=int(match["camera_id"]),
        captured_utc=captured,
    )


def load_raw(path: str | Path) -> np.ndarray:
    """Load one raw frame as a (3036, 4024) array of brightness values."""
    path = Path(path)

    # Check the size first: a wrong size means a different format or a
    # truncated file, and reshaping it would silently produce garbage.
    expected = WIDTH * HEIGHT * BYTES_PER_PIXEL
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path.name}: expected {expected} bytes, found {actual}")

    # "<u2" = little-endian, unsigned, 2 bytes per value
    image = np.fromfile(path, dtype="<u2").reshape(HEIGHT, WIDTH)

    # A 12-bit camera can't produce values above 4095, so anything higher
    # means the bytes were read in the wrong order.
    if image.max() > MAX_12_BIT:
        raise ValueError(f"{path.name}: values above the 12-bit range, byte order may be wrong")

    return image


if __name__ == "__main__":
    # Usage: python src/raw_io.py path/to/frame.raw [more frames...]
    for arg in sys.argv[1:]:
        info = parse_filename(arg)
        img = load_raw(arg)
        print(
            f"frame {info.camera_id} at {info.captured_utc.isoformat(timespec='milliseconds')}: "
            f"min={img.min()} median={int(np.median(img))} max={img.max()}"
        )
