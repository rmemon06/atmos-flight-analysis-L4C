"""Run steps 1-4 on every frame in a folder and save the results to CSV files.

Usage:
    python src/run.py data/flight
    python src/run.py data/flight --workers 4 --out results

Writes:
    results/frames/frame_XXX.csv   one file per frame, written as each finishes
    results/detections.csv         every detection from every frame, in one table
    results/summary.csv            one row per frame: time, laser on/off, counts

Each frame's file is written as soon as it finishes, so if the run stops
(crash, laptop closed), running the same command again skips the frames
that are already done and carries on from there.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from flatten import BLACK_LEVEL, flatten
from raw_io import load_raw, parse_filename

LASER_ON_MIN = 20   # a frame counts as lit if its median reading is this far above the
                    # camera's zero. Lit flight frames sit 120-210 above; dark ones 0

DETECTION_COLUMNS = ["frame", "time_utc", "x", "y", "strength", "roundness",
                     "dark_fraction", "is_particle"]
SUMMARY_COLUMNS = ["frame", "time_utc", "laser_on", "light_median", "light_max",
                   "usable_pct", "detections", "particles", "seconds"]


def is_laser_on(raw: np.ndarray) -> bool:
    return float(np.median(raw)) > BLACK_LEVEL + LASER_ON_MIN


def process_frame(raw_path: Path, out_dir: Path) -> dict:
    """Steps 1-4 on one frame. Writes its CSV and returns its summary row."""
    # Imported here so each worker process loads them itself
    import cv2
    from describe import describe
    from detect import detect

    cv2.setNumThreads(1)   # we already run one frame per CPU core; don't oversubscribe
    start = time.time()
    info = parse_filename(raw_path)
    raw = load_raw(raw_path)
    when = info.captured_utc.isoformat(timespec="milliseconds")

    summary = {"frame": info.camera_id, "time_utc": when, "laser_on": is_laser_on(raw)}
    rows = []
    if summary["laser_on"]:
        fl = flatten(raw)
        described = describe(fl, detect(fl))
        rows = [{"frame": info.camera_id, "time_utc": when, "x": d.x, "y": d.y,
                 "strength": round(d.strength, 2), "roundness": round(d.roundness, 4),
                 "dark_fraction": round(d.dark_fraction, 4), "is_particle": d.is_particle}
                for d in described]
        summary.update(light_median=round(float(np.median(fl.beam)), 1),
                       light_max=round(float(fl.beam.max()), 1),
                       usable_pct=round(100 * float(fl.usable.mean()), 1),
                       detections=len(described),
                       particles=sum(d.is_particle for d in described))
    else:
        summary.update(light_median=0, light_max=0, usable_pct=0, detections=0, particles=0)
    summary["seconds"] = round(time.time() - start, 1)

    # Write to a temporary name first, then rename: a half-written file from a
    # crash can never be mistaken for a finished frame
    path = frame_file(out_dir, info.camera_id)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        f.write("# " + ",".join(f"{k}={summary[k]}" for k in SUMMARY_COLUMNS) + "\n")
        writer = csv.DictWriter(f, fieldnames=DETECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return summary


def frame_file(out_dir: Path, camera_id: int) -> Path:
    return out_dir / "frames" / f"frame_{camera_id:03d}.csv"


def read_frame_file(path: Path) -> tuple[dict, list[dict]]:
    """A frame's summary (from its first line) and its detection rows."""
    with path.open(newline="") as f:
        first = f.readline()[2:].strip()
        summary = dict(item.split("=", 1) for item in first.split(","))
        rows = list(csv.DictReader(f))
    return summary, rows


def combine(out_dir: Path) -> tuple[int, int]:
    """Join every frame's file into detections.csv and summary.csv."""
    summaries, detections = [], []
    for path in sorted((out_dir / "frames").glob("frame_*.csv")):
        summary, rows = read_frame_file(path)
        summaries.append(summary)
        detections.extend(rows)
    for name, columns, table in (("summary.csv", SUMMARY_COLUMNS, summaries),
                                 ("detections.csv", DETECTION_COLUMNS, detections)):
        with (out_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
    return len(summaries), len(detections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ATMOS pipeline on a folder of raw frames")
    parser.add_argument("folder", type=Path, help="folder containing .raw frames (searched recursively)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="where to write results")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                        help="frames processed at the same time (default: up to 4)")
    args = parser.parse_args()

    raws = sorted(args.folder.rglob("*.raw"), key=lambda p: parse_filename(p).camera_id)
    if not raws:
        sys.exit(f"no .raw files found in {args.folder}")
    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    to_do = [p for p in raws if not frame_file(args.out, parse_filename(p).camera_id).exists()]
    print(f"{len(raws)} frames found, {len(raws) - len(to_do)} already done, "
          f"{len(to_do)} to process with {args.workers} workers")

    start = time.time()
    # "spawn" starts each worker fresh. It's how Windows always does it, so using
    # it everywhere means the script behaves the same on every computer
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(process_frame, p, args.out): p for p in to_do}
        for done, job in enumerate(as_completed(jobs), start=1):
            try:
                s = job.result()
            except Exception as error:   # one bad frame shouldn't stop the whole run
                print(f"  FAILED {jobs[job].name}: {error}")
                continue
            elapsed = time.time() - start
            left = elapsed / done * (len(to_do) - done)
            state = f"{s['particles']:3d} particles" if s["laser_on"] else "laser off, skipped"
            print(f"  [{done}/{len(to_do)}] frame {s['frame']:3d}: {state} "
                  f"({s['seconds']}s) - about {left / 60:.0f} min left")

    frames, rows = combine(args.out)
    print(f"done: {frames} frames, {rows} detections -> {args.out / 'detections.csv'}, "
          f"{args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
