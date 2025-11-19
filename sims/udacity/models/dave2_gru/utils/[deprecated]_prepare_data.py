#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare data for DAVE2-GRU training.

This script mirrors PerturbationDrive's example preprocessor (copy images, write per-frame JSON)
but augments labels with sequence metadata required for GRU training:

- meta/track_id : unique per contiguous run (prevents sequences crossing run boundaries)
- meta/topo_id  : stable topology id (0..37 for generated; constant for jungle)
- meta/frame    : 0-based frame index within the run

Credits:
- Based on the structure of PD's prepare script.
- This version only adds the extra metadata and careful track/run separation.

Usage:
  python -m sims.udacity.training.dave2_gru.prepare_data \
    --input-jsons "../../data/udacity/generated_roads/pid_2025-10-18_21-23-46/train_nominal" \
    --output "../../data/udacity/generated_roads/train_dataset_gru" \
    --map generated

For jungle:
  python -m sims.udacity.training.dave2_gru.prepare_data \
    --input-jsons "../../data/udacity/jungle/train_nominal" \
    --output "../../data/udacity/jungle/train_dataset_gru" \
    --map jungle --jungle-topo-id 100
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from typing import List, Tuple, Optional


def _list_jsons(folder: str) -> List[str]:
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")]
    )


TOPO_PATTERNS = [
    re.compile(r"(?:track|road|map|generated)[-_]?(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)$"),
]


def _infer_topo_id_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    for pat in TOPO_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def _coerce_action(raw) -> Tuple[float, float]:
    """
    PD logs often look like: pid_actions = [[[steer, throttle], ...], ...]
    Their old script does `action = action[0]`. We accept both shapes.
    """
    try:
        # action may be [ [s, t], ...] or [s, t]
        if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        steer = float(raw[0])
        throttle = float(raw[1])
        return steer, throttle
    except Exception as e:
        raise ValueError(f"Cannot parse action {raw!r}: {e}")


def _prepare_one_json(
    input_json_path: str,
    output_folder: str,
    map_kind: str,
    topo_counter: int,
    jungle_topo_id: int,
    start_idx: int,
    run_uid_counter: int,
) -> Tuple[int, int]:
    """
    Processes a single PD session JSON file and writes per-frame JSON+image outputs.

    Returns:
      (next_start_idx, next_run_uid_counter)
    """
    base_dir = os.path.dirname(input_json_path)
    base_name = os.path.basename(input_json_path).split("_logs")[0]
    image_folder_name = f"{base_name}___0_original"
    image_folder_path = os.path.join(base_dir, image_folder_name)

    if not os.path.exists(image_folder_path):
        print(f"[warn] image folder not found: {image_folder_path}")
        return start_idx, run_uid_counter

    os.makedirs(output_folder, exist_ok=True)

    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Decide topo_id for this file
    if map_kind == "generated":
        topo_id = _infer_topo_id_from_filename(input_json_path)
        if topo_id is None:
            topo_id = topo_counter  # fall back to running order
    else:  # jungle
        topo_id = jungle_topo_id

    # Each 'entry' is treated as a distinct contiguous run -> unique track_id
    # track_id uniqueness guarantees GRU windows won't cross run boundaries.
    entries = data if isinstance(data, list) else [data]

    for entry_idx, entry in enumerate(entries):
        frames = entry.get("frames", [])
        pid_actions = entry.get("pid_actions", [])
        xte = entry.get("xte", [])

        if len(frames) != len(pid_actions):
            print(f"[warn] frames vs pid_actions length mismatch: {len(frames)} vs {len(pid_actions)}")
            # continue but truncate to min length
            n = min(len(frames), len(pid_actions))
            frames, pid_actions = frames[:n], pid_actions[:n]

        # Assign a fresh run-unique track_id
        track_id = run_uid_counter
        run_uid_counter += 1

        dropped = 0
        for frame_i, (frame_name, action) in enumerate(zip(frames, pid_actions)):
            try:
                steer, throttle = _coerce_action(action)
            except Exception as e:
                print(f"[warn] skipping frame due to action parse error: {e}")
                dropped += 1
                continue

            # JSON output
            simple_json = {
                "user/angle": f"{steer}",
                "user/throttle": f"{throttle}",
                "meta/track_id": int(track_id),
                "meta/topo_id": int(topo_id),
                "meta/frame": int(frame_i),
            }

            json_out = os.path.join(output_folder, f"record_{start_idx:06d}.json")
            with open(json_out, "w") as out_file:
                json.dump(simple_json, out_file, indent=4)

            # Image copy
            img_src = os.path.join(image_folder_path, f"{frame_name}.jpg")
            img_dst = os.path.join(output_folder, f"image_{start_idx:06d}.jpg")
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)
            else:
                print(f"[warn] image not found: {img_src}")
                # remove the json we just wrote to keep 1:1 pairs
                try:
                    os.remove(json_out)
                except OSError:
                    pass
                dropped += 1
                continue

            start_idx += 1

        print(
            f"[ok] {base_name} entry#{entry_idx}: "
            f"topo_id={topo_id} track_id={track_id} "
            f"wrote up to index {start_idx-1} (dropped {dropped})"
        )

    return start_idx, run_uid_counter


def main():
    ap = argparse.ArgumentParser(description="Prepare PD data with GRU-friendly metadata")
    ap.add_argument("--input-jsons", required=True, help="Folder containing PD session JSON logs")
    ap.add_argument("--output", required=True, help="Output folder for per-frame pairs")
    ap.add_argument("--map", choices=["generated", "jungle"], required=True, help="Map kind")
    ap.add_argument("--start-idx", type=int, default=1, help="Starting global index for filenames")
    ap.add_argument("--jungle-topo-id", type=int, default=100, help="Constant topo id for jungle")
    args = ap.parse_args()

    json_files = _list_jsons(args.input_jsons)
    if not json_files:
        raise SystemExit(f"No JSON files in {args.input_jsons}")

    start_idx = args.start_idx
    topo_counter = 0  # used if we cannot infer topo id from filename
    run_uid_counter = 0  # unique track_id across all contiguous runs

    os.makedirs(args.output, exist_ok=True)

    for jf in json_files:
        start_idx, run_uid_counter = _prepare_one_json(
            input_json_path=jf,
            output_folder=args.output,
            map_kind=args.map,
            topo_counter=topo_counter,
            jungle_topo_id=args.jungle_topo_id,
            start_idx=start_idx,
            run_uid_counter=run_uid_counter,
        )
        topo_counter += 1

    print(f"[done] processed all files -> {args.output}")


if __name__ == "__main__":
    main()
