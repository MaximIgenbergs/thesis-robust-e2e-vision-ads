from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json

from scripts.udacity.logging.data_collection_runs import write_frame_record

INPUT_DIR = Path("/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/genroads/pid_20251029-175400/raw_pd_logs")
OUTPUT_DIR = Path("/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/genroads/pid_20251029-175400")


def unwrap_action(raw) -> Tuple[float, float]:
    """Accepts [s, t] or [[s, t], ...]."""
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return float(raw[0]), float(raw[1])


def find_img_dir_for_record(record: Path) -> Path | None:
    """
    Example: udacity_road_swiggly_056_logs_2025_10_29_18_04_24.json -> udacity_road_swiggly_056___0_original
    """
    stem = record.stem  # e.g. udacity_road_swiggly_056_logs_2025_10_29_18_04_24

    # Strip '_logs_...' suffix if present
    logs_idx = stem.find("_logs_")
    if logs_idx != -1:
        base_stem = stem[:logs_idx]  # e.g. udacity_road_swiggly_056
    else:
        base_stem = stem

    img_dir = record.parent / f"{base_stem}___0_original"
    if img_dir.exists():
        return img_dir

    fallback = record.parent / f"{stem}___0_original"
    if fallback.exists():
        return fallback

    print(f"[scripts:logging:conversion] image dir not found for record: {record.name} (tried '{img_dir}' and '{fallback}')")
    return None


def convert_pd_outputs(input_dir: Path, output_dir: Path) -> None:
    """
    Converts PerturbationDrive ScenarioOutcomeWriter outputs into the required data format:
    output_dir/
      image_000001.jpg
      record_000001.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Path] = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".json")
    if not records:
        print(f"[scripts:logging:conversion] no JSON logs in {input_dir}")
        return

    start_idx = 1
    run_uid_counter = 0

    for record in records:
        img_dir = find_img_dir_for_record(record)
        if img_dir is None:
            continue

        try:
            data = json.loads(record.read_text())
        except json.JSONDecodeError:
            print(f"[scripts:logging:conversion] json decode error: {record}")
            continue

        entries = [data] if isinstance(data, dict) else list(data)

        for entry in entries:
            frames = entry.get("frames", [])
            pid_actions = entry.get("pid_actions", [])
            if len(frames) != len(pid_actions):
                n = min(len(frames), len(pid_actions))
                frames = frames[:n]
                pid_actions = pid_actions[:n]

            track_id = run_uid_counter
            run_uid_counter += 1

            dropped = 0
            for i, (img_name, action) in enumerate(zip(frames, pid_actions)):
                try:
                    steer, throttle = unwrap_action(action)
                except Exception as e:
                    print(f"[scripts:logging:conversion] drop frame (action parse): {e}")
                    dropped += 1
                    continue

                src_img = img_dir / f"{img_name}.jpg"
                if not src_img.exists():
                    print(f"[scripts:logging:conversion] missing image: {src_img}")
                    dropped += 1
                    continue

                dst_img = output_dir / f"image_{start_idx:06d}.jpg"
                dst_js = output_dir / f"record_{start_idx:06d}.json"

                dst_img.write_bytes(src_img.read_bytes())
                write_frame_record(dst_js, steer, throttle, track_id, i)
                start_idx += 1

            print(f"[scripts:logging:conversion:ok] {record.name}: track_id={track_id} dropped={dropped}")

    print(f"[scripts:logging:conversion] wrote data at {output_dir}")


def main() -> None:
    print(f"[scripts:logging:conversion] input_dir  = {INPUT_DIR}")
    print(f"[scripts:logging:conversion] output_dir = {OUTPUT_DIR}")
    convert_pd_outputs(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
