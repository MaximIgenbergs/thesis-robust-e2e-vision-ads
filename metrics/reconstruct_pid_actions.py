from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Target speed used during GenRoads generalization
TARGET_SPEED = 2.0


def pid_speed21(
    road_error: float,
    angle_error: float,
    speed_error: float,
    prev_road_error: float,
    prev_angle_error: float,
    prev_speed_error: float,
    total_road_error: float,
    total_angle_error: float,
    total_speed_error: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    PID controller (pid_speed21 from udacity_simulator.py)
    Returns: (throttle, steering, prev_road_error, prev_angle_error, prev_speed_error,
              total_road_error, total_angle_error, total_speed_error)
    """
    road_error = -road_error
    if abs(road_error) > 1:
        Kp_road = 0.6
    else:
        Kp_road = 0.45

    Ki_road = 0.0
    Kd_road = 0.000

    if angle_error < 25:
        Kp_angle = 0.003
        Kd_angle = 0.000
    else:
        Kp_angle = 0.001
        Kd_angle = 0.000

    Ki_angle = 0.0

    Kp_speed = 0.1
    Ki_speed = 0.0
    Kd_speed = 0.0

    P_angle = Kp_angle * angle_error
    I_angle = Ki_angle * total_angle_error
    D_angle = Kd_angle * (angle_error - prev_angle_error)

    P_road = Kp_road * road_error
    I_road = Ki_road * total_road_error
    D_road = Kd_road * (road_error - prev_road_error)

    steering = P_angle + I_angle + D_angle
    steering = P_road + I_road + D_road + steering

    steering = max(-1, min(1, steering))

    P_speed = Kp_speed * speed_error
    I_speed = Ki_speed * total_speed_error
    D_speed = Kd_speed * (speed_error - prev_speed_error)
    throttle = P_speed + I_speed + D_speed
    throttle -= 0.6 * abs(road_error)
    throttle = max(0.1, min(0.8, throttle))

    prev_road_error = road_error
    prev_angle_error = angle_error
    prev_speed_error = speed_error
    total_road_error += road_error
    total_angle_error += angle_error
    total_speed_error += speed_error

    return (
        throttle,
        steering,
        prev_road_error,
        prev_angle_error,
        prev_speed_error,
        total_road_error,
        total_angle_error,
        total_speed_error,
    )


def safe_float(val: Any, default: float = 0.0) -> float:
    """Convert value to float, handling strings and None."""
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.strip()
        return float(val)
    except (ValueError, AttributeError):
        return default


def reconstruct_pid_actions(pd_log: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct PID actions from pid_state in pd_log.json.
    Returns a modified copy with pid_actions added to each step.
    """
    steps = pd_log.get("steps")
    if not isinstance(steps, list) or not steps:
        print("Warning: No steps found in pd_log")
        return pd_log

    # Initialize PID state variables
    prev_road_error = 0.0
    prev_angle_error = 0.0
    prev_speed_error = 0.0
    total_road_error = 0.0
    total_angle_error = 0.0
    total_speed_error = 0.0

    # Create a deep copy to avoid modifying original
    result = json.loads(json.dumps(pd_log))

    for i, step in enumerate(result["steps"]):
        if not isinstance(step, dict):
            continue

        # Extract pid_state
        pid_state = step.get("pid_state", {})
        if not isinstance(pid_state, dict):
            pid_state = {}

        # Extract current errors from pid_state
        road_error = safe_float(pid_state.get("cte", 0.0))
        angle_error = safe_float(pid_state.get("angle", 0.0))
        current_speed = safe_float(pid_state.get("speed", 0.0))
        speed_error = TARGET_SPEED - current_speed

        # Compute PID actions
        (
            throttle,
            steering,
            prev_road_error,
            prev_angle_error,
            prev_speed_error,
            total_road_error,
            total_angle_error,
            total_speed_error,
        ) = pid_speed21(
            road_error,
            angle_error,
            speed_error,
            prev_road_error,
            prev_angle_error,
            prev_speed_error,
            total_road_error,
            total_angle_error,
            total_speed_error,
        )

        # Add reconstructed PID actions to step
        step["pid_action"] = {
            "steer": steering,
            "throttle": throttle,
        }

        # Also add as separate fields for easier extraction
        step["pid_steer"] = steering
        step["pid_throttle"] = throttle

    return result


def process_run_directory(run_dir: Path, dry_run: bool = False) -> None:
    """
    Process all pd_log.json files in a run directory and create new_log.json
    with reconstructed PID actions.
    """
    if not run_dir.exists():
        print(f"Error: Directory does not exist: {run_dir}")
        return

    # Find all pd_log.json files
    pd_logs = list(run_dir.rglob("pd_log.json"))

    if not pd_logs:
        print(f"No pd_log.json files found in {run_dir}")
        return

    print(f"Found {len(pd_logs)} pd_log.json files in {run_dir}")

    processed = 0
    failed = 0

    for pd_log_path in pd_logs:
        try:
            print(f"Processing: {pd_log_path.relative_to(run_dir)}")

            # Read original pd_log
            with pd_log_path.open("r", encoding="utf-8") as f:
                pd_log = json.load(f)

            # Reconstruct PID actions
            new_log = reconstruct_pid_actions(pd_log)

            # Write to new_log.json in same directory
            new_log_path = pd_log_path.parent / "new_log.json"

            if dry_run:
                print(f"  [DRY RUN] Would write to: {new_log_path}")
                # Show sample of first step's new fields
                if new_log.get("steps") and len(new_log["steps"]) > 0:
                    first_step = new_log["steps"][0]
                    if "pid_action" in first_step:
                        print(f"  Sample pid_action: {first_step['pid_action']}")
            else:
                with new_log_path.open("w", encoding="utf-8") as f:
                    json.dump(new_log, f, indent=2)
                print(f"  ✓ Written to: {new_log_path}")

            processed += 1

        except Exception as e:
            print(f"  ✗ Error processing {pd_log_path}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Summary for {run_dir.name}:")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pd_logs)}")
    print(f"{'='*60}\n")


def main() -> None:
    """
    Process all GenRoads generalization run directories.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct PID actions from pd_log.json files"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., /path/to/runs/genroads/generalization/dave2_20251217_225857)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually writing files",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()

    print(f"{'='*60}")
    print(f"PID Action Reconstruction")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Target speed: {TARGET_SPEED}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    process_run_directory(run_dir, dry_run=args.dry_run)

    print("\n[DONE] PID action reconstruction complete")
    print(
        "Next step: Update io_udacity.py to read from 'new_log.json' or 'pd_log.json'"
    )


if __name__ == "__main__":
    main()