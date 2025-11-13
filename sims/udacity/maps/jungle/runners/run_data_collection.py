# sims/udacity/maps/jungle/runners/run_data_collection.py
"""
Collect nominal training data on the Jungle map using a PID controller.

- Output dir: paths.DATA_DIR / pid_YYYYMMDD-HHMMSS
- Stops after reaching TARGET_SECTOR_SPAN or MAX_STEPS.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None # add project root to path

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent import PIDUdacityAgent_Angle
from sims.udacity.maps.configs.run import HOST, PORT
from sims.udacity.maps.jungle.configs import paths, run
from sims.udacity.maps.jungle.configs.data_collection import TARGET_SECTOR_SPAN, MAX_STEPS, TARGET_SPEED
from sims.udacity.logging.data_collection import DataRunLogger, make_run_dir, save_image, write_frame_record

def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()

def _ensure_jungle(env: UdacityGym, weather: str = "sunny", daytime: str = "day", timeout_s: float = 30.0) -> None:
    env.reset(track="jungle", weather=weather, daytime=daytime)
    t0 = time.perf_counter()
    last_emit = 0.0
    obs = env.observe()
    while True:
        if obs and obs.input_image is not None:
            return
        now = time.perf_counter()
        if now - last_emit > 1.0:
            try:
                env.simulator.sim_executor.send_track("jungle", weather, daytime)
            except Exception:
                pass
            last_emit = now
        if now - t0 > timeout_s:
            raise RuntimeError("Failed to enter Jungle map before timeout.")
        time.sleep(0.2)
        obs = env.observe()

SectorState = Optional[Tuple[int, int, int]]  # (initial, last, progress)

def update_sector_progress(state: SectorState, sector: int, target_span: int = 140, max_backwrap: int = 200) -> Tuple[bool, SectorState]:
    """Return (should_stop, new_state). Counts only small forward jumps; stops on wrap-around or when progress >= target."""
    s = int(sector)
    if state is None:
        return False, (s, s, 0)
    initial, last, progress = state
    delta = s - last
    if 1 <= delta <= 5:
        progress += delta
        last = s
    elif delta == 0:
        pass
    else:
        if delta < -max_backwrap:
            return True, (initial, last, progress)
        last = s
    return (progress >= int(target_span)), (initial, last, progress)


def main() -> None:
    sim_app = _abs(getattr(paths, "SIM", getattr(paths, "SIM", "")))
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sims/udacity/maps/jungle/configs/paths.py")

    base_dir: Path = Path(paths.DATA_DIR).expanduser().resolve()
    out_dir = make_run_dir(base_dir, prefix="pid")

    # manifest logger
    extras = {
        "sector_span_target": int(TARGET_SECTOR_SPAN),
        "max_steps_cap": int(MAX_STEPS),
        "target_speed": float(TARGET_SPEED),
        "weather": getattr(run, "WEATHER", "sunny"),
        "daytime": getattr(run, "DAYTIME", "day"),
    }
    mlog = DataRunLogger(run_dir=out_dir, map_name="jungle", source="jungle",
                         sim_app=sim_app, data_dir=out_dir, raw_pd_dir=None, extras=extras)

    sim = UdacitySimulator(sim_exe_path=str(sim_app), host=HOST, port=PORT)
    env = UdacityGym(simulator=sim)
    sim.start()

    _ensure_jungle(env, weather=extras["weather"], daytime=extras["daytime"])

    agent = PIDUdacityAgent_Angle(
        target_speed=TARGET_SPEED, track="jungle",
        before_action_callbacks=[], after_action_callbacks=[]
    )

    idx = steps = 0
    dropped = 0
    obs = env.observe()
    sector_state: SectorState = None

    print(f"[collect:jungle] writing to: {out_dir}")
    try:
        while steps < MAX_STEPS:
            if obs is None or obs.input_image is None:
                time.sleep(0.005)
                obs = env.observe()
                continue

            try:
                sector_now = int(getattr(obs, "sector", 0))
            except Exception:
                sector_now = 0

            should_stop, sector_state = update_sector_progress(
                sector_state, sector_now, TARGET_SECTOR_SPAN, 200
            )
            if should_stop:
                progress = sector_state[2] if sector_state is not None else 0
                print(f"[collect:jungle] sector span reached ({progress} >= {TARGET_SECTOR_SPAN}); stopping.")
                break

            action = agent(obs)
            if not isinstance(action, UdacityAction):
                try:
                    action = UdacityAction(steering_angle=float(action[0]), throttle=float(action[1]))
                except Exception:
                    action = UdacityAction(steering_angle=0.0, throttle=0.1)

            last_time = obs.time
            obs, reward, terminated, truncated, info = env.step(action)
            while obs.time == last_time:
                obs = env.observe()
                time.sleep(0.0025)

            try:
                img = np.asarray(obs.input_image, dtype=np.uint8)
                img_path = out_dir / f"image_{idx:06d}.jpg"
                js_path  = out_dir / f"record_{idx:06d}.json"

                save_image(img, img_path)
                write_frame_record(
                    js_path,
                    steer=float(action.steering_angle),
                    throttle=float(action.throttle),
                    track_id=1,
                    topo_id=1,
                    frame_idx_in_run=int(steps),
                )
                idx += 1
                mlog.add_frames(1)
            except Exception:
                dropped += 1
                mlog.add_dropped(1)

            steps += 1

        print(f"[collect:jungle] wrote {idx} frames (dropped {dropped}) -> {out_dir}")

    except KeyboardInterrupt:
        print("\n[collect:jungle] Ctrl-C — stopping.")
    finally:
        try:
            sim.close()
            env.close()
        except Exception:
            pass
        print("[collect:jungle] done.")

if __name__ == "__main__":
    main()
