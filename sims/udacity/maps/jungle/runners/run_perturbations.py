"""
Generic Udacity driver using adapters + PerturbationDrive.
- Runs on jungle map
- Uses perturbations from sims/udacity/configs/perturbations.py (LIST / CHUNKS)
- Selects model via sims/udacity/configs/jungle/run.py (MODEL_NAME)
"""

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import List, Union
import numpy as np

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD_DIR = ROOT / "external" / "perturbation-drive"
if str(PD_DIR) not in sys.path:
    sys.path.insert(0, str(PD_DIR))

from sims.udacity.maps.configs import perturbations
from sims.udacity.maps.configs.run import HOST, PORT
from sims.udacity.maps.jungle.configs import paths, run
from sims.udacity.logging.eval_runs import RunLogger, prepare_run_dir, module_public_dict, best_effort_git_sha, pip_freeze
from sims.udacity.adapters.dave2_adapter import Dave2Adapter
from sims.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent_callback import PDPreviewCallback
from external.udacity_gym.logger import ScenarioOutcomeWriter, ScenarioOutcomeLite, ScenarioLite

from perturbationdrive import ImagePerturbation


def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _build_adapter(model_name: str, image_size_hw, ckpt_path: Path | None):
    if model_name == "dave2":
        return Dave2Adapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    elif model_name == "dave2_gru":
        return Dave2GRUAdapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims/udacity/configs/jungle/run.py")


def _force_start_episode(env: UdacityGym, track="jungle", weather="sunny", daytime="day", timeout_s=30.0):
    """
    Robustly enter the jungle map without restarting the Unity process.
    Keeps re-sending 'start_episode' until frames arrive.
    """
    env.reset(track=track, weather=weather, daytime=daytime)  # set sim_state
    t0 = time.perf_counter()
    last_emit = 0.0
    while True:
        obs = env.observe()
        if obs and obs.input_image is not None:
            return
        now = time.perf_counter()
        if now - last_emit > 1.0:
            try:
                env.simulator.sim_executor.send_track(track, weather, daytime)
            except Exception:
                pass
            last_emit = now
        if now - t0 > timeout_s:
            raise RuntimeError("Failed to enter Jungle map before timeout.")
        time.sleep(0.2)


def _run_episode(env: UdacityGym,
                 adapter,
                 preview: PDPreviewCallback,
                 controller: ImagePerturbation,
                 pert_name: str,
                 severity: int,
                 max_steps: int,
                 save_images: bool) -> ScenarioOutcomeLite:

    # Make sure the car is spawned on Jungle before the episode starts
    _force_start_episode(env,
                         track="jungle",
                         weather=getattr(run, "WEATHER", "sunny"),
                         daytime=getattr(run, "DAYTIME", "day"))

    frames, xte, speeds, actions, pos = [], [], [], [], []
    original_images, perturbed_images = ([] if save_images else None), ([] if save_images else None)

    offtrack = False
    step = 0
    t0 = time.perf_counter()
    obs = env.observe()

    try:
        while (step < max_steps) and not offtrack:
            if obs is None or obs.input_image is None:
                time.sleep(0.005)
                obs = env.observe()
                continue

            # Writable copy (some PD dynamic filters write in-place)
            img_np = np.asarray(obs.input_image, dtype=np.uint8).copy()
            if save_images:
                original_images.append(img_np)

            # Apply perturbation (controller carries masks for THIS chunk only)
            pert_np = controller.perturbation(img_np, pert_name, int(severity))
            if save_images:
                perturbed_images.append(pert_np)

            # Adapter -> UdacityAction
            try:
                out = getattr(adapter, "predict")(pert_np)
            except AttributeError:
                out = adapter(pert_np)

            if isinstance(out, UdacityAction):
                action = out
            elif isinstance(out, dict):
                steer = out.get("steer", out.get("steering", out.get("steering_angle"), 0.0))
                thr   = out.get("throttle", out.get("accel", out.get("throttle_cmd"), 0.0))
                action = UdacityAction(
                    steering_angle=float(np.clip(float(steer), -1.0, 1.0)),
                    throttle=float(np.clip(float(thr),   0.0, 1.0)),
                )
            else:
                arr = np.array(out).reshape(-1)
                action = UdacityAction(
                    steering_angle=float(np.clip(float(arr[0]), -1.0, 1.0)),
                    throttle=float(np.clip(float(arr[1]),    0.0, 1.0)),
                )

            # Preview overlay (perturbed ego)
            preview(obs, display_image_np=pert_np, action=action, perturbation=pert_name)

            last_time = obs.time
            obs, reward, terminated, truncated, info = env.step(action)

            # Log step
            frames.append(step)
            xte_val = float(obs.cte)
            xte.append(xte_val)
            speeds.append(float(obs.speed))
            actions.append([float(action.steering_angle), float(action.throttle)])
            px, py, pz = obs.position
            pos.append([float(px), float(py), float(pz)])

            # Early stop if off track (|cte| >= 4)
            if abs(xte_val) >= 4.0:
                offtrack = True
                break

            # Sync to next tick
            while obs.time == last_time:
                time.sleep(0.0025)
                obs = env.observe()

            step += 1

    finally:
        wall = time.perf_counter() - t0
        print(f"[episode] {pert_name}@{severity}: steps={step} wall={wall:.1f}s offtrack={offtrack}")

    is_success = (not offtrack) and (max([abs(v) for v in xte]) < 4.0 if xte else False)

    return ScenarioOutcomeLite(
        frames=frames,
        pos=pos,
        xte=xte,
        speeds=speeds,
        actions=actions,
        pid_actions=[],
        scenario=ScenarioLite(perturbation_function=pert_name, perturbation_scale=int(severity), road="jungle"),
        isSuccess=bool(is_success),
        timeout=False,
        original_images=original_images,
        perturbed_images=perturbed_images,
    )


def main() -> int:
    sim_app = _abs(getattr(paths, "SIM", getattr(paths, "SIM", "")))
    model_name = getattr(run, "MODEL_NAME", "dave2")

    if model_name == "dave2":
        ckpt = _abs(paths.DAVE2_CKPT) if getattr(paths, "DAVE2_CKPT", None) else None
    elif model_name == "dave2_gru":
        ckpt = _abs(paths.DAVE2_GRU_CKPT) if getattr(paths, "DAVE2_GRU_CKPT", None) else None
    else:
        raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims/udacity/configs/jungle/run.py")

    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sims/udacity/configs/jungle/paths.py")
    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()}_CKPT not found: {ckpt}\nEdit sims/udacity/configs/jungle/paths.py")

    runs_root = _abs(paths.RUNS_DIR) / "robustness" / model_name
    adapter = _build_adapter(model_name, image_size_hw=run.IMAGE_SIZE, ckpt_path=ckpt)

    # ---- logging ----
    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(
        map_name="jungle",
        test_type="robustness",
        model_name=model_name,
        tag=ckpt_name,
    )
    print(f"[drive:jungle] model={model_name} logs -> {run_dir}")
    
    git_info = {
        "thesis_sha": best_effort_git_sha(ROOT),
        "perturbation_drive_sha": best_effort_git_sha(PD_DIR),
    }
    logger = RunLogger(
        run_dir=run_dir,
        model=model_name,
        checkpoint=(str(ckpt) if ckpt else None),
        sim_name="udacity",
        git_info=git_info,
    )
    logger.snapshot_configs(
        sim_app=sim_app,
        ckpt=ckpt,
        cfg_paths=module_public_dict(paths),
        cfg_roads={"map": "jungle"},
        cfg_perturbations=module_public_dict(perturbations),
        cfg_run=module_public_dict(run),
        cfg_host_port={"host": HOST, "port": PORT},
    )
    logger.snapshot_env(pip_freeze())

    chunks: List[List[str]] = perturbations.CHUNKS if len(perturbations.CHUNKS) > 0 else [perturbations.LIST]
    severities = list(getattr(perturbations, "SEVERITIES", [0, 3, 5]))
    episodes = int(getattr(perturbations, "EPISODES", 1))
    max_steps = int(getattr(run, "STEPS", 2000))
    save_images = bool(getattr(run, "SAVE_IMAGES", False))
    show_image = bool(getattr(run, "SHOW_IMAGE", True))

    sim = UdacitySimulator(sim_exe_path=str(sim_app), host=HOST, port=PORT)
    env = UdacityGym(simulator=sim)
    sim.start()
    _force_start_episode(env,
                         track="jungle",
                         weather=getattr(run, "WEATHER", "sunny"),
                         daytime=getattr(run, "DAYTIME", "day"))
    preview = PDPreviewCallback(enabled=show_image)

    ep_idx = 0
    try:
        for chunk in chunks:
            # Load masks for THIS chunk only (lazy exact set)
            controller = ImagePerturbation(funcs=list(chunk), image_size=(run.IMAGE_SIZE[0], run.IMAGE_SIZE[1]))

            # After heavy mask load, respawn cleanly on Jungle to avoid rolling off
            _force_start_episode(env,
                                 track="jungle",
                                 weather=getattr(run, "WEATHER", "sunny"),
                                 daytime=getattr(run, "DAYTIME", "day"))

            for pert in chunk:
                for sev in severities:
                    for ep in range(episodes):
                        ep_idx += 1
                        meta = {
                            "road": "jungle",
                            "angles": None,
                            "segs": None,
                            "start": None,
                            "perturbation": pert,
                            "severity": int(sev),
                            "image_size": {"h": run.IMAGE_SIZE[0], "w": run.IMAGE_SIZE[1]},
                            "episodes": int(episodes),
                            "ckpt_name": ckpt_name,
                        }
                        eid, ep_dir = logger.new_episode(ep_idx, meta)
                        writer = ScenarioOutcomeWriter(str(ep_dir / "logs.json"), overwrite_logs=True)
                        log_file = ep_dir / "pd_log.json"

                        t0 = time.perf_counter()
                        try:
                            outcome = _run_episode(
                                env=env,
                                adapter=adapter,
                                preview=preview,
                                controller=controller,
                                pert_name=pert,
                                severity=int(sev),
                                max_steps=max_steps,
                                save_images=save_images,
                            )
                            writer.write([outcome], images=save_images)
                            with open(log_file, "w") as f:
                                f.write("{}\n")

                            status = "ok" if outcome.isSuccess else "fail_offtrack"
                            logger.complete_episode(eid, status=status, wall_time_s=(time.perf_counter() - t0))

                        except Exception as e:
                            logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=(time.perf_counter() - t0))
                            raise

                        # Soft reset between episodes to ensure stable spawn
                        try:
                            adapter.reset()
                        except Exception:
                            pass
                        _force_start_episode(env,
                                             track="jungle",
                                             weather=getattr(run, "WEATHER", "sunny"),
                                             daytime=getattr(run, "DAYTIME", "day"))

    except KeyboardInterrupt:
        print("\n[drive:jungle] Ctrl-C — stopping.")
    finally:
        try:
            preview.close()
        except Exception:
            pass
        try:
            sim.close()
            env.close()
        except Exception:
            pass
        print("[drive:jungle] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
