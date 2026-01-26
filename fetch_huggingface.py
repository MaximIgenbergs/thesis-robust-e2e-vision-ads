"""
Download thesis resources from Hugging Face.

Examples:
    # Download DAVE-2 model for GenRoads
    python -m fetch_huggingface --type model --model dave2 --map genroads

    # Download training data for ViT on GenRoads
    python -m fetch_huggingface --type data --model vit --map genroads

    # Download evaluation run
    python -m fetch_huggingface --type run --model dave2 --map jungle --test robustness

    # Download CARLA TCP tiny generalization
    python -m fetch_huggingface --type run --model tcp --map carla_tiny --test generalization

    # Download Udacity simulator (from HF dataset maxim-igenbergs/udacity-binaries)
    python -m fetch_huggingface --type sim --map jungle --platform linux
    python -m fetch_huggingface --type sim --map jungle --platform macos_silicon
    python -m fetch_huggingface --type sim --map jungle --platform macos_intel_64-bit
    python -m fetch_huggingface --type sim --map genroads --platform linux
    python -m fetch_huggingface --type sim --map genroads --platform macos_silicon
    python -m fetch_huggingface --type sim --map jungle --platform unity_project

    # List available resources
    python -m fetch_huggingface --list

Output layout (default --output "."):
  checkpoints/<modelname>/
  runs/<mapname>/<testtype>/<modelname>/
  data/<mapname>/
  (special) jungle unity_project downloads/extracts to "."
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from huggingface_hub import hf_hub_download, snapshot_download


# Configuration

# Model repositories
MODELS = {
    "dave2": {
        "repo": "maxim-igenbergs/dave2",
        "checkpoints": {
            "genroads": "genroads_20251028-145557",
            "jungle": "jungle_20251209-175046",
        },
    },
    "dave2-gru": {
        "repo": "maxim-igenbergs/dave2-gru",
        "checkpoints": {
            "genroads": "genroads_20251215-174930",
            "jungle": "jungle_20251201-142321",
        },
    },
    "vit": {
        "repo": "maxim-igenbergs/vit",
        "checkpoints": {
            "genroads": "genroads_20251202-152358",
            "jungle": "jungle_20251201-132938",
        },
    },
    "tcp": {
        "repo": "maxim-igenbergs/tcp-carla-repro",
        "checkpoints": {
            "carla": None,  # Root level - download best_model.ckpt directly
        },
        "files": ["best_model.ckpt"],
    },
}

# Training data
DATA_REPO = "maxim-igenbergs/thesis-data"
DATA_FILES = {
    "jungle": {
        "dave2": "jungle_20251029-174507.tar.zst",
        "dave2-gru": "jungle_20251029-174507.tar.zst",
        "vit": "jungle_20251029-174507.tar.zst",
    },
    "genroads": {
        "dave2": "genroads_gru.tar.zst",
        "dave2-gru": "genroads_gru.tar.zst",
        "vit": "genroads_20251201-163211.tar.zst",
    },
}

# Evaluation runs
RUNS_REPO = "maxim-igenbergs/thesis-runs"
RUNS = {
    "jungle": {
        "robustness": {
            "vit": ["runs/jungle/robustness/vit_20251218_172745.tar.zst"],
            "dave2": ["runs/jungle/robustness/dave2_20251219_165635.tar.zst"],
            "dave2-gru": ["runs/jungle/robustness/dave2_gru_20251220_113323.tar.zst"],
        },
        "generalization": {
            "vit": ["runs/jungle/generalization/vit_20251219_030941.tar.zst"],
            "dave2": ["runs/jungle/generalization/dave2_20251220_225512.tar.zst"],
            "dave2-gru": ["runs/jungle/generalization/dave2_gru_20251221_000824.tar.zst"],
        },
    },
    "genroads": {
        "robustness": {
            "vit": ["runs/genroads/robustness/vit_20251208_181344.tar.zst"],
            "dave2": ["runs/genroads/robustness/dave2_20251214_184315.tar.zst"],
            "dave2-gru": ["runs/genroads/robustness/dave2_gru_20251221_193038.tar.zst"],
        },
        "generalization": {
            "vit": ["runs/genroads/generalization/vit_20251217_182632.tar.zst"],
            "dave2": ["runs/genroads/generalization/dave2_20251217_225857.tar.zst"],
            "dave2-gru": ["runs/genroads/generalization/dave2_gru_20251218_031746.tar.zst"],
        },
    },
    "carla": {
        "robustness": {
            "tcp": [
                "runs/carla/robustness/tcp/20251212_185734.tar.zst.part_aa",
                "runs/carla/robustness/tcp/20251212_185734.tar.zst.part_ab",
                "runs/carla/robustness/tcp/20251212_185734.tar.zst.part_ac",
            ],
        },
        "generalization": {
            "tcp": ["runs/carla/generalization/tcp/20251201_120732.tar.zst"],
        },
    },
    "carla_tiny": {
        "generalization": {
            "tcp": ["runs/carla/generalization/tcp/tiny_20251231_045136.tar.zst"],
        },
    },
}

# Udacity simulator binaries
SIM_REPO = "maxim-igenbergs/udacity-binaries"
SIMULATORS = {
    "jungle": {
        "linux": "jungle/udacity_linux.zip",
        "macos_silicon": "jungle/udacity_macos_silicon.zip",
        "macos_intel_64-bit": "jungle/udacity_macos_intel-64-bit.zip",
        "unity_project": "self-driving-car-sim.zip",
    },
    "genroads": {
        "linux": "genroads/udacity_linux.zip",
        "macos_silicon": "genroads/udacity_macos_silicon.zip",
    },
}


# Download functions

def download_model(model: str, map_name: str, output_dir: Path) -> bool:
    """Download a model checkpoint from Hugging Face."""
    if model not in MODELS:
        print(f"[fetch][WARN] Unknown model: {model}")
        print(f"[fetch][INFO] Available: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model]
    repo = model_info["repo"]
    checkpoints = model_info["checkpoints"]

    if map_name not in checkpoints:
        print(f"[fetch][WARN] Model '{model}' has no checkpoint for map '{map_name}'")
        print(f"[fetch][INFO] Available maps: {', '.join(checkpoints.keys())}")
        return False

    checkpoint_path = checkpoints[map_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch][INFO] Downloading model: {model} ({map_name})")
    print(f"[fetch][INFO] Repository: {repo}")

    try:
        if checkpoint_path is None:
            # TCP case - download specific files
            files = model_info.get("files", [])
            for f in files:
                print(f"[fetch][INFO] Downloading: {f}")
                hf_hub_download(
                    repo_id=repo,
                    filename=f,
                    repo_type="model",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                )
        else:
            # Download checkpoint folder contents
            print(f"[fetch][INFO] Downloading checkpoint: {checkpoint_path}/")
            snapshot_download(
                repo_id=repo,
                repo_type="model",
                local_dir=output_dir,
                allow_patterns=f"{checkpoint_path}/*",
                local_dir_use_symlinks=False,
            )
        return True
    except Exception as e:
        print(f"[fetch][WARN] Download failed: {e}")
        return False


def download_data(model: str, map_name: str, output_dir: Path) -> bool:
    """Download training data from Hugging Face."""
    if map_name not in DATA_FILES:
        print(f"[fetch][WARN] Unknown map: {map_name}")
        print(f"[fetch][INFO] Available: {', '.join(DATA_FILES.keys())}")
        return False

    map_data = DATA_FILES[map_name]
    if model not in map_data:
        print(f"[fetch][WARN] No training data for model '{model}' on map '{map_name}'")
        print(f"[fetch][INFO] Available models: {', '.join(map_data.keys())}")
        return False

    filename = map_data[model]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch][INFO] Downloading training data: {filename}")
    print(f"[fetch][INFO] Repository: {DATA_REPO}")

    try:
        hf_hub_download(
            repo_id=DATA_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as e:
        print(f"[fetch][WARN] Download failed: {e}")
        return False


def download_run(model: str, map_name: str, test_type: str, output_dir: Path) -> bool:
    """Download evaluation run from Hugging Face.

    Saves archives directly into:
      runs/<map>/<test>/<model>/<archive files>
    (i.e., flattens the HF-internal subfolders like runs/...).
    """
    if map_name not in RUNS:
        print(f"[fetch][WARN] Unknown map: {map_name}")
        print(f"[fetch][INFO] Available: {', '.join(RUNS.keys())}")
        return False

    map_runs = RUNS[map_name]
    if test_type not in map_runs:
        print(f"[fetch][WARN] No '{test_type}' runs for map '{map_name}'")
        print(f"[fetch][INFO] Available test types: {', '.join(map_runs.keys())}")
        return False

    test_runs = map_runs[test_type]
    if model not in test_runs:
        print(f"[fetch][WARN] No run for model '{model}' on {map_name}/{test_type}")
        print(f"[fetch][INFO] Available models: {', '.join(test_runs.keys())}")
        return False

    files = test_runs[model]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch][INFO] Downloading run: {model} / {map_name} / {test_type}")
    print(f"[fetch][INFO] Repository: {RUNS_REPO}")
    print(f"[fetch][INFO] Files: {len(files)}")

    staging_dir = output_dir / ".hf_download"
    staging_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    try:
        for f in files:
            print(f"[fetch][INFO] Downloading: {f}")
            p = hf_hub_download(
                repo_id=RUNS_REPO,
                filename=f,
                repo_type="dataset",
                local_dir=staging_dir,
                local_dir_use_symlinks=False,
            )

            src = Path(p)
            dst = output_dir / Path(f).name  # flatten into output_dir
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            downloaded.append(dst)

        shutil.rmtree(staging_dir, ignore_errors=True)

        # If chunked, print reassembly instructions using local paths
        is_chunked = len(downloaded) > 1 and any(".part_" in p.name for p in downloaded)
        if is_chunked:
            first = str(downloaded[0])
            base = first.rsplit(".part_", 1)[0]  # .../20251212_185734.tar.zst
            print("[fetch][INFO] Chunked archive downloaded. To reassemble:")
            print(f"[fetch][INFO]   cat {base}.part_* > {base}")
            print(f"[fetch][INFO]   tar --use-compress-program=unzstd -xf {base}")

        return True
    except Exception as e:
        print(f"[fetch][WARN] Download failed: {e}")
        return False


def download_sim(map_name: str, platform: str, output_dir: Path) -> bool:
    """Download Udacity simulator binaries from Hugging Face."""
    if map_name not in SIMULATORS:
        print(f"[fetch][WARN] Unknown map: {map_name}")
        print(f"[fetch][INFO] Available: {', '.join(SIMULATORS.keys())}")
        return False

    platforms = SIMULATORS[map_name]
    if platform not in platforms:
        print(f"[fetch][WARN] Unknown platform for {map_name}: {platform}")
        print(f"[fetch][INFO] Available: {', '.join(platforms.keys())}")
        return False

    filename = platforms[platform]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch][INFO] Downloading simulator: {map_name} ({platform})")
    print(f"[fetch][INFO] Repository: {SIM_REPO}")
    print(f"[fetch][INFO] File: {filename}")

    try:
        local_path = hf_hub_download(
            repo_id=SIM_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        local_path = Path(local_path)

        # Extract into a dedicated folder to avoid mixing multiple builds
        if map_name == "jungle" and platform == "unity_project":
            extract_to = output_dir  # special: extract to "."
        else:
            extract_to = output_dir / map_name / platform
            extract_to.mkdir(parents=True, exist_ok=True)

        if local_path.suffix.lower() == ".zip":
            print(f"[fetch][INFO] Extracting to: {extract_to}")
            try:
                with ZipFile(local_path, "r") as zf:
                    zf.extractall(extract_to)
            except BadZipFile as e:
                print(f"[fetch][WARN] ZIP extract failed (bad zip): {e}")
                return False
        else:
            print(f"[fetch][INFO] Not a .zip file; downloaded to: {local_path}")

        return True
    except Exception as e:
        print(f"[fetch][WARN] Download/extract failed: {e}")
        return False


def list_resources() -> None:
    """Print all available resources."""
    print("\nModels:")
    for model, info in MODELS.items():
        maps = ", ".join(info["checkpoints"].keys())
        print(f"  {model}: {maps}")

    print("\nTraining data:")
    for map_name, files in DATA_FILES.items():
        models = ", ".join(files.keys())
        print(f"  {map_name}: {models}")

    print("\nEvaluation runs:")
    for map_name, tests in RUNS.items():
        for test_type, models in tests.items():
            model_list = ", ".join(models.keys())
            chunked = any(
                any(".part_" in f for f in files) and len(files) > 1
                for files in models.values()
            )
            note = " (chunked)" if chunked else ""
            print(f"  {map_name} {test_type}: {model_list}{note}")

    print("\nSimulators:")
    for map_name, platforms in SIMULATORS.items():
        platform_list = ", ".join(platforms.keys())
        print(f"  {map_name}: {platform_list}")

    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download thesis resources from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fetch_huggingface --type model --model dave2 --map genroads
  python -m fetch_huggingface --type data --model vit --map genroads
  python -m fetch_huggingface --type run --model tcp --map carla_tiny --test generalization
  python -m fetch_huggingface --type sim --map jungle --platform linux
  python -m fetch_huggingface --list
        """,
    )
    ap.add_argument(
        "--type",
        choices=["model", "data", "run", "sim"],
        help="Type of resource to download",
    )
    ap.add_argument(
        "--model",
        choices=["dave2", "dave2-gru", "vit", "tcp"],
        help="Model name",
    )
    ap.add_argument(
        "--map",
        choices=["genroads", "jungle", "carla", "carla_tiny"],
        help="Map/environment",
    )
    ap.add_argument(
        "--test",
        choices=["robustness", "generalization"],
        help="Test type (for runs)",
    )
    ap.add_argument(
        "--platform",
        choices=["linux", "macos_silicon", "macos_intel_64-bit", "unity_project"],
        default="linux",
        help="Platform (for sim, default: linux)",
    )
    ap.add_argument(
        "--output",
        default=".",
        help="Output directory (default: .)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List all available resources",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_resources()
        return 0

    if not args.type:
        print("[fetch][WARN] --type is required (model, data, run, sim)")
        print("[fetch][INFO] Use --list to see available resources")
        return 1

    output_dir = Path(args.output)

    if args.type == "model":
        if not args.model:
            print("[fetch][WARN] --model is required for type 'model'")
            return 1
        if not args.map:
            print("[fetch][WARN] --map is required for type 'model'")
            return 1
        success = download_model(
            args.model,
            args.map,
            output_dir / "checkpoints" / args.model,
        )

    elif args.type == "data":
        if not args.model:
            print("[fetch][WARN] --model is required for type 'data'")
            return 1
        if not args.map:
            print("[fetch][WARN] --map is required for type 'data'")
            return 1
        success = download_data(
            args.model,
            args.map,
            output_dir / "data" / args.map,
        )

    elif args.type == "run":
        if not args.model:
            print("[fetch][WARN] --model is required for type 'run'")
            return 1
        if not args.map:
            print("[fetch][WARN] --map is required for type 'run'")
            return 1
        if not args.test:
            print("[fetch][WARN] --test is required for type 'run'")
            return 1
        success = download_run(
            args.model,
            args.map,
            args.test,
            output_dir / "runs" / args.map / args.test / args.model,
        )

    elif args.type == "sim":
        if not args.map:
            print("[fetch][WARN] --map is required for type 'sim'")
            return 1

        # Special case: jungle unity project should be downloaded/extracted to "."
        sim_out = output_dir if (args.map == "jungle" and args.platform == "unity_project") else (output_dir / "sim")
        success = download_sim(args.map, args.platform, sim_out)

    else:
        print(f"[fetch][WARN] Unknown type: {args.type}")
        return 1

    if success:
        print(f"[fetch][INFO] Download complete → {output_dir}")
        return 0

    print("[fetch][WARN] Download failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
