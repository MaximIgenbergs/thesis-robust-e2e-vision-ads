import platform
from scripts import ROOT

system = platform.system()
machine = platform.machine()

if system == "Darwin":
    SIM = ROOT / "binaries/jungle/udacity_macos_silicon/udacity.app"
elif system == "Linux":
    SIM = ROOT / "binaries/jungle/udacity_linux/udacity.x86_64"
else:
    raise RuntimeError(f"Unsupported platform: {system}")

RUNS_DIR = ROOT / "runs" / "jungle"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "jungle"
DATA_DIR.mkdir(parents=True, exist_ok=True)