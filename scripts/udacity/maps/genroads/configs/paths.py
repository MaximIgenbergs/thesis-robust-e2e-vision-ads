import platform
from scripts import ROOT

system = platform.system()
machine = platform.machine()

if system == "Darwin":
    SIM = ROOT / "binaries/genroads/udacity_macos_silicon/udacity_sim_weather_sky_ready_angles_fortuna.app"
elif system == "Linux":
    SIM = ROOT / "binaries/genroads/udacity_linux/udacity_binary.x86_64"
else:
    raise RuntimeError(f"Unsupported platform: {system}")

RUNS_DIR = ROOT / "runs" / "genroads"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "genroads"
DATA_DIR.mkdir(parents=True, exist_ok=True)