from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CKPTS_DIR = REPO_ROOT / "sims" / "udacity" / "checkpoints"
CKPTS_DIR.mkdir(parents=True, exist_ok=True)
