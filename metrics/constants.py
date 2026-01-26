"""Centralized constants for metrics computation."""

from __future__ import annotations

# Baseline detection
BASELINE_NAME_DEFAULT = "baseline"
BASELINE_NAMES = {"baseline", "clean", "none", ""}

# Action keys for normalization
STEER_KEYS = ["steer", "steering", "angle", "model_steer", "actual_steer"]
THROTTLE_KEYS = ["throttle", "thr", "model_throttle", "actual_throttle"]

# Metric computation
TARGET_SPEED = 2.0

# CARLA specific
IGNORE_LAV = True
ALLOWED_PREFIXES = ("simulation_results",)
ALLOWED_SUFFIXES = (".json",)