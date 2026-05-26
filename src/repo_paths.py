#!/usr/bin/env python3

########################################################################
# Imports
########################################################################

from __future__ import annotations

from pathlib import Path


########################################################################
# Repository Paths
########################################################################

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
BUILD_DIR = ROOT_DIR / "build"

LIVE_INPUT_DIR = INPUTS_DIR / "live"
LIVE_PROFILE_DIR = INPUTS_DIR / "profiles" / "user"
LIVE_PROFILE_PATH = LIVE_PROFILE_DIR / "live-config.json"
LIVE_MODEL_DIR = LIVE_INPUT_DIR / "model"
LIVE_CONFIG_PATH = LIVE_INPUT_DIR / "config.ini"
LIVE_OUTPUT_DIR = OUTPUTS_DIR / "live"
LIVE_SESSION_DIR = LIVE_OUTPUT_DIR / "sessions"
LIVE_ACTIVE_PATH = LIVE_OUTPUT_DIR / "latest_active.json"

CLUSTERING_INPUT_DIR = INPUTS_DIR / "clustering"
CLUSTERING_OUTPUT_DIR = OUTPUTS_DIR / "clustering"

NATIVE_SRC_DIR = SRC_DIR / "native"
NATIVE_ENGINE_DIR = NATIVE_SRC_DIR / "engine"
NATIVE_HOST_DIR = NATIVE_SRC_DIR / "host"
NATIVE_BUILD_DIR = BUILD_DIR / "native"
NATIVE_TUNE_BIN = NATIVE_BUILD_DIR / "gradbot_tune"
NATIVE_BINANCE_BIN = NATIVE_BUILD_DIR / "gradbot_binance"


########################################################################
# Helpers
########################################################################

def rootPath(rawPath: str | Path) -> Path:
    # Resolve a user path relative to the repository root.
    path = Path(rawPath)
    return path if path.is_absolute() else ROOT_DIR / path


def livePath(rawPath: str | Path) -> Path:
    # Resolve a live runtime path against the standard live input directory.
    path = Path(rawPath)
    return path if path.is_absolute() else LIVE_INPUT_DIR / path


def liveProfilePath(rawPath: str | Path | None = None) -> Path:
    # Resolve the live profile path with the repo standard default.
    if rawPath is None or str(rawPath).strip() == "":
        return LIVE_PROFILE_PATH
    return rootPath(rawPath)


def liveOutputPath(rawPath: str | Path) -> Path:
    # Resolve a live output path under outputs/live unless absolute.
    path = Path(rawPath)
    return path if path.is_absolute() else LIVE_OUTPUT_DIR / path
