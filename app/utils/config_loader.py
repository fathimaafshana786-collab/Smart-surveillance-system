"""
config_loader.py
-----------------
Loads config/config.yaml into a plain Python dictionary.

Why a separate module for this (instead of every file calling yaml.load itself):
- Single place to validate the config and fail with a clear error message
  if something important is missing.
- If we ever switch from YAML to JSON or environment variables, only this
  file changes - nothing else in the project needs to know.
"""

import os
import yaml


class ConfigError(Exception):
    """Raised when the config file is missing or malformed."""
    pass


def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise ConfigError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config YAML: {e}")

    # Basic sanity checks - fail early and clearly instead of crashing
    # deep inside the pipeline later with a confusing error.
    required_top_level_keys = ["video", "model", "output", "logging"]
    for key in required_top_level_keys:
        if key not in config:
            raise ConfigError(f"Missing required config section: '{key}'")

    return config
