"""
test_config_loader.py
-----------------------
Tests for app/utils/config_loader.py. Verifies that we fail clearly and
early on bad input, rather than crashing confusingly deep in the pipeline
later - the whole point of having a dedicated config loader.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app.utils.config_loader import load_config, ConfigError


def write_temp_yaml(content: str) -> str:
    """Helper: writes a string to a temp .yaml file and returns its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestConfigLoader:
    def test_missing_file_raises_config_error(self):
        with pytest.raises(ConfigError):
            load_config("this/path/does/not/exist.yaml")

    def test_valid_config_loads_successfully(self):
        path = write_temp_yaml("""
video:
  source: webcam
model:
  weights_path: models/yolov8n.pt
  confidence_threshold: 0.4
  iou_threshold: 0.45
  classes: [person]
output:
  save_video: false
  output_dir: data/output
  log_dir: data/output/logs
logging:
  level: INFO
""")
        try:
            config = load_config(path)
            assert config["video"]["source"] == "webcam"
            assert config["model"]["confidence_threshold"] == 0.4
        finally:
            os.remove(path)

    def test_missing_required_section_raises_config_error(self):
        path = write_temp_yaml("""
video:
  source: webcam
output:
  output_dir: data/output
  log_dir: data/output/logs
logging:
  level: INFO
""")
        try:
            with pytest.raises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_malformed_yaml_raises_config_error(self):
        path = write_temp_yaml("video: [unclosed_list\nmodel: {broken")
        try:
            with pytest.raises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)