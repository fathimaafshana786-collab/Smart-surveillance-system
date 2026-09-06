"""
logger_setup.py
----------------
Sets up Python's built-in `logging` module once, so every other file in the
project can just do:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")

...and it will automatically show up both on the console and in a log file,
with consistent formatting (timestamp, level, module name, message).

Why this matters for a CV pipeline specifically:
- Model loading failures, corrupt video files, missing webcams, and bad
  frames are all real, common failure points. Logging lets us see WHERE
  and WHY something failed instead of just seeing a crash traceback.
"""

import logging
import os
from datetime import datetime


def setup_logging(log_dir: str = "data/output/logs", level: str = "INFO"):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # also print to console
        ],
    )

    logger = logging.getLogger("smart_surveillance")
    logger.info(f"Logging initialized. Writing to: {log_filename}")
    return logger
