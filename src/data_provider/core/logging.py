from __future__ import annotations

import logging

from utils.logging_utils import init_logger


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return the shared global logger to keep output consistent."""

    return init_logger(level=level)
