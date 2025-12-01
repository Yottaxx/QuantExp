import logging
import os
from datetime import datetime
from typing import Optional

_GLOBAL_LOGGER: Optional[logging.Logger] = None
_LOGGER_CREATED_AT: Optional[str] = None
_LOGGER_EXPERIMENT: Optional[str] = None


def _coerce_level(level: str) -> int:
    try:
        return getattr(logging, level.upper())
    except Exception:
        return logging.INFO


def init_logger(experiment_name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Initialize and return a shared logger.

    The logger name is composed of the current timestamp and the provided experiment
    name. Subsequent calls reuse the same logger instance while allowing log level
    updates.
    """

    global _GLOBAL_LOGGER, _LOGGER_CREATED_AT, _LOGGER_EXPERIMENT

    exp_name = (experiment_name or os.environ.get("EXPERIMENT_NAME") or "default").strip() or "default"

    if _GLOBAL_LOGGER is None:
        _LOGGER_CREATED_AT = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOGGER_EXPERIMENT = exp_name
        logger_name = f"{_LOGGER_CREATED_AT}_{_LOGGER_EXPERIMENT}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(_coerce_level(level))

        handler = logging.StreamHandler()
        handler.setLevel(_coerce_level(level))
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        _GLOBAL_LOGGER = logger
    else:
        # Keep a single global logger instance; only update metadata/level.
        if exp_name and exp_name != (_LOGGER_EXPERIMENT or ""):
            _LOGGER_EXPERIMENT = exp_name
            # Update the displayed name without replacing the logger instance.
            _GLOBAL_LOGGER.name = f"{_LOGGER_CREATED_AT or datetime.now().strftime('%Y%m%d_%H%M%S')}_{_LOGGER_EXPERIMENT}"

        _GLOBAL_LOGGER.setLevel(_coerce_level(level))
        for handler in _GLOBAL_LOGGER.handlers:
            handler.setLevel(_coerce_level(level))

    return _GLOBAL_LOGGER


def get_logger() -> logging.Logger:
    """Return the shared logger, initializing it with defaults if necessary."""

    return _GLOBAL_LOGGER or init_logger()
