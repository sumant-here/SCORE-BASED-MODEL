"""Structured logging configuration for research runs."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "score_models",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a structured logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
