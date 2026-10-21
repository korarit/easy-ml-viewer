"""Structured logging — single uvicorn-compatible logger for the backend."""

from __future__ import annotations

import logging
import sys

_configured = False


def get_logger(name: str = "easy_ml_viewer") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _configured = True
    return logger
