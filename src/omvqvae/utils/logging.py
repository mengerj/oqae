"""
Centralized logging configuration for OQAE.

This module provides a consistent logging interface across all OQAE modules
with rich formatting and configurable output levels.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance with rich formatting.

    Parameters
    ----------
    name : str
        Name of the logger, typically __name__ from the calling module
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        Defaults to INFO

    Returns
    -------
    logging.Logger
        Configured logger instance with rich formatting

    Examples
    --------
    >>> from omvqvae.utils.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    """
    # Set default level
    if level is None:
        level = "INFO"

    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create rich handler with console
    console = Console(stderr=True)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
    )

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def configure_logging(level: str = "INFO", quiet: bool = False) -> None:
    """
    Configure global logging settings for OQAE.

    Parameters
    ----------
    level : str, default "INFO"
        Global logging level
    quiet : bool, default False
        If True, suppress all logging output

    Examples
    --------
    >>> from omvqvae.utils.logging import configure_logging
    >>> configure_logging(level="DEBUG")
    """
    if quiet:
        logging.disable(logging.CRITICAL)
        return

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add rich handler
    console = Console(stderr=True)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )

    formatter = logging.Formatter(
        fmt="%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
