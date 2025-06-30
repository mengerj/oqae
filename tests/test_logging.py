"""
Test module for OQAE logging utilities.

This module tests the centralized logging configuration and ensures
proper integration with the rich logging system.
"""

import logging
import pytest
from unittest.mock import patch

from omvqvae.utils.logging import get_logger, configure_logging


class TestLogging:
    """Test suite for logging utilities."""

    def test_get_logger_default_level(self) -> None:
        """Test that get_logger returns a logger with INFO level by default."""
        logger = get_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert logger.name == "test_logger"

    def test_get_logger_custom_level(self) -> None:
        """Test that get_logger respects custom logging levels."""
        logger = get_logger("test_logger_debug", level="DEBUG")

        assert logger.level == logging.DEBUG

    def test_get_logger_no_duplicate_handlers(self) -> None:
        """Test that multiple calls to get_logger don't create duplicate handlers."""
        logger1 = get_logger("test_duplicate")
        initial_handlers = len(logger1.handlers)

        logger2 = get_logger("test_duplicate")

        assert logger1 is logger2
        assert len(logger2.handlers) == initial_handlers

    def test_configure_logging_quiet_mode(self) -> None:
        """Test that quiet mode disables logging."""
        with patch('logging.disable') as mock_disable:
            configure_logging(quiet=True)
            mock_disable.assert_called_once_with(logging.CRITICAL)

    def test_configure_logging_level_setting(self) -> None:
        """Test that configure_logging sets the correct level."""
        configure_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
