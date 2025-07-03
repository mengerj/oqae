"""
Integration tests for SystemMonitor with W&B.

This module provides focused integration tests for the SystemMonitor
class with W&B integration.
"""

import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from omvqvae.utils.system_monitor import SystemMonitor


def _is_wandb_available() -> bool:
    """Check if W&B is available and properly configured."""
    try:
        import wandb  # noqa: F401

        # Check if W&B has essential functions
        if not hasattr(wandb, "log"):
            return False

        # Try offline mode to test basic functionality
        try:
            wandb.init(mode="offline", project="test-project", reinit=True)
            wandb.finish()
            return True
        except Exception:
            # Check for API key as fallback
            return bool(os.environ.get("WANDB_API_KEY"))

    except ImportError:
        return False


class TestSystemMonitorIntegration:
    """Integration tests for SystemMonitor."""

    def test_basic_monitoring_workflow(self) -> None:
        """Test complete monitoring workflow."""
        monitor = SystemMonitor(interval=1)

        # Start monitoring
        monitor.start()

        # Log some events
        monitor.log_event("Test started")
        time.sleep(1.1)
        monitor.log_event("Test in progress")
        time.sleep(1.1)
        monitor.log_event("Test completed")

        # Stop monitoring
        monitor.stop()

        # Verify data collection
        assert len(monitor.events) == 3
        assert monitor.events[0]["message"] == "Test started"
        assert monitor.events[2]["message"] == "Test completed"

        # Verify resource data was collected
        assert len(monitor.cpu_usage) > 0
        assert len(monitor.memory_usage) > 0

    def test_monitoring_with_summary(self) -> None:
        """Test monitoring with summary generation."""
        monitor = SystemMonitor(interval=1)

        monitor.start()
        time.sleep(1.1)
        monitor.stop()

        summary = monitor.summarize()

        # Check summary structure
        assert "cpu_usage_mean" in summary
        assert "memory_usage_mean" in summary
        assert "gpu_metrics" in summary
        assert isinstance(summary["cpu_usage_mean"], float)

    def test_monitoring_with_file_output(self) -> None:
        """Test monitoring with file output."""
        monitor = SystemMonitor(interval=1)

        # Add some data
        monitor.cpu_usage = [(time.time(), 50.0)]
        monitor.memory_usage = [(time.time(), 2.5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test plotting (with mocked matplotlib)
            with (
                patch("omvqvae.utils.system_monitor.plt.figure"),
                patch("omvqvae.utils.system_monitor.plt.savefig") as mock_savefig,
            ):
                monitor.plot_metrics(save_dir=temp_dir)
                assert mock_savefig.called

            # Test metrics saving (with mocked pandas)
            with patch("pandas.DataFrame") as mock_df:
                mock_df.return_value.to_csv = Mock()
                monitor.save_metrics(temp_dir)
                assert mock_df.called


class TestWandBIntegration:
    """Tests for W&B integration with SystemMonitor."""

    def test_wandb_availability_check(self) -> None:
        """Test W&B availability detection."""
        available = _is_wandb_available()
        assert isinstance(available, bool)

        # Test should work regardless of W&B availability
        monitor = SystemMonitor()
        monitor.log_event("Test event")
        assert len(monitor.events) == 1

    @pytest.mark.skipif(
        not _is_wandb_available(), reason="W&B not available or not configured"
    )
    def test_wandb_custom_monitor(self) -> None:
        """Test custom W&B monitor implementation."""
        try:
            import wandb  # noqa: F401

            class TestWandBMonitor(SystemMonitor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.logged_metrics = []

                def log_event(self, message: str) -> None:
                    super().log_event(message)

                    # Custom logging logic
                    summary = self.summarize()
                    self.logged_metrics.append(
                        {
                            "event": message,
                            "cpu_usage": summary.get("cpu_usage_mean", 0),
                            "memory_usage": summary.get("memory_usage_mean", 0),
                        }
                    )

            # Test the custom monitor
            monitor = TestWandBMonitor(interval=1)

            monitor.start()
            time.sleep(1.1)
            monitor.log_event("Custom W&B test")
            monitor.stop()

            # Verify custom logging worked
            assert len(monitor.logged_metrics) == 1
            assert monitor.logged_metrics[0]["event"] == "Custom W&B test"
            assert "cpu_usage" in monitor.logged_metrics[0]

        except ImportError:
            pytest.skip("wandb not available")

    @pytest.mark.skipif(
        not _is_wandb_available(), reason="W&B not available or not configured"
    )
    @patch("wandb.log")
    def test_wandb_mock_integration(self, mock_wandb_log: Mock) -> None:
        """Test W&B integration with mocked wandb.log."""
        try:
            import wandb  # noqa: F401

            class MockWandBMonitor(SystemMonitor):
                def log_event(self, message: str) -> None:
                    super().log_event(message)

                    # Mock W&B logging
                    summary = self.summarize()
                    wandb.log(
                        {
                            "system/cpu_usage": summary.get("cpu_usage_mean", 0),
                            "system/memory_usage": summary.get("memory_usage_mean", 0),
                            "event": message,
                        }
                    )

            monitor = MockWandBMonitor()

            # Add some test data
            monitor.cpu_usage = [(time.time(), 45.0)]
            monitor.memory_usage = [(time.time(), 3.2)]

            # Log event
            monitor.log_event("Mock W&B test")

            # Verify wandb.log was called
            mock_wandb_log.assert_called_once()

        except ImportError:
            pytest.skip("wandb not available")


# Test configuration
@pytest.fixture
def sample_monitor():
    """Create a SystemMonitor with sample data."""
    monitor = SystemMonitor()

    # Add sample data
    current_time = time.time()
    monitor.cpu_usage = [(current_time, 50.0)]
    monitor.memory_usage = [(current_time, 2.5)]

    return monitor
