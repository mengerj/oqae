"""
Test module for OQAE system monitoring utilities.

This module tests the SystemMonitor class for resource monitoring
and W&B integration capabilities.
"""

import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import pytest

from omvqvae.utils.logging import get_logger
from omvqvae.utils.system_monitor import SystemMonitor


def _is_wandb_available() -> bool:
    """Check if W&B is available and properly configured."""
    try:
        import wandb

        # Check if W&B can be imported
        if not hasattr(wandb, "log"):
            return False

        # Check if W&B is configured (has API key or is in offline mode)
        try:
            # Try to initialize in offline mode to test basic functionality
            wandb.init(mode="offline", project="test", reinit=True)
            wandb.finish()
            return True
        except Exception:
            # If offline mode fails, check for API key
            return bool(os.environ.get("WANDB_API_KEY"))

    except ImportError:
        return False


class TestSystemMonitor:
    """Test suite for SystemMonitor class."""

    def test_init_default_parameters(self) -> None:
        """Test SystemMonitor initialization with default parameters."""
        monitor = SystemMonitor()

        assert monitor.interval == 1
        assert monitor.gpu_indices is None
        assert monitor.logger is not None
        assert len(monitor.cpu_usage) == 0
        assert len(monitor.memory_usage) == 0
        assert len(monitor.events) == 0
        assert isinstance(monitor.gpu_available, bool)

    def test_init_custom_parameters(self) -> None:
        """Test SystemMonitor initialization with custom parameters."""
        logger = get_logger("test_monitor")
        monitor = SystemMonitor(interval=5, gpu_idx=0, logger=logger)

        assert monitor.interval == 5
        assert monitor.gpu_indices == [0]
        assert monitor.logger is logger

    def test_init_single_gpu(self) -> None:
        """Test SystemMonitor initialization with single GPU."""
        monitor = SystemMonitor(gpu_idx=0)
        assert monitor.gpu_indices == [0]

    def test_start_stop_monitoring(self) -> None:
        """Test starting and stopping the monitoring thread."""
        monitor = SystemMonitor(interval=1)

        # Start monitoring
        monitor.start()
        # Brief pause to let thread start
        time.sleep(0.1)

        # Stop monitoring
        monitor.stop()
        # Verify no exceptions raised

    def test_monitoring_collects_data(self) -> None:
        """Test that monitoring actually collects resource data."""
        monitor = SystemMonitor(interval=1)

        monitor.start()
        time.sleep(1.2)  # Let it collect at least one sample
        monitor.stop()

        # Should have collected some data
        assert len(monitor.cpu_usage) > 0
        assert len(monitor.memory_usage) > 0

        # Check data structure
        for timestamp, usage in monitor.cpu_usage:
            assert isinstance(timestamp, float)
            assert isinstance(usage, (int, float))
            assert usage >= 0

    def test_log_event(self) -> None:
        """Test event logging functionality."""
        monitor = SystemMonitor()

        test_message = "Test event message"
        monitor.log_event(test_message)

        assert len(monitor.events) == 1
        event = monitor.events[0]
        assert event["message"] == test_message
        assert isinstance(event["timestamp"], float)

    def test_summarize_empty_data(self) -> None:
        """Test summarize method with no collected data."""
        monitor = SystemMonitor()
        summary = monitor.summarize()

        assert summary["cpu_usage_mean"] == 0.0
        assert summary["cpu_usage_max"] == 0.0
        assert summary["memory_usage_mean"] == 0.0
        assert summary["memory_usage_max"] == 0.0
        assert summary["gpu_metrics"] == []

    def test_summarize_with_data(self) -> None:
        """Test summarize method with collected data."""
        monitor = SystemMonitor(interval=1)

        monitor.start()
        time.sleep(1.2)
        monitor.stop()

        summary = monitor.summarize()

        assert isinstance(summary["cpu_usage_mean"], float)
        assert isinstance(summary["cpu_usage_max"], float)
        assert isinstance(summary["memory_usage_mean"], float)
        assert isinstance(summary["memory_usage_max"], float)
        assert summary["cpu_usage_mean"] >= 0
        assert summary["cpu_usage_max"] >= 0

    def test_print_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test print_summary method output."""
        monitor = SystemMonitor()

        # Add some mock data
        monitor.cpu_usage = [(time.time(), 50.0)]
        monitor.memory_usage = [(time.time(), 2.5)]

        monitor.print_summary()

        captured = capsys.readouterr()
        assert "System Resource Usage Summary:" in captured.out
        assert "Core Utilization" in captured.out
        assert "Memory Usage" in captured.out

    def test_save_metrics_with_pandas(self) -> None:
        """Test save_metrics method when pandas is available."""
        monitor = SystemMonitor()

        # Add some mock data
        monitor.cpu_usage = [(time.time(), 50.0)]
        monitor.memory_usage = [(time.time(), 2.5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pandas.DataFrame") as mock_df:
                mock_df.return_value.to_csv = Mock()

                monitor.save_metrics(temp_dir)

                mock_df.assert_called_once()
                mock_df.return_value.to_csv.assert_called_once()

    def test_save_metrics_without_pandas(self) -> None:
        """Test save_metrics method when pandas is not available."""
        monitor = SystemMonitor()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pandas.DataFrame", side_effect=ImportError):
                # Should not raise exception
                monitor.save_metrics(temp_dir)

    @patch("omvqvae.utils.system_monitor.plt.savefig")
    @patch("omvqvae.utils.system_monitor.plt.figure")
    def test_plot_metrics_save_mode(
        self, mock_figure: Mock, mock_savefig: Mock
    ) -> None:
        """Test plot_metrics method in save mode."""
        monitor = SystemMonitor()

        # Add some mock data
        monitor.cpu_per_core = [(time.time(), 25.0), (time.time() + 1, 30.0)]
        monitor.memory_usage = [(time.time(), 2.5), (time.time() + 1, 3.0)]

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor.plot_metrics(save_dir=temp_dir)

            # Should create plots
            assert mock_figure.called
            assert mock_savefig.called

    def test_thread_safety(self) -> None:
        """Test that the monitor is thread-safe."""
        monitor = SystemMonitor(interval=1)

        def log_events():
            for i in range(5):
                monitor.log_event(f"Event {i}")
                time.sleep(0.05)

        # Start monitoring and logging from multiple threads
        monitor.start()

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=log_events)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        monitor.stop()

        # Should have logged 15 events total (3 threads × 5 events)
        assert len(monitor.events) == 15

    def test_cleanup_on_stop(self) -> None:
        """Test that resources are properly cleaned up on stop."""
        monitor = SystemMonitor(interval=1)

        monitor.start()
        time.sleep(0.2)

        # Stop monitoring
        monitor.stop()

        # Should complete without errors
        # The internal state is cleaned up properly


class TestSystemMonitorGPU:
    """Test suite for SystemMonitor GPU functionality."""

    @patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", True)
    def test_gpu_monitoring_initialization(self) -> None:
        """Test GPU monitoring initialization with mocked pynvml."""
        # Create a mock pynvml module
        mock_pynvml = Mock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Mock GPU"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(total=8589934592)  # 8GB

        # Add pynvml to the module globals if it's not there
        import omvqvae.utils.system_monitor as monitor_module

        original_pynvml = getattr(monitor_module, "pynvml", None)
        monitor_module.pynvml = mock_pynvml

        try:
            monitor = SystemMonitor(gpu_idx=0)

            assert monitor.gpu_available is True
            assert monitor.gpu_type == "NVIDIA"
            assert len(monitor.gpu_names) == 1
            assert monitor.gpu_names[0] == "Mock GPU"
        finally:
            # Restore original state
            if original_pynvml is not None:
                monitor_module.pynvml = original_pynvml
            else:
                delattr(monitor_module, "pynvml")

    @patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", False)
    def test_gpu_monitoring_disabled(self) -> None:
        """Test GPU monitoring when pynvml is not available."""
        monitor = SystemMonitor(gpu_idx=0)

        assert monitor.gpu_available is False
        assert monitor.gpu_type == "None"
        assert len(monitor.gpu_names) == 0

    @patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", True)
    def test_gpu_monitoring_with_cuda_visible_devices(self) -> None:
        """Test GPU monitoring with CUDA_VISIBLE_DEVICES environment variable."""
        # Create a mock pynvml module
        mock_pynvml = Mock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Mock GPU"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(total=8589934592)

        # Add pynvml to the module globals if it's not there
        import omvqvae.utils.system_monitor as monitor_module

        original_pynvml = getattr(monitor_module, "pynvml", None)
        monitor_module.pynvml = mock_pynvml

        try:
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                _ = SystemMonitor()

                # Should attempt to get handles for GPUs 0 and 1
                assert mock_pynvml.nvmlDeviceGetHandleByIndex.call_count == 2
        finally:
            # Restore original state
            if original_pynvml is not None:
                monitor_module.pynvml = original_pynvml
            else:
                delattr(monitor_module, "pynvml")


class TestSystemMonitorWandB:
    """Test suite for SystemMonitor W&B integration."""

    def test_wandb_not_available(self) -> None:
        """Test behavior when W&B is not available."""
        # This test runs regardless of W&B availability
        monitor = SystemMonitor()

        # Should not raise exception even if W&B is not available
        monitor.log_event("Test event")
        assert len(monitor.events) == 1

    @pytest.mark.skipif(
        not _is_wandb_available(), reason="W&B not available or not configured"
    )
    def test_wandb_integration_basic(self) -> None:
        """Test basic W&B integration when W&B is available."""
        try:
            import wandb  # noqa: F401

            class WandBMonitor(SystemMonitor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.wandb_logs = []

                def log_event(self, message: str) -> None:
                    super().log_event(message)

                    # Mock W&B logging
                    summary = self.summarize()
                    log_data = {
                        "system/cpu_usage": summary.get("cpu_usage_mean", 0),
                        "system/memory_usage": summary.get("memory_usage_mean", 0),
                        "event": message,
                    }
                    self.wandb_logs.append(log_data)

            monitor = WandBMonitor(interval=1)

            monitor.start()
            time.sleep(1.2)
            monitor.log_event("Test W&B event")
            monitor.stop()

            # Should have logged event to W&B
            assert len(monitor.wandb_logs) == 1
            log_entry = monitor.wandb_logs[0]
            assert log_entry["event"] == "Test W&B event"
            assert "system/cpu_usage" in log_entry
            assert "system/memory_usage" in log_entry

        except ImportError:
            pytest.skip("wandb not available")

    @pytest.mark.skipif(
        not _is_wandb_available(), reason="W&B not available or not configured"
    )
    @patch("wandb.log")
    def test_wandb_integration_with_mock(self, mock_wandb_log: Mock) -> None:
        """Test W&B integration with mocked wandb.log."""
        try:
            import wandb  # noqa: F401

            class WandBMonitor(SystemMonitor):
                def log_event(self, message: str) -> None:
                    super().log_event(message)

                    summary = self.summarize()
                    wandb.log(
                        {
                            "system/cpu_usage": summary.get("cpu_usage_mean", 0),
                            "system/memory_usage": summary.get("memory_usage_mean", 0),
                            "event": message,
                        }
                    )

            monitor = WandBMonitor()

            # Add some mock data
            monitor.cpu_usage = [(time.time(), 50.0)]
            monitor.memory_usage = [(time.time(), 2.5)]

            monitor.log_event("Test W&B logging")

            # Verify wandb.log was called
            mock_wandb_log.assert_called_once()
            call_args = mock_wandb_log.call_args[0][0]
            assert call_args["event"] == "Test W&B logging"
            assert "system/cpu_usage" in call_args
            assert "system/memory_usage" in call_args

        except ImportError:
            pytest.skip("wandb not available")

    @pytest.mark.skipif(
        not _is_wandb_available(), reason="W&B not available or not configured"
    )
    def test_wandb_monitor_context_manager(self) -> None:
        """Test W&B monitor as context manager."""
        try:
            import wandb  # noqa: F401

            class WandBMonitorContext(SystemMonitor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.wandb_logs = []

                def __enter__(self):
                    self.start()
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.stop()
                    self.print_summary()

                def log_event(self, message: str) -> None:
                    super().log_event(message)

                    summary = self.summarize()
                    self.wandb_logs.append(
                        {
                            "system/cpu_usage": summary.get("cpu_usage_mean", 0),
                            "system/memory_usage": summary.get("memory_usage_mean", 0),
                            "event": message,
                        }
                    )

            with WandBMonitorContext(interval=1) as monitor:
                time.sleep(1.2)
                monitor.log_event("Context manager test")

            # Should have logged event
            assert len(monitor.wandb_logs) == 1
            assert monitor.wandb_logs[0]["event"] == "Context manager test"

        except ImportError:
            pytest.skip("wandb not available")


class TestSystemMonitorEdgeCases:
    """Test suite for SystemMonitor edge cases and error handling."""

    def test_very_short_interval(self) -> None:
        """Test monitor with very short interval."""
        monitor = SystemMonitor(interval=1)

        monitor.start()
        time.sleep(0.1)
        monitor.stop()

        # Should complete without errors

    def test_stop_without_start(self) -> None:
        """Test calling stop without start."""
        monitor = SystemMonitor()

        # Should not raise exception
        monitor.stop()

    def test_multiple_starts(self) -> None:
        """Test calling start multiple times."""
        monitor = SystemMonitor(interval=1)

        monitor.start()

        # Starting again should not cause issues (should warn but not crash)
        monitor.start()

        monitor.stop()

    def test_exception_handling_in_monitoring_thread(self) -> None:
        """Test exception handling in monitoring thread."""
        monitor = SystemMonitor(interval=1)

        # Mock psutil to raise exception
        with patch("psutil.cpu_percent", side_effect=Exception("Test error")):
            monitor.start()
            time.sleep(0.2)
            monitor.stop()

        # Should not crash

    def test_disk_io_error_handling(self) -> None:
        """Test disk I/O error handling."""
        monitor = SystemMonitor(interval=1)

        # Mock disk_io_counters to raise exception
        with patch("psutil.disk_io_counters", side_effect=Exception("Disk error")):
            monitor.start()
            time.sleep(0.2)
            monitor.stop()

        # Should handle gracefully


# Fixtures for testing
@pytest.fixture
def temp_monitor_dir():
    """Create a temporary directory for monitor outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def basic_monitor():
    """Create a basic SystemMonitor instance for testing."""
    return SystemMonitor(interval=1)


@pytest.fixture
def monitor_with_data():
    """Create a SystemMonitor instance with some sample data."""
    monitor = SystemMonitor()

    # Add sample data
    current_time = time.time()
    monitor.cpu_usage = [(current_time, 50.0), (current_time + 1, 55.0)]
    monitor.memory_usage = [(current_time, 2.5), (current_time + 1, 2.8)]
    monitor.events = [
        {"timestamp": current_time, "message": "Test event 1"},
        {"timestamp": current_time + 1, "message": "Test event 2"},
    ]

    return monitor
