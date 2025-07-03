"""Advanced tests for SystemMonitor class to improve test coverage.

This test file focuses on areas that need better coverage:
- GPU detection and monitoring edge cases
- Event logging when no data is available
- Save metrics error handling and pandas fallback
- Plotting functionality with different scenarios
- Complex summarize scenarios with GPU metrics
- Error handling in monitoring thread
"""

import os
import tempfile
import time
from unittest.mock import Mock, patch

from omvqvae.utils.system_monitor import SystemMonitor


class TestSystemMonitorAdvanced:
    """Advanced tests to improve SystemMonitor test coverage."""

    def test_gpu_initialization_with_cuda_visible_devices(self):
        """Test GPU initialization with CUDA_VISIBLE_DEVICES environment variable."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", True):
                with patch("omvqvae.utils.system_monitor.pynvml") as mock_pynvml:
                    # Mock successful GPU detection
                    mock_pynvml.nvmlInit.return_value = None
                    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
                        Mock(),
                        Mock(),  # Two GPU handles
                    ]
                    mock_pynvml.nvmlDeviceGetName.side_effect = [
                        b"NVIDIA GPU 0",
                        b"NVIDIA GPU 1",
                    ]
                    mock_memory_info = Mock()
                    mock_memory_info.total = 8 * 1024**3  # 8GB
                    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

                    monitor = SystemMonitor()

                    assert monitor.gpu_available is True
                    assert monitor.gpu_type == "NVIDIA"
                    assert len(monitor.gpu_names) == 2
                    assert monitor.gpu_names[0] == "NVIDIA GPU 0"
                    assert monitor.gpu_names[1] == "NVIDIA GPU 1"

    def test_gpu_initialization_with_uuid_in_cuda_visible_devices(self):
        """Test GPU initialization with UUID in CUDA_VISIBLE_DEVICES."""
        mock_uuid = "GPU-12345678-1234-1234-1234-123456789012"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mock_uuid}):
            with patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", True):
                with patch("omvqvae.utils.system_monitor.pynvml") as mock_pynvml:
                    mock_pynvml.nvmlInit.return_value = None
                    mock_handle = Mock()
                    mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
                    mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GPU"
                    mock_memory_info = Mock()
                    mock_memory_info.total = 8 * 1024**3
                    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

                    monitor = SystemMonitor()

                    assert monitor.gpu_available is True
                    mock_pynvml.nvmlDeviceGetHandleByUUID.assert_called_with(mock_uuid)

    def test_apple_gpu_detection_on_macos(self):
        """Test Apple GPU detection on macOS when NVIDIA fails."""
        with patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", True):
            with patch("omvqvae.utils.system_monitor.pynvml") as mock_pynvml:
                with patch("platform.system", return_value="Darwin"):
                    with patch("platform.platform", return_value="macOS-13.0"):
                        # Make NVIDIA detection fail
                        mock_pynvml.nvmlInit.side_effect = Exception("No NVIDIA GPU")

                        monitor = SystemMonitor()

                        assert monitor.gpu_available is False
                        assert monitor.gpu_type == "Apple"
                        assert monitor.gpu_name == "Apple Integrated GPU"

    def test_monitor_nvidia_gpu_error_handling(self):
        """Test error handling in _monitor_nvidia_gpu method."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_handles = [Mock()]
        monitor.gpu_usage = [[]]
        monitor.gpu_memory_usage = [[]]

        with patch("omvqvae.utils.system_monitor.pynvml") as mock_pynvml:
            # Make GPU monitoring fail
            mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = Exception(
                "GPU error"
            )

            # Should not raise an exception
            monitor._monitor_nvidia_gpu(time.time())

            # Should have logged an error
            assert len(monitor.gpu_usage[0]) == 0  # No data added due to error

    def test_monitor_apple_gpu(self):
        """Test _monitor_apple_gpu method."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "Apple"

        timestamp = time.time()
        monitor._monitor_apple_gpu(timestamp)

        # Apple GPU monitoring adds None values
        assert len(monitor.gpu_usage) == 1
        assert len(monitor.gpu_memory_usage) == 1
        assert monitor.gpu_usage[0] == (timestamp, None)
        assert monitor.gpu_memory_usage[0] == (timestamp, None)

    def test_log_event_with_no_data_collected(self):
        """Test log_event when no monitoring data has been collected yet."""
        monitor = SystemMonitor()

        # Should not raise an exception even with no data
        monitor.log_event("Test event with no data")

        assert len(monitor.events) == 1
        assert monitor.events[0]["message"] == "Test event with no data"

    def test_log_event_with_gpu_but_no_gpu_data(self):
        """Test log_event when GPU is available but no GPU data collected."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = ["Test GPU"]
        monitor.gpu_usage = [[]]  # Empty GPU data
        monitor.gpu_memory_usage = [[]]

        monitor.log_event("Test event with GPU but no data")

        assert len(monitor.events) == 1

    def test_log_event_with_apple_gpu(self):
        """Test log_event with Apple GPU (non-NVIDIA)."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "Apple"
        monitor.gpu_name = "Apple Integrated GPU"

        monitor.log_event("Test event with Apple GPU")

        assert len(monitor.events) == 1

    def test_save_metrics_without_pandas(self):
        """Test save_metrics when pandas is not available."""
        monitor = SystemMonitor()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the instance logger to avoid logging interference
            with patch.object(monitor, "logger") as mock_logger:
                # Mock only pandas import
                import builtins

                real_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "pandas":
                        raise ImportError("No pandas")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    # Should not raise an exception
                    monitor.save_metrics(temp_dir)

                    # No CSV file should be created
                    csv_path = os.path.join(temp_dir, "sys_metrics.csv")
                    assert not os.path.exists(csv_path)

                    # Should have logged warning about pandas not being available
                    mock_logger.warning.assert_called_once_with(
                        "pandas not available - skipping CSV export"
                    )

    def test_save_metrics_file_error(self):
        """Test save_metrics when file operations fail."""
        monitor = SystemMonitor()

        with patch("pandas.DataFrame") as mock_df_class:
            mock_df = Mock()
            mock_df.to_csv.side_effect = OSError("Permission denied")
            mock_df_class.return_value = mock_df

            # Should not raise an exception
            monitor.save_metrics("/invalid/path/that/should/not/exist")

    def test_save_alias_method(self):
        """Test that save() is an alias for save_metrics()."""
        monitor = SystemMonitor()

        with patch.object(monitor, "save_metrics") as mock_save_metrics:
            monitor.save("test_dir")
            mock_save_metrics.assert_called_once_with("test_dir")

    def test_plot_metrics_with_save_dir(self):
        """Test plot_metrics functionality when saving to directory."""
        monitor = SystemMonitor()

        # Add some test data
        monitor.cpu_per_core = [(time.time(), 50.0), (time.time() + 1, 60.0)]
        monitor.memory_usage = [(time.time(), 2.0), (time.time() + 1, 2.5)]
        monitor.disk_io = [(time.time(), 10.0, 5.0), (time.time() + 1, 15.0, 8.0)]
        monitor.events = [{"timestamp": time.time(), "message": "Test event"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("omvqvae.utils.system_monitor.plt") as mock_plt:
                mock_plt.figure.return_value = Mock()
                mock_plt.plot.return_value = Mock()
                mock_plt.savefig.return_value = Mock()
                mock_plt.close.return_value = Mock()

                monitor.plot_metrics(save_dir=temp_dir)

                # Should have called savefig for each plot
                assert mock_plt.savefig.call_count >= 3  # CPU, memory, disk I/O
                mock_plt.close.assert_called()

    def test_plot_metrics_without_save_dir(self):
        """Test plot_metrics functionality when showing plots."""
        monitor = SystemMonitor(show_plots=False)

        # Add minimal test data
        monitor.cpu_per_core = [(time.time(), 50.0)]
        monitor.memory_usage = [(time.time(), 2.0)]
        monitor.events = []

        with patch("omvqvae.utils.system_monitor.plt") as mock_plt:
            mock_plt.figure.return_value = Mock()
            mock_plt.plot.return_value = Mock()
            mock_plt.close.return_value = Mock()

            monitor.plot_metrics(save_dir=None)

            # Should have called close instead of show when show_plots=False
            mock_plt.close.assert_called()

    def test_plot_metrics_with_gpu_data(self):
        """Test plot_metrics with GPU data."""
        monitor = SystemMonitor(show_plots=False)
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = ["Test GPU"]

        # Add GPU data
        timestamp = time.time()
        monitor.gpu_usage = [[(timestamp, 80.0), (timestamp + 1, 85.0)]]
        monitor.gpu_memory_usage = [[(timestamp, 4.0), (timestamp + 1, 4.5)]]

        with patch("omvqvae.utils.system_monitor.plt") as mock_plt:
            monitor.plot_metrics(save_dir=None)

            # Should include GPU plots
            assert mock_plt.figure.call_count >= 2  # At least GPU usage and memory

    def test_summarize_with_nvidia_gpu_data(self):
        """Test summarize method with NVIDIA GPU data."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = ["Test GPU"]
        monitor.gpu_total_memory = [8.0]

        # Add GPU monitoring data
        timestamp = time.time()
        monitor.gpu_usage = [[(timestamp, 80.0), (timestamp + 1, 85.0)]]
        monitor.gpu_memory_usage = [[(timestamp, 4.0), (timestamp + 1, 4.5)]]

        summary = monitor.summarize()

        assert "gpu_metrics" in summary
        assert len(summary["gpu_metrics"]) == 1
        gpu_metric = summary["gpu_metrics"][0]
        assert gpu_metric["name"] == "Test GPU"
        assert gpu_metric["usage_mean"] == 82.5
        assert gpu_metric["usage_max"] == 85.0
        assert gpu_metric["memory_usage_mean"] == 4.25
        assert gpu_metric["memory_usage_max"] == 4.5
        assert gpu_metric["total_memory"] == 8.0
        assert gpu_metric["gpu_id"] == 0

    def test_summarize_with_apple_gpu_data(self):
        """Test summarize method with Apple GPU (non-NVIDIA) data."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "Apple"
        monitor.gpu_name = "Apple Integrated GPU"
        monitor.gpu_total_memory = [16.0]

        summary = monitor.summarize()

        assert "gpu_metrics" in summary
        assert len(summary["gpu_metrics"]) == 1
        gpu_metric = summary["gpu_metrics"][0]
        assert gpu_metric["name"] == "Apple Integrated GPU"
        assert gpu_metric["usage_mean"] is None
        assert gpu_metric["usage_max"] is None
        assert gpu_metric["memory_usage_mean"] is None
        assert gpu_metric["memory_usage_max"] is None

    def test_summarize_with_empty_gpu_data(self):
        """Test summarize method when GPU data lists are empty."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = ["Test GPU"]
        monitor.gpu_total_memory = [8.0]
        monitor.gpu_usage = [[]]  # Empty data
        monitor.gpu_memory_usage = [[]]

        summary = monitor.summarize()

        gpu_metric = summary["gpu_metrics"][0]
        assert "usage_mean" not in gpu_metric  # Should not be added if no data

    def test_print_summary_with_gpu_metrics(self):
        """Test print_summary method with GPU metrics."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = ["Test GPU"]
        monitor.gpu_total_memory = [8.0]

        # Add some data to get meaningful summary
        timestamp = time.time()
        monitor.cpu_usage = [(timestamp, 50.0)]
        monitor.cpu_per_core = [(timestamp, 25.0)]
        monitor.memory_usage = [(timestamp, 2.0)]
        monitor.disk_io = [(timestamp, 10.0, 5.0)]
        monitor.gpu_usage = [[(timestamp, 80.0)]]
        monitor.gpu_memory_usage = [[(timestamp, 4.0)]]

        # Capture print output
        with patch("builtins.print") as mock_print:
            monitor.print_summary()

            # Should have printed GPU metrics
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            gpu_section_found = any("GPU Metrics:" in call for call in print_calls)
            assert gpu_section_found

    def test_print_summary_without_gpu(self):
        """Test print_summary method when no GPU is available."""
        monitor = SystemMonitor()
        monitor.gpu_available = False

        # Add minimal data
        timestamp = time.time()
        monitor.cpu_usage = [(timestamp, 50.0)]
        monitor.cpu_per_core = [(timestamp, 25.0)]
        monitor.memory_usage = [(timestamp, 2.0)]
        monitor.disk_io = [(timestamp, 10.0, 5.0)]

        with patch("builtins.print") as mock_print:
            monitor.print_summary()

            # Should have printed no GPU message
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            no_gpu_found = any(
                "No supported GPU detected" in call for call in print_calls
            )
            assert no_gpu_found

    def test_monitoring_thread_exception_handling(self):
        """Test exception handling in the monitoring thread."""
        monitor = SystemMonitor()

        # Mock psutil.Process to raise an exception
        with patch("psutil.Process") as mock_process:
            mock_process.side_effect = Exception("Process error")

            # Start and stop monitoring
            monitor.start()
            time.sleep(0.2)  # Let it try to monitor
            monitor.stop()

            # Should have handled the exception gracefully
            assert not monitor._thread.is_alive()

    def test_stop_with_gpu_shutdown_error(self):
        """Test stop method when GPU shutdown fails."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"

        with patch("omvqvae.utils.system_monitor._PYNVML_AVAILABLE", True):
            with patch("omvqvae.utils.system_monitor.pynvml") as mock_pynvml:
                mock_pynvml.nvmlShutdown.side_effect = Exception("Shutdown error")

                # Should not raise an exception
                monitor.stop()

    def test_cpu_and_memory_metrics_with_empty_data(self):
        """Test summarize with empty CPU and memory data."""
        monitor = SystemMonitor()

        # Clear all data
        monitor.cpu_usage = []
        monitor.cpu_per_core = []
        monitor.memory_usage = []
        monitor.disk_io = []

        summary = monitor.summarize()

        assert summary["cpu_usage_mean"] == 0.0
        assert summary["cpu_usage_max"] == 0.0
        assert summary["core_usage_mean"] == 0.0
        assert summary["core_usage_max"] == 0.0
        assert summary["memory_usage_mean"] == 0.0
        assert summary["memory_usage_max"] == 0.0

    def test_disk_io_metrics_with_empty_data(self):
        """Test summarize with empty disk I/O data."""
        monitor = SystemMonitor()
        monitor.disk_io = []

        summary = monitor.summarize()

        assert summary["disk_read_mb_s_mean"] == 0.0
        assert summary["disk_read_mb_s_max"] == 0.0
        assert summary["disk_write_mb_s_mean"] == 0.0
        assert summary["disk_write_mb_s_max"] == 0.0

    def test_monitoring_thread_cleanup_on_stop(self):
        """Test that monitoring thread is properly cleaned up on stop."""
        monitor = SystemMonitor()

        # Start monitoring
        monitor.start()
        assert monitor._thread.is_alive()

        # Stop monitoring
        monitor.stop()

        # Thread should be stopped
        assert not monitor._thread.is_alive()

    def test_multiple_gpu_handling(self):
        """Test handling of multiple GPUs."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = ["GPU 0", "GPU 1"]
        monitor.gpu_total_memory = [8.0, 16.0]

        # Add data for multiple GPUs
        timestamp = time.time()
        monitor.gpu_usage = [[(timestamp, 70.0)], [(timestamp, 80.0)]]  # GPU 0  # GPU 1
        monitor.gpu_memory_usage = [
            [(timestamp, 3.0)],  # GPU 0
            [(timestamp, 8.0)],  # GPU 1
        ]

        summary = monitor.summarize()

        assert len(summary["gpu_metrics"]) == 2
        assert summary["gpu_metrics"][0]["name"] == "GPU 0"
        assert summary["gpu_metrics"][1]["name"] == "GPU 1"
        assert summary["total_gpu_memory"] == 24.0  # 8 + 16

    def test_bytes_name_decoding_in_gpu_metrics(self):
        """Test that GPU names are properly decoded if they come as bytes."""
        monitor = SystemMonitor()
        monitor.gpu_available = True
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_names = [b"Test GPU"]  # Bytes instead of string
        monitor.gpu_total_memory = [8.0]
        monitor.gpu_usage = [[(time.time(), 80.0)]]
        monitor.gpu_memory_usage = [[(time.time(), 4.0)]]

        summary = monitor.summarize()

        gpu_metric = summary["gpu_metrics"][0]
        assert gpu_metric["name"] == "Test GPU"  # Should be decoded to string

    def test_context_manager_protocol(self):
        """Test that SystemMonitor can be used as a context manager."""
        with patch.object(SystemMonitor, "start") as mock_start:
            with patch.object(SystemMonitor, "stop") as mock_stop:
                # Test the actual context manager implementation
                monitor = SystemMonitor()

                with monitor:
                    # Start should be called on entering context
                    mock_start.assert_called_once()

                # Stop should be called when exiting context
                mock_stop.assert_called_once()
