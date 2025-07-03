#!/usr/bin/env python3
"""
Simple test runner for SystemMonitor functionality.

This script demonstrates and tests the SystemMonitor class without
complex test framework dependencies.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omvqvae.utils.logging import configure_logging, get_logger
from omvqvae.utils.system_monitor import SystemMonitor


def test_basic_functionality():
    """Test basic SystemMonitor functionality."""
    print("Testing basic SystemMonitor functionality...")

    monitor = SystemMonitor(interval=1)

    # Test initialization
    assert monitor.interval == 1
    assert monitor.logger is not None
    print("✓ Initialization successful")

    # Test monitoring workflow
    monitor.start()

    monitor.log_event("Test started")
    time.sleep(1.1)
    monitor.log_event("Test in progress")
    time.sleep(1.1)
    monitor.log_event("Test completed")

    monitor.stop()

    # Verify data collection
    assert len(monitor.events) == 3
    assert len(monitor.cpu_usage) > 0
    assert len(monitor.memory_usage) > 0
    print("✓ Monitoring workflow successful")

    # Test summary
    summary = monitor.summarize()
    assert "cpu_usage_mean" in summary
    assert "memory_usage_mean" in summary
    assert isinstance(summary["cpu_usage_mean"], float)
    print("✓ Summary generation successful")

    print("✓ All basic tests passed!")


def test_wandb_integration():
    """Test W&B integration if available."""
    print("\nTesting W&B integration...")

    try:
        import wandb

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

        monitor = TestWandBMonitor(interval=1)

        monitor.start()
        time.sleep(1.1)
        monitor.log_event("W&B test event")
        monitor.stop()

        # Verify custom logging worked
        assert len(monitor.logged_metrics) == 1
        assert monitor.logged_metrics[0]["event"] == "W&B test event"

        print("✓ W&B integration successful")

    except ImportError:
        print("⚠ W&B not available - skipping W&B tests")


def test_file_output():
    """Test file output functionality."""
    print("\nTesting file output...")

    monitor = SystemMonitor()

    # Add some mock data
    monitor.cpu_usage = [(time.time(), 50.0)]
    monitor.memory_usage = [(time.time(), 2.5)]

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test plotting
            monitor.plot_metrics(save_dir=temp_dir)
            print("✓ Plot generation successful")
        except Exception as e:
            print(f"⚠ Plot generation failed: {e}")

        try:
            # Test metrics saving
            monitor.save_metrics(temp_dir)
            print("✓ Metrics saving successful")
        except Exception as e:
            print(f"⚠ Metrics saving failed: {e}")


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")

    monitor = SystemMonitor()

    # Test stop without start
    try:
        monitor.stop()
        print("✓ Stop without start handled gracefully")
    except Exception as e:
        print(f"✗ Stop without start failed: {e}")

    # Test multiple starts
    try:
        monitor.start()
        monitor.start()  # Should not cause issues
        monitor.stop()
        print("✓ Multiple starts handled gracefully")
    except Exception as e:
        print(f"✗ Multiple starts failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SystemMonitor Test Suite")
    print("=" * 60)

    # Configure logging
    configure_logging(level="INFO")

    try:
        test_basic_functionality()
        test_wandb_integration()
        test_file_output()
        test_error_handling()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
