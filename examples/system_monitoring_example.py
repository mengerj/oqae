#!/usr/bin/env python3
"""
Example script demonstrating the SystemMonitor class for resource monitoring.

This script shows how to use the SystemMonitor to track system resources
during data processing operations.
"""

import time

import numpy as np

from omvqvae.utils.logging import get_logger
from omvqvae.utils.system_monitor import SystemMonitor


def simulate_data_processing():
    """Simulate some data processing work."""
    logger = get_logger(__name__)

    # Initialize monitor
    monitor = SystemMonitor(interval=2, logger=logger)

    try:
        # Start monitoring
        monitor.start()
        monitor.log_event("Starting data processing simulation")

        # Simulate loading large dataset
        logger.info("Simulating large dataset loading...")
        large_array = np.random.rand(10000, 1000)
        monitor.log_event("Large dataset loaded")

        # Simulate computation
        logger.info("Simulating computation...")
        for i in range(5):
            result = np.dot(large_array, large_array.T)
            time.sleep(2)
            monitor.log_event(f"Computation step {i+1}/5 completed")

        # Simulate memory-intensive operation
        logger.info("Simulating memory-intensive operation...")
        memory_intensive_data = [np.random.rand(1000, 1000) for _ in range(10)]
        monitor.log_event("Memory-intensive operation completed")

        # Clean up
        del large_array, result, memory_intensive_data
        monitor.log_event("Data processing simulation completed")

    finally:
        # Stop monitoring
        monitor.stop()

        # Show summary
        monitor.print_summary()

        # Save plots
        monitor.plot_metrics(save_dir="./monitoring_results")

        # Save metrics to CSV
        monitor.save_metrics("./monitoring_results")


if __name__ == "__main__":
    simulate_data_processing()
