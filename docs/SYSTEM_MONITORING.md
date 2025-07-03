# System Resource Monitoring

The OQAE project includes a comprehensive system monitoring utility for tracking resource usage during data processing and training operations.

## Features

- **CPU Monitoring**: Track CPU usage per core and total utilization
- **Memory Monitoring**: Monitor memory usage relative to baseline
- **Disk I/O Monitoring**: Track read/write rates
- **GPU Monitoring**: Monitor NVIDIA GPU usage and memory (optional)
- **Event Logging**: Log events with resource usage context
- **Visualization**: Generate plots of resource usage over time
- **Export**: Save metrics to CSV files

## Installation

The basic monitoring functionality is included with the standard installation:

```bash
pip install oqae
```

For GPU monitoring, install with monitoring extras:

```bash
pip install "oqae[monitoring]"
```

## Usage

### Basic Usage

```python
from omvqvae.utils.system_monitor import SystemMonitor

# Initialize monitor
monitor = SystemMonitor(interval=5)

# Start monitoring
monitor.start()

# Your data processing code here...
monitor.log_event("Data loading started")
# ... load data ...
monitor.log_event("Data loading completed")

# Stop monitoring
monitor.stop()

# View summary
monitor.print_summary()

# Save plots and metrics
monitor.plot_metrics(save_dir="./results")
```

### Advanced Usage

```python
from omvqvae.utils.system_monitor import SystemMonitor
from omvqvae.utils.logging import get_logger

# Custom logger and GPU monitoring
logger = get_logger(__name__)
monitor = SystemMonitor(
    interval=2,
    gpu_idx=0,  # Monitor specific GPU
    logger=logger
)

# Context manager style
class MonitoredProcess:
    def __init__(self):
        self.monitor = SystemMonitor(interval=1)

    def __enter__(self):
        self.monitor.start()
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        self.monitor.print_summary()

# Usage
with MonitoredProcess() as monitor:
    # Your processing code
    monitor.log_event("Processing started")
    # ... do work ...
    monitor.log_event("Processing completed")
```

## Integration with Weights & Biases

The SystemMonitor can be integrated with W&B for remote monitoring:

```python
import wandb
from omvqvae.utils.system_monitor import SystemMonitor

# Initialize W&B
wandb.init(project="oqae-monitoring")

# Custom callback for W&B logging
class WandBMonitor(SystemMonitor):
    def log_event(self, message):
        super().log_event(message)

        # Log current metrics to W&B
        summary = self.summarize()
        wandb.log({
            "cpu_usage": summary.get("cpu_usage_mean", 0),
            "memory_usage": summary.get("memory_usage_mean", 0),
            "event": message
        })

# Use with W&B
monitor = WandBMonitor(interval=10)
monitor.start()
# ... your code ...
monitor.stop()
```

## Configuration

### Monitor Parameters

- `interval`: Monitoring interval in seconds (default: 1)
- `gpu_idx`: GPU index(es) to monitor (default: None for all GPUs)
- `logger`: Custom logger instance (default: creates new logger)

### GPU Monitoring

GPU monitoring requires `pynvml` package:

```bash
pip install pynvml
```

Supports:
- NVIDIA GPUs via pynvml
- Apple GPUs (detection only, limited metrics)
- Multiple GPU monitoring
- CUDA_VISIBLE_DEVICES environment variable

## Output Files

When using `plot_metrics(save_dir="./results")`:

- `cpu_usage.png`: CPU usage over time
- `memory_usage.png`: Memory usage over time
- `disk_io.png`: Disk I/O rates over time
- `gpu_usage.png`: GPU usage over time (if available)
- `gpu_memory.png`: GPU memory usage over time (if available)

When using `save_metrics(save_dir="./results")`:

- `system_metrics_summary.csv`: Summary statistics

## Best Practices

1. **Choose appropriate intervals**:
   - Use 1-2 seconds for short operations
   - Use 5-10 seconds for long-running processes

2. **Monitor selectively**:
   - Specify GPU indices for multi-GPU systems
   - Use custom loggers for integration

3. **Log meaningful events**:
   - Mark major processing phases
   - Include context about what's being processed

4. **Resource cleanup**:
   - Always call `stop()` to clean up monitoring thread
   - Use context managers for automatic cleanup

## Troubleshooting

### Common Issues

1. **pynvml import errors**: Install with `pip install pynvml`
2. **GPU not detected**: Check CUDA installation and GPU visibility
3. **High monitoring overhead**: Increase interval or reduce logging frequency
4. **Memory errors**: Monitor may consume additional memory for data storage

### Performance Impact

The monitor runs in a separate thread and has minimal impact:
- CPU overhead: <1% for typical intervals
- Memory usage: ~1MB per hour of monitoring
- Disk I/O: Minimal, only during plot/CSV generation
