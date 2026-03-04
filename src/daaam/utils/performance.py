from time import perf_counter_ns
import logging
import time
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar, Optional, Dict, List
from pathlib import Path
import threading
import csv
import numpy as np
import torch

try:
    from typing import ParamSpec  # available in Python 3.10+
except ImportError:
    from typing_extensions import ParamSpec  # python 3.8 , 3.9

R = TypeVar("R")
P = ParamSpec("P")


class PerformanceTracker:
	"""Thread-safe performance statistics collector for runtime profiling."""

	def __init__(self):
		self._measurements: Dict[str, List[float]] = {}
		self._lock = threading.Lock()

	def record(self, operation: str, duration_ns: int) -> None:
		"""Record a timing measurement in nanoseconds (thread-safe)."""
		with self._lock:
			if operation not in self._measurements:
				self._measurements[operation] = []
			# Convert nanoseconds to milliseconds for storage
			self._measurements[operation].append(duration_ns / 1_000_000.0)

	def get_statistics(self) -> Dict[str, Dict[str, float]]:
		"""Compute statistics for all recorded operations."""
		statistics = {}

		with self._lock:
			# Copy data to minimize lock time
			measurements_copy = {op: list(times) for op, times in self._measurements.items()}

		for operation, times in measurements_copy.items():
			if not times:
				continue

			times_array = np.array(times)

			statistics[operation] = {
				'count': len(times),
				'mean_ms': float(np.mean(times_array)),
				'std_ms': float(np.std(times_array)),
				'min_ms': float(np.min(times_array)),
				'max_ms': float(np.max(times_array)),
				'total_ms': float(np.sum(times_array)),
				'p50_ms': float(np.percentile(times_array, 50)),
				'p95_ms': float(np.percentile(times_array, 95)),
				'p99_ms': float(np.percentile(times_array, 99))
			}

		return statistics

	def export_csv(self, output_path: Path) -> None:
		"""Export statistics to CSV file."""
		statistics = self.get_statistics()

		if not statistics:
			logging.warning("No performance statistics to export")
			return

		# Ensure parent directory exists
		output_path.parent.mkdir(parents=True, exist_ok=True)

		# Write CSV with consistent column order
		fieldnames = ['operation', 'count', 'mean_ms', 'std_ms', 'min_ms', 'max_ms',
		              'total_ms', 'p50_ms', 'p95_ms', 'p99_ms']

		with open(output_path, 'w', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()

			# Sort by operation name for consistent output
			for operation in sorted(statistics.keys()):
				row = {'operation': operation}
				row.update(statistics[operation])
				writer.writerow(row)


class performance_measure:
    """
    A class that measures the execution time of a code block.
    Usage:
    with performance_measure("name of code block"):
        # code block

    Avoid usage with parallel or async code
    """

    def __init__(self, name, logger_print = None, tracker: Optional[PerformanceTracker] = None) -> None:
        self.name = name
        if logger_print is None:
            self.logger_print = logging.getLogger(__name__).info
        else:
            self.logger_print = logger_print
        self.tracker = tracker

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, *args):
        self.end_time = perf_counter_ns()
        self.duration = self.end_time - self.start_time

        self.logger_print(f"{self.name} - execution time: {(self.duration)/1000000:.2f} ms")

        # Record to tracker if provided
        if self.tracker is not None:
            self.tracker.record(self.name, self.duration)


class performance_measure_torch:
    """
    Time a code block, reporting both total wall‐time and (if on CUDA)
    the GPU‐only elapsed time (via CUDA events).
    """
    def __init__(self, name, logger_print=None, device=None, synchronize=True):
        self.name        = name
        self.synchronize = synchronize

        self.logger = logger_print or logging.getLogger(__name__).info

        # device
        if device is None and torch.cuda.is_available():
            device = torch.device('cuda', torch.cuda.current_device())
        self.device = device

        self.use_cuda_events = (
            self.synchronize
            and isinstance(self.device, torch.device)
            and self.device.type == 'cuda'
        )

        if self.use_cuda_events:
            # create events on correct device
            with torch.cuda.device(self.device):
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.use_cuda_events:
            # flush prior work on device
            torch.cuda.synchronize(self.device)
            self.start_event.record()
        self.start_time = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize(self.device)

            gpu_ms = self.start_event.elapsed_time(self.end_event)
            self.logger(f"{self.name} - GPU time: {gpu_ms:.2f} ms")

        else:
            # if we still want to flush CUDA so CPU timer doesn't
            # return too early
            if self.synchronize and torch.cuda.is_available():
                torch.cuda.synchronize()

        end_time = perf_counter_ns()
        wall_ms = (end_time - self.start_time) / 1_000_000
        self.logger(f"{self.name} - Wall time: {wall_ms:.2f} ms")


def time_execution_sync(
    additional_text: str = "", logger: Callable = logging.debug
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger(f"{additional_text} Execution time: {execution_time:.2f} seconds")
            return result

        return wrapper

    return decorator


def time_execution_async(
    additional_text: str = "", logger: Callable = logging.debug
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]]
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger(f"{additional_text} Execution time: {execution_time:.2f} seconds")
            return result

        return wrapper

    return decorator


def singleton(cls):
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper
