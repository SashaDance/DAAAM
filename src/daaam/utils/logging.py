"""
Unified logging system for the entire pipeline.
Provides consistent logging interface for both main process and worker processes.
"""
import logging
import multiprocessing as mp
import threading
import queue
import sys
import time
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime
import signal
import atexit

class bcolors:
	"""ANSI color codes for terminal output."""
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


class PipelineLogger:
	"""
	Base logger interface.
	All loggers in the system implement this interface.
	"""
	
	def info(self, message: str) -> None:
		"""Log info message."""
		raise NotImplementedError
		
	def warning(self, message: str) -> None:
		"""Log warning message."""
		raise NotImplementedError
		
	def debug(self, message: str) -> None:
		"""Log debug message."""
		raise NotImplementedError
		
	def error(self, message: str) -> None:
		"""Log error message."""
		raise NotImplementedError


class ConsoleLogger(PipelineLogger):
	"""Simple console logger with colored output (fallback logger)."""
	
	def info(self, message: str) -> None:
		print(f"[INFO] {message}")
		
	def warning(self, message: str) -> None:
		print(f"{bcolors.WARNING}[WARNING] {message}{bcolors.ENDC}")
		
	def debug(self, message: str) -> None:
		print(f"{bcolors.OKGREEN}[DEBUG] {message}{bcolors.ENDC}")
		
	def error(self, message: str) -> None:
		print(f"{bcolors.FAIL}[ERROR] {message}{bcolors.ENDC}")


class FileLogger(PipelineLogger):
	"""Simple file-based logger for worker processes."""
	
	def __init__(self, name: str, log_file: Path):
		"""
		Initialize file logger.
		
		Args:
			name: Logger name (e.g., "AssignmentWorker-0")
			log_file: Path to log file
		"""
		self.name = name
		self.log_file = log_file
		
		# Create logger
		self.logger = logging.getLogger(name)
		self.logger.setLevel(logging.DEBUG)
		self.logger.handlers.clear()
		
		# Create file handler
		file_handler = logging.FileHandler(log_file, mode='a')
		file_handler.setLevel(logging.DEBUG)
		
		# Set formatter
		formatter = logging.Formatter(
			'%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)
		file_handler.setFormatter(formatter)
		
		self.logger.addHandler(file_handler)
		self.logger.propagate = False
		
		# Log initialization
		self.logger.info(f"Logger initialized for {name} (PID: {mp.current_process().pid})")
	
	def info(self, message: str) -> None:
		self.logger.info(message)
		
	def warning(self, message: str) -> None:
		self.logger.warning(message)
		
	def debug(self, message: str) -> None:
		self.logger.debug(message)
		
	def error(self, message: str) -> None:
		self.logger.error(message)


class MainProcessLogger(PipelineLogger):
	"""Logger for main process that outputs to both console and file."""

	def __init__(self, name: str, log_processor: 'LogProcessor', use_async: bool = True):
		"""
		Initialize main process logger.

		Args:
			name: Logger name
			log_processor: LogProcessor instance for handling logs
			use_async: Whether to use async handler (default: True)
		"""
		self.name = name
		self.logger = logging.getLogger(name)
		self.logger.setLevel(logging.DEBUG)
		self.logger.handlers.clear()

		# Choose handler based on async flag
		if use_async:
			handler = AsyncLogProcessorHandler(log_processor.log_queue, batch_size=50, flush_interval=0.05)
		else:
			handler = LogProcessorHandler(log_processor.log_queue)
		handler.setLevel(logging.DEBUG)
		self.logger.addHandler(handler)
		self.logger.propagate = False
		
	def info(self, message: str) -> None:
		self.logger.info(message)
		
	def warning(self, message: str) -> None:
		self.logger.warning(message)
		
	def debug(self, message: str) -> None:
		self.logger.debug(message)
		
	def error(self, message: str) -> None:
		self.logger.error(message)


class LogProcessorHandler(logging.Handler):
	"""Handler that sends log records to a multiprocessing queue."""

	def __init__(self, log_queue: mp.Queue):
		super().__init__()
		self.log_queue = log_queue

	def emit(self, record):
		"""Send log record to queue."""
		try:
			# Add process and thread info
			record.process_name = mp.current_process().name
			record.thread_name = threading.current_thread().name
			self.log_queue.put_nowait(record)
		except queue.Full:
			pass  # Don't block if queue is full
		except Exception:
			pass  # Ignore any other errors


class AsyncLogProcessorHandler(logging.Handler):
	"""Asynchronous handler that batches log records before sending to multiprocessing queue."""

	def __init__(self, log_queue: mp.Queue, batch_size: int = 50, flush_interval: float = 0.1):
		"""
		Initialize async handler.

		Args:
			log_queue: Multiprocessing queue for log records
			batch_size: Number of records to batch before sending
			flush_interval: Maximum time (seconds) before flushing batch
		"""
		super().__init__()
		self.log_queue = log_queue
		self.batch_size = batch_size
		self.flush_interval = flush_interval

		# Thread-safe queue for buffering log records
		self.buffer_queue = queue.Queue(maxsize=10000)

		# Stats for monitoring
		self.dropped_count = 0
		self.submitted_count = 0

		# Background thread for processing
		self.processing_thread = None
		self.stop_event = threading.Event()

		# Start background thread
		self._start_processing_thread()

		# Register cleanup on exit
		atexit.register(self.close)

	def _start_processing_thread(self):
		"""Start the background processing thread."""
		if self.processing_thread is None or not self.processing_thread.is_alive():
			self.processing_thread = threading.Thread(
				target=self._process_buffer,
				name="AsyncLogProcessor",
				daemon=True
			)
			self.processing_thread.start()

	def _process_buffer(self):
		"""Background thread that processes buffered log records."""
		batch = []
		last_flush = time.time()

		while not self.stop_event.is_set():
			try:
				# Try to get a record with timeout
				timeout = max(0.01, self.flush_interval - (time.time() - last_flush))
				try:
					record = self.buffer_queue.get(timeout=timeout)
					batch.append(record)
				except queue.Empty:
					pass

				# Check if we should flush the batch
				current_time = time.time()
				should_flush = (
					len(batch) >= self.batch_size or
					(len(batch) > 0 and current_time - last_flush >= self.flush_interval)
				)

				if should_flush and batch:
					self._flush_batch(batch)
					batch = []
					last_flush = current_time

			except Exception:
				# Ignore exceptions to keep thread running
				pass

		# Final flush on shutdown
		if batch:
			self._flush_batch(batch)

	def _flush_batch(self, batch: List[logging.LogRecord]):
		"""Send a batch of records to the multiprocessing queue."""
		for record in batch:
			try:
				# Add process and thread info
				record.process_name = mp.current_process().name
				record.thread_name = threading.current_thread().name
				self.log_queue.put_nowait(record)
				self.submitted_count += 1
			except queue.Full:
				self.dropped_count += 1
			except Exception:
				self.dropped_count += 1

	def emit(self, record):
		"""Add log record to buffer (non-blocking)."""
		try:
			# Try to add to buffer without blocking
			self.buffer_queue.put_nowait(record)
		except queue.Full:
			# Buffer is full, drop the record
			self.dropped_count += 1
		except Exception:
			# Ignore any other errors
			pass

	def flush(self):
		"""Flush any buffered records."""
		# Signal the thread to flush by adding a sentinel
		try:
			# Wait a bit for the buffer to be processed
			time.sleep(0.2)
		except:
			pass

	def close(self):
		"""Clean shutdown of the handler."""
		if self.stop_event.is_set():
			return

		self.stop_event.set()
		if self.processing_thread and self.processing_thread.is_alive():
			self.processing_thread.join(timeout=1.0)

		# Log stats if any messages were dropped
		if self.dropped_count > 0:
			print(f"[AsyncLogHandler] Warning: Dropped {self.dropped_count} log messages", file=sys.stderr)


class LogProcessor:
	"""Processes log records from the main process and outputs to console/file."""

	def __init__(self, output_dir: Path, console_level: int = logging.INFO):
		self.output_dir = output_dir
		self.log_queue = mp.Queue(maxsize=10000)  # Increased queue size
		self.stop_event = mp.Event()
		self.processor_process: Optional[mp.Process] = None
		self.console_level = console_level

		# Formatters
		self.console_formatter = logging.Formatter(
			'[%(asctime)s] %(levelname)s [%(process_name)s:%(thread_name)s] %(name)s: %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)

		self.file_formatter = logging.Formatter(
			'%(asctime)s | %(levelname)-8s | %(process_name)-20s | %(thread_name)-15s | %(name)-25s | %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)
		
	def start(self) -> None:
		"""Start the log processor."""
		if self.processor_process is not None:
			return
			
		self.processor_process = mp.Process(
			target=self._process_logs,
			name="LogProcessor"
		)
		self.processor_process.start()
		
	def stop(self) -> None:
		"""Stop the log processor."""
		if self.processor_process is None:
			return

		self.stop_event.set()
		# Give more time for the processor to drain the queue
		self.processor_process.join(timeout=5.0)

		if self.processor_process.is_alive():
			# Still alive after 5 seconds, force terminate
			print(f"[LogProcessor] Warning: Force terminating log processor (queue may have {self.log_queue.qsize()} items)", file=sys.stderr)
			self.processor_process.terminate()
			self.processor_process.join(timeout=3.0)

		self.processor_process = None
		
	def _process_logs(self) -> None:
		"""Process log records in a separate process."""

		# Ignore SIGINT (Ctrl+C) in the log processor - we want graceful shutdown via stop_event
		signal.signal(signal.SIGINT, signal.SIG_IGN)

		# Create output directory
		self.output_dir.mkdir(parents=True, exist_ok=True)

		# Setup handlers with buffering
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(self.console_level)  # Set console log level
		console_handler.setFormatter(self.console_formatter)

		file_handlers = {}

		# Batch processing for efficiency
		batch_size = 10
		records_batch = []

		# Process logs until stop event is set
		while not self.stop_event.is_set():
			try:
				# Try to get multiple records quickly
				timeout = 0.1 if len(records_batch) == 0 else 0.01
				record = self.log_queue.get(timeout=timeout)
				records_batch.append(record)

				# Try to get more records without blocking
				while len(records_batch) < batch_size:
					try:
						record = self.log_queue.get_nowait()
						records_batch.append(record)
					except queue.Empty:
						break

				# Process the batch
				for record in records_batch:
					# Log to console (only if level meets threshold)
					if record.levelno >= self.console_level:
						console_handler.emit(record)

					# Log to process-specific file
					if hasattr(record, 'process_name'):
						process_name = record.process_name

						if process_name not in file_handlers:
							log_file = self.output_dir / f"{process_name}.log"
							file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
							file_handler.setFormatter(self.file_formatter)
							file_handlers[process_name] = file_handler

						file_handlers[process_name].emit(record)

				# Clear batch
				records_batch = []

			except queue.Empty:
				# Process any remaining records in batch
				if records_batch:
					for record in records_batch:
						if record.levelno >= self.console_level:
							console_handler.emit(record)
						if hasattr(record, 'process_name'):
							process_name = record.process_name
							if process_name not in file_handlers:
								log_file = self.output_dir / f"{process_name}.log"
								file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
								file_handler.setFormatter(self.file_formatter)
								file_handlers[process_name] = file_handler
							file_handlers[process_name].emit(record)
					records_batch = []
				continue
			except Exception as e:
				print(f"Error processing log: {e}", file=sys.stderr)
				records_batch = []  # Clear batch on error

		# CRITICAL: Drain any remaining messages in the queue after stop event
		print(f"[LogProcessor] Draining queue before shutdown...", file=sys.stderr)
		drained_count = 0
		max_drain_time = 3.0  # Maximum 3 seconds to drain
		drain_start = time.time()

		while (time.time() - drain_start) < max_drain_time:
			try:
				record = self.log_queue.get_nowait()  # Non-blocking get
				drained_count += 1

				# Process the record just like above
				if record.levelno >= self.console_level:
					console_handler.emit(record)

				if hasattr(record, 'process_name'):
					process_name = record.process_name

					if process_name not in file_handlers:
						log_file = self.output_dir / f"{process_name}.log"
						file_handler = logging.FileHandler(log_file)
						file_handler.setFormatter(self.file_formatter)
						file_handlers[process_name] = file_handler

					file_handlers[process_name].emit(record)

			except queue.Empty:
				# Queue is empty, we're done
				break
			except Exception as e:
				print(f"Error draining log: {e}", file=sys.stderr)

		if drained_count > 0:
			print(f"[LogProcessor] Drained {drained_count} messages from queue during shutdown", file=sys.stderr)

		# Flush all handlers before closing
		for handler in file_handlers.values():
			handler.flush()

		# Close all file handlers
		for handler in file_handlers.values():
			handler.close()


class LoggingManager:
	"""Manages logging for the main process."""

	def __init__(self, output_dir: Path, console_level: int = logging.INFO, use_async: bool = True):
		self.output_dir = output_dir
		self.log_processor = LogProcessor(output_dir, console_level)
		self.loggers: Dict[str, MainProcessLogger] = {}
		self._started = False
		self.use_async = use_async
		
	def start(self) -> None:
		"""Start the logging system."""
		if self._started:
			return
			
		self.log_processor.start()
		self._started = True
		
	def stop(self) -> None:
		"""Stop the logging system."""
		if not self._started:
			return

		self.log_processor.stop()
		self._started = False

	def flush_all_handlers(self) -> None:
		"""Flush all log handlers to ensure all messages are written to disk."""
		if not self._started:
			return

		# Flush all logger handlers
		for logger in self.loggers.values():
			for handler in logger.logger.handlers:
				handler.flush()
				# Close async handlers properly
				if isinstance(handler, AsyncLogProcessorHandler):
					handler.close()

		# Give the log processor time to process remaining messages
		time.sleep(1.0)  # Wait longer for queue to be processed

	def get_logger(self, name: str) -> MainProcessLogger:
		"""Get a logger for the given name."""
		if not self._started:
			self.start()

		if name not in self.loggers:
			self.loggers[name] = MainProcessLogger(name, self.log_processor, self.use_async)

		return self.loggers[name]


# Public API functions

def setup_main_logging(output_dir: Path, console_level: int = logging.INFO, use_async: bool = True) -> LoggingManager:
	"""
	Setup logging for the main process.

	Args:
		output_dir: Directory for log files
		console_level: Minimum logging level for console output (default: logging.INFO)
		               Use logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, or logging.CRITICAL
		use_async: Whether to use asynchronous logging (default: True)

	Returns:
		LoggingManager instance
	"""
	manager = LoggingManager(output_dir, console_level, use_async)
	manager.start()
	return manager


def setup_worker_logging(log_dir: str) -> PipelineLogger:
	"""
	Setup logging for a worker process.
	
	Args:
		log_dir: Directory path for log files (from config)
		
	Returns:
		Logger instance for the worker
	"""
	process_name = mp.current_process().name
	log_path = Path(log_dir) / f"{process_name}.log"
	
	# Ensure directory exists
	log_path.parent.mkdir(parents=True, exist_ok=True)
	
	return FileLogger(process_name, log_path)


def get_default_logger() -> PipelineLogger:
	"""Get a default console logger (for fallback/testing)."""
	return ConsoleLogger()


# Backward compatibility alias
DefaultLogger = ConsoleLogger