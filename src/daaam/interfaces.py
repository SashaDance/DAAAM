import multiprocessing as mp
from multiprocessing.synchronize import Event
import queue
import time
from typing import Any, Optional, Callable
import logging

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Union
import json
import traceback


class QueueWorkerProcessInterface(mp.Process, ABC):
	"""
	Blueprint for a multiprocessing Process that reads from an input queue
	and writes to an output queue.
	
	Supports two shutdown mechanisms:
	1. Shared stop_event (recommended for most cases)
	2. Poison pill (useful for specific scenarios)
	"""
	
	def __init__(
		self,
		input_queue: mp.Queue,
		output_queue: mp.Queue,
		stop_event: Optional[Event] = None,
		name: Optional[str] = None,
		timeout: float = 1.0,
		poison_pill: Any = None
	):
		"""
		Initialize the queue worker process.
		
		Args:
			input_queue: Queue to read items from
			output_queue: Queue to write processed items to
			stop_event: Shared event to signal process termination
			name: Optional name for the process
			timeout: Timeout for queue operations in seconds
			poison_pill: Optional sentinel value for queue-based termination
		"""
		super().__init__(name=name)
		self.input_queue = input_queue
		self.output_queue = output_queue
		self.timeout = timeout
		self.poison_pill = poison_pill
		
		# provided stop_event or create internal one
		self.stop_event = stop_event if stop_event is not None else mp.Event()
		
		self.logger = logging.getLogger(self.name or self.__class__.__name__)
	
	@classmethod
	def create_worker(cls, input_queue, output_queue, stop_event, config, ready_queue=None):
		"""Entry point for worker process."""
		worker = cls(
			input_queue,
			output_queue,
			stop_event,
			config,
			ready_queue
		)
		worker.run()

		
	@abstractmethod
	def process_item(self, item: Any) -> Any:
		"""
		Process a single item from the input queue.
				
		Args:
			item: The item to process
			
		Returns:
			The processed item (or None to skip writing to output queue)
		"""
		return item
	
	def run(self):
		"""
		Main process loop - reads from input queue, processes items,
		and writes to output queue.
		"""
		self.logger.info(f"Starting {self.name or 'worker'}")
		
		try:
			while not self.stop_event.is_set():
				try:
					# get item from input queue with timeout
					item = self.input_queue.get(timeout=self.timeout)
					
					# check for poison pill (optional)
					if self.poison_pill is not None and item is self.poison_pill:
						self.logger.info(f"Received poison pill, shutting down")
						break

					# process item
					try:
						result = self.process_item(item)
						
						# to output queue if not None
						if result is not None:
							if isinstance(result, list):
								for res in result:
									self.output_queue.put(res)
							else:
								self.output_queue.put(result)

					except Exception as e:
						self.logger.error(f"Error processing item: {e}", exc_info=True)
						
				except queue.Empty:
					# continue loop
					continue
					
		except KeyboardInterrupt:
			self.logger.info("Received keyboard interrupt")
		finally:
			self.logger.info(f"Shutting down {self.name or 'worker'}")
	
	def stop(self):
		"""Signal the process to stop."""
		self.stop_event.set()

