from typing import Optional, Dict, List, Any
import multiprocessing as mp
from multiprocessing.synchronize import Event
import threading
import queue
import time

from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import WorkerConfig


class AssignmentService:
	"""Service for handling frame assignment to grounding workers."""
	
	def __init__(self, config: WorkerConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.workers: List[mp.Process] = []
		self.stop_event: Optional[Event] = None
		self.assignment_task_queue: Optional[mp.Queue] = None
		self.selected_groups_queue: Optional[mp.Queue] = None
		self.worker_ready_queue: Optional[mp.Queue] = None
		self._running = False
	
	def start(self, assignment_task_queue: mp.Queue, selected_groups_queue: mp.Queue, pipeline_config, log_dir=None) -> None:
		"""Start assignment worker processes."""
		if self._running:
			self.logger.warning("Assignment service already running")
			return
		
		self.assignment_task_queue = assignment_task_queue
		self.selected_groups_queue = selected_groups_queue
		self.stop_event = mp.Event()
		self.worker_ready_queue = mp.Queue()
		
		# Get worker-specific configuration - ensure proper parameter passing
		if hasattr(pipeline_config, 'get_worker_config'):
			worker_config = pipeline_config.get_worker_config(self.config.assignment_worker)
		else:
			# Fallback: construct worker config manually
			worker_config = {
				"assignment_worker": self.config.assignment_worker,
				"min_obs_per_track": self.config.assignment_config.min_obs_per_track,
				"N_masks_per_batch": self.config.assignment_config.N_masks_per_batch,
			}
		
		# Add log_dir to worker config if provided
		if log_dir:
			worker_config['log_dir'] = log_dir
		
		# Start assignment worker processes
		for i in range(self.config.num_assignment_workers):
			worker = mp.Process(
				target=self._get_assignment_worker_process(),
				args=(
					self.assignment_task_queue,
					self.selected_groups_queue,
					self.stop_event,
					worker_config,
					self.worker_ready_queue,
				),
				name=f"AssignmentWorker-{i}"
			)
			worker.start()
			self.workers.append(worker)
		
		self._running = True
		self.logger.info(f"Started {self.config.num_assignment_workers} {self.config.assignment_worker} assignment workers")
	
	def stop(self) -> None:
		"""Stop assignment worker processes."""
		if not self._running:
			return
		
		if self.stop_event:
			self.stop_event.set()
		
		# Wait for workers to finish
		for worker in self.workers:
			worker.join(timeout=5.0)
			if worker.is_alive():
				self.logger.warning(f"Force terminating assignment worker {worker.name}")
				worker.terminate()
				worker.join()
		
		self.workers.clear()
		self._running = False
		self.logger.info("Stopped assignment workers")
	
	def _get_assignment_worker_process(self):
		"""Get assignment worker process function dynamically."""
		assignment_worker_name = self.config.assignment_worker
		
		if assignment_worker_name == "min_frames":
			from .workers.min_frames import MinFramesAssignmentWorker
			return MinFramesAssignmentWorker.create_worker
		elif assignment_worker_name == "min_frames_max_size":
			from .workers.min_frames_max_size import MinFramesMaxSizeAssignmentWorker
			return MinFramesMaxSizeAssignmentWorker.create_worker
		else:
			self.logger.error(f"Unknown assignment worker: {assignment_worker_name}")
			raise ValueError(f"Unknown assignment worker: {assignment_worker_name}")
	
	def is_running(self) -> bool:
		"""Check if assignment service is running."""
		return self._running
	
	def get_worker_health(self) -> Dict[str, Any]:
		"""Get health status of assignment workers."""
		health_status = {
			"running": self._running,
			"num_workers": len(self.workers),
			"workers": []
		}
		
		for worker in self.workers:
			health_status["workers"].append({
				"name": worker.name,
				"pid": worker.pid,
				"is_alive": worker.is_alive(),
				"is_ready": self.check_worker_ready(worker.name),
				"exitcode": worker.exitcode
			})
		
		return health_status
	
	def check_worker_ready(self, worker_name: str) -> bool:
		"""Check if a specific worker has reported ready."""
		if not self.worker_ready_queue:
			return False
			
		# Check for ready messages without blocking
		ready_workers = set()
		try:
			while True:
				ready_name = self.worker_ready_queue.get_nowait()
				ready_workers.add(ready_name)
		except:
			pass
			
		# Put messages back for future checks
		for name in ready_workers:
			self.worker_ready_queue.put(name)
			
		return worker_name in ready_workers