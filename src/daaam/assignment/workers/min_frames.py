import numpy as np
from typing import Dict, List, Tuple, Union, Set, Optional

from daaam.assignment.interfaces import AssignmentWorkerInterface
from daaam.utils.performance import performance_measure
from daaam.utils.logging import PipelineLogger
from daaam.grounding.models import ObjectAnnotation
from daaam.tracking.models import SimplifiedTrack
from daaam.assignment.models import AssignmentTask, SelectedGroup


class MinFramesAssignmentWorker(AssignmentWorkerInterface):
	"""DAM-specific assignment worker that minimizes frames and covers all unlabeled tracks."""
	
	def __init__(self, assignment_task_queue, selected_groups_queue, stop_event, config, name=None):
		super().__init__(assignment_task_queue, selected_groups_queue, stop_event, config, name)
		self.min_obs_per_track = config.get("min_obs_per_track", 6)
		self.N_masks_per_batch = config.get("N_masks_per_batch", 64)
		self.worker_logger.info(f"Initialized with min_obs_per_track: {self.min_obs_per_track}, N_masks_per_batch: {self.N_masks_per_batch}")
	
	def process_assignment_task(self, assignment_task: AssignmentTask) -> Optional[List[SelectedGroup]]:
		"""Process assignment task and return selected groups."""
		frame_history = assignment_task.track_history
		corrections = assignment_task.corrections
		frame_dims = assignment_task.frame_dims
		object_labels = assignment_task.object_labels
		prompted_track_ids = set(assignment_task.prompted_track_ids)
		start_frame_count = assignment_task.start_frame_count
		
		self.worker_logger.info(f"Data unpacked: History length={len(frame_history)}, StartFrame={start_frame_count}")
		
		if not frame_history:
			self.worker_logger.warning("Received empty frame history.")
			return None
		
		return self.select_prompt_images(
			frame_history,
			corrections,
			frame_dims,
			object_labels,
			prompted_track_ids,
			start_frame_count
		)
	
	def select_prompt_images(
		self,
		frame_history: List[List[SimplifiedTrack]],
		corrections: Dict[int, ObjectAnnotation],
		frame_dims: Tuple[int, int],
		object_labels: Dict[int, int],
		prompted_track_ids: Set[int],
		start_frame_count: int
	) -> Optional[List[SelectedGroup]]:
		"""DAM-specific selection that minimizes frames and covers all unlabeled tracks without overlap constraints."""
		
		with performance_measure(
			"[DAMAssignmentWorker]: setup optimization problem", self.worker_logger.debug
		):
			self.worker_logger.debug("Starting DAM optimization setup...")
			
			if not frame_dims:
				self.worker_logger.error("Frame dimensions missing!")
				return None
			frame_height, frame_width = frame_dims

			# Build track observations and filter qualified tracks
			tracks_by_frame = {}
			track_observations = {}
			valid_tracks = []
			for frame_idx, record in enumerate(frame_history):
				valid_tracks_frame = []
				for track in record:
					if isinstance(track, SimplifiedTrack):
						valid_tracks_frame.append(track)
						track_id = track.id
						if track_id not in track_observations:
							track_observations[track_id] = set()
						track_observations[track_id].add(frame_idx)
				if valid_tracks_frame:
					tracks_by_frame[frame_idx] = valid_tracks_frame
					valid_tracks.extend(valid_tracks_frame)

			# filter qualified tracks, only tracks that haven't been previously labeled
			def is_track_qualified(track):
				tid = track.id
				semantic_id = object_labels.get(tid, -1)

				# if correction for this semantic_id, check if "unknown"
				if semantic_id in corrections:
					correction = corrections[semantic_id]
					# allow re-prompting if correction label "unknown"
					is_unknown = correction.semantic_label.lower() == "unknown"
					
					if not is_unknown:
						return False  # skip tracks with non-unknown corrections
					
				if not track.depth_valid:
					return False
				# other conditions
				return (len(track_observations[tid]) >= self.min_obs_per_track 
						and tid not in prompted_track_ids)

			qualified_track_ids = {
				track.id for track in valid_tracks if is_track_qualified(track)
			}

			self.worker_logger.info(f"Qualified tracks count: {len(qualified_track_ids)}")

			if not qualified_track_ids:
				self.worker_logger.warning("No qualified tracks found. Skipping optimization.")
				return None

			# DAM assignment logic: minimize frames while covering all tracks
			self.worker_logger.debug("Using DAM assignment logic (greedy frame selection)...")
			
			uncovered_track_ids = qualified_track_ids.copy()
			selected_groups = []
			
			while uncovered_track_ids:
				# frame that covers the most uncovered tracks
				best_local_frame_idx = None
				best_tracks_in_frame = []
				best_coverage = 0
				
				for frame_idx, frame_tracks in tracks_by_frame.items():
					tracks_in_frame = [t.id for t in frame_tracks if t.id in uncovered_track_ids]
					
					if len(tracks_in_frame) > best_coverage:
						best_coverage = len(tracks_in_frame)
						best_local_frame_idx = frame_idx
						best_tracks_in_frame = tracks_in_frame
				
				if best_local_frame_idx is None or not best_tracks_in_frame:
					self.worker_logger.debug("No more frames can cover remaining tracks.")
					break
				
				# split tracks into batches of N_masks_per_batch (mostly len(best_tracks_in_frame) <= N_masks_per_batch anyway, just a safeguard)
				for i in range(0, len(best_tracks_in_frame), self.N_masks_per_batch):
					batch_tracks = best_tracks_in_frame[i:i + self.N_masks_per_batch]
					global_frame_id = start_frame_count + int(best_local_frame_idx)
					selected_groups.append(SelectedGroup(
						global_frame_id=global_frame_id,
						track_ids=batch_tracks,
					))
					self.worker_logger.debug(f"Added batch: frame_idx={best_local_frame_idx}, tracks={len(batch_tracks)}")
				
				# remove covered tracks
				uncovered_track_ids.difference_update(best_tracks_in_frame)
				self.worker_logger.debug(f"Covered {len(best_tracks_in_frame)} tracks, {len(uncovered_track_ids)} remaining")

		self.worker_logger.info(f"DAM assignment complete. Selected {len(selected_groups)} groups.")
		return selected_groups
