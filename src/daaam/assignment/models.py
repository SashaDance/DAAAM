"""Data models for the assignment service."""
from dataclasses import dataclass
from typing import List, Dict, Any, Union
import numpy as np

from daaam.tracking.models import SimplifiedTrack
from daaam.grounding.models import ObjectAnnotation

@dataclass 
class AssignmentTask:
	"""Task data sent to assignment workers."""
	track_history: List[List[SimplifiedTrack]]
	frame_dims: tuple
	object_labels: Dict[int, int]
	corrections: Dict[int, ObjectAnnotation]
	prompted_track_ids: List[int]
	start_frame_count: int
	frame_id_mapping: Dict[int, int] = None  # local_idx -> global_frame_id


@dataclass
class SelectedGroup:
	"""
	Group of tracks selected for grounding.
	
	Attributes:
		global_frame_id: int	# Global frame ID (absolute, not relative)
		track_ids: List[int]	# IDs of the tracks in this group
	"""
	global_frame_id: int
	track_ids: List[int]