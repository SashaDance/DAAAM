from abc import ABC, abstractmethod
from typing import List
import numpy as np


class TrackerInterface(ABC):
	"""Abstract interface for object tracking."""
	
	@abstractmethod
	def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
		"""
		Update tracker with new detections.
		
		Args:
			detections: N x 6 array of [x1, y1, x2, y2, conf, cls]
			frame: Current RGB frame
			
		Returns:
			M x 8 array of [x1, y1, x2, y2, track_id, conf, cls, mask_idx]
		"""
		pass
	
	@abstractmethod 
	def initialize(self, config: dict = None) -> None:
		"""Initialize the tracker with configuration."""
		pass