from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class SegmenterInterface(ABC):
	"""Abstract interface for segmentation models."""
	
	@abstractmethod
	def __call__(self, source: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
		"""
		Perform segmentation on input image.
		
		Args:
			source: Input RGB image as numpy array
			
		Returns:
			Tuple of (detection_boxes, segmentation_masks)
			- detection_boxes: N x 6 array of [x1, y1, x2, y2, conf, cls]
			- segmentation_masks: List of N binary masks as numpy arrays
		"""
		pass
	
	@abstractmethod
	def initialize(self, model_path: str, config_path: str = None, device: str = None) -> None:
		"""Initialize the segmentation model."""
		pass