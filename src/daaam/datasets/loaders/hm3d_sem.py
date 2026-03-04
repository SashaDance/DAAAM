"""Image sequence dataset loader for HOV-SG habitat matterport 3D data."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from collections import deque

from daaam.datasets.interfaces import BaseDataset, DatasetFrame
from daaam.datasets.loaders.image_sequence import ImageSequenceDataset


class HM3DSemDataset(ImageSequenceDataset):
	"""Dataset loader for HOV-SG habitat matterport 3D data."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _load_poses_from_txt(self, file_path: Path) -> List[np.ndarray]:
		"""Load poses from a single text file."""
		poses = []
		with open(file_path, 'r') as f:
			for line in f:
				if line.strip():
					# Parse 7D pose [x, y, z, qx, qy, qz, qw]
					values = [float(x) for x in line.strip().split()]
					# 4x4 transformation matrix - convert to pose
					T = np.array(values).reshape(4, 4)

					T = self.revert_hovsg_pose_transform(T)

					pose = self._matrix_to_pose(T)
					poses.append(pose)
		return poses
		
	def _load_individual_poses(self, pose_files: List[Path]) -> List[np.ndarray]:
		"""Load poses from individual files."""
		poses = []
		for file_path in pose_files:
			if file_path.suffix == ".txt":
				with open(file_path, 'r') as f:
					values = [float(x) for x in f.read().strip().split()]
					if len(values) == 7:
						poses.append(np.array(values))
					elif len(values) == 16:
						T = np.array(values).reshape(4, 4)
						
						T = self.revert_hovsg_pose_transform(T)
						
						pose = self._matrix_to_pose(T)
						poses.append(pose)
			elif file_path.suffix == ".json":
				with open(file_path, 'r') as f:
					data = json.load(f)
					if "pose" in data:
						poses.append(np.array(data["pose"]))
					elif "transform" in data:
						poses.append(np.array(data["transform"]))
		return poses
	
	def revert_hovsg_pose_transform(self, pose: np.ndarray) -> np.ndarray:
		C = np.eye(4)
		C[1, 1] = -1  # Convert from Habitat's Y-up
		C[2, 2] = -1  # Convert from Habitat's Z-up

		pose = np.matmul(pose, C)
		
		return pose