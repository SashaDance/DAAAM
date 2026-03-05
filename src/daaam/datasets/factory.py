"""Dataset factory for automatic dataset type detection and creation."""

from pathlib import Path
from typing import Optional, Dict, Any, Type
import json

from .interfaces import BaseDataset
from .loaders import ImageSequenceDataset, CodaDataset


class DatasetFactory:
	"""Factory for creating dataset instances based on path structure."""

	# Registry of dataset types
	_dataset_types: Dict[str, Type[BaseDataset]] = {
		"image_sequence": ImageSequenceDataset,
		"coda": CodaDataset,
	}
	
	@classmethod
	def create(
		cls,
		data_path: Path,
		dataset_type: Optional[str] = None,
		config: Optional[Dict[str, Any]] = None
	) -> BaseDataset:
		"""Create a dataset instance.
		
		Args:
			data_path: Path to dataset
			dataset_type: Explicit dataset type. If None, auto-detect.
			config: Configuration dictionary
			
		Returns:
			Dataset instance
			
		Raises:
			ValueError: If dataset type cannot be determined or is unsupported
		"""
		data_path = Path(data_path)
		
		# Auto-detect if not specified
		if dataset_type is None:
			dataset_type = cls._detect_dataset_type(data_path)
			
		if dataset_type not in cls._dataset_types:
			raise ValueError(
				f"Unknown dataset type: {dataset_type}. "
				f"Supported types: {list(cls._dataset_types.keys())}"
			)
			
		# Create dataset instance
		dataset_class = cls._dataset_types[dataset_type]
		return dataset_class(data_path, config)
		
	@classmethod
	def _detect_dataset_type(cls, data_path: Path) -> str:
		"""Auto-detect dataset type from path structure.
		
		Args:
			data_path: Path to dataset
			
		Returns:
			Detected dataset type string
			
		Raises:
			ValueError: If dataset type cannot be determined
		"""
		if not data_path.exists():
			raise ValueError(f"Dataset path does not exist: {data_path}")
			
		# Check for CODa structure (2d_rect/ directory with camera subdirs)
		if data_path.is_dir():
			rect_dir = data_path / "2d_rect"
			if rect_dir.exists() and rect_dir.is_dir():
				return "coda"

		# Check for image sequence structure
		if data_path.is_dir():
			rgb_dir = data_path / "rgb"
			if rgb_dir.exists() and rgb_dir.is_dir():
				return "image_sequence"
				
			# Check for config file that specifies type
			config_file = data_path / "dataset_config.json"
			if config_file.exists():
				with open(config_file, 'r') as f:
					config = json.load(f)
					if "dataset_type" in config:
						return config["dataset_type"]
						
		# Check for rosbag file
		if data_path.is_file():
			if data_path.suffix in [".bag", ".db3"]:
				return "rosbag"
				
		raise ValueError(
			f"Could not determine dataset type for: {data_path}. "
			"Expected folder structure with 'rgb/' subdirectory for image sequences, "
			"or .bag/.db3 file for rosbags."
		)
		
	@classmethod
	def register_dataset_type(
		cls,
		name: str,
		dataset_class: Type[BaseDataset]
	):
		"""Register a new dataset type.
		
		Args:
			name: Name identifier for the dataset type
			dataset_class: Dataset class that inherits from BaseDataset
		"""
		if not issubclass(dataset_class, BaseDataset):
			raise TypeError(f"{dataset_class} must inherit from BaseDataset")
		cls._dataset_types[name] = dataset_class
		
	@classmethod
	def list_types(cls) -> list[str]:
		"""List all registered dataset types."""
		return list(cls._dataset_types.keys())