from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
import numpy as np
import spark_dsg as sdsg

from daaam.scene_understanding.config import SceneUnderstandingConfig, ToolConfig
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.scene_understanding.models import Response

class SceneUnderstandingInterface(ABC):
	"""Service for handling segmentation operations."""
	
	def __init__(self, config: SceneUnderstandingConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.scene_graph = sdsg.DynamicSceneGraph()
		self.client = None
		self.model_name = config.model_name
		self.tools: List[Dict] = []

	@abstractmethod
	def answer_query(self, query: str, *args, **kwargs) -> Tuple[Response, Dict[str, Any]]:
		"""
		Answer a query about the scene by using provided tools.
		"""

		raise NotImplementedError("This method should be overridden by subclasses.")
	

class Tool(ABC):
	"""
	Base class for tools used in scene understanding.
	
	Attributes:
		name (str): Name of the tool.
		description (str): Description of the tool's functionality.
		signature (dict): OpenAI-compatible function signature.
		config (Optional[ToolConfig]): Tool configuration object.
	"""
	def __init__(self, config: Optional[ToolConfig] = None):
		self.name = "ToolName"
		self.description = "Description of the tool's functionality."
		self.scene_graph = None  # Will be set by the service
		self.config = config  # Tool configuration
		self._build_signature()
	
	def _build_signature(self):
		"""Build the OpenAI-compatible function signature."""
		self.signature = {
			"type": "function",
			"name": self.name,
			"description": self.description,
			"parameters": self._get_parameters_schema(),
			"strict": True  # Enable strict mode for reliable adherence
		}
	
	@abstractmethod
	def _get_parameters_schema(self) -> Dict[str, Any]:
		"""Return the JSON schema for the tool's parameters.
		
		Returns:
			Dict with type, properties, required, and additionalProperties fields.
		"""
		return {
			"type": "object",
			"properties": {},
			"required": [],
			"additionalProperties": False
		}
	
	@abstractmethod
	def execute(self, **kwargs) -> Any:
		"""Execute the tool's function with keyword arguments.
		
		Args:
			**kwargs: Tool-specific arguments.
		
		Returns:
			Tool-specific result.
		"""
		raise NotImplementedError("This method should be overridden by subclasses.")
	
	def set_scene_graph(self, scene_graph: sdsg.DynamicSceneGraph):
		"""Set the scene graph reference for the tool."""
		self.scene_graph = scene_graph

	def set_config(self, config: Any):
		"""Set or update the tool configuration."""
		self.config = config

	def set_embedding_handlers(self, clip_handler: Optional[Any], sentence_handler: Optional[Any]):
		"""Set shared embedding handlers (optional, for tools that need them)."""
		pass  # Default no-op implementation

	def _get_current_robot_position(self) -> Optional[np.ndarray]:
		"""Get current robot position from latest agent node in scene graph.

		This corresponds to the 'current location' mentioned in questions.
		Returns the position of the most recent agent pose node based on timestamp.

		Returns:
			Position [x, y, z] as numpy array, or None if unavailable.
		"""
		if self.scene_graph is None:
			return None

		# Check if agent layer exists (layer 2, prefix 97)
		if not self.scene_graph.has_layer(2, 97):
			return None

		agent_layer = self.scene_graph.get_layer(2, 97)

		# Find latest agent node by timestamp
		latest_node = None
		latest_timestamp = -float('inf')

		for node in agent_layer.nodes:
			metadata = node.attributes.metadata.get()
			node_timestamp = metadata.get('timestamp')

			if node_timestamp is not None and node_timestamp > latest_timestamp:
				latest_timestamp = node_timestamp
				latest_node = node

		if latest_node is None:
			return None

		return latest_node.attributes.position