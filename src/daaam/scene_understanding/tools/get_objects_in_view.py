import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set

import spark_dsg as sdsg
from spark_dsg import DsgLayers

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import retrieve_objects_from_scene_graph
from daaam.scene_understanding.tools.utils import (
	extract_background_objects,
	UnifiedObject
)


class GetObjectsInView(Tool):
	"""Tool to retrieve ALL objects visible from a specific viewpoint (no diversity sampling)."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_objects_in_view"
		self.description = (
			"Retrieve ALL objects that were visible from a specific viewpoint in the robot's trajectory. "
			"Given a timestamp or position, finds the closest agent node (viewpoint) and returns all "
			"objects that were observed from that viewpoint. "
			"Returns complete object metadata including name, position, and observation timestamps."
		)
		self._build_signature()

		self.scene_graph = None

		# Cached data computed once per scene graph
		self._agent_trajectory: Optional[List[Tuple[float, np.ndarray, int]]] = None
		self._all_objects: Optional[Dict[int, Any]] = None
		self._all_background_objects: Optional[Dict[int, UnifiedObject]] = None

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {
				"timestamp": {
					"type": ["number", "null"],
					"description": "Timestamp in seconds from recording start (t=0). Provide either timestamp or position, not both."
				},
				"position": {
					"type": ["array", "null"],
					"items": {"type": "number"},
					"minItems": 3,
					"maxItems": 3,
					"description": "3D position [x, y, z] in meters. Provide either timestamp or position, not both."
				}
			},
			"required": ["timestamp", "position"],
			"additionalProperties": False
		}

	def set_scene_graph(self, scene_graph: sdsg.DynamicSceneGraph):
		"""Set scene graph and precompute cached data."""
		self.scene_graph = scene_graph
		self._precompute_scene_graph_data()

	def _precompute_scene_graph_data(self):
		"""Precompute scene graph-specific data that doesn't change between queries."""
		if self.scene_graph is None:
			return

		# Precompute agent trajectory
		self._agent_trajectory = []
		if self.scene_graph.has_layer(2, 97):
			for agent_node in self.scene_graph.get_layer(2, 97).nodes:
				metadata = agent_node.attributes.metadata.get()
				timestamp = metadata.get("timestamp")
				if timestamp is not None:
					self._agent_trajectory.append((
						timestamp,
						agent_node.attributes.position,
						agent_node.id.value
					))

			# Sort by timestamp for efficient lookups
			self._agent_trajectory.sort(key=lambda x: x[0])

		# Precompute all objects
		self._all_objects = retrieve_objects_from_scene_graph(self.scene_graph)

		# Precompute all background objects
		self._all_background_objects = extract_background_objects(self.scene_graph)

	def execute(
		self,
		timestamp: Optional[float] = None,
		position: Optional[List[float]] = None
	) -> Dict[str, Any]:
		"""Retrieve ALL objects visible from a specific viewpoint.

		Args:
			timestamp: Timestamp in seconds from recording start (t=0).
			position: 3D position [x, y, z] in meters.

		Returns:
			Dict with agent viewpoint info and ALL visible objects.
		"""
		# Validation: exactly one of timestamp/position must be provided
		if (timestamp is None) == (position is None):
			return {"error": "Provide exactly one of: timestamp or position"}

		if self.scene_graph is None:
			return {"error": "Scene graph not available"}

		if not self._agent_trajectory:
			return {"error": "No agent trajectory available in scene graph"}

		# Find closest agent node
		try:
			agent_t, agent_pos, agent_node_id = self._find_closest_agent_node(timestamp, position)
		except ValueError as e:
			return {"error": str(e)}

		# Get observed semantic labels from agent node
		observed_labels = self._get_observed_semantic_labels(agent_node_id)

		if not observed_labels:
			return {
				"agent_timestamp": agent_t,
				"agent_position": agent_pos.tolist() if isinstance(agent_pos, np.ndarray) else agent_pos,
				"objects": [],
				"count": 0,
				"note": "No observed semantic labels found for this viewpoint"
			}

		# Map semantic labels to ALL object IDs
		object_ids = self._map_semantic_labels_to_objects(observed_labels)

		# Format ALL objects (no diversity sampling, no limit)
		objects_info = self._format_objects_output(object_ids, agent_pos)

		return {
			"agent_timestamp": agent_t,
			"agent_position": agent_pos.tolist() if isinstance(agent_pos, np.ndarray) else agent_pos,
			"objects": objects_info,
			"count": len(objects_info)
		}

	def _find_closest_agent_node(
		self,
		timestamp: Optional[float],
		position: Optional[List[float]]
	) -> Tuple[float, np.ndarray, int]:
		"""Find the closest agent node to given timestamp or position.

		Returns:
			Tuple of (agent_timestamp, agent_position, agent_node_id)
		"""
		if not self._agent_trajectory:
			raise ValueError("No agent trajectory available")

		if timestamp is not None:
			# Find agent node with closest timestamp
			closest = min(self._agent_trajectory, key=lambda x: abs(x[0] - timestamp))
			return closest

		if position is not None:
			# Find agent node with closest position
			pos_array = np.array(position)
			closest = min(self._agent_trajectory, key=lambda x: np.linalg.norm(x[1] - pos_array))
			return closest

		raise ValueError("Either timestamp or position must be provided")

	def _get_observed_semantic_labels(self, agent_node_id: int) -> Set[int]:
		"""Extract observed semantic labels from agent node.

		Returns:
			Set of semantic label IDs observed from this viewpoint.
		"""
		agent_node = self.scene_graph.get_node(agent_node_id)

		if not hasattr(agent_node.attributes, 'observed_semantic_labels'):
			return set()

		observed_labels = agent_node.attributes.observed_semantic_labels

		# Handle different data structures
		if isinstance(observed_labels, dict):
			labels = set(observed_labels.keys())
		elif isinstance(observed_labels, (set, list)):
			labels = set(observed_labels)
		else:
			return set()

		# Filter out unknown label (0)
		labels.discard(0)

		return labels

	def _map_semantic_labels_to_objects(self, observed_labels: Set[int]) -> List[int]:
		"""Map semantic labels to object IDs.

		Args:
			observed_labels: Set of semantic label IDs.

		Returns:
			List of object IDs with matching semantic labels.
		"""
		object_ids = []

		# Check regular objects
		if self.scene_graph.has_layer(DsgLayers.OBJECTS):
			objects_layer = self.scene_graph.get_layer(DsgLayers.OBJECTS)
			for obj_node in objects_layer.nodes:
				semantic_label = obj_node.attributes.semantic_label
				if semantic_label in observed_labels:
					object_ids.append(obj_node.id.category_id)

		# Check background objects
		if self.scene_graph.has_layer("BACKGROUND_OBJECTS"):
			bg_layer = self.scene_graph.get_layer("BACKGROUND_OBJECTS")
			for bg_node in bg_layer.nodes:
				semantic_label = bg_node.attributes.semantic_label
				if semantic_label in observed_labels:
					object_ids.append(bg_node.id.category_id)

		return object_ids

	def _format_objects_output(self, object_ids: List[int], reference_position: np.ndarray) -> List[Dict[str, Any]]:
		"""Format object IDs into output structure with distances.

		Args:
			object_ids: List of object IDs to format.
			reference_position: Reference position for distance calculation.

		Returns:
			List of object info dictionaries.
		"""
		objects_info = []

		# Determine which objects are background
		background_obj_ids = set(self._all_background_objects.keys())

		for obj_id in object_ids:
			is_background = obj_id in background_obj_ids

			# Try to get from regular objects first
			if obj_id in self._all_objects:
				obj_data = self._all_objects[obj_id]
				obj_position = np.array(obj_data.object_info.position)
				distance = float(np.linalg.norm(obj_position - reference_position))

				objects_info.append({
					"id": str(obj_id),
					"name": obj_data.object_info.description,
					"position": list(obj_data.object_info.position) if isinstance(obj_data.object_info.position, tuple) else obj_data.object_info.position,
					"first_observed": obj_data.object_info.first_observed,
					"last_observed": obj_data.object_info.last_observed,
					"is_background": is_background,
					"distance_to_current": distance
				})
			elif obj_id in self._all_background_objects:
				# Get from background objects
				bg_obj = self._all_background_objects[obj_id]
				bg_position = bg_obj.position if isinstance(bg_obj.position, np.ndarray) else np.array(bg_obj.position)
				distance = float(np.linalg.norm(bg_position - reference_position))

				objects_info.append({
					"id": f"o({obj_id})",
					"name": bg_obj.name,
					"position": bg_obj.position.tolist() if isinstance(bg_obj.position, np.ndarray) else list(bg_obj.position),
					"first_observed": bg_obj.first_observed,
					"last_observed": bg_obj.last_observed,
					"is_background": True,
					"distance_to_current": distance
				})

		return objects_info
