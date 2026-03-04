import numpy as np
from typing import Dict, Any, List, Optional

import spark_dsg as sdsg

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import retrieve_objects_from_scene_graph
from daaam.scene_understanding.tools.utils import (
	UnifiedObject,
	get_unified_objects_list,
	format_unified_object_output
)


class GetObjectsInRadius(Tool):
	"""Tool to retrieve all objects within a specified radius of a position."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_objects_in_radius"
		self.description = (
			f"Retrieve all objects within a specified radius ({self.config.min_radius}m to {self.config.max_radius}m) of a 3D position. "
			"Returns objects sorted by distance from the query position. "
			"Returns up to a fixed maximum number of closest objects. "
			"Output format: List of objects with 'position' ([x,y,z] in meters), "
			"'distance' (meters from query position), and observation timestamps."
		)
		self._build_signature()

		self.scene_graph = None

		# Cached data computed once per scene graph
		self._unified_objects: Optional[List[UnifiedObject]] = None
		self._positions: Optional[np.ndarray] = None

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {
				"position": {
					"type": "array",
					"items": {"type": "number"},
					"minItems": 3,
					"maxItems": 3,
					"description": "3D position [x, y, z] in meters (center of search)"
				},
				"radius": {
					"type": "number",
					"description": f"Search radius in meters (min: {self.config.min_radius}m, max: {self.config.max_radius}m)",
				}
			},
			"required": ["position", "radius"],
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

		# Get all regular objects
		all_objects_dict = retrieve_objects_from_scene_graph(self.scene_graph)

		# Get unified objects (regular + background)
		self._unified_objects = get_unified_objects_list(self.scene_graph, all_objects_dict)

		# Precompute positions array for vectorized distance calculation
		self._positions = np.array([obj.position for obj in self._unified_objects])

	def execute(self, position: List[float], radius: float) -> List[Dict[str, Any]]:
		"""Retrieve objects within radius of position.

		Args:
			position: 3D position [x, y, z] in meters.
			radius: Search radius in meters.

		Returns:
			List of objects within radius, sorted by distance.
		"""
		radius = np.clip(radius, self.config.min_radius, self.config.max_radius)  # Clamp radius to [5m, 20m]

		if self.scene_graph is None:
			return []

		if self._unified_objects is None or len(self._unified_objects) == 0:
			return []

		pos = np.array(position)

		# Vectorized distance computation
		distances = np.linalg.norm(self._positions - pos, axis=1)

		# Filter by radius
		within_radius = distances <= radius

		if not np.any(within_radius):
			return []

		# Get indices of objects within radius, sorted by distance
		sorted_indices = np.argsort(distances[within_radius])
		filtered_indices = np.where(within_radius)[0][sorted_indices]

		# # Limit to top-k to avoid returning too many objects
		# top_k = min(self.config.default_top_k, len(filtered_indices))
		# filtered_indices = filtered_indices[:top_k]

		# Format and return
		results = []
		for idx in filtered_indices:
			obj = self._unified_objects[idx]
			result = format_unified_object_output(obj)
			result['distance'] = float(distances[idx])
			results.append(result)

		return results
