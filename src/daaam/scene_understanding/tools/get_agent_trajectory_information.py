from typing import Dict, Any, List, Optional

import numpy as np
import spark_dsg as sdsg

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import quaternion_to_heading_degrees


class GetAgentTrajectoryInformation(Tool):
	"""Tool to get agent trajectory information with position and heading over a time range."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_agent_trajectory_information"
		self.description = (
			"Get agent trajectory information between start_time and end_time. "
			f"Samples {config.trajectory_sample_points} trajectory points uniformly across the time range. "
			"Each point includes timestamp, 3D position [x, y, z], and heading angle in degrees. "
			"Coordinate system is right-handed with z up. "
			"Use this tool with questions about the your movement, path, turns, or stationary periods. "
			"A good rule of thumb is to use this tool for a timeframe of at least 30s. "
			"Input: start_time and end_time in seconds from recording start (t=0). "
			"Output: List of trajectory points with timestamp, position, and heading."
		)
		self._build_signature()

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {
				"start_time": {
					"type": "number",
					"description": "Start time in seconds from recording start (t=0)"
				},
				"end_time": {
					"type": "number",
					"description": "End time in seconds from recording start (t=0)"
				}
			},
			"required": ["start_time", "end_time"],
			"additionalProperties": False
		}

	def execute(self, start_time: float, end_time: float) -> Optional[List[Dict[str, Any]]]:
		"""Get agent trajectory information between start_time and end_time.

		Args:
			start_time: Start time in seconds from recording start.
			end_time: End time in seconds from recording start.

		Returns:
			List of trajectory points, each containing:
				- timestamp: float (seconds from start)
				- position: [x, y, z] (meters)
				- heading_degrees: float (yaw angle in degrees)
			Returns None if scene_graph is not set or no trajectory data available.
		"""
		if self.scene_graph is None:
			return None

		assert start_time <= end_time, f"start_time ({start_time}) must be <= end_time ({end_time})"

		# Get number of sample points from config (default 10)
		n_samples = self.config.trajectory_sample_points if self.config else 10

		# Collect all agent nodes with their timestamps
		agent_nodes_with_time = []
		for agent_node in self.scene_graph.get_layer(2, 97).nodes:
			metadata = agent_node.attributes.metadata.get()
			if "timestamp" not in metadata:
				continue
			timestamp = metadata["timestamp"]

			# Only include nodes within the time range
			if start_time <= timestamp <= end_time:
				agent_nodes_with_time.append((timestamp, agent_node))

		if not agent_nodes_with_time:
			return []

		# Sort by timestamp for efficient lookup
		agent_nodes_with_time.sort(key=lambda x: x[0])

		# Generate uniformly spaced sample times
		sample_times = np.linspace(start_time, end_time, n_samples)

		# For each sample time, find the nearest agent node and extract info
		trajectory_points = []
		for sample_t in sample_times:
			# Find nearest agent node by timestamp
			nearest_node = min(agent_nodes_with_time, key=lambda x: abs(x[0] - sample_t))
			timestamp, agent_node = nearest_node

			# Extract position
			position = agent_node.attributes.position.tolist()

			# Extract heading angle from quaternion
			heading_degrees = quaternion_to_heading_degrees(agent_node.attributes.world_R_body)

			trajectory_points.append({
				"timestamp": round(float(timestamp), 3),
				"position": position,
				"heading_degrees": round(heading_degrees, 3)
			})

		return trajectory_points
