from typing import Dict, Any, List, Optional

import spark_dsg as sdsg

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import get_robot_position_at_timestamp


class GetRobotLocation(Tool):
	"""Tool to get robot location at a timestamp."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_robot_location"
		self.description = (
			"Get the robot's 3D position at a specific timestamp. "
			"Input: timestamp in seconds from recording start (t=0). "
			"Output: [x, y, z] position in meters."
		)
		self._build_signature()

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {
				"timestamp": {
					"type": "number",
					"description": "Timestamp in seconds from recording start (t=0)"
				}
			},
			"required": ["timestamp"],
			"additionalProperties": False
		}

	def execute(self, timestamp: float) -> Optional[List[float]]:
		"""Get robot location."""
		if self.scene_graph is None:
			return None

		position = get_robot_position_at_timestamp(self.scene_graph, timestamp)
		return position.tolist()
