import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import copy

import spark_dsg as sdsg
from spark_dsg import DsgLayers, SceneGraphNode

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import get_time_texas_from_sdsg_timestamp


class GetRegionInformation(Tool):
	"""Tool to retrieve information about all regions with current and neighbor annotations."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_region_information"
		self.description = (
			"Retrieve region descriptions and entry / exit times of regions in the scene. "
			"Regions are clusters of traversable space (rooms, outdoor areas, buildings). "
			"Each region includes 'is_current' field (true for the region you are currently in), "
			"and 'is_neighbor' field (true for regions with direct edges to the current region). "
			"Region descriptions are semantic summaries of the regions based on 10 diversity-sampled objects each and the label of the floor in the region. "
			"This tool is NOT aware of individual objects or object counts! It helps reasoning about when and where you entered/exited regions. "
			"Only use this tool if the question explicitly asks about a region/room/building in the environment. "
			"This tool does NOT provide information about the agent's trajectory (turns, headings, stops)! "
			"Output: Dict with 'regions' and a timeline of your journey through regions. "
			"Only call this tool once per query, the returned information is the same."
		)
		self._build_signature()

		self.scene_graph = None

		# Cached data computed once per scene graph
		self._all_regions: Optional[List[Dict[str, Any]]] = None

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {},
			"required": [],
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

		# Get all region nodes
		region_node_list = list(self.scene_graph.get_layer(DsgLayers.ROOMS).nodes)

		# Precompute all region information
		self._all_regions = self._get_all_regions(region_node_list)

	def _get_current_region_id(self) -> Optional[int]:
		"""Determine which region robot is currently in.

		Returns:
			Region ID if within threshold of any region, None otherwise.
		"""
		current_position = self._get_current_robot_position()
		if current_position is None:
			return None

		# For each region, check if current position is within threshold
		for region_node in self.scene_graph.get_layer(DsgLayers.ROOMS).nodes:
			region_id = region_node.id.value

			# Get place positions for this region
			place_nodes = [self.scene_graph.get_node(i) for i in region_node.children()]
			place_positions = [p.attributes.position for p in place_nodes]

			if not place_positions:
				continue

			# Check if current position is within threshold of any place (2D XY distance)
			min_distance = min([np.linalg.norm(current_position[:2] - pos[:2])
							   for pos in place_positions])

			if min_distance < self.config.in_region_threshold:
				return region_id

		return None

	def _get_neighboring_region_ids(self, region_id: int) -> List[int]:
		"""Find regions that are neighbors to given region using room-to-room edges.

		Args:
			region_id: Region to find neighbors for.

		Returns:
			List of neighboring region IDs.
		"""
		neighbors = []

		# Get room layer and iterate through its edges
		rooms_layer = self.scene_graph.get_layer(DsgLayers.ROOMS)

		for edge in rooms_layer.edges:
			source_id = edge.source
			target_id = edge.target

			# If edge connects to our region, add the other endpoint
			if source_id == region_id:
				neighbors.append(target_id)
			elif target_id == region_id:
				neighbors.append(source_id)

		return neighbors

	def _adjust_first_last_visit_times(self, annotated_regions: List[Dict[str, Any]]) -> None:
		"""Adjust first and last visit times across all regions.

		Sets the chronologically first visit's entered_at time to 0.0.
		Sets the chronologically last visit's left_at time to NaN (still in region).

		Args:
			annotated_regions: List of all regions with visit data (modified in place).
		"""
		# Collect all visits with references to their parent region and visit index
		all_visit_refs = []
		for region in annotated_regions:
			for visit_idx, visit in enumerate(region["visits"]):
				all_visit_refs.append({
					"region": region,
					"visit_idx": visit_idx,
					"visit": visit,
					"entered_at_time": visit["entered_at"]["time"]
				})

		if not all_visit_refs:
			return

		# Sort chronologically by entry time
		all_visit_refs.sort(key=lambda v: v["entered_at_time"])

		# Modify first visit: set entered_at to 0.0 and recalculate duration
		first_ref = all_visit_refs[0]
		first_visit = first_ref["visit"]
		first_visit["entered_at"]["time"] = 0.0
		# Recalculate duration: new duration = left_at - 0.0
		left_at_time = first_visit["left_at"]["time"]
		first_visit["duration"] = round(left_at_time, 2)

		# Modify last visit: set left_at to NaN and duration to NaN
		last_ref = all_visit_refs[-1]
		last_visit = last_ref["visit"]
		last_visit["left_at"]["time"] = "did not leave."
		# last_visit["duration"] = float('nan')

	def _generate_visit_timeline_summary(self, annotated_regions: List[Dict[str, Any]]) -> str:
		"""Generate a natural language summary of the robot's journey through regions.

		Args:
			annotated_regions: List of all regions with visit data.

		Returns:
			String paragraph describing the chronological sequence of region visits.
		"""
		# Collect all visits with their region category IDs
		all_visits = []
		for region in annotated_regions:
			category_id = region["category_id"]
			for visit in region["visits"]:
				all_visits.append({
					"category_id": category_id,
					"entered_at_time": visit["entered_at"]["time"],
					"left_at_time": visit["left_at"]["time"]
				})

		if not all_visits:
			return "You have not visited any regions."

		# Sort visits chronologically by entry time
		all_visits.sort(key=lambda v: v["entered_at_time"])

		# Build natural language sequence
		if len(all_visits) == 1:
			return f"You visited R({all_visits[0]['category_id']})."

		# Build sequence description
		sequence_parts = []
		sequence_parts.append(f"started in R({all_visits[0]['category_id']})")

		for i in range(1, len(all_visits) - 1):
			sequence_parts.append(f"went to R({all_visits[i]['category_id']})")

		# Last visit
		last_category_id = all_visits[-1]['category_id']
		if len(all_visits) > 1:
			sequence_parts.append(f"finally ended in R({last_category_id})")

		# Join with proper grammar
		if len(sequence_parts) == 2:
			summary = f"You {sequence_parts[0]} and {sequence_parts[1]}."
		else:
			summary = f"You {sequence_parts[0]}, " + ", then ".join(sequence_parts[1:-1]) + f", and {sequence_parts[-1]}."

		return summary

	def execute(self) -> Dict[str, Any]:
		"""Get information about all regions with current and neighbor annotations.

		Returns all regions, with 'is_current' and 'is_neighbor' fields indicating
		which region the robot is in and which regions are neighbors to it.

		Returns:
			Dict with 'regions' (list of all regions with annotations) and 'visit_timeline_summary' (natural language paragraph describing the robot's journey through regions).
		"""
		if self.scene_graph is None:
			return {
				"regions": [],
				"visit_timeline_summary": "You have not visited any regions."
			}

		# Determine current region
		current_region_id = self._get_current_region_id()

		# Get neighboring regions using scene graph edges
		neighbor_ids = set()
		if current_region_id is not None:
			neighbor_ids = set(self._get_neighboring_region_ids(current_region_id))

		# Add is_current and is_neighbor fields to all regions
		# Deep copy visits to avoid modifying cached data
		annotated_regions = []
		for region in self._all_regions:
			region_copy = region.copy()
			region_copy["visits"] = copy.deepcopy(region["visits"])
			region_copy["summary"] = region["summary"].copy()
			region_id = int(region["id"])
			region_copy["is_current"] = (region_id == current_region_id)
			region_copy["is_neighbor"] = (region_id in neighbor_ids)
			annotated_regions.append(region_copy)

		# Adjust first and last visit times
		self._adjust_first_last_visit_times(annotated_regions)

		# Generate visit timeline summary
		timeline_summary = self._generate_visit_timeline_summary(annotated_regions)

		return {
			"regions": annotated_regions,
			"visit_timeline_summary": timeline_summary
		}

	def _get_region_description_from_metadata(self, region_node: SceneGraphNode) -> Optional[Dict[str, str]]:
		"""Extract region description from node metadata if available.

		Args:
			region_node: Region node to extract metadata from.

		Returns:
			Dict with 'label' and 'description' keys if metadata exists, None otherwise.
		"""
		if not hasattr(region_node.attributes, 'metadata'):
			return None

		metadata = region_node.attributes.metadata.get()
		if not metadata:
			return None

		# Check for summarize_regions.py output fields
		if 'region_label' in metadata and 'region_description' in metadata:
			return {
				'label': metadata['region_label'],
				'description': metadata['region_description']
			}

		# Fallback to generic description field
		if 'description' in metadata:
			return {
				'label': 'region',
				'description': metadata['description']
			}

		return None

	def _get_region_label_from_daaam(self, region_node: SceneGraphNode) -> str:
		"""Extract region label from daaam labels (legacy method).

		Args:
			region_node: Region node to extract label from.

		Returns:
			String label based on most frequent daaam label.
		"""
		places_nodes = [self.scene_graph.get_node(i) for i in region_node.children()]
		region_labels = {}

		for place_node in places_nodes:
			daaam_labels = place_node.attributes.label_weights
			for sem_id, weight in daaam_labels.items():
				if sem_id not in region_labels:
					region_labels[sem_id] = []
				region_labels[sem_id].append(weight)

		if not region_labels:
			return "unknown"

		aggregated_labels = {k: np.sum(v) for k, v in region_labels.items()}
		max_label = max(aggregated_labels, key=aggregated_labels.get)
		return self.scene_graph.get_labelspace(3, 2).labels_to_names[max_label]

	def _get_all_regions(self, region_node_list: List[SceneGraphNode]) -> List[Dict[str, Any]]:
		"""Get all regions in the scene graph with their annotations.

		Prioritizes descriptions from region metadata (e.g., from summarize_regions.py),
		falling back to daaam labels if metadata unavailable.
		"""
		all_regions = []

		for region_node in region_node_list:
			region_id = region_node.id.value
			category_id = region_node.id.category_id

			# Try to get description from metadata first
			metadata_desc = self._get_region_description_from_metadata(region_node)

			if metadata_desc:
				label = metadata_desc['label']
				description = metadata_desc['description']
			else:
				# Fallback to daaam labels
				label = self._get_region_label_from_daaam(region_node)
				description = None

			visit_data = self._compute_robot_region_visits(region_id)

			region_dict = {
				"id": str(region_id),
				"category_id": category_id,
				"label": label,
				"visits": visit_data["visits"],
				"summary": visit_data["summary"]
			}

			if description:
				region_dict["description"] = description

			all_regions.append(region_dict)

		return all_regions

	def _compute_robot_region_visits(self, region_id: int) -> Dict[str, Any]:
		"""Compute all visits to a region with entry/exit times and durations.

		Returns:
			Dict with 'visits' list and 'summary' statistics
		"""
		# Get region place positions
		all_places_nodes = [self.scene_graph.get_node(i) for i in self.scene_graph.get_node(region_id).children()]
		all_region_positions = []
		for place_node in all_places_nodes:
			all_region_positions.append(place_node.attributes.position)

		# Build trajectory
		trajectory = []
		for agent_node in self.scene_graph.get_layer(2, 97).nodes:
			# Try to get timestamp from metadata first
			agent_t = None
			if hasattr(agent_node.attributes, 'metadata'):
				metadata = agent_node.attributes.metadata.get() if hasattr(agent_node.attributes.metadata, 'get') else {}
				if metadata and 'timestamp' in metadata:
					agent_t = metadata["timestamp"]

			# Fall back to direct timestamp attribute
			if agent_t is None and hasattr(agent_node.attributes, 'timestamp'):
				ts = agent_node.attributes.timestamp
				if hasattr(ts, 'total_seconds'):
					agent_t = get_time_texas_from_sdsg_timestamp(ts)
				else:
					agent_t = ts

			if agent_t is not None:
				agent_position = agent_node.attributes.position
				trajectory.append((agent_t, agent_position))

		# Detect entry/exit events using state machine
		visits = []
		in_region = False
		current_entry = None
		current_visit_positions = []

		for agent_t, agent_position in trajectory:
			# Check if currently in region
			is_in = min([np.linalg.norm(agent_position[:2] - np.array(region_pos)[:2])
						 for region_pos in all_region_positions]) < self.config.in_region_threshold

			if is_in and not in_region:
				# ENTRY EVENT
				current_entry = {"time": agent_t, "position": agent_position.tolist()}
				current_visit_positions = [agent_position[:2]]
				in_region = True

			elif is_in and in_region:
				# Still in region, track position for distance calculation
				current_visit_positions.append(agent_position[:2])

			elif not is_in and in_region:
				# EXIT EVENT
				visit_duration = agent_t - current_entry["time"]

				# Calculate distance covered during this visit
				distance_covered = 0.0
				for i in range(1, len(current_visit_positions)):
					distance_covered += np.linalg.norm(current_visit_positions[i] - current_visit_positions[i-1])

				visits.append({
					"visit_number": len(visits) + 1,
					"entered_at": current_entry,
					"left_at": {"time": agent_t, "position": agent_position.tolist()},
					"duration": round(visit_duration, 2),
					"distance_covered": round(distance_covered, 2)
				})
				in_region = False
				current_visit_positions = []

		# Handle case where robot is still in region at trajectory end
		if in_region and current_entry:
			last_t, last_pos = trajectory[-1]
			visit_duration = last_t - current_entry["time"]

			# Calculate distance covered during this visit
			distance_covered = 0.0
			for i in range(1, len(current_visit_positions)):
				distance_covered += np.linalg.norm(current_visit_positions[i] - current_visit_positions[i-1])

			visits.append({
				"visit_number": len(visits) + 1,
				"entered_at": current_entry,
				"left_at": {"time": last_t, "position": last_pos.tolist()},
				"duration": round(visit_duration, 2),
				"distance_covered": round(distance_covered, 2)
			})

		# Compute summary statistics
		summary = {
			"total_visits": len(visits),
			"total_time_spent": round(sum(v["duration"] for v in visits), 2) if visits else 0,
			"total_distance_covered": round(sum(v["distance_covered"] for v in visits), 2) if visits else 0,
			"first_visit_time": visits[0]["entered_at"]["time"] if visits else None,
			"last_visit_time": visits[-1]["entered_at"]["time"] if visits else None
		}

		return {"visits": visits, "summary": summary}
