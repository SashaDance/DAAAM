import json
from attr import attrs
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from time import localtime
import spark_dsg as sdsg
from spark_dsg import SceneGraphNode
from scipy.spatial.transform import Rotation
import traceback
from pathlib import Path

from daaam.scene_understanding.models import ObjectInfo, ObjectData

START_TIMES = {
	0: 1673884185.589118,
	3: 1673985684.662767,
	4: 1674087704.955325,
	6: 1674764557.790565,
	16: 1675881269.033398,
	21: 1676052119.125703,
	22: 1676064138.910544
}

def get_time_texas_from_sdsg_timestamp(timestamp: timedelta) -> float:
	"""Convert S-DSG timestamp (seconds) to nanoseconds."""
	datetime_time = (datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timestamp).astimezone(timezone(timedelta(hours=-5)))
	time_s = datetime_time.timestamp()
	return time_s


def retrieve_objects_from_scene_graph(scene_graph: sdsg.DynamicSceneGraph) -> Dict[int, ObjectData]:
	"""Retrieve objects from the scene graph.
	
	Args:
		scene_graph: The dynamic scene graph.
		layer_id: Optional layer ID to filter objects.
	
	Returns:
		List of objects from the scene graph.
	"""
	objects = list(scene_graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes)

	objects_data = {}
	for obj in objects:
		try: 
			data = ObjectData.from_scene_graph_node(obj)
			objects_data[data.object_info.id] = data
		except Exception as e:
			print(f"Error processing object {obj.id}: {e}")
		
	return objects_data


def compute_path_in_scene_graph(scene_graph: sdsg.DynamicSceneGraph, 
								start_pos: np.ndarray, 
								end_pos: np.ndarray) -> Optional[List[np.ndarray]]:
	"""Compute a path between two positions in the scene graph.
	
	Args:
		scene_graph: The dynamic scene graph.
		start_pos: Starting 3D position.
		end_pos: Ending 3D position.
	
	Returns:
		List of waypoints or None if no path exists.
	"""
	# TODO: Need to implement pathfinding logic, relies on 2D places working

	pass 


def quaternion_to_heading_degrees(quat) -> float:
	"""Convert a quaternion to heading angle in degrees.

	Args:
		quat: spark_dsg Quaternion object with x, y, z, w components.

	Returns:
		Heading angle in degrees (yaw around z-axis in right-handed z-up coordinate system).
	"""
	quat_array = np.array([quat.x, quat.y, quat.z, quat.w])
	rot = Rotation.from_quat(quat_array)
	euler = rot.as_euler('xyz', degrees=True)
	return float(euler[2])


def get_robot_position_at_timestamp(scene_graph: sdsg.DynamicSceneGraph,
								timestamp: float) -> Optional[np.ndarray]:
	"""Get robot pose at a specific timestamp.

	Args:
		scene_graph: The dynamic scene graph.
		timestamp: Timestamp in seconds.

	Returns:
		3D position of robot or None if not found.
	"""
	timestamps = []
	for agent_node in scene_graph.get_layer(2,97).nodes:
		node_t = agent_node.attributes.metadata.get()["timestamp"]
		timestamps.append((node_t, agent_node.attributes.position))

	return min(timestamps, key=lambda x: abs(x[0] - timestamp))[1]


def get_objects_in_radius(objects: List[ObjectData],
						  center_position: np.ndarray,
						  radius: float) -> List[ObjectInfo]:
	"""Get all objects within a specified radius from a center position.
	
	Args:
		scene_graph: The dynamic scene graph.
		center_position: 3D center position.
		radius: Maximum distance from center.
	
	Returns:
		List of object IDs within radius.
	"""
	objects_in_radius = []

	for obj_id, obj in objects:
		obj_pos = np.array(obj.object_info.position)
		if obj_pos is not None:
			distance = np.linalg.norm(obj_pos - center_position)
			if distance <= radius:
				objects_in_radius.append(obj.object_info)

	return objects_in_radius


def get_objects_in_timespan(scene_graph: sdsg.DynamicSceneGraph,
							start_time: Optional[float] = None,
							end_time: Optional[float] = None) -> List[ObjectInfo]:
	"""Get all objects observed within a specified time window.
	
	Args:
		scene_graph: The dynamic scene graph.
		start_time: Start of time window (None for no lower bound).
		end_time: End of time window (None for no upper bound).
	
	Returns:
		List of object IDs within time window.
	"""
	objects_in_timespan = []
	objects = retrieve_objects_from_scene_graph(scene_graph)

	for obj_id, obj in objects:

		if start_time is not None and start_time > obj.object_info.last_observed:
			continue
		elif end_time is not None and end_time < obj.object_info.first_observed:
			continue
		else:
			objects_in_timespan.append(obj.object_info)
	
	return objects_in_timespan


def load_background_objects_temporal_data(yaml_path: Path) -> Dict[int, Dict]:
	"""Load temporal data for background objects from YAML file.

	Args:
		yaml_path: Path to background_objects.yaml file

	Returns:
		Dictionary mapping semantic_id to temporal data dict with keys:
		{first_observed, last_observed, observations}
	"""
	if not yaml_path.exists():
		print(f"Warning: Background objects YAML not found: {yaml_path}")
		return {}

	try:
		with open(yaml_path, 'r') as f:
			bg_data = yaml.safe_load(f)

		temporal_map = {}
		objects = bg_data.get('objects', [])

		for obj in objects:
			semantic_id = obj.get('semantic_id')
			if semantic_id is None:
				continue

			temporal_map[semantic_id] = {
				'first_observed': obj.get('first_observed'),
				'last_observed': obj.get('last_observed'),
				'observations': obj.get('observations', 0)
			}

		print(f"Loaded temporal data for {len(temporal_map)} background objects from {yaml_path}")
		return temporal_map

	except Exception as e:
		print(f"Error loading background objects YAML from {yaml_path}: {e}")
		traceback.print_exc()
		return {}
	

def enrich_background_objects_temporal_data(
	sg: sdsg.DynamicSceneGraph,
	bg_temporal_map: Dict[int, Dict],
	start_time: float
) -> int:
	"""Enrich background objects with temporal data from YAML mapping.

	Args:
		sg: Scene graph containing BACKGROUND_OBJECTS layer
		bg_temporal_map: Mapping of semantic_id to temporal data
		start_time: Unix timestamp to subtract for relative timestamps

	Returns:
		Count of enriched background object nodes
	"""
	enriched_count = 0
	missing_ids = []

	for node in sg.get_layer("BACKGROUND_OBJECTS").nodes:
		semantic_id = node.attributes.semantic_label

		# Lookup temporal data
		if semantic_id not in bg_temporal_map:
			missing_ids.append(semantic_id)
			continue

		temporal_data = bg_temporal_map[semantic_id]
		first_observed = temporal_data.get('first_observed')
		last_observed = temporal_data.get('last_observed')
		observations = temporal_data.get('observations', 0)

		# Skip if temporal data is None
		if first_observed is None or last_observed is None:
			continue

		# Get existing metadata (mappingproxy is read-only, need a mutable copy)
		metadata = dict(node.attributes.metadata.get())

		# Adjust timestamps to be relative to start_time
		# (YAML timestamps are in the same coordinate system as DSG on disk)
		first_observed_relative = first_observed - start_time
		last_observed_relative = last_observed - start_time

		# Create temporal_history dict
		temporal_history = {
			"first_observed": first_observed_relative,
			"last_observed": last_observed_relative,
			"observation_count": observations,
			"timestamps": [],  # Not available in YAML
			"frame_ids": []    # Not available in YAML
		}

		# Update metadata
		metadata["temporal_history"] = temporal_history
		node.attributes.metadata.set(metadata)
		enriched_count += 1

	if missing_ids:
		print(f"Warning: {len(missing_ids)} background object semantic IDs not found in YAML: {missing_ids[:10]}...")

	return enriched_count


def load_question_metadata(seq_id: int, qa_data_dir: Path) -> Dict[str, Dict]:
	"""Load question metadata from human_qa.json for the given sequence.

	Args:
		seq_id: Sequence ID
		qa_data_dir: Base directory containing question data

	Returns:
		Dictionary mapping question ID to metadata (start_time, end_time, etc.)
	"""
	qa_file = qa_data_dir / str(seq_id) / "human_qa.json"

	if not qa_file.exists():
		print(f"Warning: Question metadata file not found: {qa_file}")
		return {}

	try:
		with open(qa_file, 'r') as f:
			qa_data = json.load(f)

		metadata_map = {}
		for question in qa_data.get('data', []):
			question_id = question.get('id')
			if question_id:
				metadata_map[question_id] = {
					'start_time': question.get('start_time') - START_TIMES[seq_id],
					'end_time': question.get('end_time') - START_TIMES[seq_id],
					'length': question.get('length'),
					'length_category': question.get('length_category')
				}

		print(f"Loaded metadata for {len(metadata_map)} questions from {qa_file}")
		return metadata_map

	except Exception as e:
		print(f"Error loading question metadata from {qa_file}: {e}")
		return {}

def preprocess_scene_graph(
	sg: sdsg.DynamicSceneGraph,
	start_time: float,
	bg_objects_yaml_path: Optional[Path] = None
) -> sdsg.DynamicSceneGraph:
	"""Update timestamps in the DSG to be relative to start_time.

	Args:
		sg: Scene graph with Unix timestamps (absolute)
		start_time: Unix timestamp to use as reference (t=0)
		bg_objects_yaml_path: Optional path to background_objects.yaml for temporal enrichment

	Returns:
		Scene graph with timestamps relative to start_time
	"""
	# Validation: start_time should be a Unix timestamp (very large number > 1e9)
	if start_time < 1e9:
		print(f"WARNING: start_time={start_time} doesn't look like a Unix timestamp. "
			  f"Expected value > 1e9 (year 2001+). This may indicate a coordinate system bug.")

	first_observeds = []
	skipped_objects = 0
	skipped_background = 0

	### Object nodes:
	for node in sg.get_layer(sdsg.DsgLayers.OBJECTS).nodes:
		# set object timestamps in scene graph to timestamp - start_time
		metadata = node.attributes.metadata.get()
		if metadata == {}:
			continue
		temporal_history = metadata.get("temporal_history", {})

		# Skip objects with incomplete temporal history
		if not temporal_history or "first_observed" not in temporal_history:
			skipped_objects += 1
			continue

		temporal_history["first_observed"] -= start_time
		temporal_history["last_observed"] -= start_time
		temporal_history["timestamps"] = [t - start_time for t in temporal_history.get("timestamps", [])]
		metadata["temporal_history"].update(temporal_history)
		sg.get_node(node.id).attributes.metadata.set(dict(metadata))
		first_observeds.append(temporal_history["first_observed"])

	# print(f"Timestamps in SG: P{first_observeds}")

	### Enrich background objects with temporal data from YAML if available:
	if bg_objects_yaml_path is not None and bg_objects_yaml_path.exists():
		bg_temporal_map = load_background_objects_temporal_data(bg_objects_yaml_path)
		if bg_temporal_map:
			enriched_count = enrich_background_objects_temporal_data(sg, bg_temporal_map, start_time)
			print(f"Enriched {enriched_count} background objects with temporal data from YAML")

	### Background object nodes:
	for node in sg.get_layer("BACKGROUND_OBJECTS").nodes:
		# set object timestamps in scene graph to timestamp - start_time
		metadata = node.attributes.metadata.get()
		if metadata == {}:
			continue
		temporal_history = dict(metadata.get("temporal_history", {}))

		# Skip objects with incomplete temporal history
		if not temporal_history or "first_observed" not in temporal_history:
			skipped_background += 1
			continue

		temporal_history["first_observed"] -= start_time
		temporal_history["last_observed"] -= start_time
		temporal_history["timestamps"] = [t - start_time for t in temporal_history.get("timestamps", [])]
		metadata["temporal_history"].update(temporal_history)
		sg.get_node(node.id).attributes.metadata.set(dict(metadata))
		first_observeds.append(temporal_history["first_observed"])

	### Traversability places: 
	for node in sg.get_layer(3,2).nodes:
		# set place timestamps in scene graph to timestamp - start_time
		node.attributes.first_observed_ns -= int(start_time * 1e9)
		node.attributes.last_observed_ns -= int(start_time * 1e9)

	### Agent nodes:
	for agent_node in sg.get_layer(2,97).nodes:
		timestamp_sg = agent_node.attributes.timestamp
		timestamp_texas = get_time_texas_from_sdsg_timestamp(timestamp_sg)
		agent_node.attributes.metadata.set({"timestamp": timestamp_texas - start_time})

	# Log skipped objects if any
	if skipped_objects > 0 or skipped_background > 0:
		print(f"Note: Skipped {skipped_objects} objects and {skipped_background} background objects "
			  f"with incomplete temporal_history during preprocessing")

	return sg
