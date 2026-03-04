import numpy as np
from typing import Any


def round_floats(obj: Any, decimals: int = 4) -> Any:
	if isinstance(obj, float):
		return round(obj, decimals)
	if isinstance(obj, dict):
		return {k: round_floats(v, decimals) for k, v in obj.items()}
	if isinstance(obj, list):
		return [round_floats(v, decimals) for v in obj]
	return obj
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import spark_dsg as sdsg
from spark_dsg import DsgLayers, NodeSymbol

from daaam.scene_understanding.models import ObjectData
from daaam.scene_understanding.utils import retrieve_objects_from_scene_graph
from daaam.scene_understanding.config import ToolConfig
from daaam.utils.embedding import get_combined_embedding


@dataclass
class UnifiedObject:
	"""Common representation for both regular and background objects."""
	id: int
	name: str
	position: np.ndarray
	first_observed: Optional[float]
	last_observed: Optional[float]
	observation_timestamps: List[float]
	is_background: bool
	semantic_label: int


def extract_background_objects(scene_graph: sdsg.DynamicSceneGraph) -> Dict[int, UnifiedObject]:
	"""Extract all background objects from scene graph as UnifiedObject instances.

	Args:
		scene_graph: Scene graph to extract from.

	Returns:
		Dictionary mapping background object ID to UnifiedObject.
	"""
	background_objects = {}

	if not scene_graph.has_layer("BACKGROUND_OBJECTS"):
		return background_objects

	bg_layer = scene_graph.get_layer("BACKGROUND_OBJECTS")

	for bg_node in bg_layer.nodes:
		bg_id = bg_node.id.category_id

		# Extract name and temporal data from metadata
		name = "unknown"
		first_observed = None
		last_observed = None
		observation_timestamps = []

		if hasattr(bg_node.attributes, 'metadata'):
			metadata = bg_node.attributes.metadata.get()
			name = metadata.get('description', 'unknown')

			temporal_history = metadata.get('temporal_history', {})
			if temporal_history:
				first_observed = temporal_history.get('first_observed')
				last_observed = temporal_history.get('last_observed')
				observation_timestamps = temporal_history.get('timestamps', [])

		background_objects[bg_id] = UnifiedObject(
			id=bg_id,
			name=name,
			position=bg_node.attributes.position,
			first_observed=first_observed,
			last_observed=last_observed,
			observation_timestamps=observation_timestamps,
			is_background=True,
			semantic_label=bg_node.attributes.semantic_label
		)

	return background_objects


def get_unified_objects_list(
	scene_graph: sdsg.DynamicSceneGraph,
	all_objects_dict: Dict[int, ObjectData]
) -> List[UnifiedObject]:
	"""Merge regular objects and background objects into unified list.

	Args:
		scene_graph: Scene graph to extract background objects from.
		all_objects_dict: Dictionary of regular objects from retrieve_objects_from_scene_graph.

	Returns:
		List of UnifiedObject instances (regular + background).
	"""
	unified_objects = []

	# Add regular objects
	objects_layer = scene_graph.get_layer(DsgLayers.OBJECTS)
	for obj_id, obj_data in all_objects_dict.items():
		obj_node_symbol = NodeSymbol('O', obj_id)

		if not objects_layer.has_node(obj_node_symbol):
			continue

		obj_node = objects_layer.get_node(obj_node_symbol)

		unified_objects.append(UnifiedObject(
			id=obj_id,
			name=obj_data.object_info.description,
			position=np.array(obj_data.object_info.position),
			first_observed=obj_data.object_info.first_observed,
			last_observed=obj_data.object_info.last_observed,
			observation_timestamps=obj_data.observation_timestamps,
			is_background=False,
			semantic_label=obj_node.attributes.semantic_label
		))

	# Add background objects
	background_objects = extract_background_objects(scene_graph)
	for bg_obj in background_objects.values():
		unified_objects.append(bg_obj)

	return unified_objects


def precompute_unified_embeddings(
	unified_objects: List[UnifiedObject],
	features_dict: Dict[str, Dict[str, Any]],
	config: ToolConfig
) -> Tuple[np.ndarray, Dict[int, int]]:
	"""Precompute embeddings for all unified objects.

	Args:
		unified_objects: List of UnifiedObject instances.
		features_dict: Dictionary mapping semantic label to features.
		config: Tool configuration with embedding weights.

	Returns:
		Tuple of:
			- embeddings: (N, D) normalized embedding array
			- obj_id_to_idx: mapping from object ID to embedding index
	"""
	embeddings_list = []
	obj_id_to_idx = {}

	for idx, obj in enumerate(unified_objects):
		features = features_dict.get(str(obj.semantic_label))

		if not features:
			continue

		clip_feature = features.get('clip_feature')
		sentence_feature = features.get('sentence_embedding_feature')

		if not clip_feature or not sentence_feature:
			continue

		# Combine embeddings
		embedding = get_combined_embedding(
			clip_embedding=np.array(clip_feature),
			sentence_embedding=np.array(sentence_feature),
			clip_weight=config.clip_weight,
			sentence_weight=config.sentence_weight
		)

		# Normalize
		embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

		embeddings_list.append(embedding)
		obj_id_to_idx[obj.id] = len(embeddings_list) - 1

	if not embeddings_list:
		return np.array([]), {}

	embeddings = np.stack(embeddings_list, axis=0)
	return embeddings, obj_id_to_idx


def format_unified_object_output(obj: UnifiedObject) -> Dict[str, Any]:
	"""Standard output format for unified objects.

	Args:
		obj: UnifiedObject instance.

	Returns:
		Dictionary with standard output fields.
	"""
	return {
		'id': f'o({obj.id})' if obj.is_background else str(obj.id),
		'name': obj.name,
		'position': obj.position.tolist() if isinstance(obj.position, np.ndarray) else obj.position,
		'first_observed': obj.first_observed,
		'last_observed': obj.last_observed,
		'is_background': obj.is_background
	}
