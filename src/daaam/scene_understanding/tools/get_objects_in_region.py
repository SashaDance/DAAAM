import numpy as np
from typing import Dict, Any, List, Optional

import torch
import spark_dsg as sdsg
from spark_dsg import DsgLayers, NodeSymbol

from daaam.utils.embedding import (
	CLIPHandler,
	SentenceEmbeddingHandler,
	get_combined_embedding
)
from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import retrieve_objects_from_scene_graph
from daaam.utils.scene_graph_utils import objects_by_region
from daaam.scene_understanding.tools.utils import (
	UnifiedObject,
	extract_background_objects,
	format_unified_object_output
)


class GetObjectsInRegion(Tool):
	"""Tool to retrieve objects within a specified region using semantic search."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_objects_in_region"
		self.description = (
			"Retrieve objects within a specified region (room) using semantic search. "
			"Given a region ID and a semantic description, returns the top matching objects "
			"within that region only. "
			"Use this when you know which region to search and want to find specific types of objects. Only use it if the question explicitly refers to a region. "
			"Do not call the a function with the same arguments as you have before, as this will lead to repeated information."
			"Returns object metadata including name, position, and observation timestamps."
		)
		self._build_signature()

		self.scene_graph = None

		# Initialize handler references (will be set via dependency injection)
		self.clip_handler = None
		self.sentence_handler = None

		# Cached data computed once per scene graph
		self._all_objects: Optional[Dict[int, Any]] = None
		self._all_background_objects: Optional[Dict[int, UnifiedObject]] = None
		self._objs_by_region: Optional[Dict[int, List[int]]] = None
		self._background_objs_by_region: Optional[Dict[int, List[int]]] = None
		self._features_dict: Optional[Dict[str, Dict[str, Any]]] = None
		self._region_objects: Optional[Dict[int, List[UnifiedObject]]] = None
		self._region_embeddings: Optional[Dict[int, np.ndarray]] = None

	def set_embedding_handlers(self, clip_handler, sentence_handler):
		"""Set shared embedding handlers."""
		self.clip_handler = clip_handler
		self.sentence_handler = sentence_handler

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {
				"region_id": {
					"type": "string",
					"description": "The ID of the region (room) to search within"
				},
				"description": {
					"type": "string",
					"description": "Semantic description of objects to find (e.g., 'chairs', 'tables', 'red objects')"
				}
			},
			"required": ["region_id", "description"],
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

		# Get all objects
		self._all_objects = retrieve_objects_from_scene_graph(self.scene_graph)

		# Get background objects
		self._all_background_objects = extract_background_objects(self.scene_graph)

		# Get object-region mappings
		self._objs_by_region, _, self._background_objs_by_region = objects_by_region(self.scene_graph)

		# Get features dict
		features_metadata = dict(self.scene_graph.metadata.get())
		self._features_dict = features_metadata.get('features', {})

		# For EACH region, precompute unified objects + embeddings
		self._region_objects = {}
		self._region_embeddings = {}

		all_region_ids = set(self._objs_by_region.keys()) | set(self._background_objs_by_region.keys())
		for region_id in all_region_ids:
			unified_objs = self._get_unified_objects_for_region(region_id)

			if unified_objs:
				# Precompute embeddings for this region's objects
				embeddings_list = []
				valid_objects = []

				for obj in unified_objs:
					embedding = self._extract_object_embedding(obj)
					if embedding is not None:
						embeddings_list.append(embedding)
						valid_objects.append(obj)

				if embeddings_list:
					self._region_objects[region_id] = valid_objects
					self._region_embeddings[region_id] = np.stack(embeddings_list, axis=0)

	def _get_unified_objects_for_region(self, region_id: int) -> List[UnifiedObject]:
		"""Get unified objects (regular + background) for a specific region.

		Args:
			region_id: Region node ID.

		Returns:
			List of UnifiedObject instances in this region.
		"""
		unified_objs = []

		objects_layer = self.scene_graph.get_layer(DsgLayers.OBJECTS)

		# Add regular objects
		for obj_id in self._objs_by_region.get(region_id, []):
			if obj_id not in self._all_objects:
				continue

			obj_data = self._all_objects[obj_id]
			obj_node_symbol = NodeSymbol('O', obj_id)

			if not objects_layer.has_node(obj_node_symbol):
				continue

			obj_node = objects_layer.get_node(obj_node_symbol)

			unified_objs.append(UnifiedObject(
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
		for bg_id in self._background_objs_by_region.get(region_id, []):
			if bg_id in self._all_background_objects:
				unified_objs.append(self._all_background_objects[bg_id])

		return unified_objs

	def _extract_object_embedding(self, obj: UnifiedObject) -> Optional[np.ndarray]:
		"""Extract and normalize embedding for an object.

		Args:
			obj: UnifiedObject instance.

		Returns:
			Normalized embedding or None if features unavailable.
		"""
		features = self._features_dict.get(str(obj.semantic_label))
		if not features:
			return None

		clip_feature = features.get('clip_feature')
		sentence_feature = features.get('sentence_embedding_feature')

		if not clip_feature or not sentence_feature:
			return None

		embedding = get_combined_embedding(
			clip_embedding=np.array(clip_feature),
			sentence_embedding=np.array(sentence_feature),
			clip_weight=self.config.clip_weight,
			sentence_weight=self.config.sentence_weight
		)

		# Normalize
		return embedding / (np.linalg.norm(embedding) + 1e-10)

	def execute(self, region_id: str, description: str) -> Dict[str, Any]:
		"""Retrieve objects in region matching semantic description.

		Args:
			region_id: String representation of region node ID.
			description: Semantic description of objects to find.

		Returns:
			Dict with matching objects and metadata, or error message.
		"""
		# Validate region
		try:
			region_node_id = int(region_id)
		except ValueError:
			return {'error': f'Invalid region ID format: {region_id}'}

		if region_node_id not in self._region_objects:
			return {'error': f'Region {region_id} not found or has no objects'}

		if not description:
			return {'error': 'Description cannot be empty'}

		# Get current robot position for distance calculation
		current_position = self._get_current_robot_position()

		# Get query embedding
		query_embedding = self._get_query_embedding(description)
		if query_embedding is None:
			return {'error': 'Failed to compute query embedding'}

		# Compute semantic scores ONLY for objects in this region
		scores = self._region_embeddings[region_node_id] @ query_embedding

		# Get top-k
		top_k = min(self.config.default_top_k, len(scores))
		top_k_indices = np.argsort(scores)[::-1][:top_k]

		# Format and return
		results = []
		for idx in top_k_indices:
			obj = self._region_objects[region_node_id][idx]
			result = format_unified_object_output(obj)
			result['semantic_score'] = float(scores[idx])

			# Add distance to current position
			if current_position is not None:
				distance = float(np.linalg.norm(obj.position - current_position))
				result['distance_to_current'] = distance

			results.append(result)

		return {
			'region_id': region_id,
			'objects': results,
			'count': len(results)
		}

	def _get_query_embedding(self, description: str) -> Optional[np.ndarray]:
		"""Get embedding for query description.

		Args:
			description: Text description to encode.

		Returns:
			Normalized embedding vector or None if no model available.
		"""
		# Lazy initialization fallback (for standalone usage)
		if self.sentence_handler is None:
			self.sentence_handler = SentenceEmbeddingHandler(
				model_name=self.config.sentence_embedding_model_name,
				device="cuda" if torch.cuda.is_available() else "cpu"
			)
		if self.clip_handler is None and self.config.clip_model_name:
			self.clip_handler = CLIPHandler(model_name=self.config.clip_model_name)

		clip_embedding = None
		if self.clip_handler:
			clip_embedding = self.clip_handler.extract_text_features([description])[0]

		sentence_embedding = None
		if self.sentence_handler:
			sentence_embedding = self.sentence_handler.extract_text_embeddings(
				[description], show_progress=False
			)[0]

		emb = get_combined_embedding(
			clip_embedding=clip_embedding,
			sentence_embedding=sentence_embedding,
			clip_weight=self.config.clip_weight,
			sentence_weight=self.config.sentence_weight
		)

		if emb is not None:
			# Normalize
			emb = emb / (np.linalg.norm(emb) + 1e-10)

		return emb
