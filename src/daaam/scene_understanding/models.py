from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass
import numpy as np
from typing import Annotated, Optional
import time

from shapely import node
from spark_dsg import SceneGraphNode


class Response(BaseModel):
	reasoning: str = Field(
		...,
		description="""Description of the context in the scene,
		explanations of the classifications in the image.""",
	)
	answer: Any = Field(
		..., description="The answer to the query."
	)

class LocationResponse(Response):
	answer: List[float] = Field(
		..., description="List of three floats representing the 3D position [x, y, z]."
	)

class TextResponse(Response):
	answer: str = Field(
		..., description="A textual answer to the query."
	)

class TrajectoryResponse(Response):
	answer: List[List[float]] = Field(
		..., description="List of 3D points [x,y,z] representing the trajectory."
	)

class TrajectoryInstructionsResponse(Response):
	answer: str = Field(
		..., description="Natural language instructions how to go from A to B."
	)

class TimeResponse(Response):
	answer: float = Field(
		..., description="[seconds] (negative values indicate the past (e.g., -57.67))." 
	)

class DistanceResponse(Response):
	answer: float = Field(
		..., description="[meters]." 
	)

class BinaryResponse(Response):
	answer: str = Field(
		..., description="yes / no ." 
	)


class ObjectInfo(BaseModel):
	id: int = Field(..., description="Unique identifier for the object.")
	description: str = Field(..., description="Textual description of the object.")
	position: Tuple[float, float, float] = Field(..., description="3D position of the object in space.")
	dimensions: Tuple[float, float, float] = Field(..., description="Dimensions of the object (width, height, depth).")
	first_observed: float = Field(..., description="Timestamp when the object was first observed.")
	last_observed: float = Field(..., description="Timestamp when the object was last observed.")

	@classmethod
	def from_scene_graph_node(cls, node: SceneGraphNode) -> 'ObjectInfo':
		"""Create ObjectInfo from a scene graph node.
		
		Args:
			scene_graph: The dynamic scene graph.
			node_id: ID of the node.
		
		Returns:
			ObjectInfo instance.
		"""

		return cls(
			id=node.id.category_id,
			description=node.attributes.metadata.get()["description"],
			position=tuple(node.attributes.position) if node.attributes.position is not None else (0.0, 0.0, 0.0),
			dimensions=tuple(node.attributes.bounding_box.dimensions) if node.attributes.bounding_box is not None else (0.0, 0.0, 0.0),
			first_observed=node.attributes.metadata.get()["temporal_history"]["first_observed"],
			last_observed=node.attributes.metadata.get()["temporal_history"]["last_observed"],
		)


@dataclass
class ObjectData:
	object_info: ObjectInfo
	sentence_embedding_feature: Optional[np.ndarray]
	clip_feature: Optional[np.ndarray]
	observation_timestamps: Optional[List[float]]

	"""Container for object data including embeddings and timestamps.
	
	Args:
		object_info: Basic object information.
		sentence_embedding_feature: Sentence embedding vector.
		clip_feature: CLIP embedding vector.
		observation_timestamps: List of timestamps when the object was observed.
	
	"""

	@classmethod
	def from_scene_graph_node(cls, node: SceneGraphNode) -> 'ObjectData':
		"""Create ObjectData from a scene graph node and embeddings.
		
		Args:
			scene_graph: The dynamic scene graph.
			node_id: ID of the node.
			sentence_embedding_feature: Optional sentence embedding override.
			clip_feature: Optional CLIP feature override.
		
		Returns:
			ObjectData instance.
		"""
		return cls(
			object_info=ObjectInfo.from_scene_graph_node(node),
			sentence_embedding_feature=np.array(node.attributes.metadata.get().get("sentence_embedding_feature", [])),
			clip_feature=np.array(node.attributes.metadata.get().get("selectframe_clip_feature", [])),
			observation_timestamps=np.array(node.attributes.metadata.get().get("temporal_history", {}).get("timestamps", [])),
		)