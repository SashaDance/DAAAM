import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import spark_dsg as sdsg
from sentence_transformers import SentenceTransformer

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import (
	retrieve_objects_from_scene_graph,
	get_robot_position_at_timestamp,
	compute_path_in_scene_graph
)

from .get_matching_subjects import GetMatchingSubjects
from .get_objects_in_radius import	GetObjectsInRadius
from .get_robot_location import GetRobotLocation
from .get_region_information import GetRegionInformation
from .get_objects_in_region import GetObjectsInRegion
from .get_objects_in_view import GetObjectsInView
from .get_agent_trajectory_information import GetAgentTrajectoryInformation


class ToolRegistry:
	"""Class to manage and provide tools for scene understanding."""
	
	def __init__(self):
		self.tools: Dict[str, Tool] = {}
		
	def register_tool(self, tool: Tool):
		"""Register a new tool."""
		self.tools[tool.name] = tool
		
	def get_tool(self, name: str) -> Optional[Tool]:
		"""Retrieve a tool by name."""
		return self.tools.get(name, None)
		
	def list_tools(self) -> List[str]:
		"""List all registered tools."""
		return list(self.tools.keys())
	
	def get_openai_tools(self) -> List[Dict[str, Any]]:
		"""Get all tools in OpenAI format."""
		return [tool.signature for tool in self.tools.values()]
		
	def call_tool(self, name: str, arguments: str) -> Any:
		"""Call a registered tool with JSON arguments."""
		tool = self.get_tool(name)
		if tool:
			args = json.loads(arguments)
			return tool.execute(**args)
		else:
			raise ValueError(f"Tool '{name}' not found.")
	
	def set_scene_graph(self, scene_graph: sdsg.DynamicSceneGraph):
		"""Set scene graph for all tools."""
		for tool in self.tools.values():
			tool.set_scene_graph(scene_graph)

	def set_embedding_handlers(self, clip_handler: Optional[Any], sentence_handler: Optional[Any]):
		"""Set embedding handlers on all tools that support them."""
		for tool in self.tools.values():
			if hasattr(tool, 'set_embedding_handlers'):
				tool.set_embedding_handlers(clip_handler, sentence_handler)


def create_default_tool_registry(config: Optional[ToolConfig] = None, 
                                tools_to_include: Optional[List[str]] = None) -> ToolRegistry:
	"""Create a tool store with specified tools registered.
	
	Args:
		config: Optional tool configuration to pass to all tools.
		tools_to_include: List of tool names to register. If None, all tools are registered.
	
	Returns:
		ToolRegistry with specified tools registered and configured.
		
	Raises:
		ValueError: If an unknown tool name is specified.
	"""
	store = ToolRegistry()
	
	# Define available tools
	available_tools = {
		"get_matching_subjects": lambda: GetMatchingSubjects(config),
		"get_objects_in_radius": lambda: GetObjectsInRadius(config),
		"get_robot_location": lambda: GetRobotLocation(config),
		"get_region_information": lambda: GetRegionInformation(config),
		"get_objects_in_region": lambda: GetObjectsInRegion(config),
		"get_objects_in_view": lambda: GetObjectsInView(config),
		"get_agent_trajectory_information": lambda: GetAgentTrajectoryInformation(config),
	}

	# If no specific tools requested, register all
	if tools_to_include is None:
		tools_to_include = list(available_tools.keys())
	
	# Validate and register requested tools
	for tool_name in tools_to_include:
		if tool_name not in available_tools:
			raise ValueError(f"Unknown tool requested: {tool_name}. Available tools: {list(available_tools.keys())}")
		
		tool = available_tools[tool_name]()
		store.register_tool(tool)

	return store