"""Utilities for loading and formatting prompt templates."""

from pathlib import Path
from typing import Dict, Any, Optional
from string import Template
import os

from daaam import ROOT_DIR
from daaam.scene_understanding.models import (
	LocationResponse,
	TextResponse,
	TrajectoryResponse,
	TrajectoryInstructionsResponse,
	TimeResponse,
	DistanceResponse,
	BinaryResponse,
	Response
)

# Map response types to their template files
RESPONSE_TEMPLATE_MAP = {
	LocationResponse: "location_response.txt",
	TextResponse: "text_response.txt",
	TrajectoryResponse: "trajectory_response.txt",
	TrajectoryInstructionsResponse: "text_response.txt",  # Use text template for instructions
	TimeResponse: "time_response.txt",
	DistanceResponse: "distance_response.txt",
	BinaryResponse: "binary_response.txt",
}

# Default template directory
DEFAULT_TEMPLATE_DIR = ROOT_DIR / "config" / "prompt_templates"


def load_template(template_name: str, template_dir: Optional[Path] = None) -> str:
	"""Load a prompt template from file.
	
	Args:
		template_name: Name of the template file (with or without .txt extension).
		template_dir: Directory containing templates. Uses default if None.
	
	Returns:
		Template content as string.
	
	Raises:
		FileNotFoundError: If template file doesn't exist.
	"""
	if template_dir is None:
		template_dir = DEFAULT_TEMPLATE_DIR
	
	# Add .txt extension if not present
	if not template_name.endswith('.txt'):
		template_name = f"{template_name}.txt"
	
	template_path = Path(template_dir) / template_name
	
	if not template_path.exists():
		raise FileNotFoundError(f"Template not found: {template_path}")
	
	with open(template_path, 'r') as f:
		return f.read()


def format_template(template: str, variables: Dict[str, Any]) -> str:
	"""Format a template string with variables.
	
	Args:
		template: Template string with $variable placeholders.
		variables: Dictionary of variable values.
	
	Returns:
		Formatted template string.
	"""
	tmpl = Template(template)
	return tmpl.safe_substitute(variables)


def get_response_format_instructions(response_type: type) -> str:
	"""Get formatting instructions for a specific response type.
	
	Args:
		response_type: The response class (e.g., LocationResponse, TimeResponse).
	
	Returns:
		Formatting instructions as a string.
	"""
	template_file = RESPONSE_TEMPLATE_MAP.get(response_type)
	
	if template_file is None:
		# Default generic instructions
		return (
			"Format your final response as JSON with the following fields:\n"
			"{\n"
			"  \"reasoning\": \"Your step-by-step analysis\",\n"
			"  \"answer\": <your answer>\n"
			"}"
		)
	
	try:
		return load_template(template_file)
	except FileNotFoundError:
		# Fallback to generic instructions if template not found
		return (
			"Format your final response as JSON with the following fields:\n"
			"{\n"
			"  \"reasoning\": \"Your step-by-step analysis\",\n"
			"  \"answer\": <your answer>\n"
			"}"
		)


def build_system_prompt(response_type: type, additional_context: Optional[str] = None) -> str:
	"""Build a complete system prompt for the scene understanding agent.
	
	Args:
		response_type: The expected response type.
		additional_context: Optional additional context to include.
	
	Returns:
		Complete system prompt string.
	"""
	try:
		base_template = load_template("scene_understanding_system")
	except FileNotFoundError:
		raise FileNotFoundError("Base system prompt template 'scene_understanding_system.txt' not found.")
	
	format_instructions = get_response_format_instructions(response_type)
	
	variables = {
		"response_format_instructions": format_instructions
	}
	
	system_prompt = format_template(base_template, variables)
	
	if additional_context:
		system_prompt += f"\n\nAdditional Context:\n{additional_context}"
	
	return system_prompt


def get_tool_output_format_hint(tool_name: str) -> str:
	"""Get a hint about the output format for a specific tool.
	
	Args:
		tool_name: Name of the tool.
	
	Returns:
		Format hint string to append to tool descriptions.
	"""
	format_hints = {
		"get_robot_location": "Returns [x, y, z] position in meters.",
		"get_closest_robot_location": "Returns (timestamp_seconds, [x, y, z]) tuple.",
		"get_matching_subjects": "Returns list of objects with timestamps in seconds from start.",
		"get_euclidean_distance": "Returns distance in meters.",
		"get_walking_distance": "Returns walking distance in meters.",
		"get_shortest_path": "Returns list of [x, y, z] waypoints.",
		"get_region_information": "Returns region details with timestamps in seconds.",
	}
	
	return format_hints.get(tool_name, "")