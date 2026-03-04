from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json
import base64
from io import BytesIO
from pydantic import BaseModel

import spark_dsg as sdsg
import numpy as np
from PIL import Image

from daaam.scene_understanding.config import SceneUnderstandingConfig
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.scene_understanding.models import Response
from daaam.scene_understanding.interfaces import (
    SceneUnderstandingInterface
)
from daaam.scene_understanding.tools import create_default_tool_registry
from daaam.scene_understanding.tools.utils import round_floats
from daaam.scene_understanding.prompt_utils import (
    build_system_prompt,
    get_response_format_instructions
)
from daaam.scene_understanding.providers import (
	Provider, detect_provider, create_client,
	openai_tools_to_anthropic, run_anthropic_tool_loop,
	build_anthropic_initial_messages,
)

class SceneUnderstandingAgent(SceneUnderstandingInterface):
	"""Service for handling scene understanding with tool calling."""
	
	def __init__(self, config: SceneUnderstandingConfig, logger: Optional[PipelineLogger] = None):
		super().__init__(config, logger)

		# Initialize provider-aware client and tool store with configuration
		self.provider = detect_provider(config.model_name)
		self.client = create_client(self.provider)
		self.tool_registry = create_default_tool_registry(
			config.tool_config,
			tools_to_include=config.available_tools
		)

		# Create shared embedding handlers (expensive models, load once)
		import torch
		from daaam.utils.embedding import CLIPHandler, SentenceEmbeddingHandler

		self.clip_handler = None
		if config.tool_config.clip_model_name and config.tool_config.clip_backend:
			self.clip_handler = CLIPHandler(model_name=config.tool_config.clip_model_name, backend=config.tool_config.clip_backend)

		self.sentence_handler = SentenceEmbeddingHandler(
			model_name=config.tool_config.sentence_embedding_model_name,
			device="cuda" if torch.cuda.is_available() else "cpu"
		)

		# Inject shared handlers into all tools
		self.tool_registry.set_embedding_handlers(self.clip_handler, self.sentence_handler)

		# Don't set scene graph here - it's empty! Wait for update_scene_graph() to be called
		

	def answer_query(self, ResponseFormat: type, query: str, image: Optional[Any] = None, **kwargs) -> Tuple[Response, Dict[str, Any]]:
		"""
		Answer a query about the scene by using provided tools.

		Args:
			query: The user's query about the scene.
			image: Optional image (PIL Image, numpy array, or path) to include.
			**kwargs: Additional context.

		Returns:
			Tuple[Response, Dict[str, Any]]:
				Response: The final answer and reasoning.
				Dict[str, Any]: The full interaction history.
		"""
		# Ensure scene graph has been set
		if not self.scene_graph or self.scene_graph.num_nodes() == 0:
			raise RuntimeError("Scene graph not set. Call update_scene_graph() before answer_query().")

		if self.config.verbose:
			self.logger.info(f"Processing query: {query}")

		if self.provider == Provider.ANTHROPIC:
			return self._answer_query_anthropic(ResponseFormat, query, image)
		return self._answer_query_openai(ResponseFormat, query, image)

	def _answer_query_openai(self, ResponseFormat: type, query: str, image: Optional[Any] = None):
		"""OpenAI Responses API tool-calling loop."""
		messages = self._build_initial_messages(query, image, ResponseFormat)
		system_prompt = messages.pop(0)["content"]
		tools = self.tool_registry.get_openai_tools()
		assert tools, f"No tools registered — check available_tools config"

		if self.config.verbose:
			self.logger.info(f"Registered {len(tools)} tools: {[t['name'] for t in tools]}")

		interaction_history = {
			"query": query,
			"iterations": [],
			"final_response": None
		}

		for iteration in range(self.config.max_iterations):
			if self.config.verbose:
				self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")

			print(f"Iteration {iteration + 1}/{self.config.max_iterations}")
			response = self.client.responses.create(
				model=self.config.model_name,
				instructions=system_prompt,
				input=messages,
				tools=tools,
				tool_choice="auto",
			)

			output_types = [item.type for item in response.output]
			print(f"  Response output types: {output_types}")

			iteration_data = {
				"iteration": iteration + 1,
				"response": response
			}

			# Append all output items first (canonical Responses API pattern)
			messages += response.output

			# Then process function calls and append their outputs
			for item in response.output:
				if item.type == "function_call":
					result = self.tool_registry.call_tool(item.name, item.arguments)

					text_result = result
					if isinstance(result, dict) and "_images" in result:
						text_result = {k: v for k, v in result.items() if k != "_images"}

					messages.append({
						"type": "function_call_output",
						"call_id": item.call_id,
						"output": json.dumps(round_floats(text_result))
					})

					iteration_data["tool_results"] = text_result

			interaction_history["iterations"].append(iteration_data)

			has_function_calls = any(item.type == "function_call" for item in response.output)
			if not has_function_calls:
				# Build parse input: user query for context + model's complete final output
				# response.output includes reasoning + message items as a valid pair
				user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
				parse_input = user_msgs + list(response.output)

				format_instructions = get_response_format_instructions(ResponseFormat)
				final_response = self.client.responses.parse(
					model=self.config.model_name,
					instructions=f"Respond only in the given format. {format_instructions}",
					input=parse_input,
					text_format=ResponseFormat,
				)
				iteration_data["response"] = final_response

				interaction_history["iterations"].append(iteration_data)
				interaction_history["final_response"] = final_response.output_parsed
				messages += final_response.output

				return final_response.output_parsed, interaction_history, messages

		# Max iterations reached
		self.logger.warning("Max iterations reached without final answer")

		filtered_messages = self._filter_messages_for_parse(messages)
		filtered_messages.append({
			"role": "user",
			"content": "Max iterations reached. Based on all the information gathered, please provide your final answer now."
		})

		format_instructions = get_response_format_instructions(ResponseFormat)
		final_response = self.client.responses.parse(
			model=self.config.model_name,
			instructions=f"Respond only in the given format. {format_instructions}",
			input=filtered_messages,
			text_format=ResponseFormat,
		)
		iteration_data["response"] = final_response

		interaction_history["iterations"].append(iteration_data)
		interaction_history["final_response"] = final_response.output_parsed
		messages += final_response.output

		return final_response.output_parsed, interaction_history, messages

	def _answer_query_anthropic(self, ResponseFormat: type, query: str, image: Optional[Any] = None):
		"""Anthropic Messages API tool-calling loop."""
		system_prompt = build_system_prompt(ResponseFormat)
		image_data = self._process_image(image) if image else None
		messages = build_anthropic_initial_messages(query, image_data)
		tools = openai_tools_to_anthropic(self.tool_registry.get_openai_tools())
		format_instructions = get_response_format_instructions(ResponseFormat)

		return run_anthropic_tool_loop(
			client=self.client,
			model_name=self.config.model_name,
			system_prompt=system_prompt,
			messages=messages,
			tools=tools,
			tool_registry=self.tool_registry,
			max_iterations=self.config.max_iterations,
			response_schema=ResponseFormat,
			format_instructions=format_instructions,
			logger=self.logger,
			verbose=self.config.verbose,
		)
	
	def _filter_messages_for_parse(self, messages: list) -> list:
		"""Keep user messages and model output (reasoning + message).
		Strips function_call/function_call_output items to avoid
		orphaned call_id references in the parse call."""
		filtered = []
		for msg in messages:
			if isinstance(msg, dict):
				if msg.get("role") == "user":
					filtered.append(msg)
			elif hasattr(msg, "type") and msg.type in ("message", "reasoning"):
				filtered.append(msg)
		return filtered

	def _build_initial_messages(self, query: str, image: Optional[Any] = None, response_type: Optional[type] = None) -> List[Dict[str, Any]]:
		"""Build initial messages for the conversation.
		
		Args:
			query: User's query.
			image: Optional image to include.
		
		Returns:
			List of message dictionaries.
		"""
		# Build system prompt using response-specific template
		if response_type:
			system_content = build_system_prompt(response_type)
		else:
			# Fallback to generic prompt if no response type specified
			system_content = build_system_prompt(Response)

		messages = [
			{
				"role": "system",
				"content": system_content
			}
		]
		
		# Build user message
		user_content = []
		
		# Add text query
		user_content.append({
			"type": "input_text",
			"text": query
		})
		
		# Add image if provided
		if image is not None:
			image_data = self._process_image(image)
			if image_data:
				user_content.append({
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{image_data}"
					}
				})
		
		messages.append({
			"role": "user",
			"content": user_content
		})
		
		return messages
	

	def _process_image(self, image: Any) -> Optional[str]:
		"""Process image to base64 string.
		
		Args:
			image: PIL Image, numpy array, or file path.
		
		Returns:
			Base64 encoded image string or None.
		"""
		try:
			# Handle different image types
			if isinstance(image, str) or isinstance(image, Path):
				# Load from file path
				pil_image = Image.open(image)
			elif isinstance(image, np.ndarray):
				# Convert numpy array to PIL
				pil_image = Image.fromarray(image)
			elif hasattr(image, 'save'):
				# Assume it's already a PIL Image
				pil_image = image
			else:
				self.logger.warning(f"Unknown image type: {type(image)}")
				return None
			
			# Convert to base64
			buffered = BytesIO()
			pil_image.save(buffered, format="JPEG")
			img_str = base64.b64encode(buffered.getvalue()).decode()
			
			return img_str
			
		except Exception as e:
			self.logger.error(f"Failed to process image: {e}")
			return None
	
	def _parse_response(self, content: str) -> Response:
		"""Parse the model's response into a Response object.
		
		Args:
			content: Model's response content.
		
		Returns:
			Response object.
		"""
		try:
			# Try to parse as JSON
			if '{' in content and '}' in content:
				# Extract JSON from content
				start = content.index('{')
				end = content.rindex('}') + 1
				json_str = content[start:end]
				data = json.loads(json_str)
				
				return Response(
					reasoning=data.get("reasoning", "No reasoning provided"),
					answer=data.get("answer", content)
				)
		except:
			pass
		
		# Fallback: use content as answer
		return Response(
			reasoning="Direct response without structured format",
			answer=content
		)
	
	def update_scene_graph(self, new_graph: sdsg.DynamicSceneGraph):
		"""Update the internal scene graph representation."""
		self.scene_graph = new_graph
		# Update tool store with new scene graph
		self.tool_registry.set_scene_graph(new_graph)