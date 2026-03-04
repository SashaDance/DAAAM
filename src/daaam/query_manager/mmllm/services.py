import os
import base64
import httpx
import sys
import asyncio
from PIL import Image
from typing import List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pydantic import BaseModel

from langchain_openai import ChatOpenAI

# only import if python version is 3.9+
if sys.version_info >= (3, 9):
    from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from daaam.utils.language import image_to_base64, parse_json


class Agent:

    def __init__(
        self, model_name: str = "gpt-4.1", structure: Optional[BaseModel] = None
    ):

        self.context = []
        self.model_name = model_name

        # Select the appropriate model provider based on model name
        if model_name.startswith(("gpt-", "o")):
            if model_name.startswith("o"):
                self.model = ChatOpenAI(
                    model=model_name,
                    max_retries=2,
                    disabled_params={"parallel_tool_calls": None},
                )
            else:
                self.model = ChatOpenAI(model=model_name, max_retries=2)
        elif model_name.startswith(("gemini-", "g-")) and sys.version_info >= (3, 9):
            self.model = ChatGoogleGenerativeAI(model=model_name, max_retries=2)
        elif model_name.startswith("claude-"):
            self.model = ChatAnthropic(model=model_name, max_retries=2)
        else:
            # Default to OpenAI for unknown prefixes
            self.model = ChatOpenAI(model=model_name, max_retries=2)

        self.structure = structure
        if self.structure is not None:
            self.model = self.model.with_structured_output(structure)

    def query_batch(self, prompts: List[str], max_parallel: int = 5) -> List[Any]:
        """Query the model with multiple prompts in parallel.

        Args:
            prompts: List of text prompts to send to the model
            max_parallel: Maximum number of parallel API calls

        Returns:
            List of model responses (structured or unstructured) in the same order as prompts

        Raises:
            ValueError: If prompts list is empty or contains invalid prompts
            Exception: For model API errors or response parsing failures
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        if not all(p and p.strip() for p in prompts):
            raise ValueError("All prompts must be non-empty")
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all queries
            futures = [executor.submit(self.query, prompt) for prompt in prompts]
            
            # Collect results in order
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Re-raise with context about which prompt failed
                    idx = futures.index(future)
                    raise Exception(f"Batch query failed for prompt {idx}: {str(e)}")
        
        return results

    async def query_batch_async(self, prompts: List[str], max_parallel: int = 5) -> List[Any]:
        """Asynchronously query the model with multiple prompts in parallel.

        Args:
            prompts: List of text prompts to send to the model
            max_parallel: Maximum number of parallel API calls

        Returns:
            List of model responses (structured or unstructured) in the same order as prompts

        Raises:
            ValueError: If prompts list is empty or contains invalid prompts
            Exception: For model API errors or response parsing failures
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        if not all(p and p.strip() for p in prompts):
            raise ValueError("All prompts must be non-empty")
        
        # semaphore limit concurrent requests
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def query_with_semaphore(prompt: str, index: int):
            async with semaphore:
                loop = asyncio.get_event_loop()
                try:
                    # synchronous query in thread pool
                    result = await loop.run_in_executor(None, self.query, prompt)
                    return index, result
                except Exception as e:
                    raise Exception(f"Batch query failed for prompt {index}: {str(e)}")
        
        # all tasks
        tasks = [query_with_semaphore(prompt, i) for i, prompt in enumerate(prompts)]
        
        # concurrent execution
        results_with_indices = await asyncio.gather(*tasks)
        
        # sort by index
        results_with_indices.sort(key=lambda x: x[0])
        
        return [result for _, result in results_with_indices]

    def query(self, prompt: str) -> Any:
        """Query the model with a prompt.

        Args:
            prompt: Text prompt to send to the model

        Returns:
            Model response (structured or unstructured)

        Raises:
            ValueError: If prompt is empty or invalid
            Exception: For model API errors or response parsing failures
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            response = self.model.invoke(prompt)

            if self.structure is not None:
                try:
                    assert isinstance(response, self.structure)
                except Exception as e:
                    raise ValueError(f"Invalid response format from model: {str(e)}")
                return response
            else:
                if not hasattr(response, "content"):
                    raise ValueError("Model response missing content")
                return response.content

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise Exception(f"Query failed: {str(e)}")

    def query_multimodal_batch(
        self,
        prompts: List[str],
        images: List[Union[Image.Image, List[Image.Image]]],
        max_parallel: int = 5,
        show_progress: bool = True
    ) -> List[Any]:
        """Query the model with multiple prompt+image pairs in parallel.

        Args:
            prompts: List of text prompts to send to the model
            images: List of images (PIL Image or list of PIL Images) corresponding to each prompt
            max_parallel: Maximum number of parallel API calls
            show_progress: Whether to show a progress bar

        Returns:
            List of model responses (structured or unstructured) in the same order as prompts

        Raises:
            ValueError: If prompts and images have different lengths or are invalid
            Exception: For model API errors or response parsing failures
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        if len(prompts) != len(images):
            raise ValueError(f"Prompts ({len(prompts)}) and images ({len(images)}) must have same length")

        if not all(p and p.strip() for p in prompts):
            raise ValueError("All prompts must be non-empty")

        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all queries and track their indices
            future_to_idx = {
                executor.submit(self.query_multimodal, prompt, img): i
                for i, (prompt, img) in enumerate(zip(prompts, images))
            }

            # Collect results as they complete, with progress bar
            results = [None] * len(prompts)
            iterator = as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc="Querying LLM")

            for future in iterator:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    raise Exception(f"Batch multimodal query failed for prompt {idx}: {str(e)}")

        return results

    def query_multimodal(
        self, prompt: str, images: Union[Image.Image, List[Image.Image]]
    ) -> Any:
        """Query the model with a prompt and one or more images.

        Args:
            prompt: Text prompt to send to the model
            images: A PIL Image object or list of PIL Image objects

        Returns:
            Model response (structured or unstructured)

        Raises:
            ValueError: If prompt is empty or images are invalid
            Exception: For model API errors or response parsing failures
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if isinstance(images, Image.Image):
            images = [images]

        if not isinstance(images, list) or not all(
            isinstance(img, Image.Image) for img in images
        ):
            raise ValueError(
                "Invalid image input; must be a PIL Image or list of PIL Images"
            )

        assert (
            len(images) <= 10
        ), "Attempting to prompt too many images, most models only support up to 10 images"

        try:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_to_base64(img)}"
                    },
                }
                for img in images
            ]

            message = HumanMessage(
                content=[{"type": "text", "text": prompt}] + image_contents
            )

            response = self.model.invoke([message])

            if self.structure is not None:
                try:
                    assert isinstance(response, self.structure)
                except Exception as e:
                    raise ValueError(f"Invalid response format from model: {str(e)}")
                return response
            else:
                if not hasattr(response, "content"):
                    raise ValueError("Model response missing content")
                return response.content

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise Exception(f"Query failed: {str(e)}")