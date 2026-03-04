import numpy as np
import multiprocessing as mp
import queue
from typing import List, Dict, Set, Optional, Union
from PIL import Image as PILImage

from daaam.tracking.models import Track
from daaam.grounding.models import ImageAnnotation, ObjectAnnotation


def prepare_masks_from_tracks(
	prompt_tracks: List[Track],
	object_labels: Dict[int, int],
	worker_id: str,
) -> tuple[List[PILImage.Image], List[int]]:
	"""Prepare PIL masks and semantic IDs from track data."""
	masks = []
	semantic_ids = []
	
	for track in prompt_tracks:
		semantic_id = object_labels.get(track.id)
		if semantic_id is not None:
			if isinstance(track.segmentation, np.ndarray):
				# Convert to PIL mask (binary)
				mask_pil = PILImage.fromarray((track.segmentation * 255).astype(np.uint8), mode='L')
				masks.append(mask_pil)
				semantic_ids.append(semantic_id)
			else:
				print(
					f"[{worker_id}] WARN: Track {track.id} has invalid segmentation type: {type(track.segmentation)}",
					flush=True,
				)
	
	return masks, semantic_ids


def create_corrections_from_descriptions(
	descriptions: List[str],
	semantic_ids: List[int],  
	worker_id: str,
) -> List[ObjectAnnotation]:
	"""Convert descriptions to ObjectAnnotation objects."""
	corrections = []
	for i, description in enumerate(descriptions):
		if i < len(semantic_ids):
			semantic_id = semantic_ids[i]

			if semantic_id == -1:
				correction = ImageAnnotation(
					semantic_label=description.strip() if description else "unknown",
					confidence=0.0,
				)
			else:
				correction = ObjectAnnotation(
					semantic_id=semantic_id,
					semantic_label=description.strip() if description else "unknown",
					confidence=0.0,
				)
			corrections.append(correction)
			print(
				f"[{worker_id}] Created correction for semantic_id {semantic_id}: '{description.strip()}'",
				flush=True,
			)
	return corrections