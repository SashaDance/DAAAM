from pydantic import BaseModel, Field, field_validator
from typing import Union

from daaam.grounding.models import (
        Annotation, 
        ObjectAnnotation, 
        ImageAnnotation
    )

def annotation_from_dict(data: dict) -> Union[ObjectAnnotation, ImageAnnotation]:
    """Create the appropriate annotation type based on data."""
    if "transform" in data:
        return ImageAnnotation.model_validate(data)
    return ObjectAnnotation.model_validate(data)

def annotation_to_json_serializable(annotation: Annotation) -> dict:
    """Convert a list of annotations to JSON-serializable dictionaries."""
    return annotation.model_dump(mode="json")