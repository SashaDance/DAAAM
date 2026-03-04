import json
from typing import List
from PIL import Image
import base64
from io import BytesIO

def image_to_base64(img: Image) -> str:
    """Convert PIL Image to base64 string.
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        ValueError: If image conversion fails
    """
    buffered = BytesIO()
    # Use JPEG format if none specified
    format = img.format if img.format else 'JPEG'
    
    try:
        img.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")

def parse_json(json_output: str) -> List[dict]:
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break 
    try:
        parsed_json = json.loads(json_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

