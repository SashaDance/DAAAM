from typing import List, Optional, Dict, Union, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import pandas as pd

from daaam.utils.logging import PipelineLogger, get_default_logger

class BoundingBox(BaseModel):
    xmin: int = Field(description="Minimum x-coordinate")
    ymin: int = Field(description="Minimum y-coordinate")
    xmax: int = Field(description="Maximum x-coordinate")
    ymax: int = Field(description="Maximum y-coordinate")

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @classmethod
    def from_xyxy(cls, xyxy: List[float]) -> "BoundingBox":
        """Initialize BoundingBox from [xmin, ymin, xmax, ymax] format.

        Args:
            xyxy: List containing [xmin, ymin, xmax, ymax]

        Returns:
            BoundingBox instance
        """
        return cls(
            xmin=int(xyxy[0]), ymin=int(xyxy[1]), xmax=int(xyxy[2]), ymax=int(xyxy[3])
        )


class DetectionResult(BaseModel):
    score: float = Field(
        description="Confidence score of the detection (0-100)", ge=0.0, le=1.0
    )
    label: str = Field(description="Label of the detected object")
    box: BoundingBox = Field(description="Bounding box of the detected object")

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )

    def __post_init__(self, __context):
        self.label = self.label.lower()


@dataclass
class SegmentationResult:
    detection_res: DetectionResult
    seg_score: Optional[float] = None
    mask: Optional[np.ndarray] = None
    label: Optional[str] = None

    def __post_init__(self):
        if self.seg_score is None:
            self.seg_score = self.detection_res.score
        if self.label is None:
            self.label = self.detection_res.label


def fast_median_depth(depth_image: np.ndarray, mask: np.ndarray) -> float:
    depths = depth_image[mask]
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size == 0:
        return -1.0
    return np.median(valid)


def fast_create_output_images(combined_mask, semantic_lookup, color_lookup):
    label_image = semantic_lookup[combined_mask]
    color_image = color_lookup[combined_mask]
    # zero out where mask is 0
    zero_mask = combined_mask == 0
    label_image[zero_mask] = 0
    color_image[zero_mask] = 0
    return label_image, color_image


def annotate_labeled_2d(
    frame: np.ndarray,
    boxes: List[List[int]],
    colors: List[Tuple[int, int, int]],
    labels: List[str],
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw 2D bounding boxes with labels in image.

    Args:
        frame: Image to draw on
        boxes: List of [xmin, ymin, xmax, ymax] coordinates
        colors: List of (B, G, R) colors for each box
        labels: List of text labels for each box
        line_thickness: Thickness of box lines

    Returns:
        Annotated image
    """
    height, width = frame.shape[:2]

    for i, (box, color, label) in enumerate(zip(boxes, colors, labels)):

        xmin, ymin, xmax, ymax = map(int, box)

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness
        )

        other_boxes = boxes[:i] + boxes[i + 1 :]
        text_x, text_y = find_label_position(
            xmin,
            ymin,
            xmax,
            ymax,
            width,
            height,
            text_width,
            text_height,
            other_boxes,
        )

        assert isinstance(text_x, int)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, line_thickness)

        # label background
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + 2),
            color,
            -1,
        )

        # label text
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            line_thickness,
        )

    return frame


def find_label_position(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_width: int,
    img_height: int,
    text_width: int,
    text_height: int,
    other_boxes: List[List[int]],
) -> Tuple[int, int]:
    """Find best position for label that's in frame and doesn't overlap other boxes."""
    # positions to try, order of preference:
    positions = [
        (xmin, ymin - 2),  # above
        (xmin, ymax + text_height),  # below
        (xmax - text_width, ymin - 2),  # above right
        (xmax - text_width, ymax + text_height),  # below right
    ]

    def text_box(x, y):
        return [x, y, x + text_width, y + text_height]

    for x, y in positions:
        # check if text in frame
        if x < 0 or x + text_width > img_width or y < 0 or y + text_height > img_height:
            continue

        text_bb = text_box(x, y)
        if not any(
            do_boxes_overlap(BoundingBox.from_xyxy(text_bb), BoundingBox.from_xyxy(box))
            for box in other_boxes
        ):
            return x, y

    # no good position found
    x = max(0, min(xmin, img_width - text_width))
    y = max(text_height, ymin)
    return x, y


def bounding_box_from_mask(
    mask: np.ndarray, smooth: bool = False, kernel_size: int = 5
) -> np.ndarray:
    """Extract bounding box coordinates from a binary mask.

    Args:
        mask: Binary mask (2D numpy array)
        smooth: Apply median filtering to remove noise
        kernel_size: Size of the median filter kernel if smoothing is applied

    Returns:
        numpy array [xmin, ymin, xmax, ymax]
    """
    if smooth:
        mask = cv2.medianBlur(mask.astype(np.uint8), kernel_size)

    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    bbox = [x, y, x + w, y + h]

    return np.array(bbox)


def do_boxes_overlap(box1: BoundingBox, box2: BoundingBox) -> bool:
    """Check if two boxes overlap."""
    return not (
        box1.xyxy[2] < box2.xyxy[0]
        or box1.xyxy[0] > box2.xyxy[2]
        or box1.xyxy[3] < box2.xyxy[1]
        or box1.xyxy[1] > box2.xyxy[3]
    )


def highlight_objects_mask_bbox(frame, masks, colors, boxes, labels):
    """Highlight objects in frame with combination of segmentation masks, boxes and labels.

    Args:
        frame: Input image as numpy array
        masks: Binary masks for each object with shape (n_objects, height, width)
        colors: Colors for each object, either as list of tuples or numpy array
        boxes: List of bounding boxes in format [xmin, ymin, xmax, ymax]
        labels: List of text labels for each object

    Returns:
        Frame with highlighted objects
    """
    colors = np.array(colors)

    if len(masks) == 0:
        return frame

    # Create mask overlay
    mask_overlay = np.zeros_like(frame, dtype=np.uint8)

    # Find pixels where any mask has a True value
    valid = masks.any(axis=0)

    if not np.any(valid):
        print("No valid pixels found in masks")
        return frame

    # Get index of the topmost mask at each pixel
    indices = np.argmax(masks, axis=0)

    for i in range(3):  # For each color channel
        for obj_idx in range(len(masks)):
            obj_pixels = np.logical_and(valid, indices == obj_idx)
            if np.any(obj_pixels):
                mask_overlay[obj_pixels, i] = colors[obj_idx, i]

    frame = overlay_mask(frame, mask_overlay, alpha=0.2)
    frame = annotate_labeled_2d(frame, boxes, colors.tolist(), labels, line_thickness=2)

    return frame


def highlight_objects_mask_contour(frame, masks, colors, boxes, labels, line_thickness=2):
    """Highlight objects in frame using contours and labels adjacent to them.

    Args:
        frame: Input image as numpy array
        masks: Binary masks for each object with shape (n_objects, height, width)
        colors: Colors for each object, either as list of tuples or numpy array
        boxes: List of bounding boxes (unused, for compatibility)
        labels: List of text labels for each object
        line_thickness: Thickness for contours and text

    Returns:
        Frame with highlighted objects
    """
    colors = np.array(colors)
    height, width = frame.shape[:2]
    placed_label_boxes = [] # Keep track of label bounding boxes to avoid overlap

    if len(masks) == 0:
        return frame

    for i, (mask, color, label) in enumerate(zip(masks, colors, labels)):
        # Ensure mask is uint8 for findContours
        mask_uint8 = mask.astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Find the largest contour
        cnt = max(contours, key=cv2.contourArea)

        # Draw only the largest contour
        cv2.drawContours(frame, [cnt], -1, color.tolist(), line_thickness)

        # Find the topmost point of the largest contour
        topmost_point = tuple(cnt[cnt[:, :, 1].argmin()][0])
        top_x, top_y = topmost_point

        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness
        )

        # Use find_label_position to determine best placement near the topmost point
        # Pass the topmost point as a zero-area box and the list of already placed label boxes
        text_x, text_y = find_label_position(
            top_x, # xmin
            top_y, # ymin
            top_x, # xmax
            top_y, # ymax
            width,
            height,
            text_width,
            text_height,
            placed_label_boxes, # Pass boxes of labels already placed
        )


        # Draw background rectangle for the label
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_height - 2), # Top-left corner of background
            (text_x + text_width + 2, text_y + 2), # Bottom-right corner of background
            color.tolist(), # Use contour color for background
            -1, # Filled rectangle
        )

        # Add the bounding box of this label (including padding) to the list
        placed_label_boxes.append([
            text_x - 2, text_y - text_height - 2,
            text_x + text_width + 2, text_y + 2
        ])

        # Draw label text (black)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            line_thickness,
        )

    return frame


def overlay_mask(
    image: Union[np.ndarray, Image.Image],
    mask_image: Union[np.ndarray, Image.Image],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a mask image on an image.

    Args:
        image: Input image (cv2 or PIL)
        mask_image: 3 channel label image (unique color per label)
        alpha: Transparency of the overlay (0-1)
        color: RGB color tuple for the mask

    Returns:
        image with overlay (np.ndarray)
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if isinstance(mask_image, Image.Image):
        mask_image = cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR)

    # Blend images
    return cv2.addWeighted(image, 1, mask_image, alpha, 0)


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors using HSV color space.

    Args:
        n: Number of colors to generate

    Returns:
        List of RGB color tuples
    """
    colors = []
    while len(colors) < n:
        i = len(colors)
        # golden ratio
        hue = (i * 137.5) % 360
        saturation = np.random.uniform(0.6, 1.0)
        value = np.random.uniform(0.6, 1.0)

        h = hue / 360.0
        s = saturation
        v = value

        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if i % 6 == 0:
            r, g, b = v, t, p
        elif i % 6 == 1:
            r, g, b = q, v, p
        elif i % 6 == 2:
            r, g, b = p, v, t
        elif i % 6 == 3:
            r, g, b = p, q, v
        elif i % 6 == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        color = (int(r * 255), int(g * 255), int(b * 255))
        if color not in colors:
            colors.append(color)

    return colors


def create_label_files(output_path: str, num_labels: int) -> None:
    """Create both CSV and YAML files for pseudo label configuration.

    Args:
        output_path: Base path for output files (without extension)
        num_labels: Number of labels to generate colors for
    """
    colors = generate_distinct_colors(num_labels)

    # Create CSV file
    csv_path = f"{output_path}.csv"
    print(f"Saving label map CSV to: {csv_path}")
    with open(csv_path, "w") as f:
        f.write("name,red,green,blue,alpha,id\n")
        for i, (r, g, b) in enumerate(colors):
            f.write(f"label_{i},{r},{g},{b},255,{i}\n")

    # Create YAML file
    yaml_path = f"{output_path}.yaml"
    print(f"Saving label space YAML to: {yaml_path}")
    yaml_content = {
        "total_semantic_labels": num_labels,
        "dynamic_labels": [],
        "invalid_labels": [],
        "object_labels": list(range(num_labels)),
        "surface_places_labels": [],
        "label_names": [{"label": i, "name": f"label_{i}"} for i in range(num_labels)],
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(
            yaml_content,
            f,
            sort_keys=False,
            default_flow_style=True,
        )


def create_label_map(output_path: str, num_labels: int) -> None:
    """Create label configuration files.

    Args:
        output_path: Path where to save the label map CSV file
        num_labels: Number of labels to generate colors for
    """
    # Strip extension if present
    base_path = str(Path(output_path).with_suffix(""))
    create_label_files(base_path, num_labels)


def load_semantic_config(semantic_config_path: Path) -> Dict:
    """Load semantic configuration with auto-creation if needed."""
    
    with open(semantic_config_path, 'r') as f:
        semantic_config = yaml.safe_load(f)

    return semantic_config

def load_color_map(
        labelspace_colors_path: Optional[str], 
        logger: PipelineLogger = get_default_logger()
        ) -> Tuple[Dict[int, Tuple[int, int, int]]]:
    """Load color map from CSV file"""
    if not labelspace_colors_path:
        logger.warning("No labelspace colors path provided, using default colors")
        return {}
    
    colors_path = Path(labelspace_colors_path)
    if not colors_path.exists():
        logger.warning(f"Color map file not found: {labelspace_colors_path}")
        return {}
    
    try:
        labelspace_colors = pd.read_csv(labelspace_colors_path)
        # color map
        color_map = {
            row["id"]: (row["red"], row["green"], row["blue"])
            for _, row in labelspace_colors.iterrows()
        }
        logger.info(f"Loaded color map with {len(color_map)} colors")
    except Exception as e:
        logger.error(f"Failed to load color map: {e}")
        color_map = {}

    return color_map


def mask_to_polygons(mask: np.ndarray, epsilon_factor: float = 0.001) -> List[np.ndarray]:
	"""Convert binary mask to polygon contours with optional approximation.

	Args:
		mask: Binary mask (2D numpy array)
		epsilon_factor: Approximation factor for Douglas-Peucker algorithm.
						0 = no approximation, higher values = more approximation.
						Typical values: 0.001-0.01

	Returns:
		List of polygon contours (each contour is an array of points)
	"""
	# Ensure mask is binary uint8
	if mask.dtype != np.uint8:
		mask = mask.astype(np.uint8)

	# Find contours using CHAIN_APPROX_SIMPLE for initial compression
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Apply Douglas-Peucker approximation if requested
	if epsilon_factor > 0 and len(contours) > 0:
		h, w = mask.shape
		epsilon = epsilon_factor * np.sqrt(h*h + w*w)  # Scale by image diagonal
		contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

	return contours


def polygons_to_mask(contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
	"""Reconstruct binary mask from polygon contours.

	Args:
		contours: List of polygon contours
		shape: Shape of the output mask (height, width)

	Returns:
		Binary mask as boolean numpy array
	"""
	mask = np.zeros(shape, dtype=np.uint8)
	if contours:  # Check if contours list is not empty
		cv2.fillPoly(mask, contours, 1)
	return mask.astype(bool)


def compute_polygon_area(contours: List[np.ndarray]) -> int:
	"""Compute total area covered by polygon contours.

	Args:
		contours: List of polygon contours

	Returns:
		Total area in pixels
	"""
	total_area = 0
	for contour in contours:
		total_area += cv2.contourArea(contour)
	return int(total_area)
