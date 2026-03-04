"""Geometry utilities for 3D position computation from 2D masks and depth."""
import cv2
import numpy as np
from typing import Tuple, Optional


def compute_mask_centroid(mask: np.ndarray, method: str = "median") -> Optional[Tuple[int, int]]:
	"""
	Compute a representative center point for a binary mask.

	For concave shapes (U-shape, donut, etc.), the geometric centroid might fall outside
	the mask. This function provides multiple methods to find a point guaranteed to be
	within the mask.

	Args:
		mask: Binary mask (2D numpy array)
		method: Method to use for center computation:
			- "centroid": Geometric centroid (may be outside mask)
			- "median": Median of all mask points (always in mask, robust)
			- "max_distance": Point with maximum distance from edges (skeleton-based)
			- "centroid_safe": Centroid with fallback to nearest point if outside

	Returns:
		Tuple of (cx, cy) pixel coordinates, or None if mask is empty
	"""
	# Convert to uint8 if needed
	if mask.dtype != np.uint8:
		mask = mask.astype(np.uint8)

	# Get mask points
	mask_points = np.where(mask > 0)
	if len(mask_points[0]) == 0:
		return None

	if method == "median":
		# Use median of all mask points - fast and always in mask
		cy = int(np.median(mask_points[0]))
		cx = int(np.median(mask_points[1]))

		# Median might still be outside mask for weird shapes, verify
		if mask[cy, cx] == 0:
			# Fall back to finding closest point to median (should be rare)
			return find_nearest_mask_point_fast(mask, cx, cy)
		return (cx, cy)

	elif method == "max_distance":
		# Use distance transform to find point furthest from edges
		# This gives the "most central" point in the mask
		dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

		# Find the point with maximum distance from edges
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)

		# max_loc is (x, y) format
		return max_loc if max_val > 0 else None

	elif method == "centroid":
		# Standard geometric centroid (may be outside mask)
		moments = cv2.moments(mask)
		if moments['m00'] == 0:
			return None
		cx = int(moments['m10'] / moments['m00'])
		cy = int(moments['m01'] / moments['m00'])
		return (cx, cy)

	elif method == "centroid_safe":
		# Geometric centroid with safety check
		moments = cv2.moments(mask)
		if moments['m00'] == 0:
			return None

		cx = int(moments['m10'] / moments['m00'])
		cy = int(moments['m01'] / moments['m00'])

		# Check bounds and mask membership
		h, w = mask.shape
		if cx < 0 or cx >= w or cy < 0 or cy >= h or mask[cy, cx] == 0:
			# Fall back to median method which is fast and reliable
			cy = int(np.median(mask_points[0]))
			cx = int(np.median(mask_points[1]))

			# Double-check median is in mask
			if mask[cy, cx] == 0:
				# Use max distance as last resort
				dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
				return max_loc if max_val > 0 else None

		return (cx, cy)

	else:
		raise ValueError(f"Unknown method: {method}")


def find_nearest_mask_point_fast(mask: np.ndarray, cx: int, cy: int) -> Optional[Tuple[int, int]]:
	"""
	Find the nearest point within the mask to a given point using distance transform.
	Much faster than brute-force distance computation.

	Args:
		mask: Binary mask
		cx, cy: Target point coordinates

	Returns:
		Nearest point (x, y) within the mask, or None if mask is empty
	"""
	# Create a target image with a single point
	h, w = mask.shape
	target = np.zeros((h, w), dtype=np.uint8)

	# Clip coordinates to image bounds
	cy_clipped = np.clip(cy, 0, h-1)
	cx_clipped = np.clip(cx, 0, w-1)
	target[cy_clipped, cx_clipped] = 255

	# Compute distance from target point to all pixels
	dist_from_target = cv2.distanceTransform(255 - target, cv2.DIST_L2, 5)

	# Mask the distances to only consider points in the mask
	masked_distances = dist_from_target.copy()
	masked_distances[mask == 0] = np.inf

	# Find minimum distance point
	min_idx = np.unravel_index(np.argmin(masked_distances), masked_distances.shape)

	if masked_distances[min_idx] == np.inf:
		return None

	return (int(min_idx[1]), int(min_idx[0]))


def unproject_pixel_to_3d(u: int, v: int, depth: float,
						  fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
	"""
	Convert pixel coordinates + depth to 3D camera coordinates.

	Uses standard pinhole camera model:
		x = (u - cx) * z / fx
		y = (v - cy) * z / fy
		z = depth

	Args:
		u, v: Pixel coordinates
		depth: Depth value at (u, v) in meters
		fx, fy: Focal lengths
		cx, cy: Principal point coordinates

	Returns:
		3D point in camera frame as numpy array [x, y, z]
	"""
	x = (u - cx) * depth / fx
	y = (v - cy) * depth / fy
	z = depth

	return np.array([x, y, z])


def transform_point_to_world(point_camera: np.ndarray,
							 world_T_camera: np.ndarray) -> np.ndarray:
	"""
	Transform 3D point from camera to world frame.

	Args:
		point_camera: 3D point in camera frame [x, y, z]
		world_T_camera: 4x4 transformation matrix from camera to world

	Returns:
		3D point in world frame [x, y, z]
	"""
	# Convert to homogeneous coordinates
	point_homo = np.append(point_camera, 1.0)

	# Transform to world
	point_world_homo = world_T_camera @ point_homo

	# Return Euclidean coordinates
	return point_world_homo[:3]


def get_median_depth_at_mask(depth_image: np.ndarray, mask: np.ndarray,
							 min_valid_ratio: float = 0.25) -> Optional[float]:
	"""
	Get median depth value within a mask region.

	Args:
		depth_image: Depth image
		mask: Binary mask
		min_valid_ratio: Minimum ratio of valid (>0) depth pixels required

	Returns:
		Median depth value, or None if insufficient valid pixels
	"""
	# Get depth values within mask
	depth_values = depth_image[mask > 0]

	# Filter out invalid (zero) depths
	valid_depths = depth_values[depth_values > 0]

	# Check if we have enough valid depth pixels
	if len(valid_depths) < min_valid_ratio * len(depth_values):
		return None

	return float(np.median(valid_depths))


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
	"""
	Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix.

	Args:
		pose: 7-element pose array [translation, quaternion]

	Returns:
		4x4 homogeneous transformation matrix
	"""
	from scipy.spatial.transform import Rotation

	T = np.eye(4)
	T[:3, 3] = pose[:3]  # Translation
	T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()  # Rotation

	return T