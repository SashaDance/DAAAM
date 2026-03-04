#!/usr/bin/env python3
"""Main script to run the daaam pipeline without ROS.

This script provides a command-line interface to process datasets through
the MMLLM Grounded SAM pipeline with Hydra integration for scene graph generation.

Usage:
	python run_pipeline.py /path/to/dataset --config config/pipeline_config.yaml
	python run_pipeline.py /path/to/dataset --hydra-config habitat --target-fps 10
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json
import numpy as np

# Add parent directory to path to import daaam
sys.path.insert(0, str(Path(__file__).parent.parent))

from daaam.config import PipelineConfig
from daaam.datasets import ImageSequenceDataset, HM3DSemDataset
from daaam.hydra.runner import HydraPipelineRunner
from daaam import ROOT_DIR


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description="Run daaam pipeline on datasets without ROS"
	)
	
	# Required arguments
	parser.add_argument(
		"data_path",
		type=str,
		help="Path to dataset (folder for image sequences, file for rosbags)"
	)
	
	# Configuration
	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Path to pipeline configuration YAML file"
	)
	
	parser.add_argument(
		"--config-overrides",
		type=str,
		nargs='+',
		help="Configuration overrides in key=value format (e.g., workers.num_grounding_workers=8)"
	)
	
	# Dataset options
	parser.add_argument(
		"--dataset-type",
		type=str,
		choices=["ImageSequenceDataset", "HM3DSemDataset"],
		default="ImageSequenceDataset",
		help="Dataset class name to use"
	)
	
	parser.add_argument(
		"--dataset-name",
		type=str,
		default=None,
		help="Dataset name for output directory structure (e.g., 'clio', 'habitat')"
	)
	
	# Model configuration (matching launch file defaults)
	parser.add_argument(
		"--agent-model-name",
		type=str,
		default="gpt-4.1-mini",
		help="Agent model name for grounding"
	)
	
	parser.add_argument(
		"--sam-model",
		type=str,
		default="fastsam/FastSAM-s.pt",
		help="SAM model path"
	)
	
	parser.add_argument(
		"--sam-model-config-path",
		type=str,
		default="fastsam/fastsam_config.yaml",
		help="SAM model config path"
	)
	
	parser.add_argument(
		"--sentence-embedding-model",
		type=str,
		default="sentence-transformers/sentence-t5-large",
		help="Sentence embedding model"
	)
	
	parser.add_argument(
		"--depth-scale",
		type=float,
		default=1.0,
		help="Scale factor to convert depth to meters"
	)
	
	parser.add_argument(
		"--fps",
		type=float,
		default=30.0,
		help="Dataset framerate"
	)
	
	# Processing parameters (matching launch file)
	parser.add_argument(
		"--query-interval-frames",
		type=int,
		default=60,
		help="Query interval for grounding"
	)
	
	parser.add_argument(
		"--num-assignment-workers",
		type=int,
		default=1,
		help="Number of assignment workers"
	)
	
	parser.add_argument(
		"--num-grounding-workers",
		type=int,
		default=4,
		help="Number of grounding workers"
	)
	
	parser.add_argument(
		"--assignment-worker",
		type=str,
		default="min_frames_max_size",
		help="Assignment worker type"
	)
	
	parser.add_argument(
		"--grounding-worker",
		type=str,
		default="dam_multi_image",
		help="Grounding worker type"
	)
	
	parser.add_argument(
		"--min-mask-region-area",
		type=int,
		default=100,
		help="Minimum mask region area"
	)
	
	# Depth filtering
	parser.add_argument(
		"--depth-lb",
		type=float,
		default=0.25,
		help="Depth lower bound in meters"
	)
	
	parser.add_argument(
		"--depth-ub",
		type=float,
		default=5.0,
		help="Depth upper bound in meters"
	)
	
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Maximum number of frames to process"
	)
	
	# Processing options
	parser.add_argument(
		"--target-fps",
		type=float,
		default=None,
		help="Target framerate for processing (default: no throttling)"
	)
	
	parser.add_argument(
		"--no-wait-workers",
		action="store_true",
		help="Don't wait for workers to initialize"
	)
	
	parser.add_argument(
		"--no-throttle",
		action="store_true",
		help="Disable framerate throttling (overrides --target-fps)"
	)
	
	parser.add_argument(
		"--hydra-config-path",
		type=str,
		default="/path/to/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml",
		help="Path to Hydra config YAML file"
	)
	
	parser.add_argument(
		"--labelspace-path",
		type=str,
		default=None,
		help="Path to labelspace YAML file (optional)"
	)
	
	parser.add_argument(
		"--labelspace-colors",
		type=str,
		default=None,
		help="Path to labelspace colors CSV file (optional)"
	)
	
	parser.add_argument(
		"--zmq-url",
		type=str,
		default="tcp://127.0.0.1:8001",
		help="ZMQ URL for DSG publishing (use 'none' to disable)"
	)
	
	# Output options
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Output directory for results"
	)
	
	parser.add_argument(
		"--save-images",
		action="store_true",
		help="Save segmentation images"
	)
	
	parser.add_argument(
		"--save-interval",
		type=int,
		default=100,
		help="Save images every N frames"
	)
	
	# Display options
	parser.add_argument(
		"--no-progress",
		action="store_true",
		help="Disable progress bar"
	)
	
	parser.add_argument(
		"--no-logging",
		action="store_true",
		help="Disable logging"
	)
	
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose output"
	)
	
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Load dataset and config but don't run pipeline"
	)
	
	# Configuration files
	parser.add_argument(
		"--semantic-config",
		type=str,
		default="config/labels_pseudo.yaml",
		help="Semantic configuration file"
	)
	
	parser.add_argument(
		"--labelspace-colors",
		type=str,
		default="config/labels_pseudo.csv",
		help="Labelspace colors file"
	)
	
	return parser.parse_args()


def apply_config_overrides(config: PipelineConfig, overrides: list[str]):
	"""Apply configuration overrides from command line.
	
	Args:
		config: Pipeline configuration
		overrides: List of key=value override strings
	"""
	if not overrides:
		return
		
	for override in overrides:
		if '=' not in override:
			print(f"Warning: Invalid override format: {override}")
			continue
			
		key, value = override.split('=', 1)
		keys = key.split('.')
		
		# Navigate to the correct nested config
		obj = config
		for k in keys[:-1]:
			if hasattr(obj, k):
				obj = getattr(obj, k)
			else:
				print(f"Warning: Unknown config key: {'.'.join(keys[:keys.index(k)+1])}")
				break
		else:
			# Set the final value
			final_key = keys[-1]
			if hasattr(obj, final_key):
				# Try to parse the value type
				current_value = getattr(obj, final_key)
				if isinstance(current_value, bool):
					value = value.lower() in ['true', '1', 'yes']
				elif isinstance(current_value, int):
					value = int(value)
				elif isinstance(current_value, float):
					value = float(value)
				# else keep as string
				
				setattr(obj, final_key, value)
				print(f"Applied override: {key} = {value}")
			else:
				print(f"Warning: Unknown config key: {key}")


def create_frame_callback(save_images: bool, save_interval: int, output_dir: Path):
	"""Create a callback function for frame processing.
	
	Args:
		save_images: Whether to save images
		save_interval: Save every N frames
		output_dir: Output directory
		
	Returns:
		Callback function
	"""
	if not save_images:
		return None
		
	images_dir = output_dir / "images"
	images_dir.mkdir(exist_ok=True)
	
	def callback(frame_id, frame, label_image, color_image, stats):
		"""Save images periodically."""
		if frame_id % save_interval == 0:
			import cv2
			
			# Save color segmentation
			color_path = images_dir / f"segmentation_{frame_id:06d}.png"
			cv2.imwrite(str(color_path), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
			
			# Save label image
			label_path = images_dir / f"labels_{frame_id:06d}.png"
			cv2.imwrite(str(label_path), label_image.astype(np.uint16))
			
			print(f"Saved images for frame {frame_id}")
			
	return callback


def main():
	"""Main entry point."""
	args = parse_args()
	
	# Load or create configuration
	if args.config:
		config_path = Path(args.config)
		if not config_path.is_absolute():
			config_path = Path(ROOT_DIR) / config_path
		print(f"Loading configuration from: {config_path}")
		config = PipelineConfig.from_yaml(str(config_path))
	else:
		print("Using default configuration")
		config = PipelineConfig()
		
	# Apply model and processing parameters from args
	config.segmentation.model_name = args.sam_model
	config.segmentation.model_config_path = args.sam_model_config_path
	config.segmentation.min_mask_region_area = args.min_mask_region_area
	
	config.grounding.agent_model_name = args.agent_model_name
	config.grounding.query_interval_frames = args.query_interval_frames
	config.grounding.sentence_embedding_model = args.sentence_embedding_model
	
	config.workers.num_assignment_workers = args.num_assignment_workers
	config.workers.num_grounding_workers = args.num_grounding_workers
	config.workers.assignment_worker = args.assignment_worker
	config.workers.grounding_worker = args.grounding_worker
	
	config.depth.depth_lb = args.depth_lb
	config.depth.depth_ub = args.depth_ub
	
	config.semantic_config_path = args.semantic_config
	config.labelspace_colors_path = args.labelspace_colors
	
	# Apply dataset configuration
	config.dataset.data_path = args.data_path
	config.dataset.dataset_type = args.dataset_type
	config.dataset.depth_scale = args.depth_scale
	config.dataset.fps = args.fps
	config.dataset.max_frames = args.max_frames
	
	# Apply Hydra configuration (always enabled)
	config.hydra.enable_hydra = True
	config.hydra.hydra_config_path = args.hydra_config_path
	config.hydra.labelspace_path = args.labelspace_path or config.semantic_config_path
	config.hydra.labelspace_colors = args.labelspace_colors or config.labelspace_colors_path
	config.hydra.zmq_url = None if args.zmq_url.lower() == 'none' else args.zmq_url
		
	# Apply command-line overrides
	apply_config_overrides(config, args.config_overrides)
	
	# Set output directory
	if args.output_dir:
		config.output_dir = args.output_dir
		
	# Create dataset using the specified class
	print(f"Loading dataset from: {args.data_path}")
	print(f"Using dataset type: {args.dataset_type}")
	
	dataset_config = {
		"depth_scale": args.depth_scale,
		"fps": args.fps,
		"compute_velocities": getattr(config.dataset, 'compute_velocities', True),
		"velocity_window": getattr(config.dataset, 'velocity_window', 10),
		"velocity_alpha": getattr(config.dataset, 'velocity_alpha', 0.4)
	}

	try:
		# Get dataset class by name
		if args.dataset_type == "HM3DSemDataset":
			dataset = HM3DSemDataset(
				Path(args.data_path),
				config=dataset_config,
				depth_scale=args.depth_scale,
				compute_velocities=dataset_config["compute_velocities"],
				velocity_window=dataset_config["velocity_window"],
				velocity_alpha=dataset_config["velocity_alpha"]
			)
		else:  # Default to ImageSequenceDataset
			dataset = ImageSequenceDataset(
				Path(args.data_path),
				config=dataset_config,
				depth_scale=args.depth_scale,
				compute_velocities=dataset_config["compute_velocities"],
				velocity_window=dataset_config["velocity_window"],
				velocity_alpha=dataset_config["velocity_alpha"]
			)
		print(f"Loaded {len(dataset)} frames from dataset")
	except Exception as e:
		print(f"Error loading dataset: {e}")
		sys.exit(1)
		
	# Validate dataset
	if not dataset.validate():
		print("Dataset validation failed")
		sys.exit(1)
	
	# Check for dry run
	if args.dry_run:
		print("\nDry run complete. Dataset and configuration loaded successfully.")
		print(f"Would process {len(dataset)} frames")
		print(f"Hydra config path: {config.hydra.hydra_config_path}")
		print(f"Labelspace path: {config.hydra.labelspace_path}")
		return
		
	# Create Hydra pipeline runner (always used)
	output_dir = Path(config.output_dir) if config.output_dir else None
	
	print("Creating Hydra pipeline runner...")
	print(f"Hydra config path: {config.hydra.hydra_config_path}")
	print(f"Labelspace path: {config.hydra.labelspace_path}")
	
	# Handle target FPS from args
	target_fps = args.target_fps if not args.no_throttle else None
	
	runner = HydraPipelineRunner(
		config=config,
		dataset=dataset,
		hydra_config_path=config.hydra.hydra_config_path,
		labelspace_path=config.hydra.labelspace_path,
		labelspace_colors=config.hydra.labelspace_colors,
		output_dir=output_dir,
		enable_logging=not args.no_logging,
		show_progress=not args.no_progress,
		target_fps=target_fps,
		wait_for_workers=not args.no_wait_workers,
		match_ros_log_dir=True,  # Always match ROS log structure for consistency
		zmq_url=config.hydra.zmq_url,
		glog_level=getattr(config.hydra, 'glog_level', 0),
		verbosity=getattr(config.hydra, 'verbosity', 0),
		dataset_name=args.dataset_name
	)
		
	# Create callbacks
	frame_callback = create_frame_callback(
		args.save_images, 
		args.save_interval,
		runner.output_dir
	)
	
	# Run pipeline
	print(f"Starting pipeline processing...")
	print(f"Output will be saved to: {runner.output_dir}")
	
	try:
		stats = runner.run(
			max_frames=config.dataset.max_frames,
			frame_callback=frame_callback
		)
		
		# Print summary statistics
		print("\n" + "="*50)
		print("Processing Complete!")
		print("="*50)
		print(f"Frames processed: {stats['frames_processed']}/{stats['total_frames']}")
		
		if 'avg_processing_time' in stats:
			print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
			print(f"Total processing time: {stats['total_processing_time']:.1f}s")
			
		if 'cv_avg_time' in stats:
			print(f"CV average time: {stats['cv_avg_time']:.3f}s")
			
		if 'hydra_avg_time' in stats:
			print(f"Hydra average time: {stats['hydra_avg_time']:.3f}s")
			
		print(f"Tracks created: {stats.get('tracks_created', 0)}")
		print(f"Corrections applied: {stats.get('corrections_applied', 0)}")
		
		if 'dsg_nodes' in stats:
			print(f"DSG nodes: {stats['dsg_nodes']}")
			print(f"DSG edges: {stats['dsg_edges']}")
			
		print(f"\nResults saved to: {runner.output_dir}")
		
		# Save final statistics
		stats_file = runner.output_dir / "final_stats.json"
		with open(stats_file, 'w') as f:
			# Convert numpy values for JSON
			json_stats = {}
			for key, value in stats.items():
				if isinstance(value, (np.ndarray, np.float32, np.float64, np.int32, np.int64)):
					json_stats[key] = float(value) if np.isscalar(value) else value.tolist()
				else:
					json_stats[key] = value
			json.dump(json_stats, f, indent=2)
			
	except KeyboardInterrupt:
		print("\n\nProcessing interrupted by user")
		runner.shutdown()
		sys.exit(1)
	except Exception as e:
		print(f"\nError during processing: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()