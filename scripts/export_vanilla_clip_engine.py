#!/usr/bin/env python3
"""
Export vanilla CLIP (general-purpose) to TensorRT .engine for ReID.

This creates a CLIP model with:
- Base ViT-B-16 visual encoder from OpenAI (pretrained on 400M image-text pairs)
- Default-initialized bottleneck layers (near-identity BatchNorm)
- No ReID-specific fine-tuning (works for any object class)

Usage:
    python scripts/export_vanilla_clip_engine.py --dynamic              # Dynamic batch (required for ReID)
    python scripts/export_vanilla_clip_engine.py --dynamic --trt-fp16   # Dynamic + FP16
"""
import os
os.environ["TORCH_ONNX_USE_LEGACY_EXPORTER"] = "1"

import argparse
import sys
import time
import torch
import onnx
from pathlib import Path

from boxmot.appearance.backbones.clip.config.defaults import _C as cfg
from boxmot.appearance.backbones.clip.make_model import make_model
from boxmot.utils.torch_utils import select_device
from boxmot.utils import logger as LOGGER

from daaam import ROOT_DIR


def parse_args():
	parser = argparse.ArgumentParser(description="Export vanilla CLIP to TensorRT")
	parser.add_argument("--output-dir", type=Path,
						default=Path(f"checkpoints/reid_weights"),
						help="Output directory for exported models")
	parser.add_argument("--name", type=str, default="clip_general",
						help="Base name for output files")
	parser.add_argument("--batch-size", type=int, default=128,
						help="Batch size for export")
	parser.add_argument("--imgsz", nargs="+", type=int, default=[256, 128],
						help="Image size (h, w)")
	parser.add_argument("--device", default="0",
						help="CUDA device for TensorRT build")
	parser.add_argument("--dynamic", default=True,
						help="Enable dynamic batch sizes (required for ReID)")
	parser.add_argument("--trt-fp16", action="store_true",
						help="Build FP16 TensorRT engine")
	parser.add_argument("--workspace", type=int, default=4,
						help="TensorRT workspace size (GB)")
	parser.add_argument("--skip-engine", action="store_true",
						help="Only export .pt and .onnx, skip TensorRT")
	return parser.parse_args()


def build_vanilla_clip_model():
	"""Build CLIP model with base OpenAI weights (no ReID fine-tuning)."""
	LOGGER.info("Building vanilla CLIP model...")
	LOGGER.info(f"  Backbone: {cfg.MODEL.NAME}")
	LOGGER.info(f"  Input size: {cfg.INPUT.SIZE_TRAIN}")

	# Build model on CPU - this downloads base CLIP from OpenAI
	# num_class=1 since classifier is unused at inference
	model = make_model(cfg, num_class=1, camera_num=1, view_num=1)
	model.eval()

	# Count parameters
	n_params = sum(p.numel() for p in model.parameters())
	LOGGER.info(f"  Parameters: {n_params / 1e6:.1f}M")

	return model


def export_onnx(model, im, onnx_path, dynamic=False):
	"""Export model to ONNX format."""
	LOGGER.info(f"Exporting to ONNX: {onnx_path}")
	LOGGER.info(f"  Dynamic batch: {dynamic}")

	# Export on CPU to avoid device mismatch issues
	model_cpu = model.cpu()
	im_cpu = im.cpu()

	# Dynamic axes for variable batch size
	dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None

	torch.onnx.export(
		model_cpu,
		im_cpu,
		onnx_path,
		verbose=False,
		opset_version=14,
		do_constant_folding=True,
		input_names=["images"],
		output_names=["output"],
		dynamic_axes=dynamic_axes,
		dynamo=False,
	)

	# Verify
	model_onnx = onnx.load(onnx_path)
	onnx.checker.check_model(model_onnx)
	onnx.save(model_onnx, onnx_path)
	LOGGER.info(f"ONNX export successful: {onnx_path}")
	return onnx_path


def export_tensorrt(onnx_path, engine_path, im_shape, workspace_gb=4, fp16=False, dynamic=False):
	"""Build TensorRT engine from ONNX file."""
	try:
		import tensorrt as trt
	except ImportError:
		LOGGER.error("TensorRT not found. Install with: pip install nvidia-tensorrt")
		return None

	LOGGER.info(f"Building TensorRT engine: {engine_path}")
	LOGGER.info(f"  TensorRT version: {trt.__version__}")
	LOGGER.info(f"  Workspace: {workspace_gb} GB")
	LOGGER.info(f"  FP16: {fp16}")
	LOGGER.info(f"  Dynamic: {dynamic}")

	is_trt10 = int(trt.__version__.split(".")[0]) >= 10
	logger = trt.Logger(trt.Logger.INFO)

	builder = trt.Builder(logger)
	config = builder.create_builder_config()

	workspace = int(workspace_gb * (1 << 30))
	if is_trt10:
		config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
	else:
		config.max_workspace_size = workspace

	flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	network = builder.create_network(flag)
	parser = trt.OnnxParser(network, logger)

	if not parser.parse_from_file(str(onnx_path)):
		for i in range(parser.num_errors):
			LOGGER.error(f"ONNX parse error: {parser.get_error(i)}")
		return None

	# Log network info
	LOGGER.info("Network Description:")
	for i in range(network.num_inputs):
		inp = network.get_input(i)
		LOGGER.info(f'  input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
	for i in range(network.num_outputs):
		out = network.get_output(i)
		LOGGER.info(f'  output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

	# Dynamic batch size optimization profile
	if dynamic:
		max_batch = im_shape[0]
		LOGGER.info(f"Setting up dynamic batch profile: min=1, opt={max_batch//2}, max={max_batch}")
		profile = builder.create_optimization_profile()
		for i in range(network.num_inputs):
			inp = network.get_input(i)
			# (batch, channels, height, width)
			min_shape = (1, *im_shape[1:])
			opt_shape = (max(1, max_batch // 2), *im_shape[1:])
			max_shape = (max_batch, *im_shape[1:])
			profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
			LOGGER.info(f'  {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}')
		config.add_optimization_profile(profile)

	# FP16 mode
	if fp16 and builder.platform_has_fast_fp16:
		config.set_flag(trt.BuilderFlag.FP16)
		LOGGER.info("Building FP16 engine")
	else:
		LOGGER.info("Building FP32 engine")

	# Build engine
	build = builder.build_serialized_network if is_trt10 else builder.build_engine
	with build(network, config) as engine:
		if engine is None:
			LOGGER.error("Failed to build TensorRT engine")
			return None
		with open(engine_path, "wb") as f:
			f.write(engine if is_trt10 else engine.serialize())

	LOGGER.info(f"TensorRT engine saved: {engine_path}")
	return engine_path


def main():
	args = parse_args()
	t_start = time.time()

	# Warn if not using dynamic batch (required for ReID)
	if not args.dynamic:
		LOGGER.warning("Building with fixed batch size. Use --dynamic for variable batch sizes (required for ReID).")

	# Setup
	output_dir = ROOT_DIR / args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	# Build vanilla CLIP model (on CPU)
	model = build_vanilla_clip_model()

	# Create dummy input on CPU
	im = torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1])

	# Warmup forward pass
	LOGGER.info("Running warmup inference...")
	with torch.no_grad():
		for _ in range(2):
			y = model(im)

	output_shape = tuple(y.shape)
	LOGGER.info(f"Output shape: {output_shape}")

	# Save PyTorch model
	pt_path = output_dir / f"{args.name}.pt"
	torch.save(model.state_dict(), pt_path)
	pt_size = pt_path.stat().st_size / 1e6
	LOGGER.info(f"Saved PyTorch model: {pt_path} ({pt_size:.1f} MB)")

	# Export to ONNX
	onnx_path = output_dir / f"{args.name}.onnx"
	export_onnx(model, im, onnx_path, dynamic=args.dynamic)
	onnx_size = onnx_path.stat().st_size / 1e6
	LOGGER.info(f"ONNX model size: {onnx_size:.1f} MB")

	if args.skip_engine:
		LOGGER.info("Skipping TensorRT export (--skip-engine)")
		return 0

	# Export to TensorRT
	engine_path = output_dir / f"{args.name}.engine"
	result = export_tensorrt(
		onnx_path,
		engine_path,
		im_shape=im.shape,
		workspace_gb=args.workspace,
		fp16=args.trt_fp16,
		dynamic=args.dynamic
	)

	if result:
		engine_size = engine_path.stat().st_size / 1e6
		LOGGER.info(f"\nExport complete ({time.time() - t_start:.1f}s)")
		LOGGER.info(f"  PyTorch:  {pt_path} ({pt_size:.1f} MB)")
		LOGGER.info(f"  ONNX:     {onnx_path} ({onnx_size:.1f} MB)")
		LOGGER.info(f"  TensorRT: {engine_path} ({engine_size:.1f} MB)")
		LOGGER.info(f"\nUsage in config:")
		LOGGER.info(f'  reid_weights: "checkpoints/reid_weights/{args.name}.engine"')
		LOGGER.info(f'  reid_half: false  # CLIP requires FP32 input')
	else:
		LOGGER.error("TensorRT export failed!")
		return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
