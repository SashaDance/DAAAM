import torch
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import yaml

from daaam.utils.performance import time_execution_sync
from daaam.utils.logging import PipelineLogger, get_default_logger

try:
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator
except ImportError:
    print("[WARNING] 'efficientvit' library not found. EfficientViT models will not be functional.")
    # Dummy classes to avoid NameErrors
    class create_efficientvit_sam_model:
        def __init__(self, *args, **kwargs):
            raise ImportError("'efficientvit' library not found.")
    class EfficientViTSamAutomaticMaskGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("'efficientvit' library not found.")

try:
    from ultralytics import FastSAM
except ImportError:
    print("[WARNING] 'ultralytics' library not found or FastSAM is not available. FastSAM models will not be functional.")
    #dummy class
    class FastSAM:
        def __init__(self, *args, **kwargs):
            raise ImportError("'ultralytics' library not found or FastSAM not available.")
        def to(self, device):
            raise ImportError("'ultralytics' library not found or FastSAM not available.")
        def __call__(self, *args, **kwargs):
            raise ImportError("'ultralytics' library not found or FastSAM not available.")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("[WARNING] 'segment_anything' library not found. SAM (ViT) models will not be functional.")
    print("Please install it using: pip install git+https://github.com/facebookresearch/segment-anything.git")

    class sam_model_registry:
        @staticmethod
        def __getitem__(item):
            raise ImportError("segment_anything not installed")
    class SamAutomaticMaskGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("segment_anything not installed")

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("[WARNING] 'sam2' library not found. SAM2 models will not be functional.")

    class build_sam2:
        def __init__(self, *args, **kwargs):
            raise ImportError("'sam2' library not found.")
    class SAM2AutomaticMaskGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("'sam2' library not found.")


class UniversalSegmenter:
    """
    A universal segmenter class that wraps FastSAM, SAM (ViT), SAM2.
    Determines the model type from the checkpoint path and provides a consistent
    __call__ interface returning (dets, masks). (in the format required by boxmox tracker)
    """
    def __init__(self,
                 model_checkpoint_path: str,
                 model_config_path: str = "",
                 device: str = "cuda",
                 min_mask_region_area: int = 300,
                 imgsz: Optional[Tuple[int, int]] = None,
                 logger: Optional[PipelineLogger] = None,
                ):
        """
        Args:
            model_checkpoint_path: Full path to the segmentation model checkpoint.
            model_config_path: Path to the model configuration file.
            device: Device to run models on (e.g., "cuda", "cpu").
            min_mask_region_area: Minimum mask area in pixels to keep (filters smaller masks).
            logger: Optional logger function.
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.device = device
        self.model_config_path = model_config_path
        self.min_mask_region_area = min_mask_region_area
        self.imgsz = imgsz
        self.model_type: Optional[str] = None
        self.model_instance: Any = None
        self.is_tensorrt_model: bool = False  # Flag for TensorRT/ONNX models
        self.logger = logger or get_default_logger()

        if not model_config_path:
            self.logger.warning("No model config path provided, using default params.")
            self.model_config = {}
        elif not Path(model_config_path).exists():
            self.logger.error(f"Model config path does not exist: {model_config_path}")
            raise FileNotFoundError(f"Model config path does not exist: {model_config_path}")
        else:
            with open(model_config_path, 'r') as f:
               self.model_config = yaml.safe_load(f)

        # Add imgsz to model config if specified
        if self.imgsz:
            self.model_config['fastsam_imgsz'] = self.imgsz

        self.logger.info(f"loaded SAM model config with parameters: {self.model_config}")

        self._initialize_model()

    def _initialize_model(self):
        """Determines model type and initializes the appropriate segmenter."""
        filename_lower = Path(self.model_checkpoint_path).name.lower()
        
        if "efficientvit" in filename_lower:
            self._initialize_efficientvit(filename_lower)
        elif "fastsam" in filename_lower:
            self._initialize_fastsam()
        elif "sam_vit" in filename_lower:
            self._initialize_sam_vit(filename_lower)
        elif "sam2" in filename_lower:
            self._initialize_sam2()
        else:
            raise ValueError(
                f"Could not determine segmenter type from checkpoint filename: {self.model_checkpoint_path}. "
                "Expected 'efficientvit', 'fastsam', 'sam_vit', or 'sam2' in the name."
            )
        
    def _initialize_efficientvit(self, filename_lower: str = ""):
        self.model_type = "efficientvit"
        if not Path(self.model_checkpoint_path).exists():
            raise FileNotFoundError(f"EfficientViT checkpoint not found: {self.model_checkpoint_path}")
        
        if "_l0" in filename_lower:
            efficientvit_arch_type = "efficientvit-sam-l0"
        elif "_l1" in filename_lower:
            efficientvit_arch_type = "efficientvit-sam-l1"
        elif "_l2" in filename_lower:
            efficientvit_arch_type = "efficientvit-sam-l2"
        elif "_xl0" in filename_lower:
            efficientvit_arch_type = "efficientvit-sam-xl0"
        elif "_xl1" in filename_lower:
            efficientvit_arch_type = "efficientvit-sam-xl1"
        else:
            raise ValueError(
                f"Could not determine EfficientViT SAM architecture type from checkpoint filename: {self.model_checkpoint_path}. "
                "Expected '-l0', '-l1', '-l2', '-xl0', or '-xl1' in the name."
            )

        try:
            efficientvit_sam = create_efficientvit_sam_model(
                name=efficientvit_arch_type,
                weight_url=self.model_checkpoint_path,
                pretrained=True,
                # **self.model_config
            )
            efficientvit_sam.to(self.device)
            if self.device == "cuda":
                efficientvit_sam = efficientvit_sam.cuda().eval()
            self.model_instance = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam)

            self.logger.info(f"Initialized EfficientViT SAM with {self.model_checkpoint_path} on {self.device}")
        except ImportError:
            self.logger.error("EfficientViT library could not be imported. EfficientViT models are unavailable.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize EfficientViT SAM: {e}")
            raise

    def _initialize_fastsam(self):
        self.model_type = "fastsam"
        if not Path(self.model_checkpoint_path).exists():
            raise FileNotFoundError(f"FastSAM checkpoint not found: {self.model_checkpoint_path}")

        filename_lower = Path(self.model_checkpoint_path).name.lower()

        # Check if this is a TensorRT/ONNX model
        self.is_tensorrt_model = filename_lower.endswith(('.engine', '.onnx'))

        # Auto-detect TensorRT and extract dimensions from filename
        if filename_lower.endswith('.engine'):
            self.logger.info(f"Detected TensorRT model: {filename_lower}")
            # Extract dimensions if in filename (e.g., "1024x1248")
            # Filename convention is widthxheight, ultralytics expects (height, width)
            match = re.search(r'(\d+)x(\d+)', filename_lower)
            if match and not self.imgsz:
                width = int(match.group(1))
                height = int(match.group(2))
                # For "1024x1248" engine: width=1024, height=1248
                # But this is actually 1248 width × 1024 height!
                # The filename convention appears to be swapped
                self.imgsz = (width, height)  # Keep as-is since filename seems to already be (h,w)
                self.logger.info(f"Auto-detected TensorRT imgsz from filename: {self.imgsz} (h,w)")
                # Update model config with detected imgsz
                self.model_config['fastsam_imgsz'] = self.imgsz

        try:
            self.model_instance = FastSAM(self.model_checkpoint_path)
            # Only call .to() for PyTorch models, not TensorRT/ONNX
            if not self.is_tensorrt_model:
                self.model_instance.to(self.device)
            self.logger.info(f"Initialized FastSAM with {self.model_checkpoint_path} on {self.device}")
        except ImportError:
            self.logger.error("Ultralytics FastSAM could not be imported. FastSAM models are unavailable.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize FastSAM: {e}")
            raise

    def _initialize_sam_vit(self, filename_lower: str):
        self.model_type = "sam_vit"
        if not Path(self.model_checkpoint_path).exists():
            raise FileNotFoundError(f"SAM (ViT) checkpoint not found: {self.model_checkpoint_path}")
        
        if "vit_h" in filename_lower:
            sam_arch_type = "vit_h"
        elif "vit_l" in filename_lower:
            sam_arch_type = "vit_l"
        elif "vit_b" in filename_lower:
            sam_arch_type = "vit_b"
        else:
            raise ValueError(
                f"Could not determine SAM (ViT) architecture type from checkpoint filename: {self.model_checkpoint_path}. "
                "Expected 'vit_h', 'vit_l', or 'vit_b' in the name."
            )
        
        try:
            if 'SamAutomaticMaskGenerator' not in globals() or SamAutomaticMaskGenerator.__module__ == 'builtins':
                raise ImportError("SAM (segment_anything) library not properly installed.")
            
            sam_model = sam_model_registry[sam_arch_type](checkpoint=self.model_checkpoint_path)
            sam_model.to(device=self.device)
            self.model_instance = SamAutomaticMaskGenerator(model=sam_model, **self.model_config)
            self.logger.info(f"Initialized SAM (ViT) with type {sam_arch_type} from {self.model_checkpoint_path} on {self.device}")
        except ImportError:
            self.logger.error("SAM (segment_anything) could not be imported or initialized. SAM models are unavailable.")
            raise
        except KeyError:
            self.logger.error(f"SAM model type '{sam_arch_type}' not in registry. Check checkpoint name. Available: {list(sam_model_registry.keys())}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize SAM (ViT): {e}")
            raise

    def _initialize_sam2(self):
        self.model_type = "sam2"
        if not Path(self.model_checkpoint_path).exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.model_checkpoint_path}")
        
        if not self.model_config_path or not Path(self.model_config_path).exists():
            raise ValueError("SAM2 model config file is not optional.")

        try:
            if 'SAM2AutomaticMaskGenerator' not in globals() or SAM2AutomaticMaskGenerator.__module__ == 'builtins':
                raise ImportError("SAM2 library (sam2.automatic_mask_generator) not properly installed.")
            if 'build_sam2' not in globals() or build_sam2.__module__ == 'builtins': # Check if it's the dummy
                raise ImportError("SAM2 library (sam2.build_sam) not properly installed.")

            sam2_model_loaded = build_sam2(
                "/" + str(self.model_config_path), 
                self.model_checkpoint_path, 
                device=self.device, 
                apply_postprocessing=False
            )
            
            self.model_instance = SAM2AutomaticMaskGenerator(
                model=sam2_model_loaded
            )
            self.logger.info(f"Initialized SAM2 with checkpoint {self.model_checkpoint_path} and config {self.model_config_path} on {self.device}")
        except ImportError:
            self.logger.error("SAM2 library components could not be imported or initialized. SAM2 models are unavailable.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize SAM2: {e}")
            raise

    def _filter_small_masks(self, dets_list: List, masks_list: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Filter out masks smaller than min_mask_region_area threshold."""
        if self.min_mask_region_area <= 0 or not masks_list:
            return np.array(dets_list) if dets_list else np.empty((0, 6)), masks_list
        
        filtered_dets = []
        filtered_masks = []
        
        for i, mask in enumerate(masks_list):
            mask_area = np.sum(mask)
            if mask_area >= self.min_mask_region_area:
                if i < len(dets_list):
                    filtered_dets.append(dets_list[i])
                filtered_masks.append(mask)
        
        if len(filtered_masks) < len(masks_list):
            self.logger.debug(f"Filtered {len(masks_list) - len(filtered_masks)} small masks (area < {self.min_mask_region_area} pixels)")
        
        dets_np = np.array(filtered_dets) if filtered_dets else np.empty((0, 6))
        return dets_np, filtered_masks
    
    def _preprocess_frame_for_sam(self, frame: np.ndarray) -> np.ndarray:
        """Ensures frame is uint8 RGB for SAM."""
        if frame.dtype != np.uint8:
            self.logger.info(f"Converting frame to uint8 for SAM. Original dtype: {frame.dtype}")
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0 and frame.min() >= 0.0:
                frame = (frame * 255).astype(np.uint8)
            elif np.issubdtype(frame.dtype, np.integer):
                
                frame = frame.astype(np.uint8)
            else:
                self.logger.warning(f"Frame dtype {frame.dtype} conversion to uint8 for SAM might be lossy or incorrect.")
                frame = frame.astype(np.uint8)
        
        if frame.ndim != 3 or frame.shape[2] != 3:
             raise ValueError(f"Input image for SAM must be 3-channel (HWC), got shape {frame.shape}")
        return frame
    
    def _process_efficientvit(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Processes frame with EfficientViT SAM and converts output."""
        frame = self._preprocess_frame_for_sam(frame) 
        
        if kwargs:
            self.logger.warning(f"EfficientViT SAM received runtime kwargs: {kwargs}, but they are currently ignored as generator is pre-configured.")

        anns = self.model_instance.generate(frame)

        if not anns:
            return np.empty((0, 6)), []
    
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        
        dets_list = []
        masks_list = []
        for ann in sorted_anns:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            score = ann.get('predicted_iou', 0.0)
            cls_id = 0

            dets_list.append([x1, y1, x2, y2, score, cls_id])
            masks_list.append(ann['segmentation'])

        # Apply mask size filtering
        dets, masks_list = self._filter_small_masks(dets_list, masks_list)
        return dets, masks_list

    def _process_fastsam(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Processes frame with FastSAM and converts output."""

        current_kwargs = {**self.model_config, **kwargs}

        # # Check if frame size matches configured imgsz (for TensorRT models)
        # if self.imgsz:
        #     # self.imgsz is already (height, width), same as numpy shape[:2]
        #     if frame.shape[:2] != self.imgsz:
        #         self.logger.warning(
        #             f"Frame size {frame.shape[:2]} (h,w) doesn't match configured imgsz {self.imgsz} (h,w). "
        #             f"FastSAM will handle resizing internally"
        #         )

        # Prepare parameters for FastSAM
        fastsam_params = {
            "source": frame,
            "retina_masks": current_kwargs.get("fastsam_retina_masks", current_kwargs.get("retina_masks", True)),
            "conf": current_kwargs.get("fastsam_conf", current_kwargs.get("conf", 0.3)),
            "iou": current_kwargs.get("fastsam_iou", current_kwargs.get("iou", 0.5)),
            "verbose": False
        }

        # Add imgsz parameter if specified
        if "fastsam_imgsz" in current_kwargs:
            fastsam_params["imgsz"] = current_kwargs["fastsam_imgsz"]
            self.logger.debug(f"Using imgsz={fastsam_params['imgsz']} for FastSAM inference")

        # For TensorRT/ONNX models, pass device directly in inference call
        if hasattr(self, 'is_tensorrt_model') and self.is_tensorrt_model:
            fastsam_params["device"] = self.device
            self.logger.debug(f"TensorRT/ONNX model detected, passing device={self.device} to inference")

        results = self.model_instance(**fastsam_params)
        
        dets_list = []
        masks_list = []
        if results and len(results) > 0 and results[0].boxes is not None: 
            r = results[0] 
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = np.zeros_like(scores) 

            if r.masks is not None:
                for mask_tensor in r.masks.data:
                    masks_list.append(mask_tensor.cpu().numpy().astype(bool))
            
            if masks_list and len(masks_list) != len(boxes):
                 self.logger.warning(f"Mismatch between number of FastSAM boxes ({len(boxes)}) and masks ({len(masks_list)}). Using minimum count.")
                 min_count = min(len(boxes), len(masks_list))
                 boxes = boxes[:min_count]
                 scores = scores[:min_count]
                 classes = classes[:min_count]
                 masks_list = masks_list[:min_count]
            elif not masks_list and len(boxes) > 0:
                 self.logger.warning("FastSAM produced boxes but no masks. Creating empty masks.")
                 masks_list = [np.zeros(frame.shape[:2], dtype=bool) for _ in range(len(boxes))]


            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                dets_list.append([x1, y1, x2, y2, scores[i], classes[i]])
        
        # Apply mask size filtering
        dets_np, masks_list = self._filter_small_masks(dets_list, masks_list)
        return dets_np, masks_list

    def _process_sam_vit(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Processes frame with SAM (ViT) and converts output."""
        frame = self._preprocess_frame_for_sam(frame) 
        
        if kwargs:
            self.logger.info(f"SAM (ViT) received runtime kwargs: {kwargs}, but they are currently ignored as generator is pre-configured.")

        raw_sam_masks = self.model_instance.generate(frame)
        
        dets_list = []
        masks_list = []
        if not raw_sam_masks:
            return np.empty((0, 6)), []

        for r_mask_info in raw_sam_masks:
            x, y, w, h = r_mask_info['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            score = r_mask_info.get('predicted_iou', 0.0)
            cls_id = 0 

            dets_list.append([x1, y1, x2, y2, score, cls_id])
            masks_list.append(r_mask_info['segmentation']) 

        # Apply mask size filtering
        dets_np, masks_list = self._filter_small_masks(dets_list, masks_list)
        return dets_np, masks_list

    def _process_sam2(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Processes frame with SAM2 and converts output."""
        frame = self._preprocess_frame_for_sam(frame) # Assuming SAM2 also needs uint8 RGB

        if kwargs:
            self.logger.info(f"SAM2 received runtime kwargs: {kwargs}, but they are currently ignored as generator is pre-configured with SAM1-like defaults.")

        raw_sam2_masks = self.model_instance.generate(frame) # Call SAM2 generator
        
        dets_list = []
        masks_list = []
        if not raw_sam2_masks:
            return np.empty((0, 6)), []

        for r_mask_info in raw_sam2_masks:
            # Output format is similar to SAM1 as per user's info
            x, y, w, h = r_mask_info['bbox'] # XYWH format
            x1, y1, x2, y2 = x, y, x + w, y + h
            score = r_mask_info.get('predicted_iou', 0.0)
            cls_id = 0 # SAM2 is class-agnostic

            dets_list.append([x1, y1, x2, y2, score, cls_id])
            masks_list.append(r_mask_info['segmentation']) # Boolean np.ndarray

        # Apply mask size filtering
        dets_np, masks_list = self._filter_small_masks(dets_list, masks_list)
        return dets_np, masks_list

    def __call__(self, source: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Processes the input frame using the initialized segmenter.
        Args:
            source: Input image as a numpy array (H, W, C).
            **kwargs: Runtime parameters to override defaults for specific models
                      (e.g., conf, iou for FastSAM).
        Returns:
            A tuple (dets, masks):
            - dets: np.ndarray of shape (N, 6) with [x1, y1, x2, y2, score, cls_id]
            - masks: List of N boolean np.ndarray masks.
        """
        if self.model_instance is None and self.model_type != "sam2":
            raise RuntimeError(f"UniversalSegmenter model_instance is not initialized for model type '{self.model_type}'. Cannot process frame.")

        if self.model_type == "efficientvit":
            return self._process_efficientvit(source, **kwargs)
        elif self.model_type == "fastsam":
            return self._process_fastsam(source, **kwargs)
        elif self.model_type == "sam_vit":
            return self._process_sam_vit(source, **kwargs)
        elif self.model_type == "sam2":
            return self._process_sam2(source, **kwargs) 
        else:
            raise RuntimeError(f"UniversalSegmenter called with an uninitialized or unknown model type: {self.model_type}")
