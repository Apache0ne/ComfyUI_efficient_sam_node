import os
import re
import torch
import numpy as np
import folder_paths
from PIL import Image

# Required imports from the EfficientViT-SAM library
# Make sure 'efficientvit' is installed in your ComfyUI environment
# pip install git+https://github.com/mit-han-lab/efficientvit.git
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor, EfficientViTSamAutomaticMaskGenerator

# Helper: Find EfficientViT-SAM models in ComfyUI/models/sam or /sams
def find_efficientvit_sam_models():
    """
    Scans for EfficientViT-SAM models inside ComfyUI/models/sam and ComfyUI/models/sams
    """
    model_filenames = set()
    
    models_dir = folder_paths.models_dir  

    sam_dir_names = ["sam", "sams"]

    for dir_name in sam_dir_names:
        full_path = os.path.join(models_dir, dir_name)
        if os.path.isdir(full_path):
            print(f"[EfficientViT-SAM Loader] Scanning for models in: {full_path}")
            for filename in os.listdir(full_path):
                if filename.startswith("efficientvit_sam_") and filename.endswith((".pt", ".pth", ".safetensors")):
                    model_filenames.add(filename)
    
    return sorted(list(model_filenames))

available_models = find_efficientvit_sam_models()

# NODE 1: Model Loader
class EfficientViTSAMLoader:
    @classmethod
    def INPUT_TYPES(s):
        if not available_models:
            raise RuntimeError(
                "No EfficientViT-SAM models found. "
                "Please place models like 'efficientvit_sam_l0.pt' inside 'ComfyUI/models/sam/'."
            )
        return {
            "required": {
                "model_name": (available_models,),
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("SAM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "EfficientViT-SAM"

    def load_model(self, model_name, device):
        ckpt_path = None
        models_dir = folder_paths.models_dir
        sam_dir_names = ["sam", "sams"]

        for dir_name in sam_dir_names:
            potential_path = os.path.join(models_dir, dir_name, model_name)
            if os.path.isfile(potential_path):
                ckpt_path = potential_path
                break

        if not ckpt_path:
            raise FileNotFoundError(
                f"Could not find checkpoint '{model_name}' in 'sam' or 'sams' under '{models_dir}'."
            )

        match = re.search(r'efficientvit_sam_(.*)\.(pt|pth|safetensors)', model_name)
        if not match:
            raise ValueError(f"Invalid model filename: {model_name}")
        model_type = match.group(1)

        full_model_name = f"efficientvit-sam-{model_type}"

        print(f"Loading EfficientViT-SAM model: {full_model_name} from {ckpt_path}")
        
        sam_model = create_efficientvit_sam_model(name=full_model_name, pretrained=False)
        sam_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        sam_model = sam_model.to(device).eval()

        sam_predictor = EfficientViTSamPredictor(sam_model)
        return (sam_predictor,)

# NODE 2: The Segmentation Predictor (BBox)
class EfficientViTSAMPredictorNode:
    """
    This node uses a loaded EfficientViT-SAM model to predict a segmentation
    mask based on a bounding box (BBOX) input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "bbox": ("BBOX",), 
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    FUNCTION = "predict_segmentation"
    CATEGORY = "EfficientViT-SAM"

    def predict_segmentation(self, sam_model, image, bbox):
            (h, w) = image.shape[1:3]
            empty_mask_tensor = torch.zeros(1, h, w, dtype=torch.float32, device="cpu")
            empty_image_tensor = torch.zeros(1, h, w, 3, dtype=torch.float32, device="cpu")
            empty_return = (empty_mask_tensor, empty_image_tensor)

            if not isinstance(bbox, list) or not bbox:
                print("[EfficientViTSAM Bbox] Warning: BBOX input is not a valid list or is empty. Returning a black mask.")
                return empty_return

            box_data = bbox[0]

            if not isinstance(box_data, (list, tuple)) or len(box_data) < 4:
                print(f"[EfficientViTSAM Bbox] Warning: Received invalid data for BBOX. Expected a list of 4 numbers, but got data of type '{type(box_data)}'. This is likely point data connected to the wrong input. Returning a black mask.")
                return empty_return
            
            predictor = sam_model
            image_np = (image[0].numpy() * 255).astype(np.uint8)
            predictor.set_image(image_np)
            bbox_np = np.array([box_data[:4]], dtype=np.float32)

            masks, scores, logits = predictor.predict(
                box=bbox_np,
                multimask_output=False,
            )

            mask_np = masks[0].astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

            seg_image = mask_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

            return (mask_tensor, seg_image)

# NODE 3: The Point-Based Segmentation Predictor
class EfficientViTSAMPointPredictorNode:
    """
    This node uses a loaded EfficientViT-SAM model to predict a segmentation
    mask based on foreground and background point prompts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "points": ("SAM_PROMPT",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    FUNCTION = "predict_segmentation_with_points"
    CATEGORY = "EfficientViT-SAM"

    def predict_segmentation_with_points(self, sam_model, image, points):
        predictor = sam_model

        image_np = (image[0].numpy() * 255).astype(np.uint8)
        predictor.set_image(image_np)

        point_coords = np.array(points.get("points", []), dtype=np.float32)
        point_labels = np.array(points.get("labels", []), dtype=np.int32)

        if point_coords.shape[0] == 0:
            h, w = image_np.shape[:2]
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device="cpu")
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32, device="cpu")
            return (empty_mask, empty_image)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        mask_np = masks[0].astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        seg_image = mask_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

        return (mask_tensor, seg_image)

# NODE 4: The Automatic Mask Generator
class EfficientViTSAMAutoMaskGeneratorNode:
    """
    This node automatically generates segmentation masks for all objects
    detected in an image, without needing any input prompts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    FUNCTION = "generate_masks"
    CATEGORY = "EfficientViT-SAM"

    def generate_masks(self, sam_model, image):
        raw_model = sam_model.model
        mask_generator = EfficientViTSamAutomaticMaskGenerator(raw_model)
        image_np = (image[0].numpy() * 255).astype(np.uint8)
        masks_list = mask_generator.generate(image_np)

        if not masks_list:
            h, w = image_np.shape[:2]
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device="cpu")
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32, device="cpu")
            return (empty_mask, empty_image)

        all_masks = [torch.from_numpy(m['segmentation']) for m in masks_list]
        mask_batch = torch.stack(all_masks).float()

        seg_images = mask_batch.unsqueeze(-1).expand(-1, -1, -1, 3)

        return (mask_batch, seg_images)

# NODE 5: Video BBox Segmentation Predictor
class EfficientViTSAMVideoPredictorNode:
    """
    Processes a batch of images (video frames) to predict segmentation masks based on a bounding box.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "bbox": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    FUNCTION = "predict_video_segmentation"
    CATEGORY = "EfficientViT-SAM"

    def predict_video_segmentation(self, sam_model, image, bbox):
        (b, h, w, c) = image.shape
        empty_mask_tensor = torch.zeros(b, h, w, dtype=torch.float32, device="cpu")
        empty_image_tensor = torch.zeros(b, h, w, 3, dtype=torch.float32, device="cpu")
        empty_return = (empty_mask_tensor, empty_image_tensor)

        if not isinstance(bbox, list) or not bbox:
            print("[EfficientViTSAM Video Bbox] Warning: BBOX input is not a valid list or is empty.")
            return empty_return

        box_data = bbox[0]
        if not isinstance(box_data, (list, tuple)) or len(box_data) < 4:
            print(f"[EfficientViTSAM Video Bbox] Warning: Invalid BBOX data type '{type(box_data)}'.")
            return empty_return

        predictor = sam_model
        bbox_np = np.array([box_data[:4]], dtype=np.float32)
        
        masks = []
        seg_images = []

        for i in range(b):
            image_np = (image[i].numpy() * 255).astype(np.uint8)
            predictor.set_image(image_np)

            frame_masks, _, _ = predictor.predict(box=bbox_np, multimask_output=False)
            
            mask_np = frame_masks[0].astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np)
            masks.append(mask_tensor)

            seg_image = mask_tensor.unsqueeze(-1).expand(-1, -1, 3)
            seg_images.append(seg_image)

        return (torch.stack(masks), torch.stack(seg_images))

# NODE 6: Video Point-Based Segmentation Predictor
class EfficientViTSAMVideoPointPredictorNode:
    """
    Processes a batch of images (video frames) to predict segmentation masks based on point prompts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "points": ("SAM_PROMPT",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    FUNCTION = "predict_video_segmentation_with_points"
    CATEGORY = "EfficientViT-SAM"

    def predict_video_segmentation_with_points(self, sam_model, image, points):
        predictor = sam_model
        (b, h, w, c) = image.shape

        point_coords = np.array(points.get("points", []), dtype=np.float32)
        point_labels = np.array(points.get("labels", []), dtype=np.int32)

        if point_coords.shape[0] == 0:
            empty_mask = torch.zeros((b, h, w), dtype=torch.float32, device="cpu")
            empty_image = torch.zeros((b, h, w, 3), dtype=torch.float32, device="cpu")
            return (empty_mask, empty_image)

        masks = []
        seg_images = []

        for i in range(b):
            image_np = (image[i].numpy() * 255).astype(np.uint8)
            predictor.set_image(image_np)

            frame_masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

            mask_np = frame_masks[0].astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np)
            masks.append(mask_tensor)

            seg_image = mask_tensor.unsqueeze(-1).expand(-1, -1, 3)
            seg_images.append(seg_image)

        return (torch.stack(masks), torch.stack(seg_images))

# NODE 7: Video Automatic Mask Generator
class EfficientViTSAMVideoAutoMaskGeneratorNode:
    """
    Automatically generates segmentation masks for all objects in a batch of images (video frames).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    FUNCTION = "generate_video_masks"
    CATEGORY = "EfficientViT-SAM"

    def generate_video_masks(self, sam_model, image):
        raw_model = sam_model.model
        mask_generator = EfficientViTSamAutomaticMaskGenerator(raw_model)
        (b, h, w, c) = image.shape

        all_masks = []
        all_seg_images = []

        for i in range(b):
            image_np = (image[i].numpy() * 255).astype(np.uint8)
            masks_list = mask_generator.generate(image_np)

            if not masks_list:
                continue

            for m in masks_list:
                mask_tensor = torch.from_numpy(m['segmentation']).float()
                all_masks.append(mask_tensor)
                
                seg_image = mask_tensor.unsqueeze(-1).expand(-1, -1, 3)
                all_seg_images.append(seg_image)

        if not all_masks:
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32, device="cpu")
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32, device="cpu")
            return (empty_mask, empty_image)

        return (torch.stack(all_masks), torch.stack(all_seg_images))