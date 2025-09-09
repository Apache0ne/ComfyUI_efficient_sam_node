from .efficient_sam_node import (
    EfficientViTSAMLoader,
    EfficientViTSAMPredictorNode,
    EfficientViTSAMPointPredictorNode,
    EfficientViTSAMAutoMaskGeneratorNode,
    EfficientViTSAMVideoPredictorNode,
    EfficientViTSAMVideoPointPredictorNode,
    EfficientViTSAMVideoAutoMaskGeneratorNode,
)

NODE_CLASS_MAPPINGS = {
    "EfficientViTSAMLoader": EfficientViTSAMLoader,
    "EfficientViTSAMPredictorNode": EfficientViTSAMPredictorNode,
    "EfficientViTSAMPointPredictorNode": EfficientViTSAMPointPredictorNode,
    "EfficientViTSAMAutoMaskGeneratorNode": EfficientViTSAMAutoMaskGeneratorNode,
    "EfficientViTSAMVideoPredictorNode": EfficientViTSAMVideoPredictorNode,
    "EfficientViTSAMVideoPointPredictorNode": EfficientViTSAMVideoPointPredictorNode,
    "EfficientViTSAMVideoAutoMaskGeneratorNode": EfficientViTSAMVideoAutoMaskGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EfficientViTSAMLoader": "EfficientViT-SAM Loader",
    "EfficientViTSAMPredictorNode": "EfficientViT-SAM Predictor (BBox)",
    "EfficientViTSAMPointPredictorNode": "EfficientViT-SAM Predictor (Points)",
    "EfficientViTSAMAutoMaskGeneratorNode": "EfficientViT-SAM Auto Mask Generator",
    "EfficientViTSAMVideoPredictorNode": "EfficientViT-SAM Video Predictor (BBox)",
    "EfficientViTSAMVideoPointPredictorNode": "EfficientViT-SAM Video Predictor (Points)",
    "EfficientViTSAMVideoAutoMaskGeneratorNode": "EfficientViT-SAM Video Auto Mask Generator",
}