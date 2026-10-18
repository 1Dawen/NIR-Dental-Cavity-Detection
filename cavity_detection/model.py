"""Faster R-CNN initialization and checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(num_classes: int, pretrained_backbone: bool = False,
                   min_size: int = 800, max_size: int = 1333):
    """Create a detector; pretrained backbone download is opt-in."""
    if num_classes < 2:
        raise ValueError("At least one foreground category is required")
    backbone_weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=backbone_weights,
        min_size=min_size, max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(path: str | Path, device: torch.device):
    # Never load checkpoints of unknown origin. weights_only is intentionally set.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    mapping = {int(k): int(v) for k, v in checkpoint["category_to_label"].items()}
    model = build_detector(
        len(mapping) + 1, pretrained_backbone=False,
        min_size=int(checkpoint.get("min_size", 800)),
        max_size=int(checkpoint.get("max_size", 1333)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), mapping
