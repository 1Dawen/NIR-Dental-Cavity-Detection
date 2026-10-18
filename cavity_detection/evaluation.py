"""Thresholded bounding-box detection metrics. These are not COCO mAP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class DetectionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "DetectionCounts") -> "DetectionCounts":
        return DetectionCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def as_dict(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": p, "recall": r, "f1": f1}


def intersection_over_union(box_a, box_b) -> float:
    x0, y0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x1, y1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0., x1 - x0) * max(0., y1 - y0)
    area_a = max(0., box_a[2] - box_a[0]) * max(0., box_a[3] - box_a[1])
    area_b = max(0., box_b[2] - box_b[0]) * max(0., box_b[3] - box_b[1])
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def match_image(prediction: dict, target: dict, *, score_threshold: float = 0.5,
                iou_threshold: float = 0.5) -> DetectionCounts:
    """Confidence-ordered greedy one-to-one matching of same-class boxes."""
    if not 0 <= score_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("Thresholds must be within [0, 1]")
    pred_boxes = prediction["boxes"].detach().cpu().tolist()
    pred_scores = prediction["scores"].detach().cpu().tolist()
    pred_labels = prediction["labels"].detach().cpu().tolist()
    gt_boxes = target["boxes"].detach().cpu().tolist()
    gt_labels = target["labels"].detach().cpu().tolist()
    order = sorted((i for i, score in enumerate(pred_scores) if score >= score_threshold),
                   key=lambda i: pred_scores[i], reverse=True)
    matched = set()
    tp = fp = 0
    for idx in order:
        candidates = (
            (intersection_over_union(pred_boxes[idx], gt_boxes[j]), j)
            for j in range(len(gt_boxes))
            if j not in matched and pred_labels[idx] == gt_labels[j]
        )
        best = max(candidates, default=(0.0, -1))
        if best[1] >= 0 and best[0] >= iou_threshold:
            matched.add(best[1])
            tp += 1
        else:
            fp += 1
    return DetectionCounts(tp, fp, len(gt_boxes) - len(matched))


def evaluate_model(model, data_loader: Iterable, device: torch.device,
                   score_threshold: float = 0.5, iou_threshold: float = 0.5):
    model.eval()
    total = DetectionCounts()
    with torch.inference_mode():
        for images, targets in data_loader:
            predictions = model([img.to(device) for img in images])
            for prediction, target in zip(predictions, targets):
                total += match_image(prediction, target, score_threshold=score_threshold,
                                     iou_threshold=iou_threshold)
    return total.as_dict()
