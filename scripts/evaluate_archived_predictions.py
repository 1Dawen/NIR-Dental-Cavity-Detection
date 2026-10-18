"""Evaluate archived COCO bbox predictions without model weights or source images.

The results are threshold-specific matching counts, NOT COCO mean average precision.
"""
import argparse
import json
from pathlib import Path

import torch

from cavity_detection.evaluation import DetectionCounts, match_image


def to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def evaluate_annotations(annotations: dict, predictions: list, confidence: float, iou: float) -> dict:
    by_image_gt = {im["id"]: [] for im in annotations["images"]}
    by_image_dt = {im["id"]: [] for im in annotations["images"]}
    for ann in annotations["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["image_id"] in by_image_gt:
            by_image_gt[ann["image_id"]].append(ann)
    for prediction in predictions:
        if prediction["image_id"] not in by_image_dt:
            raise ValueError("Prediction references an unknown test image ID")
        by_image_dt[prediction["image_id"]].append(prediction)
    total = DetectionCounts()
    for img_id in by_image_gt:
        gt = by_image_gt[img_id]
        dt = by_image_dt[img_id]
        target = {
            "boxes": torch.tensor([to_xyxy(x["bbox"]) for x in gt], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([x["category_id"] for x in gt], dtype=torch.int64),
        }
        predicted = {
            "boxes": torch.tensor([to_xyxy(x["bbox"]) for x in dt], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([x["category_id"] for x in dt], dtype=torch.int64),
            "scores": torch.tensor([x["score"] for x in dt], dtype=torch.float32),
        }
        total += match_image(predicted, target, score_threshold=confidence, iou_threshold=iou)
    return total.as_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    ground_truth = json.loads(args.annotations.read_text())
    predictions = json.loads(args.predictions.read_text())
    print(json.dumps(evaluate_annotations(ground_truth, predictions, args.confidence, args.iou), indent=2))
    print("Archived predictions may not have verifiable checkpoint/dataset provenance.")


if __name__ == "__main__":
    main()
