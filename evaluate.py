"""Evaluate a trained Faster R-CNN detector using IoU-thresholded box matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cavity_detection.data import CocoDetectionDataset, collate_detection
from cavity_detection.evaluation import evaluate_model
from cavity_detection.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fasterrcnn.pth"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping = load_checkpoint(args.checkpoint, device)
    dataset = CocoDetectionDataset(args.data / args.split,
                                   args.data / args.split / "_annotations.coco.json", mapping)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_detection)
    metrics = evaluate_model(model, loader, device,
                             score_threshold=args.confidence, iou_threshold=args.iou)
    print(json.dumps({"split": args.split, "confidence": args.confidence,
                      "iou": args.iou, "counts_and_metrics": metrics}, indent=2))
    if "demo" in args.data.parts:
        print("Synthetic results do not establish real dental-cavity detection accuracy.")


if __name__ == "__main__":
    main()
