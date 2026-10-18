"""Train Faster R-CNN on a COCO-format dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cavity_detection.data import CocoDetectionDataset, collate_detection
from cavity_detection.evaluation import evaluate_model
from cavity_detection.model import build_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo"),
                        help="Root containing train/valid/test subdirectories")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--pretrained-backbone", action="store_true",
                        help="Download ImageNet pretrained ResNet-50 backbone weights")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fasterrcnn.pth"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    torch.manual_seed(24)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    train_set = CocoDetectionDataset(args.data / "train", args.data / "train/_annotations.coco.json")
    val_set = CocoDetectionDataset(args.data / "valid", args.data / "valid/_annotations.coco.json",
                                   category_to_label=train_set.category_to_label)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_detection, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=collate_detection, num_workers=0)
    # Avoid upscaling tiny synthetic shapes to the default 800px training size.
    is_demo = "demo" in args.data.parts
    min_size, max_size = (128, 256) if is_demo else (800, 1333)
    if is_demo and device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_detector(
        len(train_set.category_to_label) + 1,
        pretrained_backbone=args.pretrained_backbone,
        min_size=min_size, max_size=max_size,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}: mean_batch_loss={total_loss / len(train_loader):.4f} "
              f"validation_P={val_metrics['precision']:.3f} "
              f"R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "category_to_label": train_set.category_to_label,
                "categories": train_set.categories,
                "epochs": args.epochs,
                "min_size": min_size, "max_size": max_size}, args.output)
    print(f"Saved checkpoint to {args.output}")
    if "demo" in args.data.parts:
        print("NOTE: Synthetic training results are not medical detection performance.")


if __name__ == "__main__":
    main()
