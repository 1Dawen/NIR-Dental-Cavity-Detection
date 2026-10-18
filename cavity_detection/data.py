"""COCO bounding-box dataset for torchvision object detection models."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class CocoDetectionDataset(Dataset):
    """Read COCO image annotations, including images without positive boxes.

    COCO category IDs are mapped to consecutive labels starting at 1 because
    torchvision reserves label 0 for the background class.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotations: str | Path,
        category_to_label: dict[int, int] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotations_path = Path(annotations)
        with self.annotations_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)
        for field in ("images", "annotations", "categories"):
            if field not in coco:
                raise ValueError(f"Invalid COCO annotations: missing '{field}'")
        self.images = coco["images"]
        ids = [int(category["id"]) for category in coco["categories"]]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("COCO categories must contain unique, nonempty IDs")
        if category_to_label is None:
            category_to_label = {id_: i + 1 for i, id_ in enumerate(sorted(ids))}
        self.category_to_label = {int(k): int(v) for k, v in category_to_label.items()}
        if set(ids) != set(self.category_to_label):
            raise ValueError("Categories differ from the training dataset")
        if set(self.category_to_label.values()) != set(range(1, len(ids) + 1)):
            raise ValueError("Category label mapping must be consecutive from 1")
        self.categories = {int(c["id"]): c["name"] for c in coco["categories"]}
        self.anns_per_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_per_image.setdefault(int(ann["image_id"]), []).append(ann)
        image_ids = [int(im["id"]) for im in self.images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("Duplicate COCO image IDs")
        if set(self.anns_per_image) - set(image_ids):
            raise ValueError("Annotations reference unknown images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        info = self.images[index]
        relative = Path(info["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe COCO image filename")
        image_path = self.image_dir / relative
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        boxes, labels, iscrowd = [], [], []
        for ann in self.anns_per_image.get(int(info["id"]), []):
            if int(ann.get("iscrowd", 0)):
                # Crowd regions require separate ignore-region handling during evaluation.
                continue
            category = int(ann["category_id"])
            if category not in self.category_to_label:
                raise ValueError(f"Unknown COCO category: {category}")
            x, y, w, h = map(float, ann["bbox"])
            left, top = max(x, 0.0), max(y, 0.0)
            right, bottom = min(x + w, width), min(y + h, height)
            if right <= left or bottom <= top:
                continue
            boxes.append([left, top, right, bottom])
            labels.append(self.category_to_label[category])
            iscrowd.append(0)
        tensor_boxes = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes else torch.empty((0, 4), dtype=torch.float32)
        )
        target = {
            "boxes": tensor_boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(int(info["id"]), dtype=torch.int64),
            "area": ((tensor_boxes[:, 2] - tensor_boxes[:, 0]) *
                     (tensor_boxes[:, 3] - tensor_boxes[:, 1])),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return to_tensor(image), target


def collate_detection(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
