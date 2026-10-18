"""Generate small synthetic COCO detection images for code testing only.

These geometric illustrations are NOT dental or patient imagery.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def generate(root: Path, seed: int = 24) -> None:
    rng = random.Random(seed)
    for split, count in (("train", 12), ("valid", 4), ("test", 4)):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for i in range(count):
            image = Image.new("RGB", (96, 96), (180, 185, 190))
            draw = ImageDraw.Draw(image)
            # Abstract background and fake object. Nothing here resembles clinical imaging.
            x, y = rng.randint(12, 35), rng.randint(12, 35)
            draw.ellipse((x - 6, y - 7, x + 42, y + 45), fill=(224, 226, 221), outline="gray")
            annotated = (i % 4 != 0)  # Some samples have no positive objects.
            if annotated:
                bx, by = x + rng.randint(7, 18), y + rng.randint(9, 19)
                draw.rectangle((bx, by, bx + 9, by + 7), fill=(80, 85, 88))
                annotations.append({
                    "id": len(annotations) + 1, "image_id": i,
                    "category_id": 1, "bbox": [bx, by, 9, 7],
                    "area": 63, "segmentation": [], "iscrowd": 0,
                })
            filename = f"synthetic_{split}_{i:03d}.png"
            image.save(folder / filename)
            images.append({"id": i, "file_name": filename, "width": 96, "height": 96})
        payload = {
            "info": {"description": "Synthetic geometric shapes; not dental data"},
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "synthetic_spot"}],
        }
        (folder / "_annotations.coco.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Synthetic demo written to {root}. Not suitable for clinical evaluation.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    generate(parser.parse_args().output)
