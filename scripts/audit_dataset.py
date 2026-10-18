"""Audit COCO image references, duplicates, augmentation-family overlap, and basic boxes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


_SPLITS = ("train", "valid", "test")


def source_family(filename: str) -> str:
    """Approximate augmentation family from Roboflow export names.

    This only identifies shared filename roots; it cannot prove unique patients.
    """
    return re.sub(r"_jpg\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$", "", filename, flags=re.I)


def audit(root: Path) -> dict:
    summary = {}
    families = {}
    image_hashes = {}
    for split in _SPLITS:
        folder = root / split
        path = folder / "_annotations.coco.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing COCO annotations: {path}")
        info = json.loads(path.read_text(encoding="utf-8"))
        images = info.get("images", [])
        annotations = info.get("annotations", [])
        indexed = {int(im["id"]): im for im in images}
        if len(indexed) != len(images):
            raise ValueError(f"Duplicate COCO image IDs in {split}")
        all_files = {p.name for p in folder.iterdir() if p.is_file()
                     and p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        references = {im["file_name"] for im in images}
        available = references & all_files
        missing = references - all_files
        orphaned = all_files - references
        hashes = {}
        for name in available:
            hashes[hashlib.sha256((folder / name).read_bytes()).hexdigest()] = name
        image_hashes[split] = hashes
        families[split] = {source_family(im["file_name"]) for im in images}
        bad_boxes = 0
        for ann in annotations:
            image = indexed.get(int(ann["image_id"]))
            if not image:
                bad_boxes += 1
                continue
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0 or x >= image["width"] or y >= image["height"] or x + w <= 0 or y + h <= 0:
                bad_boxes += 1
        summary[split] = {
            "referenced_images": len(images), "annotations": len(annotations),
            "referenced_images_found": len(available), "missing_referenced_images": len(missing),
            "unreferenced_extra_images": len(orphaned),
            "distinct_filename_families": len(families[split]),
            "invalid_or_outside_boxes": bad_boxes,
            "categories": {str(c["id"]): c["name"] for c in info.get("categories", [])},
        }
    summary["cross_split"] = {}
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        summary["cross_split"][f"{first}_vs_{second}"] = {
            "same_filename_family": len(families[first] & families[second]),
            "identical_referenced_image_hashes": len(image_hashes[first].keys() & image_hashes[second].keys()),
        }
    summary["notes"] = (
        "Checks for exact byte duplicates and filename families in the COCO-referenced images. "
        "Near duplicates and patient-level overlap require a separate image/provenance review."
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo"))
    print(json.dumps(audit(parser.parse_args().data), indent=2))
