import json

import pytest
import torch
from PIL import Image

from cavity_detection.data import CocoDetectionDataset, collate_detection


def create_coco(tmp_path, *, invalid_filename=False):
    img_dir = tmp_path / "train"
    img_dir.mkdir()
    Image.new("RGB", (32, 24)).save(img_dir / "image.png")
    payload = {
        "images": [{"id": 10, "file_name": "../outside.png" if invalid_filename else "image.png",
                    "width": 32, "height": 24}],
        "categories": [{"id": 7, "name": "cavity"}],
        "annotations": [{"image_id": 10, "category_id": 7, "bbox": [25, 17, 20, 20]}],
    }
    path = img_dir / "_annotations.coco.json"
    path.write_text(json.dumps(payload))
    return img_dir, path


def test_coco_category_remapping_and_box_clipping(tmp_path):
    img_dir, path = create_coco(tmp_path)
    dataset = CocoDetectionDataset(img_dir, path)
    image, target = dataset[0]
    assert image.shape == (3, 24, 32)
    assert dataset.category_to_label == {7: 1}
    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[25, 17, 32, 24]]
    assert target["area"].tolist() == [49]


def test_negative_image_has_no_dummy_background_box(tmp_path):
    img_dir, path = create_coco(tmp_path)
    payload = json.loads(path.read_text())
    payload["annotations"] = []
    path.write_text(json.dumps(payload))
    _, target = CocoDetectionDataset(img_dir, path)[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].numel() == 0


def test_reject_path_traversal(tmp_path):
    img_dir, path = create_coco(tmp_path, invalid_filename=True)
    with pytest.raises(ValueError, match="Unsafe"):
        CocoDetectionDataset(img_dir, path)[0]


def test_mismatched_category_ids(tmp_path):
    img_dir, path = create_coco(tmp_path)
    with pytest.raises(ValueError, match="Categories differ"):
        CocoDetectionDataset(img_dir, path, category_to_label={2: 1})


def test_collate_preserves_targets(tmp_path):
    img_dir, path = create_coco(tmp_path)
    sample = CocoDetectionDataset(img_dir, path)[0]
    imgs, tgts = collate_detection([sample, sample])
    assert len(imgs) == len(tgts) == 2
