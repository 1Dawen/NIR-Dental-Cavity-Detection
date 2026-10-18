from scripts.evaluate_archived_predictions import evaluate_annotations


def test_saved_json_evaluator_counts_detection():
    coco = {
        "images": [{"id": 0}, {"id": 1}],
        "annotations": [{"image_id": 0, "category_id": 1, "bbox": [2, 2, 4, 4]}],
    }
    predictions = [
        {"image_id": 0, "category_id": 1, "bbox": [2, 2, 4, 4], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [2, 2, 4, 4], "score": 0.8},
    ]
    out = evaluate_annotations(coco, predictions, 0.5, 0.5)
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 0


def test_saved_json_evaluator_rejects_unknown_image():
    coco = {"images": [{"id": 0}], "annotations": []}
    predictions = [{"image_id": 3, "category_id": 1, "bbox": [2, 2, 4, 4], "score": .9}]
    import pytest
    with pytest.raises(ValueError, match="unknown test image ID"):
        evaluate_annotations(coco, predictions, 0.5, 0.5)
