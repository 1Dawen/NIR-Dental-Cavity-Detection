import torch
from cavity_detection.evaluation import DetectionCounts, intersection_over_union, match_image


def cases(pred_boxes, pred_labels, scores, gt_boxes, gt_labels):
    return ({"boxes": torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(pred_labels, dtype=torch.int64),
             "scores": torch.tensor(scores, dtype=torch.float32)},
            {"boxes": torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(gt_labels, dtype=torch.int64)})


def test_iou_exact_and_disjoint():
    assert intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert intersection_over_union([0, 0, 2, 2], [5, 5, 8, 8]) == 0


def test_one_to_one_matching_counts_duplicate_predictions_as_fp():
    prediction, truth = cases([[0, 0, 8, 8], [0, 0, 8, 8]], [1, 1], [.95, .9],
                              [[0, 0, 8, 8]], [1])
    assert match_image(prediction, truth) == DetectionCounts(1, 1, 0)


def test_different_class_not_match():
    prediction, truth = cases([[0, 0, 8, 8]], [2], [.99], [[0, 0, 8, 8]], [1])
    assert match_image(prediction, truth) == DetectionCounts(0, 1, 1)


def test_no_predictions_counts_false_negatives():
    prediction, truth = cases([], [], [], [[0, 0, 8, 8]], [1])
    assert match_image(prediction, truth) == DetectionCounts(0, 0, 1)


def test_predictions_in_empty_image_are_false_positives():
    prediction, truth = cases([[0, 0, 8, 8]], [1], [.99], [], [])
    assert match_image(prediction, truth) == DetectionCounts(0, 1, 0)


def test_metric_aggregation():
    counts = DetectionCounts(2, 1, 1) + DetectionCounts(1, 0, 1)
    m = counts.as_dict()
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 2
    assert m["precision"] == .75
    assert m["recall"] == .6
