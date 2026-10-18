from scripts.audit_dataset import audit, source_family
from scripts.generate_demo import generate


def test_demo_generates_valid_coco_splits(tmp_path):
    root = tmp_path / "demo"
    generate(root)
    result = audit(root)
    assert result["train"]["referenced_images"] == 12
    assert result["valid"]["referenced_images"] == 4
    assert result["test"]["referenced_images"] == 4
    assert result["train"]["missing_referenced_images"] == 0
    assert result["train"]["invalid_or_outside_boxes"] == 0


def test_roboflow_family_name():
    assert source_family("Copia-de-0113_jpg.rf.deadbeef.jpg") == "Copia-de-0113"
