# Technical review of the archived capstone

This review is based on the supplied `Capstone.rar`. It describes what is actually present and what the maintained repository changes. It is not a clinical or independent model-performance review.

## Available material

The archive contains five substantive Python scripts (training, inference-to-JSON, scoring, Tkinter viewer and manual image cropping), plus a temporary editor fragment. It also contains COCO annotation files for training, validation and testing, images, historic prediction JSON files, several detection illustrations, and a Blender model and backup. No trained Faster R-CNN checkpoint was present.

The original Faster R-CNN setup used `fasterrcnn_resnet50_fpn`, a new predictor head for the `cavity` class, SGD (`lr=0.0025`, momentum `0.9`, weight decay `0.0005`), batch size 3, 10 epochs and a fixed CUDA device. The archive alone does not establish which historic checkpoint produced each saved prediction JSON.

## COCO-referenced dataset audit

Using `scripts/audit_dataset.py` on the extracted archive:

| Split | Referenced images | Annotations | Unreferenced JPGs | Unique source-name families |
| --- | ---: | ---: | ---: | ---: |
| Train | 603 | 1,065 | 625 | 201 |
| Validation | 61 | 120 | 56 | 61 |
| Test | 31 | 61 | 26 | 31 |

All referenced images were present and the audit found no wholly invalid/outside boxes. The training split contains three variants per source-name family. Among only the images that appear in the corresponding annotation JSON files, there are no byte-identical images or matching filename families shared by train, validation and test.

**Important distinction:** When scanning *all* folder files, including the 707 unused extras, some images were byte-identical across different split folders. These extra files are not referenced by the COCO annotations and therefore are not used by the archived COCO-based dataset loader. A simple all-files duplication check would overstate the leakage in the documented training pipeline.

This audit cannot exclude patient-level leakage or visually similar near-duplicates, because names and exact file bytes do not establish patient identity or imaging provenance. A stronger investigation would require original-source identifiers and perceptual-similarity checks.

## Original implementation concerns

1. **Empty targets:** The old dataset loader created a `[0,0,1,1]` box with label `0` when no ground truth was present. The maintained loader uses truly empty target tensors instead. Background is not modeled as a fake foreground box.
2. **Hard-coded CUDA:** The old training script used `torch.device("cuda")` without a CPU fallback. The maintained scripts select an available device or allow `--device cpu`.
3. **Validation loss:** The training script calculated validation loss while leaving the model in training mode. The maintained version instead uses inference-mode validation with explicit IoU-based metrics. These are different quantities.
4. **Data paths and categories:** The archived scripts assumed local `train/`, `valid/`, `test/` folders and assigned a cavity class ID in some locations. The maintained implementation validates COCO fields and maps category IDs to labels starting at 1; all splits must use the same category mapping.
5. **Pretrained weights:** The original used Torchvision default pretrained model weights. The maintained version makes backbone downloads opt-in, so offline smoke tests do not unexpectedly request network access. This changes training initialization and is not presented as a reproduction.
6. **Evaluation:** The archived score file computed COCO AP metrics and a separate per-image average of precision/recall/F1, skipping images without cavity ground truth. The maintained `evaluation.py` computes micro-aggregated thresholded metrics, including false positives on negative images. Do not compare these numbers as if they were the same metric.
7. **Missing checkpoint:** The archived prediction JSONs and illustrations cannot establish training provenance. They may be inspected, but reported historical performance would need verification against the exact model, dataset split and threshold.
8. **Patient data and rights:** The COCO metadata indicates a Roboflow export and a CC BY 4.0 license entry, but this metadata alone does not establish the original owners of the dental images, patient consent, or redistribution rights. The public package includes only generated synthetic images.

## What's deliberately excluded

- Original near-infrared dental JPEGs and their derived detection illustrations.
- The original COCO annotation JSONs and archived prediction JSONs, because they belong with the private image research records.
- Original model weights, which were not available.
- Blender project/backup files, which are not needed for the detection code and are large.
- The temporary editor fragment and unrelated generated outputs.

The original archive should remain the private source record. The public repository contains the maintained Python workflow, a clearly identified synthetic demonstration and historical scripts with their limitations documented.

## Reproducibility status

The public example supports COCO reading, visualization, Faster R-CNN initialization, training, inference and test-metric calculation on synthetic shapes. Unit tests check the logic without asserting medical accuracy. Tests and demo data do not validate dental cavity detection performance, clinical suitability, or the archived publication's reported experiments.

## Saved prediction JSON compatibility

The archived `results.json` contains 99 saved bounding-box predictions with image IDs compatible with the supplied test annotation file. A separate `test_detections.json` contains 120 predictions, including image IDs not found in that same test annotation file. Those two files should not be treated as equivalent runs without investigating how each was generated. Because the corresponding training checkpoint is unavailable, neither JSON file is published as a verified model benchmark.
