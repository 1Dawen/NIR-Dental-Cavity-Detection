"""Draw predicted bounding boxes on one image with a local checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from cavity_detection.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fasterrcnn.pth"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.png"))
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping = load_checkpoint(args.checkpoint, device)
    model.eval()
    with Image.open(args.image) as file:
        img = file.convert("RGB")
    with torch.inference_mode():
        output = model([to_tensor(img).to(device)])[0]
    drawing = ImageDraw.Draw(img)
    count = 0
    for box, score, label in zip(output["boxes"].cpu().tolist(),
                                  output["scores"].cpu().tolist(),
                                  output["labels"].cpu().tolist()):
        if score < args.confidence:
            continue
        drawing.rectangle(box, outline="red", width=2)
        drawing.text((box[0], max(0, box[1] - 13)),
                     f"class {label}: {score:.2f}", fill="red")
        count += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)
    print(f"Saved {count} predictions to {args.output}")
    print("Predictions are experimental and should not be used for clinical diagnosis.")


if __name__ == "__main__":
    main()
