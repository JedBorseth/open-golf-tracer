import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11s for golf ball detection.")
    parser.add_argument("--data", default="training/data.yaml", help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo11s.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use 0 for the first NVIDIA GPU or cpu.")
    parser.add_argument("--project", default="training/runs")
    parser.add_argument("--name", default="golf-ball-yolo11s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
