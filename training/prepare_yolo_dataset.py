import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a Label Studio YOLO export into Ultralytics train/val/test folders."
    )
    parser.add_argument("export_dir", type=Path, help="Label Studio YOLO export directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golf-ball"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir
    image_dir = export_dir / "images"
    label_dir = export_dir / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise SystemExit("Export must contain images/ and labels/ directories.")

    if round(args.train + args.val + args.test, 6) != 1:
        raise SystemExit("Split ratios must add up to 1.")

    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise SystemExit(f"Missing label file for {image_path.name}: {label_path}")
        _validate_label(label_path)
        samples.append((image_path, label_path))

    if not samples:
        raise SystemExit("No image samples found.")

    rng = random.Random(args.seed)
    positive_samples = [sample for sample in samples if sample[1].read_text().strip()]
    negative_samples = [sample for sample in samples if not sample[1].read_text().strip()]
    splits = {"train": [], "val": [], "test": []}
    for group in (positive_samples, negative_samples):
        rng.shuffle(group)
        train_end = int(len(group) * args.train)
        val_end = train_end + int(len(group) * args.val)
        splits["train"].extend(group[:train_end])
        splits["val"].extend(group[train_end:val_end])
        splits["test"].extend(group[val_end:])

    for split_samples in splits.values():
        rng.shuffle(split_samples)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    for split, split_samples in splits.items():
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_samples:
            shutil.copy2(image_path, args.output_dir / "images" / split / image_path.name)
            shutil.copy2(label_path, args.output_dir / "labels" / split / label_path.name)

    data_yaml = args.output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: golf_ball",
                "",
            ]
        ),
        encoding="utf-8",
    )

    positives = sum(1 for _, label_path in samples if label_path.read_text().strip())
    negatives = len(samples) - positives
    print(f"Prepared {len(samples)} images at {args.output_dir}")
    print(f"Positive labels: {positives}; empty negative labels: {negatives}")
    for split, split_samples in splits.items():
        split_positives = sum(1 for _, label_path in split_samples if label_path.read_text().strip())
        print(f"{split}: {len(split_samples)} images, {split_positives} positives")
    print(f"Data YAML: {data_yaml}")


def _validate_label(label_path: Path) -> None:
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return

    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{label_path}:{line_number} must have 5 YOLO columns.")
        class_id, *values = parts
        if class_id != "0":
            raise SystemExit(f"{label_path}:{line_number} has class {class_id}; expected 0.")
        floats = [float(value) for value in values]
        if any(value < 0 or value > 1 for value in floats):
            raise SystemExit(f"{label_path}:{line_number} has non-normalized box values.")


if __name__ == "__main__":
    main()
