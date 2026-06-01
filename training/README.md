# YOLOv11s Training

This folder holds the training entrypoint and dataset template for the `golf_ball`
detector.

## Dataset Layout

Copy `data.yaml.example` to `data.yaml`, then organize labeled YOLO data like this:

```text
data/golf-ball/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label file should use normal YOLO detection format:

```text
class_id x_center y_center width height
```

Values are normalized from `0` to `1`. For this project, `class_id` is `0` for
`golf_ball`.

## Train

If you exported from Label Studio in YOLO format, prepare the dataset first:

```bash
python training/prepare_yolo_dataset.py \
  training/project-2-at-2026-06-01-02-01-abfd969d \
  --output-dir data/golf-ball
```

On the Debian server with the GTX 1660 Ti, start conservative because the card has
6GB VRAM:

```bash
python training/train_yolo.py \
  --data data/golf-ball/data.yaml \
  --model yolo11s.pt \
  --imgsz 640 \
  --batch 8 \
  --workers 2 \
  --device 0
```

If CUDA runs out of memory, lower `--batch` to `4` or `2`, or lower `--imgsz` to
`512`.

After training, copy the best weights into the runtime model volume:

```bash
mkdir -p models
cp training/runs/golf-ball-yolo11s/weights/best.pt models/yolov11s-golf-ball.pt
```
