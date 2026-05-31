# Golf Tracer Backend

FastAPI service for upload jobs, YOLO inference, tracking, and tracer rendering.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

The default model path is `/app/models/yolov11s-golf-ball.pt` for Docker. Override it with
`MODEL_PATH` when running locally.
