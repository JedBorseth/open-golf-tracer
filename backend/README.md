# Golf Tracer Backend

FastAPI service for upload jobs, YOLO inference, tracking, and tracer rendering.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Swing tracking is motion-based and does not require model weights. Optional legacy YOLO
weights under `models/` are unused by the default pipeline.
