# Golf Tracer

Mobile web app for golf shot tracers. The frontend is a minimal TanStack Start
upload flow, and the backend is a FastAPI service that queues image/video jobs,
runs YOLOv11s golf ball detection, tracks the ball, renders a tracer, and returns
the processed result.

## Repository Layout

```text
frontend/   TanStack Start mobile upload app
backend/    FastAPI API, job state, inference, tracking, rendering
training/   YOLOv11s dataset template and training script
models/     Runtime model weights, ignored by git
uploads/    Uploaded media, ignored by git
outputs/    Processed results, ignored by git
job-store/  File-backed job metadata, ignored by git
```

## Local Development

Copy env examples:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

Run the backend directly:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
MODEL_PATH=../models/yolov11s-golf-ball.pt \
UPLOAD_DIR=../uploads \
OUTPUT_DIR=../outputs \
JOB_STORE_DIR=../job-store \
YOLO_DEVICE=cpu \
uvicorn app.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The backend listens on `http://localhost:8000` and the frontend listens on
`http://localhost:3000` by default.

## Debian Server With NVIDIA GPU

The target server uses a GTX 1660 Ti with 6GB VRAM. Install these prerequisites
on Debian:

- NVIDIA driver with working `nvidia-smi`
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit

Verify the host:

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Deploy:

```bash
cp .env.server.example .env
mkdir -p models uploads outputs job-store
docker compose up -d --build
docker compose logs -f backend
```

The server env maps the frontend to `http://borseth.ddns.net:7070` and the
backend to `http://borseth.ddns.net:7075`, keeping the app inside the server's
open `7000-8000` port range.

Verify the app and GPU:

```bash
curl http://localhost:7075/health
docker compose exec backend nvidia-smi
docker compose exec backend python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The server env sets `REQUIRE_CUDA=true`, so `/health` reports `ready: false` if
the backend container cannot see CUDA. Local development can keep
`REQUIRE_CUDA=false` and use `YOLO_DEVICE=cpu`.

## Model Weights

Place trained weights at:

```text
models/yolov11s-golf-ball.pt
```

Until weights exist, uploads should complete as failed jobs with
`model_not_found`. That is expected and still validates upload, job persistence,
polling, and frontend error display.

## API

- `POST /api/jobs` with multipart field `file`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `GET /health`

## Current MVP Limits

- Job state is stored as JSON files on disk.
- Background jobs run inside the FastAPI process.
- Video tracking is an initial YOLO plus optical-flow/Kalman implementation and
  will improve once real golf shot clips and trained weights are available.
# open-golf-tracer
