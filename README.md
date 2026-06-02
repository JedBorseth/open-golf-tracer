# Golf Tracer

Mobile web app for golf swing visualization. The frontend is a minimal TanStack
Start upload flow, and the backend is a FastAPI service that queues image/video
jobs, tracks club-head motion through the swing, renders the club path with an
impact marker, and returns the processed result.

## Repository Layout

```text
frontend/   TanStack Start mobile upload app
backend/    FastAPI API, job state, club motion tracking, rendering
training/   Optional YOLO dataset template (legacy ball model)
models/     Optional custom weights, ignored by git
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
UPLOAD_DIR=../uploads \
OUTPUT_DIR=../outputs \
JOB_STORE_DIR=../job-store \
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

## Tracking

Swing path tracing uses **motion-based club-head tracking** (frame differencing,
optical flow, Kalman smoothing) plus audio/visual **impact detection**. No YOLO
weights are required for the default pipeline.

Optional legacy ball-detector weights (`models/yolov11s-golf-ball.pt`) are unused
by the club swing pipeline but remain available under `training/` if you want to
experiment with detection-assisted tracking later.

## API

- `POST /api/jobs` with multipart field `file`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `GET /health`

## Current MVP Limits

- Job state is stored as JSON files on disk.
- Background jobs run inside the FastAPI process.
- Club tracking is motion-based and works best on clear swing videos (down-the-line
  or face-on). A dedicated club-head YOLO model can be added later for tougher angles.
# open-golf-tracer
