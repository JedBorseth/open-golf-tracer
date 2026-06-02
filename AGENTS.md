# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

Golf Tracer (`open-golf-tracer`) is a two-app repo: **FastAPI backend** (port 8000) and **TanStack Start frontend** (port 3000). Job state is JSON on disk under `job-store/`; no database container. See root `README.md` for layout and API.

### One-time VM packages

Debian images may need `python3.12-venv` before the backend virtualenv can be created:

```bash
sudo apt-get install -y python3.12-venv
```

Node **22+** is expected for `frontend/` (lockfile uses npm).

### Dependency refresh (automatic)

The VM update script reinstalls Python and Node deps only. It does **not** start servers.

### Running services (manual, use tmux)

Copy env files once if missing: `cp .env.example .env` and `cp frontend/.env.example frontend/.env`.

**Backend** (from `backend/`, with venv activated):

```bash
MODEL_PATH=../models/yolov11s-golf-ball.pt \
UPLOAD_DIR=../uploads \
OUTPUT_DIR=../outputs \
JOB_STORE_DIR=../job-store \
YOLO_DEVICE=cpu \
REQUIRE_CUDA=false \
CORS_ORIGINS='["http://localhost:3000","http://127.0.0.1:3000"]' \
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (from `frontend/`):

```bash
npm run dev
```

Restart the Vite dev server after changing `frontend/.env` so `VITE_API_BASE_URL` is picked up.

### Lint / test / build

| Area | Command | Notes |
|------|---------|--------|
| Backend tests | `cd backend && .venv/bin/pytest` | No server required |
| Backend lint | `cd backend && .venv/bin/ruff check .` | |
| Frontend lint | `cd frontend && npm run lint` | `tsc --noEmit` |
| Frontend build | `cd frontend && npm run build` | |

### Model weights and E2E

Without `models/yolov11s-golf-ball.pt`, uploads still exercise upload → queue → poll → **`failed` / `model_not_found`**, which is the expected MVP path per `README.md`.

Cloud VMs typically have no NVIDIA GPU; keep `YOLO_DEVICE=cpu` and `REQUIRE_CUDA=false`. Docker Compose in this repo requests `gpus: all` and is aimed at the GPU server workflow in `README.md`.

### Docker alternative

From repo root: `cp .env.example .env && docker compose up --build`. Requires Docker and (for GPU) NVIDIA Container Toolkit on the host.
