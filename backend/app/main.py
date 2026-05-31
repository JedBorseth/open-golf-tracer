from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import jobs
from app.services.detector import get_device_info


settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


@app.get("/health")
def health() -> dict[str, object]:
    device = get_device_info(settings.yolo_device)
    return {
        "status": "ok",
        "app": settings.app_name,
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "cuda_required": settings.require_cuda,
        "device": device,
        "ready": not settings.require_cuda or bool(device.get("cuda_available")),
    }
