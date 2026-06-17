from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"


MediaKind = Literal["image", "video"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobRecord(BaseModel):
    id: str
    status: JobStatus = JobStatus.queued
    media_kind: MediaKind
    original_filename: str
    content_type: str
    input_path: Path
    output_path: Path | None = None
    trace_path: Path | None = None
    result_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    media_kind: MediaKind
    result_url: str | None = None
    source_url: str | None = None
    trace_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobResponse":
        return cls(
            job_id=record.id,
            status=record.status,
            media_kind=record.media_kind,
            result_url=record.result_url,
            source_url=f"/api/jobs/{record.id}/source",
            trace_url=f"/api/jobs/{record.id}/trace" if record.trace_path is not None else None,
            error_code=record.error_code,
            error_message=record.error_message,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
