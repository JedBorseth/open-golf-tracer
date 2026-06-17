import json
from pathlib import Path
from threading import Lock

from app.models.job import JobRecord, JobStatus, utc_now


class JobStore:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path_for(self, job_id: str) -> Path:
        return self.store_dir / f"{job_id}.json"

    def create(self, record: JobRecord) -> JobRecord:
        return self.save(record)

    def get(self, job_id: str) -> JobRecord | None:
        path = self._path_for(job_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return JobRecord.model_validate(json.load(file))

    def save(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now()
        path = self._path_for(record.id)
        with self._lock:
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(
                    record.model_dump(mode="json"),
                    file,
                    indent=2,
                    sort_keys=True,
                )
            tmp_path.replace(path)
        return record

    def mark_running(self, job_id: str) -> JobRecord:
        record = self._require(job_id)
        record.status = JobStatus.running
        record.error_code = None
        record.error_message = None
        return self.save(record)

    def mark_complete(
        self,
        job_id: str,
        output_path: Path,
        result_url: str,
        trace_path: Path | None = None,
    ) -> JobRecord:
        record = self._require(job_id)
        record.status = JobStatus.complete
        record.output_path = output_path
        if trace_path is not None:
            record.trace_path = trace_path
        record.result_url = result_url
        record.error_code = None
        record.error_message = None
        return self.save(record)

    def mark_failed(self, job_id: str, code: str, message: str) -> JobRecord:
        record = self._require(job_id)
        record.status = JobStatus.failed
        record.error_code = code
        record.error_message = message
        return self.save(record)

    def _require(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record is None:
            raise KeyError(f"Unknown job id: {job_id}")
        return record
