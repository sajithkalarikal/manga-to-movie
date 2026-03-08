from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from modules.panel_detection import PanelDetectionService
from task_queue import build_status_payload, enqueue_manga_job, get_redis_pool, get_task_status, set_task_status

settings = get_settings()
panel_detector = PanelDetectionService(settings)


class QueueVideoResponse(BaseModel):
    status: str
    request_id: str
    task_status_url: str


class TaskStatusResponse(BaseModel):
    request_id: str
    status: str
    updated_at: str
    attempt: int | None = None
    max_attempts: int | None = None
    video_url: str | None = None
    metadata_url: str | None = None
    subtitles_url: str | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = await get_redis_pool()
    app.state.redis = redis
    yield
    await redis.close()


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/generate-video", response_model=QueueVideoResponse, status_code=202)
async def generate_video(file: UploadFile = File(...)) -> QueueVideoResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.upload_dir / f"{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    redis = app.state.redis
    await set_task_status(redis, request_id, build_status_payload(request_id, "queued", attempt=0, filename=file.filename))
    job = await enqueue_manga_job(redis, request_id=request_id, upload_path=str(upload_path))
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to enqueue video generation job.")

    return QueueVideoResponse(
        status="queued",
        request_id=request_id,
        task_status_url=f"/tasks/{request_id}",
    )


@app.get("/tasks/{request_id}", response_model=TaskStatusResponse)
async def get_task(request_id: str) -> TaskStatusResponse:
    redis = app.state.redis
    payload = await get_task_status(redis, request_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(**payload)
