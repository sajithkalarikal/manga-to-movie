from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from modules.ocr_dialogue import OCRDialogueService
from modules.panel_detection import PanelDetectionService
from modules.scene_caption import SceneCaptionService
from modules.script_generator import ScriptGeneratorService
from task_queue import build_status_payload, enqueue_manga_job, get_redis_pool, get_task_status, set_task_status

settings = get_settings()
panel_detector = PanelDetectionService(settings)
ocr_service = OCRDialogueService(settings)
caption_service = SceneCaptionService(settings)
script_service = ScriptGeneratorService(settings)


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


class OCRTextResponse(BaseModel):
    filename: str
    text: str
    text_1: str = ""
    text_2: str = ""


class SceneScript(BaseModel):
    scene_description: str
    camera_motion: str
    animation_action: str
    dialogue: str
    duration: int


class GenerateScriptResponse(BaseModel):
    request_id: str
    filename: str
    panels: int
    dialogue: list[dict[str, str]]
    captions: list[dict[str, str]]
    scene_script: SceneScript


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


@app.post("/extract-text", response_model=OCRTextResponse)
async def extract_text(file: UploadFile = File(...)) -> OCRTextResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.upload_dir / f"ocr-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)
    text_parts = await ocr_service.extract_text_parts_from_image(upload_path, limit=2)
    text = " ".join(text_parts).strip() or "[no text detected]"
    text_1 = text_parts[0] if len(text_parts) > 0 else ""
    text_2 = text_parts[1] if len(text_parts) > 1 else ""
    return OCRTextResponse(filename=file.filename, text=text, text_1=text_1, text_2=text_2)


@app.post("/generate-script", response_model=GenerateScriptResponse)
async def generate_script(file: UploadFile = File(...)) -> GenerateScriptResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.upload_dir / f"script-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / f"script-{request_id}"
    panels_dir = job_dir / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir)
    if not panels:
        raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")

    dialogue = await ocr_service.extract_dialogue(panels)
    captions = await caption_service.generate_captions(panels)
    scene_script = await script_service.generate_script(dialogue=dialogue, captions=captions)

    return GenerateScriptResponse(
        request_id=request_id,
        filename=file.filename,
        panels=len(panels),
        dialogue=dialogue,
        captions=captions,
        scene_script=SceneScript(**scene_script),
    )
