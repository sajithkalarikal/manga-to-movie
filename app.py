import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from modules.ocr_dialogue import OCRDialogueService
from modules.bubble_detector import get_bubble_detector
from modules.panel_detection import PanelDetectionService
from modules.scene_caption import SceneCaptionService
from modules.script_generator import ScriptGeneratorService
from task_queue import build_status_payload, enqueue_manga_job, get_redis_pool, get_task_status, set_task_status

settings = get_settings()
logger = logging.getLogger(__name__)
frontend_dir = Path(__file__).parent / "frontend"
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
    emotion: str | None = None
    action_level: str | None = None
    transition_type: str | None = None
    panel_flow: str | None = None
    reading_graph: str | None = None
    reading_order: str | None = None


class GenerateScriptResponse(BaseModel):
    request_id: str
    filename: str
    panels: int
    dialogue: list[dict[str, str]]
    captions: list[dict[str, str]]
    scene_script: SceneScript


class AnalyzePanelsResponse(BaseModel):
    request_id: str
    filename: str
    panels: int
    bubble_mode: str
    dialogue: list[dict[str, str]]
    captions: list[dict[str, str]]


class PanelBox(BaseModel):
    index: int
    bbox: tuple[int, int, int, int]
    image_path: str
    image_url: str


class DetectPanelsResponse(BaseModel):
    request_id: str
    filename: str
    panels: int
    panel_images_dir: str
    panel_image_urls: list[str]
    panel_boxes: list[PanelBox]


class PanelOverrideItem(BaseModel):
    speech_count: str | None = None
    narration_count: str | None = None
    sfx_count: str | None = None
    bubble_count: str | None = None
    bubble_sequence: str | None = None


class SaveOverridesRequest(BaseModel):
    request_id: str
    overrides: dict[str, PanelOverrideItem]


class SaveOverridesResponse(BaseModel):
    request_id: str
    saved: bool
    overrides_path: str


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
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui", status_code=302)


@app.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/override", include_in_schema=False)
async def override_ui() -> FileResponse:
    return FileResponse(frontend_dir / "override.html")


def _build_panel_box_response(request_id: str, panel_index: int, bbox: tuple[int, int, int, int], image_path: Path) -> PanelBox:
    image_url = f"{settings.output_base_url.rstrip('/')}/{request_id}/panels/{image_path.name}"
    return PanelBox(
        index=panel_index,
        bbox=bbox,
        image_path=str(image_path),
        image_url=image_url,
    )


@app.post("/generate-video", response_model=QueueVideoResponse, status_code=202)
async def generate_video(file: UploadFile = File(...)) -> QueueVideoResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"{request_id}{upload_suffix}"
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
    upload_path = settings.temp_dir / f"ocr-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)
    try:
        text_parts = await ocr_service.extract_text_parts_from_image(upload_path, limit=2)
        text = " ".join(text_parts).strip() or "[no text detected]"
        text_1 = text_parts[0] if len(text_parts) > 0 else ""
        text_2 = text_parts[1] if len(text_parts) > 1 else ""
        return OCRTextResponse(filename=file.filename, text=text, text_1=text_1, text_2=text_2)
    finally:
        await asyncio.to_thread(upload_path.unlink, True)


@app.post("/detect-panels", response_model=DetectPanelsResponse)
async def detect_panels(file: UploadFile = File(...)) -> DetectPanelsResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"panels-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / request_id
    panels_dir = job_dir / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    try:
        detected_panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir)
        if not detected_panels:
            raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")

        panel_boxes = [
            _build_panel_box_response(request_id, panel.index, panel.bbox, panel.image_path)
            for panel in detected_panels
        ]
        return DetectPanelsResponse(
            request_id=request_id,
            filename=file.filename,
            panels=len(detected_panels),
            panel_images_dir=str(panels_dir),
            panel_image_urls=[panel.image_url for panel in panel_boxes],
            panel_boxes=panel_boxes,
        )
    finally:
        await asyncio.to_thread(upload_path.unlink, True)


@app.post("/generate-script", response_model=GenerateScriptResponse)
async def generate_script(file: UploadFile = File(...), bubble_mode: str = Form("heuristic")) -> GenerateScriptResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"script-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / f"script-{request_id}"
    panels_dir = job_dir / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    try:
        panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir)
        if not panels:
            raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")

        dialogue = await ocr_service.extract_dialogue(panels)
        captions = await caption_service.generate_captions(panels, bubble_mode=bubble_mode)
        scene_script = await script_service.generate_script(dialogue=dialogue, captions=captions)

        return GenerateScriptResponse(
            request_id=request_id,
            filename=file.filename,
            panels=len(panels),
            dialogue=dialogue,
            captions=captions,
            scene_script=SceneScript(**scene_script),
        )
    finally:
        await asyncio.to_thread(upload_path.unlink, True)


@app.post("/analyze-panels", response_model=AnalyzePanelsResponse)
async def analyze_panels(file: UploadFile = File(...), bubble_mode: str = Form("heuristic")) -> AnalyzePanelsResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")
    if bubble_mode not in {"heuristic", "detector"}:
        raise HTTPException(status_code=400, detail="bubble_mode must be 'heuristic' or 'detector'.")
    if bubble_mode == "detector":
        detector = get_bubble_detector(settings)
        if not detector.available:
            raise HTTPException(status_code=409, detail="Object detector mode is selected, but no detector weights are loaded.")

    request_id = uuid4().hex
    phase1_started_at = datetime.now().astimezone()
    phase1_started_perf = time.perf_counter()
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"analyze-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / request_id
    panels_dir = job_dir / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    try:
        if bubble_mode == "detector":
            logger.info(
                "Phase1 detector run started request_id=%s filename=%s started_at=%s",
                request_id,
                file.filename,
                phase1_started_at.isoformat(),
            )
        panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir)
        if not panels:
            raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")

        dialogue = await ocr_service.extract_dialogue(panels)
        captions = await caption_service.generate_captions(panels, bubble_mode=bubble_mode)
        if bubble_mode == "detector":
            phase1_finished_at = datetime.now().astimezone()
            elapsed_seconds = time.perf_counter() - phase1_started_perf
            elapsed_minutes = elapsed_seconds / 60.0
            logger.info(
                "Phase1 detector run finished request_id=%s filename=%s finished_at=%s duration_seconds=%.2f duration_minutes=%.2f panels=%s",
                request_id,
                file.filename,
                phase1_finished_at.isoformat(),
                elapsed_seconds,
                elapsed_minutes,
                len(panels),
            )

        return AnalyzePanelsResponse(
            request_id=request_id,
            filename=file.filename,
            panels=len(panels),
            bubble_mode=bubble_mode,
            dialogue=dialogue,
            captions=captions,
        )
    finally:
        await asyncio.to_thread(upload_path.unlink, True)


@app.post("/panel-overrides", response_model=SaveOverridesResponse)
async def save_panel_overrides(payload: SaveOverridesRequest) -> SaveOverridesResponse:
    job_dir = settings.output_dir / payload.request_id
    if not job_dir.exists():
        analyze_job_dir = settings.output_dir / f"analyze-{payload.request_id}"
        if analyze_job_dir.exists():
            job_dir = analyze_job_dir
        else:
            raise HTTPException(status_code=404, detail="Request output directory not found.")

    overrides_path = job_dir / "panel_overrides.json"
    serialized = payload.model_dump(mode="json")
    await asyncio.to_thread(overrides_path.write_text, json.dumps(serialized, indent=2), "utf-8")
    return SaveOverridesResponse(
        request_id=payload.request_id,
        saved=True,
        overrides_path=str(overrides_path),
    )
