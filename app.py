import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from pathlib import Path
import shutil
import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from modules.annotation_workspace import (
    TARGET_CLASSES,
    available_datasets,
    build_review_queue,
    export_annotated_dataset,
    list_dataset_images,
    load_image_annotations,
    resolve_dataset_root,
    save_image_annotations,
)
from modules.ocr_dialogue import OCRDialogueService
from modules.ocr_engine import get_ocr_engine
from modules.bubble_detector import get_bubble_detector
from modules.database import (
    append_annotation_event,
    append_job_event,
    initialize_database,
    register_detected_panels,
    register_image_path,
    store_annotation_snapshot,
    sync_dataset_split_from_coco,
    upsert_job,
)
from modules.database_v2 import (
    insert_request_asset_v2,
    initialize_v2_database,
    load_request_override_v2,
    replace_request_annotations_v2,
    upsert_request_v2,
)
from modules.panel_model_detector import get_panel_model_detector
from modules.panel_detection import PanelDetectionService
from modules.scene_caption import SceneCaptionService
from modules.script_generator import ScriptGeneratorService
from task_queue import build_status_payload, enqueue_manga_job, get_redis_pool, get_task_status, set_task_status

settings = get_settings()
logger = logging.getLogger(__name__)
frontend_dir = Path(__file__).parent / "frontend"
web_dir = Path(__file__).parent / "web"
web_dist_dir = web_dir / "dist"
web_assets_dir = web_dist_dir / "assets"
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


class CaptionResponse(BaseModel):
    panel: int
    caption: str
    shot_type: str
    motion_level: str
    tone: str
    emotion: str
    action_level: str
    bubble_count: int
    bubble_candidates: int
    bubble_sequence: str
    speech_count: int
    narration_count: int
    sfx_count: int
    speech_boxes: list[list[int]]
    narration_boxes: list[list[int]]
    sfx_boxes: list[list[int]]
    transition_hint: str
    layout_role: str


class GenerateScriptResponse(BaseModel):
    request_id: str
    filename: str
    panels: int
    dialogue: list[dict[str, str]]
    captions: list[CaptionResponse]
    scene_script: SceneScript


class AnalyzePanelsResponse(BaseModel):
    request_id: str
    filename: str
    panels: int
    panel_mode: str
    bubble_mode: str
    dialogue: list[dict[str, str]]
    captions: list[CaptionResponse]


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
    source_image_url: str | None = None


class PanelOverrideItem(BaseModel):
    speech_count: str | None = None
    narration_count: str | None = None
    sfx_count: str | None = None
    bubble_count: str | None = None
    bubble_sequence: str | None = None


class PanelBoxOverrideItem(BaseModel):
    index: int | None = None
    bbox: list[int]
    points: list[list[int]] | None = None
    role: str | None = None


class PanelRegionOverrideItem(BaseModel):
    class_name: str
    bbox: list[float]


class SaveOverridesRequest(BaseModel):
    request_id: str
    overrides: dict[str, PanelOverrideItem]
    panel_boxes: list[PanelBoxOverrideItem] | None = None
    panel_regions: dict[str, list[PanelRegionOverrideItem]] | None = None


class SaveOverridesResponse(BaseModel):
    request_id: str
    saved: bool
    overrides_path: str


class LoadOverridesResponse(BaseModel):
    request_id: str
    exists: bool
    overrides_path: str | None = None
    overrides: dict[str, PanelOverrideItem] | None = None
    panel_boxes: list[PanelBoxOverrideItem] | None = None
    panel_regions: dict[str, list[PanelRegionOverrideItem]] | None = None


class DatasetImageItem(BaseModel):
    index: int
    image_id: int
    file_name: str
    width: int
    height: int
    source_annotation_count: int
    source_categories: list[str]
    override_exists: bool
    image_url: str


class DatasetImageListResponse(BaseModel):
    dataset: str
    split: str
    offset: int
    limit: int
    total: int
    items: list[DatasetImageItem]
    classes: list[str]
    available_datasets: list[dict[str, str]]


class DatasetAnnotation(BaseModel):
    id: str
    class_name: str
    bbox: list[float]
    points: list[list[float]] | None = None


class DatasetAnnotationItemResponse(BaseModel):
    dataset: str
    split: str
    index: int
    image_id: int
    file_name: str
    width: int
    height: int
    image_url: str
    classes: list[str]
    annotation_source: str
    annotations: list[DatasetAnnotation]
    available_datasets: list[dict[str, str]]


class AnnotationReviewQueueItem(BaseModel):
    queue_index: int
    split: str
    index: int
    image_id: int
    file_name: str
    width: int
    height: int
    annotation_count: int
    class_names: list[str]
    override_exists: bool
    annotation_source: str
    priority: int
    reasons: list[str]
    image_url: str


class AnnotationReviewQueueResponse(BaseModel):
    dataset: str
    total: int
    items: list[AnnotationReviewQueueItem]


class SaveDatasetAnnotationsRequest(BaseModel):
    dataset: str
    split: str
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: list[DatasetAnnotation]


class SaveDatasetAnnotationsResponse(BaseModel):
    dataset: str
    split: str
    image_id: int
    file_name: str
    saved: bool
    annotation_path: str


class ExportDatasetResponse(BaseModel):
    dataset: str
    export_mode: str
    exported: bool
    output_dir: str
    annotation_files: list[str]


class HealthRedisStatus(BaseModel):
    ok: bool
    detail: str


class HealthWorkerItem(BaseModel):
    worker_key: str
    hostname: str
    pid: int
    queue_name: str
    updated_at: str
    age_seconds: float
    status: str


class HealthWorkerStatus(BaseModel):
    ok: bool
    live_workers: int
    stale_workers: int
    workers: list[HealthWorkerItem]


class HealthModelStatus(BaseModel):
    name: str
    ok: bool
    available: bool
    weights_path: str | None = None
    device: str | None = None
    classes: list[str] = []
    load_error: str | None = None


class HealthOCRStatus(BaseModel):
    ok: bool
    tesseract_available: bool
    tesseract_cmd: str | None = None
    manga_ocr_loaded: bool
    easyocr_loaded: bool


class HealthDiskStatus(BaseModel):
    ok: bool
    total_bytes: int
    used_bytes: int
    free_bytes: int
    free_percent: float
    path: str


class HealthQueueStatus(BaseModel):
    ok: bool
    queue_name: str
    backlog: int


class HealthArtifactStatus(BaseModel):
    ok: bool
    path: str | None = None
    updated_at: str | None = None
    seconds_since_update: float | None = None
    details: dict[str, object] = {}


class SystemHealthReport(BaseModel):
    status: str
    environment: str
    checked_at: str
    redis: HealthRedisStatus
    worker: HealthWorkerStatus
    models: list[HealthModelStatus]
    ocr: HealthOCRStatus
    disk: HealthDiskStatus
    queue: HealthQueueStatus
    last_training_run: HealthArtifactStatus
    last_eval_run: HealthArtifactStatus
    evaluation_runs: list[HealthArtifactStatus] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database(settings)
    initialize_v2_database()
    try:
        sync_dataset_split_from_coco(settings, settings.annotation_dataset_root)
    except Exception:
        logger.exception("Failed to sync initial dataset split metadata into SQL")
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
if web_assets_dir.exists():
    app.mount("/ui_v2/assets", StaticFiles(directory=web_assets_dir), name="ui_v2_assets")


def _format_iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def _latest_checkpoint_path() -> Path | None:
    candidates = [
        path
        for path in (Path("models")).rglob("*.pt")
        if path.is_file() and not path.name.endswith(".latest.pt")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_eval_metrics_path() -> Path | None:
    candidates = [path for path in (settings.output_dir / "eval").glob("metrics_*.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _all_eval_metrics_paths(limit: int = 12) -> list[Path]:
    candidates = [path for path in (settings.output_dir / "eval").glob("metrics_*.json") if path.is_file()]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[:limit]


async def _queue_backlog(redis) -> int:
    try:
        return int(await redis.zcard(settings.redis_queue_name))
    except Exception:
        try:
            return int(await redis.llen(settings.redis_queue_name))
        except Exception:
            return -1


async def _read_worker_health(redis) -> HealthWorkerStatus:
    workers: list[HealthWorkerItem] = []
    try:
        keys = await redis.keys("health:worker:*")
    except Exception:
        return HealthWorkerStatus(ok=False, live_workers=0, stale_workers=0, workers=[])

    now = datetime.now().astimezone()
    for raw_key in keys:
        key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
        raw_value = await redis.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        payload = json.loads(raw_value)
        updated_at = str(payload.get("updated_at", ""))
        age_seconds = max(0.0, (now - datetime.fromisoformat(updated_at).astimezone()).total_seconds()) if updated_at else 999999.0
        workers.append(
            HealthWorkerItem(
                worker_key=str(payload.get("worker_key", key)),
                hostname=str(payload.get("hostname", "")),
                pid=int(payload.get("pid", 0)),
                queue_name=str(payload.get("queue_name", settings.redis_queue_name)),
                updated_at=updated_at,
                age_seconds=round(age_seconds, 3),
                status="alive" if age_seconds <= 45 else "stale",
            )
        )

    live_workers = sum(1 for worker in workers if worker.status == "alive")
    stale_workers = sum(1 for worker in workers if worker.status != "alive")
    return HealthWorkerStatus(
        ok=live_workers > 0,
        live_workers=live_workers,
        stale_workers=stale_workers,
        workers=workers,
    )


def _build_model_health() -> list[HealthModelStatus]:
    bubble_detector = get_bubble_detector(settings)
    panel_detector = get_panel_model_detector(settings)
    return [
        HealthModelStatus(
            name="bubble_detector",
            ok=bubble_detector.available,
            available=bubble_detector.available,
            weights_path=str(bubble_detector.weights_path) if bubble_detector.weights_path else None,
            device=str(bubble_detector.device),
            classes=list(bubble_detector.class_names),
            load_error=bubble_detector._load_error,
        ),
        HealthModelStatus(
            name="panel_detector",
            ok=panel_detector.available,
            available=panel_detector.available,
            weights_path=str(panel_detector.weights_path) if panel_detector.weights_path else None,
            device=str(panel_detector.device),
            classes=list(panel_detector.class_names),
            load_error=panel_detector._load_error,
        ),
    ]


def _build_ocr_health() -> HealthOCRStatus:
    ocr_engine = get_ocr_engine()
    return HealthOCRStatus(
        ok=ocr_service.ocr_available,
        tesseract_available=ocr_service.ocr_available,
        tesseract_cmd=str(ocr_service.settings.tesseract_cmd) if ocr_service.settings.tesseract_cmd else None,
        manga_ocr_loaded=getattr(ocr_engine, "_mocr", None) is not None,
        easyocr_loaded=getattr(ocr_engine, "_easy", None) is not None,
    )


def _build_disk_health() -> HealthDiskStatus:
    disk_path = settings.output_dir.resolve()
    usage = shutil.disk_usage(disk_path)
    free_percent = (usage.free / usage.total * 100.0) if usage.total else 0.0
    return HealthDiskStatus(
        ok=free_percent >= 10.0,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        free_percent=round(free_percent, 2),
        path=str(disk_path),
    )


def _build_artifact_health(path: Path | None, *, payload_parser=None) -> HealthArtifactStatus:
    if path is None or not path.exists():
        return HealthArtifactStatus(ok=False)
    stat = path.stat()
    updated_at = _format_iso_from_timestamp(stat.st_mtime)
    seconds_since_update = round(time.time() - stat.st_mtime, 3)
    details: dict[str, object] = {}
    if payload_parser is not None:
        try:
            details = payload_parser(path)
        except Exception:
            details = {}
    return HealthArtifactStatus(
        ok=True,
        path=str(path.resolve()),
        updated_at=updated_at,
        seconds_since_update=seconds_since_update,
        details=details,
    )


def _parse_checkpoint_details(path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        history = checkpoint.get("history")
        history_rows: list[dict[str, object]] = []
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                epoch = item.get("epoch")
                train_loss = item.get("train_loss")
                valid_loss = item.get("valid_loss")
                if epoch is None:
                    continue
                history_rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                    }
                )
        if not history_rows and checkpoint.get("epoch") is not None:
            history_rows.append(
                {
                    "epoch": checkpoint.get("epoch"),
                    "train_loss": checkpoint.get("train_loss"),
                    "valid_loss": checkpoint.get("valid_loss"),
                }
            )
        return {
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "valid_loss": checkpoint.get("valid_loss"),
            "class_names": checkpoint.get("class_names"),
            "dataset_roots": checkpoint.get("dataset_roots"),
            "detector_type": checkpoint.get("detector_type"),
            "history": history_rows,
            "history_source": "checkpoint" if isinstance(history, list) and history_rows else "current_epoch_only",
        }
    return {}


def _parse_eval_details(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text("utf-8"))
    raw_metrics = payload.get("metrics", {}) or {}
    presented_metrics = {
        "mAP_50_95": raw_metrics.get("mAP_50_95"),
        "mAP_50": raw_metrics.get("mAP_50"),
        "recall": raw_metrics.get("recall"),
    }
    return {
        "run_name": payload.get("run_name"),
        "weights": payload.get("weights"),
        "dataset_root": payload.get("dataset_root"),
        "split": payload.get("split"),
        "images": payload.get("images"),
        "predictions": payload.get("predictions"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "score_threshold": payload.get("score_threshold"),
        "max_detections": payload.get("max_detections"),
        "metrics": presented_metrics,
        "raw_metrics": raw_metrics,
        "metric_notes": {
            "precision": "Hidden in /health because older eval payloads store AP@0.50 under the precision key, not true precision."
        },
    }


async def _build_system_health_report() -> SystemHealthReport:
    checked_at = datetime.now().astimezone().isoformat()
    redis_ok = False
    redis_detail = "disconnected"
    queue_backlog = -1
    worker_status = HealthWorkerStatus(ok=False, live_workers=0, stale_workers=0, workers=[])

    try:
        redis = app.state.redis
        pong = await redis.ping()
        redis_ok = bool(pong)
        redis_detail = "connected" if redis_ok else "ping failed"
        queue_backlog = await _queue_backlog(redis)
        worker_status = await _read_worker_health(redis)
    except Exception as exc:
        redis_detail = str(exc)

    last_training_run = _build_artifact_health(_latest_checkpoint_path(), payload_parser=_parse_checkpoint_details)
    last_eval_run = _build_artifact_health(_latest_eval_metrics_path(), payload_parser=_parse_eval_details)
    evaluation_runs = [
        _build_artifact_health(path, payload_parser=_parse_eval_details)
        for path in _all_eval_metrics_paths()
    ]
    models = _build_model_health()
    ocr = _build_ocr_health()
    disk = _build_disk_health()
    queue = HealthQueueStatus(ok=queue_backlog >= 0, queue_name=settings.redis_queue_name, backlog=max(queue_backlog, 0))

    report_ok = all(
        [
            redis_ok,
            worker_status.ok,
            all(model.ok for model in models),
            ocr.ok,
            disk.ok,
            queue.ok,
        ]
    )
    return SystemHealthReport(
        status="ok" if report_ok else "degraded",
        environment=settings.app_env,
        checked_at=checked_at,
        redis=HealthRedisStatus(ok=redis_ok, detail=redis_detail),
        worker=worker_status,
        models=models,
        ocr=ocr,
        disk=disk,
        queue=queue,
        last_training_run=last_training_run,
        last_eval_run=last_eval_run,
        evaluation_runs=evaluation_runs,
    )


@app.get("/health", response_model=SystemHealthReport)
async def health_check() -> SystemHealthReport:
    return await _build_system_health_report()


@app.get("/heath", response_model=SystemHealthReport)
async def heath_check() -> SystemHealthReport:
    return await _build_system_health_report()


@app.get("/api/health/report", response_model=SystemHealthReport)
async def system_health_report() -> SystemHealthReport:
    return await _build_system_health_report()


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui", status_code=302)


@app.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/override", include_in_schema=False)
async def override_ui() -> FileResponse:
    return FileResponse(frontend_dir / "override.html")


@app.get("/annotate", include_in_schema=False)
async def annotate_ui() -> FileResponse:
    return FileResponse(
        frontend_dir / "annotate.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _get_ui_v2_index() -> Path:
    index_path = web_dist_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="UI v2 is not built yet. Run the web build first to generate web/dist.",
        )
    return index_path


@app.get("/ui_v2", include_in_schema=False)
@app.get("/ui_v2/", include_in_schema=False)
@app.get("/ui_v2/home", include_in_schema=False)
@app.get("/ui_v2/health", include_in_schema=False)
@app.get("/ui_v2/annotate", include_in_schema=False)
@app.get("/ui_v2/{request_id}/override", include_in_schema=False)
async def ui_v2_app(request_id: str | None = None) -> FileResponse:
    return FileResponse(_get_ui_v2_index())


@app.get("/ui_v2/assets/{asset_path:path}", include_in_schema=False)
async def ui_v2_asset(asset_path: str) -> FileResponse:
    asset_file = (web_assets_dir / asset_path).resolve()
    if not web_assets_dir.exists():
        raise HTTPException(
            status_code=503,
            detail="UI v2 assets are not available yet. Run the web build first.",
        )
    if web_assets_dir.resolve() not in asset_file.parents or not asset_file.exists():
        raise HTTPException(status_code=404, detail="UI v2 asset not found.")
    return FileResponse(asset_file)


@app.get("/dataset-images/{dataset}/{split}/{file_name:path}", include_in_schema=False)
async def dataset_image(dataset: str, split: str, file_name: str) -> FileResponse:
    try:
        dataset_root = resolve_dataset_root(settings, dataset)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    split_root = (dataset_root / split).resolve()
    image_path = (split_root / file_name).resolve()
    if split_root not in image_path.parents or not image_path.exists():
        raise HTTPException(status_code=404, detail="Dataset image not found.")
    return FileResponse(image_path)


@app.get("/api/annotation/datasets")
async def get_annotation_datasets() -> dict[str, object]:
    datasets = available_datasets(settings)
    return {
        "datasets": datasets,
        "default_dataset": settings.annotation_dataset_root.resolve().name,
    }


def _build_panel_box_response(request_id: str, panel_index: int, bbox: tuple[int, int, int, int], image_path: Path) -> PanelBox:
    image_url = f"{settings.output_base_url.rstrip('/')}/{request_id}/panels/{image_path.name}"
    return PanelBox(
        index=panel_index,
        bbox=bbox,
        image_path=str(image_path),
        image_url=image_url,
    )


def _build_source_image_url(request_id: str, image_path: Path) -> str:
    return f"{settings.output_base_url.rstrip('/')}/{request_id}/{image_path.name}"


def _override_database_root() -> Path:
    root = settings.annotation_database_dir / "request_overrides"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _override_database_index_path() -> Path:
    return settings.annotation_database_dir / "request_overrides_index.json"


def _override_database_record_path(request_id: str) -> Path:
    return _override_database_root() / f"{request_id}.json"


def _read_json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text("utf-8"))


def _write_override_database_record(
    *,
    payload: SaveOverridesRequest,
    job_dir: Path,
    overrides_path: Path,
) -> Path:
    database_path = _override_database_record_path(payload.request_id)
    saved_at = datetime.now().astimezone().isoformat()
    serialized = payload.model_dump(mode="json")
    database_payload = {
        "request_id": payload.request_id,
        "saved_at": saved_at,
        "request_output_dir": str(job_dir.resolve()),
        "request_override_path": str(overrides_path.resolve()),
        "database_record_path": str(database_path.resolve()),
        "overrides": serialized.get("overrides") or {},
        "panel_boxes": serialized.get("panel_boxes") or [],
        "panel_regions": serialized.get("panel_regions") or {},
    }
    database_path.write_text(json.dumps(database_payload, indent=2), "utf-8")

    index_path = _override_database_index_path()
    index_payload: dict[str, object]
    if index_path.exists():
        try:
            index_payload = _read_json_file(index_path)
        except (OSError, json.JSONDecodeError):
            index_payload = {}
    else:
        index_payload = {}
    index_payload[payload.request_id] = {
        "saved_at": saved_at,
        "request_output_dir": str(job_dir.resolve()),
        "request_override_path": str(overrides_path.resolve()),
        "database_record_path": str(database_path.resolve()),
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True), "utf-8")
    return database_path


def _resolve_request_override_path(request_id: str) -> Path | None:
    candidates = [
        settings.output_dir / request_id / "panel_overrides.json",
        settings.output_dir / f"analyze-{request_id}" / "panel_overrides.json",
        _override_database_record_path(request_id),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _relative_to_project_root(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path.resolve()


def _register_request_v2(
    *,
    request_id: str,
    status: str,
    fs_root_path: Path,
    asset_path: Path | None = None,
    asset_type: str = "source_page",
    file_name: str | None = None,
    panel_mode: str | None = None,
    bubble_mode: str | None = None,
) -> None:
    metadata = {"file_name": file_name} if file_name else {}
    upsert_request_v2(
        request_id=request_id,
        status=status,
        fs_root_path=fs_root_path,
        panel_mode=panel_mode,
        bubble_mode=bubble_mode,
        metadata=metadata,
    )
    if asset_path is not None:
        insert_request_asset_v2(
            request_id=request_id,
            asset_type=asset_type,
            rel_path=_relative_to_project_root(asset_path),
            metadata=metadata,
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
    request_root = settings.output_dir / request_id
    await asyncio.to_thread(
        _register_request_v2,
        request_id=request_id,
        status="uploaded",
        fs_root_path=request_root,
        asset_path=upload_path,
        asset_type="source_upload",
        file_name=file.filename,
    )
    image_id = await asyncio.to_thread(
        register_image_path,
        settings,
        upload_path,
        asset_key=request_id,
        source_type="upload",
    )

    redis = app.state.redis
    await set_task_status(
        redis,
        request_id,
        build_status_payload(
            request_id,
            "queued",
            attempt=0,
            filename=file.filename,
            upload_path=str(upload_path),
            image_id=image_id,
            job_type="generate_video",
        ),
    )
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
async def detect_panels(file: UploadFile = File(...), panel_mode: str = Form("heuristic")) -> DetectPanelsResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")
    if panel_mode not in {"heuristic", "detector"}:
        raise HTTPException(status_code=400, detail="panel_mode must be 'heuristic' or 'detector'.")
    if panel_mode == "detector":
        detector = get_panel_model_detector(settings)
        if not detector.available:
            raise HTTPException(status_code=409, detail="Panel detector mode is selected, but no panel detector weights are loaded.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"panels-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / request_id
    panels_dir = job_dir / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = job_dir / f"source{upload_suffix}"

    try:
        await asyncio.to_thread(source_image_path.write_bytes, file_bytes)
        await asyncio.to_thread(
            _register_request_v2,
            request_id=request_id,
            status="review_pending",
            fs_root_path=job_dir,
            asset_path=source_image_path,
            asset_type="source_page",
            file_name=file.filename,
            panel_mode=panel_mode,
        )
        image_id = await asyncio.to_thread(
            register_image_path,
            settings,
            source_image_path,
            asset_key=f"detect:{request_id}",
            source_type="upload",
        )
        detected_panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir, panel_mode=panel_mode)
        if not detected_panels:
            raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")
        await asyncio.to_thread(
            register_detected_panels,
            settings,
            image_id=image_id,
            panels=detected_panels,
            generator=f"detect-panels:{panel_mode}",
            created_by="detect_panels_endpoint",
        )

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
            source_image_url=_build_source_image_url(request_id, source_image_path),
        )
    finally:
        await asyncio.to_thread(upload_path.unlink, True)


@app.post("/generate-script", response_model=GenerateScriptResponse)
async def generate_script(
    file: UploadFile = File(...),
    bubble_mode: str = Form("heuristic"),
    panel_mode: str = Form("heuristic"),
) -> GenerateScriptResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid4().hex
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"script-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / f"script-{request_id}"
    panels_dir = job_dir / "panels"
    source_image_path = job_dir / f"source{upload_suffix}"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    try:
        await asyncio.to_thread(source_image_path.write_bytes, file_bytes)
        await asyncio.to_thread(
            _register_request_v2,
            request_id=request_id,
            status="processing",
            fs_root_path=job_dir,
            asset_path=source_image_path,
            asset_type="source_page",
            file_name=file.filename,
            panel_mode=panel_mode,
            bubble_mode=bubble_mode,
        )
        image_id = await asyncio.to_thread(
            register_image_path,
            settings,
            source_image_path,
            asset_key=f"script:{request_id}",
            source_type="upload",
        )
        panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir, panel_mode=panel_mode)
        if not panels:
            raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")
        await asyncio.to_thread(
            register_detected_panels,
            settings,
            image_id=image_id,
            panels=panels,
            generator=f"generate-script:{panel_mode}",
            created_by="generate_script_endpoint",
        )

        features = await caption_service.analyze_panel_features(panels, bubble_mode=bubble_mode)
        dialogue = await ocr_service.extract_dialogue(panels, region_hints=features)
        captions = [feature.to_caption_payload() for feature in features]
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
async def analyze_panels(
    file: UploadFile = File(...),
    bubble_mode: str = Form("heuristic"),
    panel_mode: str = Form("heuristic"),
) -> AnalyzePanelsResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")
    if bubble_mode not in {"heuristic", "detector"}:
        raise HTTPException(status_code=400, detail="bubble_mode must be 'heuristic' or 'detector'.")
    if panel_mode not in {"heuristic", "detector"}:
        raise HTTPException(status_code=400, detail="panel_mode must be 'heuristic' or 'detector'.")
    if bubble_mode == "detector":
        detector = get_bubble_detector(settings)
        if not detector.available:
            raise HTTPException(status_code=409, detail="Object detector mode is selected, but no detector weights are loaded.")
    if panel_mode == "detector":
        detector = get_panel_model_detector(settings)
        if not detector.available:
            raise HTTPException(status_code=409, detail="Panel detector mode is selected, but no panel detector weights are loaded.")

    request_id = uuid4().hex
    phase1_started_at = datetime.now().astimezone()
    phase1_started_perf = time.perf_counter()
    upload_suffix = Path(file.filename).suffix or ".png"
    upload_path = settings.temp_dir / f"analyze-{request_id}{upload_suffix}"
    file_bytes = await file.read()
    await panel_detector.save_upload(upload_path, file_bytes)

    job_dir = settings.output_dir / request_id
    panels_dir = job_dir / "panels"
    source_image_path = job_dir / f"source{upload_suffix}"
    job_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    try:
        await asyncio.to_thread(source_image_path.write_bytes, file_bytes)
        await asyncio.to_thread(
            _register_request_v2,
            request_id=request_id,
            status="review_pending",
            fs_root_path=job_dir,
            asset_path=source_image_path,
            asset_type="source_page",
            file_name=file.filename,
            panel_mode=panel_mode,
            bubble_mode=bubble_mode,
        )
        image_id = await asyncio.to_thread(
            register_image_path,
            settings,
            source_image_path,
            asset_key=f"analyze:{request_id}",
            source_type="upload",
        )
        if bubble_mode == "detector":
            logger.info(
                "Phase1 detector run started request_id=%s filename=%s started_at=%s",
                request_id,
                file.filename,
                phase1_started_at.isoformat(),
            )
        panels = await panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir, panel_mode=panel_mode)
        if not panels:
            raise HTTPException(status_code=422, detail="No panels were detected in the uploaded manga image.")
        await asyncio.to_thread(
            register_detected_panels,
            settings,
            image_id=image_id,
            panels=panels,
            generator=f"analyze-panels:{panel_mode}",
            created_by="analyze_panels_endpoint",
        )

        features = await caption_service.analyze_panel_features(panels, bubble_mode=bubble_mode)
        dialogue = await ocr_service.extract_dialogue(panels, region_hints=features)
        captions = [feature.to_caption_payload() for feature in features]
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
            panel_mode=panel_mode,
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
    database_path = await asyncio.to_thread(
        _write_override_database_record,
        payload=payload,
        job_dir=job_dir,
        overrides_path=overrides_path,
    )
    v2_result = await asyncio.to_thread(
        replace_request_annotations_v2,
        request_id=payload.request_id,
        panel_boxes=[item.model_dump(mode="json") for item in (payload.panel_boxes or [])],
        panel_regions={
            panel_key: [region.model_dump(mode="json") for region in regions]
            for panel_key, regions in (payload.panel_regions or {}).items()
        },
        created_by="override_ui",
    )
    await asyncio.to_thread(
        upsert_request_v2,
        request_id=payload.request_id,
        status="review_pending",
        fs_root_path=job_dir,
        metadata={"overrides": serialized.get("overrides") or {}},
    )
    await asyncio.to_thread(
        upsert_job,
        settings,
        {
            "request_id": payload.request_id,
            "status": "review_pending",
            "output_path": str(job_dir),
            "filename": None,
            "job_type": "panel_override",
        },
        job_type="panel_override",
    )
    await asyncio.to_thread(
        append_job_event,
        settings,
        request_id=payload.request_id,
        event_type="panel_overrides_saved",
        status="saved",
        details={
            "request_output_dir": str(job_dir),
            "request_override_path": str(overrides_path),
            "database_record_path": str(database_path),
            "override_count": len(payload.overrides),
            "panel_box_count": len(payload.panel_boxes or []),
            "panel_region_groups": len(payload.panel_regions or {}),
            "v2_annotation_count": v2_result["annotation_count"],
            "v2_version_count": v2_result["version_count"],
        },
    )
    logger.info(
        "Stored override request_id=%s request_path=%s database_path=%s v2_annotations=%s",
        payload.request_id,
        overrides_path,
        database_path,
        v2_result["annotation_count"],
    )
    return SaveOverridesResponse(
        request_id=payload.request_id,
        saved=True,
        overrides_path=str(overrides_path),
    )


@app.get("/panel-overrides/{request_id}", response_model=LoadOverridesResponse)
async def load_panel_overrides(request_id: str) -> LoadOverridesResponse:
    v2_payload = await asyncio.to_thread(load_request_override_v2, request_id)
    if v2_payload is not None:
        return LoadOverridesResponse(
            request_id=request_id,
            exists=True,
            overrides_path=str(settings.sql_database_path.parent / "app_state_v2.sqlite3"),
            overrides=v2_payload.get("overrides") or {},
            panel_boxes=v2_payload.get("panel_boxes") or [],
            panel_regions=v2_payload.get("panel_regions") or {},
        )

    overrides_path = _resolve_request_override_path(request_id)
    if overrides_path is None:
        return LoadOverridesResponse(request_id=request_id, exists=False)

    try:
        payload = json.loads(await asyncio.to_thread(overrides_path.read_text, "utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read saved overrides: {exc}") from exc

    return LoadOverridesResponse(
        request_id=request_id,
        exists=True,
        overrides_path=str(overrides_path),
        overrides=payload.get("overrides") or {},
        panel_boxes=payload.get("panel_boxes") or [],
        panel_regions=payload.get("panel_regions") or {},
    )


@app.get("/api/annotation/images", response_model=DatasetImageListResponse)
async def get_annotation_images(dataset: str | None = None, split: str = "train", offset: int = 0, limit: int = 50) -> DatasetImageListResponse:
    try:
        payload = list_dataset_images(settings, split=split, offset=offset, limit=limit, dataset_key=dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DatasetImageListResponse(**payload)


@app.get("/api/annotation/item", response_model=DatasetAnnotationItemResponse)
async def get_annotation_item(dataset: str | None = None, split: str = "train", index: int = 0) -> DatasetAnnotationItemResponse:
    try:
        payload = load_image_annotations(settings, split=split, index=index, dataset_key=dataset)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DatasetAnnotationItemResponse(**payload)


@app.get("/api/annotation/review-queue", response_model=AnnotationReviewQueueResponse)
async def get_annotation_review_queue(dataset: str | None = None) -> AnnotationReviewQueueResponse:
    try:
        payload = build_review_queue(settings, dataset_key=dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnnotationReviewQueueResponse(**payload)


@app.post("/api/annotation/save", response_model=SaveDatasetAnnotationsResponse)
async def save_annotation_item(payload: SaveDatasetAnnotationsRequest) -> SaveDatasetAnnotationsResponse:
    try:
        annotation_path = save_image_annotations(
            settings,
            dataset_key=payload.dataset,
            split=payload.split,
            image_id=payload.image_id,
            file_name=payload.file_name,
            width=payload.width,
            height=payload.height,
            annotations=[item.model_dump(mode="json") for item in payload.annotations],
        )
        dataset_root = resolve_dataset_root(settings, payload.dataset)
        await asyncio.to_thread(
            store_annotation_snapshot,
            settings,
            dataset_root=dataset_root,
            split_name=payload.split,
            file_name=payload.file_name,
            width=payload.width,
            height=payload.height,
            annotations=[item.model_dump(mode="json") for item in payload.annotations],
            created_by="annotation_ui",
        )
        dataset_root = resolve_dataset_root(settings, payload.dataset)
        image_id = await asyncio.to_thread(
            register_image_path,
            settings,
            dataset_root / payload.split / payload.file_name,
            asset_key=f"{dataset_root.name}:{payload.file_name}",
            source_type="dataset",
            width=payload.width,
            height=payload.height,
        )
        await asyncio.to_thread(
            append_annotation_event,
            settings,
            image_id=image_id,
            dataset_name=dataset_root.name,
            split_name="validation" if payload.split == "valid" else payload.split,
            event_type="annotation_saved",
            actor="annotation_ui",
            file_name=payload.file_name,
            details={
                "annotation_path": str(annotation_path),
                "annotation_count": len(payload.annotations),
                "classes": sorted({item.class_name for item in payload.annotations}),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SaveDatasetAnnotationsResponse(
        dataset=payload.dataset,
        split=payload.split,
        image_id=payload.image_id,
        file_name=payload.file_name,
        saved=True,
        annotation_path=str(annotation_path),
    )


@app.post("/api/annotation/export", response_model=ExportDatasetResponse)
async def export_annotation_dataset(dataset: str = Form(...), export_mode: str = Form("full")) -> ExportDatasetResponse:
    if export_mode == "validated_bubble_only":
        destination_root = settings.annotation_output_dir / "exported_coco" / f"{dataset}__validated_bubble_only"
        annotation_files = export_annotated_dataset(
            settings,
            destination_root,
            dataset_key=dataset,
            validated_only=True,
            bubble_only=True,
        )
    elif export_mode == "full":
        destination_root = settings.annotation_output_dir / "exported_coco" / dataset
        annotation_files = export_annotated_dataset(settings, destination_root, dataset_key=dataset)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export_mode: {export_mode}")
    await asyncio.to_thread(
        append_annotation_event,
        settings,
        image_id=None,
        dataset_name=dataset,
        split_name="all",
        event_type="dataset_exported",
        actor="annotation_ui",
        file_name=None,
        details={
            "export_mode": export_mode,
            "output_dir": str(destination_root),
            "annotation_files": [str(path) for path in annotation_files],
        },
    )
    return ExportDatasetResponse(
        dataset=dataset,
        export_mode=export_mode,
        exported=True,
        output_dir=str(destination_root),
        annotation_files=[str(path) for path in annotation_files],
    )
