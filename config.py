import logging
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_KEYS_FILE = Path("local.keys.json")


def _load_file_settings() -> dict[str, str]:
    if not _KEYS_FILE.exists():
        return {}
    try:
        payload = json.loads(_KEYS_FILE.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _default_tesseract_cmd() -> str | None:
    candidates = [
        os.getenv("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _resolve_binary(configured: str, fallback_name: str, common_paths: list[str]) -> str:
    candidates = [configured, shutil.which(configured), shutil.which(fallback_name), *common_paths]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return configured


@dataclass(slots=True)
class Settings:
    app_name: str = "manga-to-video-ai"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    temp_dir: Path = Path("temp")
    log_dir: Path = Path("logs")
    output_base_url: str = "/outputs"
    redis_url: str = "redis://localhost:6379/0"
    redis_queue_name: str = "manga-video-jobs"
    task_result_ttl_seconds: int = 86400
    job_timeout_seconds: int = 1800
    job_retry_count: int = 3
    retry_delay_seconds: int = 15
    worker_concurrency: int = 4
    tesseract_cmd: str | None = None
    blip_api_url: str = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"
    huggingface_api_token: str = ""
    llm_api_url: str = "https://api.openai.com/v1/chat/completions"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    video_provider: str = "runway"
    runway_base_url: str = "https://api.dev.runwayml.com"
    runway_api_key: str = ""
    runway_api_version: str = "2024-11-06"
    runway_model: str = "gen4_turbo"
    runway_ratio: str = "1280:720"
    runway_duration: int = 5
    fal_api_key: str = ""
    pika_model: str = "fal-ai/pika/v2.2/image-to-video"
    pika_resolution: str = "720p"
    pika_duration: int = 5
    pika_poll_seconds: int = 5
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    ffmpeg_binary: str = "/opt/homebrew/bin/ffmpeg"
    ffprobe_binary: str = "/opt/homebrew/bin/ffprobe"
    background_music_path: str = ""
    bubble_detector_weights: str = ""
    bubble_detector_score_threshold: float = 0.45
    bubble_detector_max_detections: int = 8
    bubble_detector_device: str = "auto"
    bubble_detector_max_side: int = 960
    panel_detector_weights: str = ""
    panel_detector_score_threshold: float = 0.45
    panel_detector_max_detections: int = 16
    panel_detector_device: str = "auto"
    panel_detector_max_side: int = 960
    annotation_dataset_root: Path = Path("../Manga Bubble.v4i.coco")
    annotation_output_dir: Path = Path("outputs/dataset_annotations")
    annotation_database_dir: Path = Path("outputs/dataset_annotations/database")
    sql_database_path: Path = Path("outputs/app_state.sqlite3")

    @classmethod
    def from_env(cls) -> "Settings":
        file_settings = _load_file_settings()
        return cls(
            app_env=_env_str("APP_ENV", "development"),
            host=_env_str("HOST", "0.0.0.0"),
            port=_env_int("PORT", 8000),
            log_level=_env_str("LOG_LEVEL", "INFO"),
            output_base_url=_env_str("OUTPUT_BASE_URL", "/outputs"),
            redis_url=_env_str("REDIS_URL", "redis://localhost:6379/0"),
            redis_queue_name=_env_str("REDIS_QUEUE_NAME", "manga-video-jobs"),
            task_result_ttl_seconds=_env_int("TASK_RESULT_TTL_SECONDS", 86400),
            job_timeout_seconds=_env_int("JOB_TIMEOUT_SECONDS", 1800),
            job_retry_count=_env_int("JOB_RETRY_COUNT", 3),
            retry_delay_seconds=_env_int("RETRY_DELAY_SECONDS", 15),
            worker_concurrency=_env_int("WORKER_CONCURRENCY", 4),
            tesseract_cmd=_default_tesseract_cmd(),
            blip_api_url=_env_str(
                "BLIP_API_URL",
                "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base",
            ),
            huggingface_api_token=_env_str("HUGGINGFACE_API_TOKEN", file_settings.get("HUGGINGFACE_API_TOKEN", "")),
            llm_api_url=_env_str("LLM_API_URL", file_settings.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")),
            llm_api_key=_env_str("OPENAI_API_KEY", file_settings.get("OPENAI_API_KEY", "")),
            llm_model=_env_str("LLM_MODEL", "gpt-4o-mini"),
            video_provider=_env_str("VIDEO_PROVIDER", "runway"),
            runway_base_url=_env_str("RUNWAY_BASE_URL", file_settings.get("RUNWAY_BASE_URL", "https://api.dev.runwayml.com")),
            runway_api_key=_env_str("RUNWAY_API_KEY", file_settings.get("RUNWAY_API_KEY", "")),
            runway_api_version=_env_str("RUNWAY_API_VERSION", "2024-11-06"),
            runway_model=_env_str("RUNWAY_MODEL", "gen4_turbo"),
            runway_ratio=_env_str("RUNWAY_RATIO", "1280:720"),
            runway_duration=_env_int("RUNWAY_DURATION", 5),
            fal_api_key=_env_str("FAL_KEY", file_settings.get("FAL_KEY", "")),
            pika_model=_env_str("PIKA_MODEL", "fal-ai/pika/v2.2/image-to-video"),
            pika_resolution=_env_str("PIKA_RESOLUTION", "720p"),
            pika_duration=_env_int("PIKA_DURATION", 5),
            pika_poll_seconds=_env_int("PIKA_POLL_SECONDS", 5),
            elevenlabs_api_key=_env_str("ELEVENLABS_API_KEY", file_settings.get("ELEVENLABS_API_KEY", "")),
            elevenlabs_voice_id=_env_str("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            elevenlabs_model_id=_env_str("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            ffmpeg_binary=_resolve_binary(
                _env_str("FFMPEG_BINARY", "/opt/homebrew/bin/ffmpeg"),
                "ffmpeg",
                ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"],
            ),
            ffprobe_binary=_resolve_binary(
                _env_str("FFPROBE_BINARY", "/opt/homebrew/bin/ffprobe"),
                "ffprobe",
                ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"],
            ),
            background_music_path=_env_str("BACKGROUND_MUSIC_PATH", ""),
            bubble_detector_weights=_env_str("BUBBLE_DETECTOR_WEIGHTS", ""),
            bubble_detector_score_threshold=_env_float("BUBBLE_DETECTOR_SCORE_THRESHOLD", 0.45),
            bubble_detector_max_detections=_env_int("BUBBLE_DETECTOR_MAX_DETECTIONS", 8),
            bubble_detector_device=_env_str("BUBBLE_DETECTOR_DEVICE", "auto").strip().lower(),
            bubble_detector_max_side=_env_int("BUBBLE_DETECTOR_MAX_SIDE", 960),
            panel_detector_weights=_env_str("PANEL_DETECTOR_WEIGHTS", ""),
            panel_detector_score_threshold=_env_float("PANEL_DETECTOR_SCORE_THRESHOLD", 0.45),
            panel_detector_max_detections=_env_int("PANEL_DETECTOR_MAX_DETECTIONS", 16),
            panel_detector_device=_env_str("PANEL_DETECTOR_DEVICE", "auto").strip().lower(),
            panel_detector_max_side=_env_int("PANEL_DETECTOR_MAX_SIDE", 960),
            annotation_dataset_root=Path(_env_str("ANNOTATION_DATASET_ROOT", "../Manga Bubble.v4i.coco")).expanduser(),
            annotation_output_dir=Path(_env_str("ANNOTATION_OUTPUT_DIR", "outputs/dataset_annotations")).expanduser(),
            annotation_database_dir=Path(
                _env_str("ANNOTATION_DATABASE_DIR", "outputs/dataset_annotations/database")
            ).expanduser(),
            sql_database_path=Path(_env_str("SQL_DATABASE_PATH", "outputs/app_state.sqlite3")).expanduser(),
        )

    def prepare_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_output_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_database_dir.mkdir(parents=True, exist_ok=True)
        self.sql_database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.prepare_directories()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_dir / "app.log", encoding="utf-8"),
        ],
    )
    return settings
