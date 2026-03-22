import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from modules.database import register_detected_panels, register_image_path
from modules.ocr_dialogue import OCRDialogueService
from modules.panel_detection import PanelDetectionService
from modules.scene_caption import SceneCaptionService
from modules.script_generator import ScriptGeneratorService
from modules.video_composer import VideoComposerService
from modules.video_generator import VideoGeneratorService
from modules.voice_generator import VoiceGeneratorService

logger = logging.getLogger(__name__)


class MangaVideoPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.panel_detector = PanelDetectionService(settings)
        self.ocr_service = OCRDialogueService(settings)
        self.caption_service = SceneCaptionService(settings)
        self.script_service = ScriptGeneratorService(settings)
        self.video_service = VideoGeneratorService(settings)
        self.voice_service = VoiceGeneratorService(settings)
        self.composer_service = VideoComposerService(settings)

    async def run(self, request_id: str, upload_path: Path) -> dict[str, object]:
        job_dir = self.settings.output_dir / request_id
        panels_dir = job_dir / "panels"
        job_dir.mkdir(parents=True, exist_ok=True)
        panels_dir.mkdir(parents=True, exist_ok=True)
        source_image_path = job_dir / f"source{upload_path.suffix or '.png'}"

        logger.info("Pipeline start request_id=%s", request_id)
        await asyncio.to_thread(source_image_path.write_bytes, upload_path.read_bytes())
        image_id = await asyncio.to_thread(
            register_image_path,
            self.settings,
            source_image_path,
            asset_key=request_id,
            source_type="upload",
        )
        panels = await self.panel_detector.detect_panels(upload_path=upload_path, output_dir=panels_dir)
        if not panels:
            raise RuntimeError("No panels were detected in the uploaded manga image.")
        await asyncio.to_thread(
            register_detected_panels,
            self.settings,
            image_id=image_id,
            panels=panels,
            generator="pipeline:detect_panels",
            created_by="pipeline",
        )

        dialogue = await self.ocr_service.extract_dialogue(panels)
        captions = await self.caption_service.generate_captions(panels)
        scene_script = await self.script_service.generate_script(dialogue=dialogue, captions=captions)

        raw_video_path = job_dir / "scene_video.mp4"
        voice_path = job_dir / "voice.mp3"
        subtitles_path = job_dir / "subtitles.srt"
        final_video_path = job_dir / "final_video.mp4"
        metadata_path = job_dir / "scene_metadata.json"

        await self.video_service.generate_video(
            scene_script=scene_script,
            output_path=raw_video_path,
            scene_image_path=panels[0].image_path,
        )
        await self.voice_service.generate_voice(scene_script=scene_script, output_path=voice_path)
        await self.composer_service.compose(
            scene_script=scene_script,
            video_path=raw_video_path,
            audio_path=voice_path,
            output_path=final_video_path,
            subtitles_path=subtitles_path,
        )

        metadata = {
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_image": str(source_image_path),
            "panels": [
                {"index": panel.index, "bbox": panel.bbox, "image_path": str(panel.image_path)}
                for panel in panels
            ],
            "dialogue": dialogue,
            "captions": captions,
            "scene_script": scene_script,
            "artifacts": {
                "video": str(final_video_path),
                "subtitles": str(subtitles_path),
                "voice": str(voice_path),
            },
        }
        await asyncio.to_thread(metadata_path.write_text, json.dumps(metadata, indent=2), "utf-8")
        logger.info("Pipeline complete request_id=%s", request_id)

        base = self.settings.output_base_url.rstrip("/")
        return {
            "video_url": f"{base}/{request_id}/final_video.mp4",
            "metadata_url": f"{base}/{request_id}/scene_metadata.json",
            "subtitles_url": f"{base}/{request_id}/subtitles.srt",
        }
