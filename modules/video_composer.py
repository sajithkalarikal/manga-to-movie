import logging
from pathlib import Path

from config import Settings
from utils.ffmpeg_utils import mix_background_music, mux_audio_video, write_srt_file

logger = logging.getLogger(__name__)


class VideoComposerService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def compose(
        self,
        scene_script: dict[str, object],
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        subtitles_path: Path,
    ) -> Path:
        subtitle_text = str(scene_script.get("dialogue") or scene_script.get("scene_description") or "")
        duration = int(scene_script.get("duration", 5))
        await write_srt_file(subtitles_path, subtitle_text, duration)

        final_audio_path = audio_path
        if self.settings.background_music_path:
            mixed_audio_path = output_path.with_name("voice_with_music.mp3")
            await mix_background_music(
                ffmpeg_binary=self.settings.ffmpeg_binary,
                narration_path=audio_path,
                music_path=Path(self.settings.background_music_path),
                output_path=mixed_audio_path,
            )
            final_audio_path = mixed_audio_path

        logger.info("Composing final video with FFmpeg")
        await mux_audio_video(
            ffmpeg_binary=self.settings.ffmpeg_binary,
            video_path=video_path,
            audio_path=final_audio_path,
            output_path=output_path,
            subtitles_path=subtitles_path,
        )
        return output_path
