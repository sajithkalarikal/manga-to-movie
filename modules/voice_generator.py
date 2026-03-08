import asyncio
import logging
from pathlib import Path

import aiohttp

from config import Settings
from utils.ffmpeg_utils import create_silent_audio

logger = logging.getLogger(__name__)


class VoiceGeneratorService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_voice(self, scene_script: dict[str, object], output_path: Path) -> Path:
        dialogue = str(scene_script.get("dialogue") or "")
        duration = int(scene_script.get("duration", 5))
        if not dialogue or not self.settings.elevenlabs_api_key:
            logger.info("Using silent audio fallback")
            await create_silent_audio(output_path=output_path, duration=duration, ffmpeg_binary=self.settings.ffmpeg_binary)
            return output_path

        headers = {
            "xi-api-key": self.settings.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": dialogue,
            "model_id": self.settings.elevenlabs_model_id,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
            },
        }
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}"

        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                data = await response.read()
                if response.status >= 400:
                    raise RuntimeError(f"ElevenLabs error {response.status}: {data.decode(errors='ignore')}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output_path.write_bytes, data)
        return output_path
