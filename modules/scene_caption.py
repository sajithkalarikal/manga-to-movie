import asyncio
import logging
from pathlib import Path

import aiohttp

from config import Settings
from modules.panel_detection import DetectedPanel

logger = logging.getLogger(__name__)


class SceneCaptionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_captions(self, panels: list[DetectedPanel]) -> list[dict[str, str]]:
        logger.info("Generating BLIP captions for %s panels", len(panels))
        tasks = [self.generate_caption(panel.image_path, panel.index) for panel in panels]
        return await asyncio.gather(*tasks)

    async def generate_caption(self, image_path: Path, panel_index: int | None = None) -> dict[str, str] | str:
        headers = {"Content-Type": "application/octet-stream"}
        if self.settings.huggingface_api_token:
            headers["Authorization"] = f"Bearer {self.settings.huggingface_api_token}"

        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.settings.blip_api_url, headers=headers, data=image_bytes, timeout=aiohttp.ClientTimeout(total=90)) as response:
                payload = await response.json(content_type=None)
                if response.status == 503:
                    await asyncio.sleep(5)
                    return await self.generate_caption(image_path, panel_index)
                if response.status >= 400:
                    raise RuntimeError(f"BLIP API error {response.status}: {payload}")

        caption_text = self._extract_caption(payload)
        if panel_index is None:
            return caption_text
        return {"panel": str(panel_index), "caption": caption_text}

    @staticmethod
    def _extract_caption(payload) -> str:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and first.get("generated_text"):
                return str(first["generated_text"]).strip()
        if isinstance(payload, dict) and payload.get("generated_text"):
            return str(payload["generated_text"]).strip()
        return "Dynamic manga scene with dramatic motion"
