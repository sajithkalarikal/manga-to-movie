import asyncio
import logging

from config import Settings
from modules.local_phase1 import analyze_panels
from modules.panel_detection import DetectedPanel

logger = logging.getLogger(__name__)


class SceneCaptionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def analyze_panel_features(self, panels: list[DetectedPanel], bubble_mode: str = "heuristic"):
        logger.info("Analyzing local panel features for %s panels using bubble mode=%s", len(panels), bubble_mode)
        return await asyncio.to_thread(analyze_panels, panels, bubble_mode)

    async def generate_captions(self, panels: list[DetectedPanel], bubble_mode: str = "heuristic") -> list[dict[str, str]]:
        logger.info("Generating local captions for %s panels using bubble mode=%s", len(panels), bubble_mode)
        features = await self.analyze_panel_features(panels, bubble_mode)
        return [feature.to_caption_payload() for feature in features]
