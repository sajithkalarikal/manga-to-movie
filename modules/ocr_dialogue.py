import asyncio
import logging
import shutil
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
from pytesseract import TesseractNotFoundError

from config import Settings
from modules.panel_detection import DetectedPanel

logger = logging.getLogger(__name__)


class OCRDialogueService:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        self.ocr_available = self._is_tesseract_available()
        if not self.ocr_available:
            logger.warning("Tesseract is not installed. OCR will use a placeholder fallback.")

    async def extract_dialogue(self, panels: list[DetectedPanel]) -> list[dict[str, str]]:
        logger.info("Running OCR on %s panels", len(panels))
        if not self.ocr_available:
            return [
                {"panel": str(panel.index), "text": "[ocr unavailable: install tesseract to extract dialogue]"}
                for panel in panels
            ]
        results = await asyncio.gather(*(self._read_panel(panel) for panel in panels))
        return results

    async def _read_panel(self, panel: DetectedPanel) -> dict[str, str]:
        text = await asyncio.to_thread(self._ocr_sync, panel.image_path)
        return {
            "panel": str(panel.index),
            "text": text or "[no dialogue detected]",
        }

    @staticmethod
    def _ocr_sync(image_path) -> str:
        with Image.open(image_path) as image:
            grayscale = ImageOps.grayscale(image)
            enhanced = ImageOps.autocontrast(grayscale)
            try:
                text = pytesseract.image_to_string(enhanced, config="--psm 6")
            except TesseractNotFoundError:
                return "[ocr unavailable: install tesseract to extract dialogue]"
        return " ".join(text.split())

    @staticmethod
    def _is_tesseract_available() -> bool:
        candidate = pytesseract.pytesseract.tesseract_cmd
        if candidate and Path(candidate).exists():
            return True
        return shutil.which(candidate or "tesseract") is not None
