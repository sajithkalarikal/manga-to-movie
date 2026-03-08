import logging
from dataclasses import dataclass
from pathlib import Path

from config import Settings
from utils.image_utils import crop_panels, detect_panel_boxes, write_bytes

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectedPanel:
    index: int
    image_path: Path
    bbox: tuple[int, int, int, int]


class PanelDetectionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def save_upload(self, destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        await write_bytes(destination, content)

    async def detect_panels(self, upload_path: Path, output_dir: Path) -> list[DetectedPanel]:
        logger.info("Detecting panels for %s", upload_path.name)
        boxes = await detect_panel_boxes(upload_path)
        panel_paths = await crop_panels(upload_path=upload_path, boxes=boxes, output_dir=output_dir)
        panels = [DetectedPanel(index=index, image_path=path, bbox=box) for index, (path, box) in enumerate(zip(panel_paths, boxes), start=1)]
        logger.info("Detected %s panels", len(panels))
        return panels
