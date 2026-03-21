import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from config import Settings
from modules.panel_model_detector import get_panel_model_detector
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

    async def detect_panels(self, upload_path: Path, output_dir: Path, panel_mode: str = "heuristic") -> list[DetectedPanel]:
        logger.info("Detecting panels for %s using mode=%s", upload_path.name, panel_mode)
        boxes = await self._detect_boxes(upload_path, panel_mode=panel_mode)
        panel_paths = await crop_panels(upload_path=upload_path, boxes=boxes, output_dir=output_dir)
        panels = [DetectedPanel(index=index, image_path=path, bbox=box) for index, (path, box) in enumerate(zip(panel_paths, boxes), start=1)]
        logger.info("Detected %s panels", len(panels))
        return panels

    async def _detect_boxes(self, upload_path: Path, panel_mode: str) -> list[tuple[int, int, int, int]]:
        if panel_mode == "detector":
            detected = await self._detect_boxes_from_model(upload_path)
            if detected:
                return detected
            logger.warning("Panel detector produced no usable boxes for %s; falling back to heuristic splitter", upload_path.name)
        return await detect_panel_boxes(upload_path)

    async def _detect_boxes_from_model(self, upload_path: Path) -> list[tuple[int, int, int, int]]:
        detector = get_panel_model_detector(self.settings)
        if not detector.available:
            return []

        image = await self._read_image(upload_path)
        if image is None:
            return []
        result = detector.detect(image)
        panel_items = [
            (box, score)
            for box, score, label in zip(result.boxes, result.scores, result.labels)
            if label == "panel"
        ]
        return self._normalize_panel_boxes(panel_items, image.shape[1], image.shape[0])

    async def _read_image(self, upload_path: Path):
        return await asyncio.to_thread(cv2.imread, str(upload_path))

    def _normalize_panel_boxes(
        self,
        box_items: list[tuple[tuple[int, int, int, int], float]],
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int, int, int]]:
        normalized: list[tuple[tuple[int, int, int, int], float]] = []
        page_area = image_width * image_height
        for (x1, y1, x2, y2), score in box_items:
            x1 = max(0, min(int(x1), image_width))
            x2 = max(0, min(int(x2), image_width))
            y1 = max(0, min(int(y1), image_height))
            y2 = max(0, min(int(y2), image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < page_area * 0.01:
                continue
            normalized.append(((x1, y1, x2, y2), score))

        if not normalized:
            return []

        normalized.sort(key=lambda item: item[1], reverse=True)
        deduped: list[tuple[tuple[int, int, int, int], float]] = []
        for candidate, score in normalized:
            if any(self._should_suppress_panel_box(candidate, kept) for kept, _ in deduped):
                continue
            deduped.append((candidate, score))

        row_band = max(image_height // 8, 1)
        sorted_boxes = [box for box, _ in deduped]
        sorted_boxes.sort(key=lambda box: (box[1] // row_band, -(box[0] + box[2]), box[1]))
        return sorted_boxes

    def _should_suppress_panel_box(
        self,
        candidate: tuple[int, int, int, int],
        kept: tuple[int, int, int, int],
    ) -> bool:
        if self._intersection_over_union(candidate, kept) >= 0.55:
            return True
        if self._containment_ratio(candidate, kept) >= 0.82:
            return True
        if self._containment_ratio(kept, candidate) >= 0.94:
            return True
        return False

    def _intersection_over_union(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        inter_x1 = max(first[0], second[0])
        inter_y1 = max(first[1], second[1])
        inter_x2 = min(first[2], second[2])
        inter_y2 = min(first[3], second[3])
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _containment_ratio(
        self,
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> float:
        inter_x1 = max(inner[0], outer[0])
        inter_y1 = max(inner[1], outer[1])
        inter_x2 = min(inner[2], outer[2])
        inter_y2 = min(inner[3], outer[3])
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        inner_area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
        if inner_area <= 0:
            return 0.0
        return inter_area / inner_area
