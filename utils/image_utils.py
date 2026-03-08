import asyncio
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


async def write_bytes(path: Path, content: bytes) -> None:
    await asyncio.to_thread(path.write_bytes, content)


async def detect_panel_boxes(image_path: Path) -> list[tuple[int, int, int, int]]:
    return await asyncio.to_thread(_detect_panel_boxes_sync, image_path)


def _detect_panel_boxes_sync(image_path: Path) -> list[tuple[int, int, int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    min_area = (width * height) * 0.03
    boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        boxes.append((x, y, x + w, y + h))

    if len(boxes) < 2:
        boxes = _fallback_boxes(width, height)

    return _sort_boxes(boxes, height)


def _fallback_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    panel_count = 4 if height >= width else 3
    segment_height = max(height // panel_count, 1)
    boxes = []
    for index in range(panel_count):
        top = index * segment_height
        bottom = height if index == panel_count - 1 else (index + 1) * segment_height
        boxes.append((0, top, width, bottom))
    return boxes


def _sort_boxes(boxes: list[tuple[int, int, int, int]], height: int) -> list[tuple[int, int, int, int]]:
    row_band = max(height // 8, 1)
    return sorted(boxes, key=lambda box: (box[1] // row_band, box[0], box[1]))


async def crop_panels(upload_path: Path, boxes: list[tuple[int, int, int, int]], output_dir: Path) -> list[Path]:
    return await asyncio.to_thread(_crop_panels_sync, upload_path, boxes, output_dir)


def _crop_panels_sync(upload_path: Path, boxes: list[tuple[int, int, int, int]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[Path] = []
    with Image.open(upload_path) as image:
        rgb_image = image.convert("RGB")
        for index, bbox in enumerate(boxes, start=1):
            cropped = rgb_image.crop(bbox)
            panel_path = output_dir / f"panel_{index:02d}.png"
            cropped.save(panel_path)
            panel_paths.append(panel_path)
    return panel_paths
