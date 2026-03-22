from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BubbleDetection:
    boxes: list[tuple[int, int, int, int]]
    scores: list[float]
    labels: list[str]
    source: str

    def boxes_for(self, class_name: str) -> list[tuple[int, int, int, int]]:
        return [box for box, label in zip(self.boxes, self.labels) if label == class_name]


class BubbleDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.weights_path = Path(settings.bubble_detector_weights).expanduser() if settings.bubble_detector_weights else None
        self.score_threshold = settings.bubble_detector_score_threshold
        self.max_detections = settings.bubble_detector_max_detections
        self.max_side = max(256, settings.bubble_detector_max_side)
        self.device = self._select_device()
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = ["background", "speech_bubble"]
        self._load_error: str | None = None
        self._load_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, image_bgr: np.ndarray) -> BubbleDetection:
        if self.model is None or image_bgr.size == 0:
            return BubbleDetection(boxes=[], scores=[], labels=[], source="unavailable")

        original_height, original_width = image_bgr.shape[:2]
        image_for_model, scale = self._resize_for_inference(image_bgr)
        image_rgb = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]

        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        labels: list[str] = []
        raw_boxes = output.get("boxes")
        raw_scores = output.get("scores")
        raw_labels = output.get("labels")
        if raw_boxes is None or raw_scores is None or raw_labels is None:
            return BubbleDetection(boxes=[], scores=[], labels=[], source="model")

        for box_tensor, score_tensor, label_tensor in zip(raw_boxes, raw_scores, raw_labels):
            score = float(score_tensor.item())
            label = int(label_tensor.item())
            if label <= 0 or label >= len(self.class_names) or score < self.score_threshold:
                continue
            x1, y1, x2, y2 = [int(round(value / scale)) for value in box_tensor.tolist()]
            x1 = max(0, min(x1, original_width))
            x2 = max(0, min(x2, original_width))
            y1 = max(0, min(y1, original_height))
            y2 = max(0, min(y2, original_height))
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w < 8 or h < 8:
                continue
            boxes.append((x1, y1, w, h))
            scores.append(score)
            class_name = self._normalize_class_name(self.class_names[label])
            labels.append(class_name)
            if len(boxes) >= self.max_detections:
                break

        return BubbleDetection(boxes=boxes, scores=scores, labels=labels, source="model")

    def _load_model(self) -> None:
        if not self.weights_path:
            logger.info("Bubble detector weights not configured; using heuristic bubble detection only")
            return
        if not self.weights_path.exists():
            self._load_error = f"weights not found at {self.weights_path}"
            logger.warning("Bubble detector unavailable: %s", self._load_error)
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            class_names = checkpoint.get("class_names", ["background", "Bubble with Text"]) if isinstance(checkpoint, dict) else ["background", "Bubble with Text"]
            self.class_names = [self._normalize_class_name(name) for name in class_names]
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=len(self.class_names),
            )
            state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Loaded local bubble detector from %s on %s with classes=%s", self.weights_path, self.device, self.class_names)
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load bubble detector from %s", self.weights_path)

    def _select_device(self) -> torch.device:
        configured = self.settings.bubble_detector_device
        if configured and configured != "auto":
            if configured == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if configured == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            if configured == "cpu":
                return torch.device("cpu")
            logger.warning("Unsupported or unavailable bubble detector device '%s'; falling back to auto", configured)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _normalize_class_name(self, name: str) -> str:
        if name == "Bubble with Text":
            return "speech_bubble"
        if name == "objects":
            return "speech_bubble"
        if name == "text":
            return "speech_bubble"
        return name

    def _resize_for_inference(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image_bgr.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self.max_side:
            return image_bgr, 1.0

        scale = self.max_side / float(longest_side)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return resized, scale


_DETECTOR: BubbleDetector | None = None


def get_bubble_detector(settings: Settings) -> BubbleDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = BubbleDetector(settings)
    return _DETECTOR
