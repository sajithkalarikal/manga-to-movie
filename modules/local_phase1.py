from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np

from config import get_settings
from modules.bubble_detector import get_bubble_detector
from modules.panel_detection import DetectedPanel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PanelSceneFeatures:
    panel_index: int
    area_ratio: float
    aspect_ratio: float
    ink_density: float
    edge_density: float
    bubble_count: int
    bubble_candidate_count: int
    bubble_sequence: list[str]
    speech_count: int
    narration_count: int
    sfx_count: int
    shot_type: str
    motion_level: str
    tone: str
    emotion: str
    action_level: str
    transition_hint: str
    layout_role: str

    def to_caption_payload(self) -> dict[str, str]:
        caption = (
            f"{self.layout_role.capitalize()} {self.shot_type} with a {self.tone} tone "
            f"and {self.motion_level} motion emphasis."
        )
        return {
            "panel": str(self.panel_index),
            "caption": caption,
            "shot_type": self.shot_type,
            "motion_level": self.motion_level,
            "tone": self.tone,
            "emotion": self.emotion,
            "action_level": self.action_level,
            "bubble_count": str(self.bubble_count),
            "bubble_candidates": str(self.bubble_candidate_count),
            "bubble_sequence": " | ".join(self.bubble_sequence) if self.bubble_sequence else "none",
            "speech_count": str(self.speech_count),
            "narration_count": str(self.narration_count),
            "sfx_count": str(self.sfx_count),
            "transition_hint": self.transition_hint,
            "layout_role": self.layout_role,
        }


def analyze_panels(panels: list[DetectedPanel], bubble_mode: str = "heuristic") -> list[PanelSceneFeatures]:
    if not panels:
        return []

    page_width = max(panel.bbox[2] for panel in panels)
    page_height = max(panel.bbox[3] for panel in panels)
    page_area = max(page_width * page_height, 1)
    total_panels = max(len(panels), 1)
    settings = get_settings()
    bubble_detector = get_bubble_detector(settings)
    features: list[PanelSceneFeatures] = []
    for index, panel in enumerate(panels):
        image = cv2.imread(str(panel.image_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        panel_width = max(panel.bbox[2] - panel.bbox[0], 1)
        panel_height = max(panel.bbox[3] - panel.bbox[1], 1)
        area_ratio = (panel_width * panel_height) / page_area
        aspect_ratio = panel_width / max(panel_height, 1)
        ink_density = float(np.mean(gray < 185))
        edge_density = float(np.mean(edges > 0))
        model_result = bubble_detector.detect(image) if bubble_mode == "detector" else None
        if bubble_mode == "detector" and model_result is not None:
            bubble_boxes = list(model_result.boxes)
            bubble_candidate_count = len(model_result.boxes)
        else:
            bubble_boxes, bubble_candidate_count = _detect_bubble_boxes_with_debug(gray)
        merged_regions = _promote_bubbles_from_text_regions(
            bubble_boxes=bubble_boxes,
            gray=gray,
            panel_width=panel_width,
            panel_height=panel_height,
            bubble_candidate_count=bubble_candidate_count,
        )
        classified_regions = _classify_text_regions(
            gray,
            merged_regions,
            panel_width,
            panel_height,
            speech_seed_boxes=model_result.boxes if model_result is not None else None,
        )
        speech_boxes = [item["box"] for item in classified_regions if item["kind"] == "speech"]
        narration_boxes = [item["box"] for item in classified_regions if item["kind"] == "narration"]
        sfx_boxes = [item["box"] for item in classified_regions if item["kind"] == "sfx"]
        bubble_count = len(speech_boxes)
        bubble_sequence = _describe_bubble_sequence(speech_boxes, panel_width, panel_height)
        shot_type = _classify_shot_type(area_ratio, aspect_ratio)
        motion_level = _classify_motion(edge_density, ink_density)
        tone = _classify_tone(ink_density, edge_density)
        emotion = _classify_emotion(ink_density, edge_density, bubble_count)
        action_level = _classify_action_level(edge_density, aspect_ratio, bubble_count)
        transition_hint = _classify_transition(index, total_panels, aspect_ratio)
        layout_role = _classify_layout_role(index, total_panels, area_ratio)
        features.append(
            PanelSceneFeatures(
                panel_index=panel.index,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                ink_density=ink_density,
                edge_density=edge_density,
                bubble_count=bubble_count,
                bubble_candidate_count=bubble_candidate_count,
                bubble_sequence=bubble_sequence,
                speech_count=len(speech_boxes),
                narration_count=len(narration_boxes),
                sfx_count=len(sfx_boxes),
                shot_type=shot_type,
                motion_level=motion_level,
                tone=tone,
                emotion=emotion,
                action_level=action_level,
                transition_hint=transition_hint,
                layout_role=layout_role,
            )
        )
        logger.info(
            "Phase1 panel=%s bubble_mode=%s area_ratio=%.3f aspect_ratio=%.3f bubble_candidates=%s model_bubbles=%s speech=%s narration=%s sfx=%s bubble_count=%s bubble_sequence=%s motion=%s tone=%s emotion=%s action=%s role=%s",
            panel.index,
            bubble_mode,
            area_ratio,
            aspect_ratio,
            bubble_candidate_count,
            len(model_result.boxes) if model_result is not None else 0,
            len(speech_boxes),
            len(narration_boxes),
            len(sfx_boxes),
            bubble_count,
            bubble_sequence or ["none"],
            motion_level,
            tone,
            emotion,
            action_level,
            layout_role,
        )
    return features


def build_rule_based_scene_script(dialogue: list[dict[str, str]], captions: list[dict[str, str]]) -> dict[str, object]:
    cleaned_dialogue = [
        str(item.get("text", "")).strip()
        for item in dialogue
        if str(item.get("text", "")).strip() and str(item.get("text", "")).strip() != "[no dialogue detected]"
    ]
    caption_texts = [str(item.get("caption", "")).strip() for item in captions if str(item.get("caption", "")).strip()]
    motion_levels = [str(item.get("motion_level", "")).strip() for item in captions]
    shot_types = [str(item.get("shot_type", "")).strip() for item in captions]
    tones = [str(item.get("tone", "")).strip() for item in captions]
    emotions = [str(item.get("emotion", "")).strip() for item in captions]
    action_levels = [str(item.get("action_level", "")).strip() for item in captions]
    transitions = [str(item.get("transition_hint", "")).strip() for item in captions]
    layout_roles = [str(item.get("layout_role", "")).strip() for item in captions]
    bubble_sequences = [str(item.get("bubble_sequence", "")).strip() for item in captions]

    scene_description = _build_scene_description(caption_texts, cleaned_dialogue, tones)
    camera_motion = _build_camera_motion(motion_levels, shot_types)
    animation_action = _build_animation_action(motion_levels, tones)
    dialogue_text = " ".join(cleaned_dialogue).strip()
    duration = _estimate_duration(cleaned_dialogue, caption_texts)
    dominant_emotion = _dominant_value(emotions, default="focused")
    action_level = _dominant_value(action_levels, default="measured")
    transition_type = _dominant_value(transitions, default="continuous right-to-left flow")
    panel_flow = _build_panel_flow(layout_roles, transitions)
    reading_graph = _build_reading_graph(layout_roles, bubble_sequences, transitions)
    reading_order = "right-to-left, top-to-bottom"

    return {
        "scene_description": scene_description,
        "camera_motion": camera_motion,
        "animation_action": animation_action,
        "dialogue": dialogue_text,
        "duration": duration,
        "emotion": dominant_emotion,
        "action_level": action_level,
        "transition_type": transition_type,
        "panel_flow": panel_flow,
        "reading_graph": reading_graph,
        "reading_order": reading_order,
    }


def _classify_shot_type(area_ratio: float, aspect_ratio: float) -> str:
    if area_ratio >= 0.24 or aspect_ratio >= 1.55:
        return "wide establishing shot"
    if aspect_ratio <= 0.72:
        return "tight portrait shot"
    if area_ratio <= 0.10:
        return "reaction detail shot"
    return "medium character shot"


def _classify_motion(edge_density: float, ink_density: float) -> str:
    if edge_density >= 0.12 or ink_density >= 0.42:
        return "high"
    if edge_density >= 0.06 or ink_density >= 0.28:
        return "moderate"
    return "low"


def _classify_tone(ink_density: float, edge_density: float) -> str:
    if ink_density >= 0.45:
        return "heavy dramatic"
    if edge_density >= 0.12:
        return "tense energetic"
    if ink_density <= 0.16:
        return "open airy"
    return "balanced narrative"


def _classify_emotion(ink_density: float, edge_density: float, bubble_count: int) -> str:
    if edge_density >= 0.13 and bubble_count <= 1:
        return "impact"
    if bubble_count >= 3:
        return "conversation"
    if ink_density >= 0.42:
        return "tension"
    if ink_density <= 0.16:
        return "calm"
    return "focused"


def _classify_action_level(edge_density: float, aspect_ratio: float, bubble_count: int) -> str:
    if edge_density >= 0.13:
        return "high"
    if bubble_count >= 3 and aspect_ratio < 1.2:
        return "dialogue-heavy"
    if aspect_ratio >= 1.5:
        return "broad"
    return "measured"


def _classify_transition(index: int, total_panels: int, aspect_ratio: float) -> str:
    if index == 0:
        return "opening beat"
    if index == total_panels - 1:
        return "closing beat"
    if aspect_ratio >= 1.4:
        return "pace-setting transition"
    return "continuous right-to-left flow"


def _classify_layout_role(index: int, total_panels: int, area_ratio: float) -> str:
    if index == 0 and area_ratio >= 0.18:
        return "establishing panel"
    if index == total_panels - 1:
        return "payoff panel"
    if area_ratio <= 0.10:
        return "reaction panel"
    return "story panel"


def _build_scene_description(caption_texts: list[str], cleaned_dialogue: list[str], tones: list[str]) -> str:
    if caption_texts:
        lead = caption_texts[:2]
        tone_hint = tones[0] if tones else "balanced narrative"
        return f"{' '.join(lead)} The page reads with a {tone_hint} manga rhythm."
    if cleaned_dialogue:
        return f"Dialogue-driven manga page built around: {' '.join(cleaned_dialogue[:2])}"
    return "Structured manga scene with sequential panel flow and visual emphasis."


def _build_camera_motion(motion_levels: list[str], shot_types: list[str]) -> str:
    if motion_levels.count("high") >= 2:
        return "Right-to-left tracking move with short punch-in accents between panels"
    if any("wide" in shot for shot in shot_types):
        return "Slow push-in from the opening panel followed by measured lateral drift"
    return "Gentle panel-to-panel push with restrained parallax"


def _build_animation_action(motion_levels: list[str], tones: list[str]) -> str:
    if motion_levels.count("high") >= 2:
        return "Emphasize impact lines, layered parallax, and quick reaction holds"
    if any("dramatic" in tone for tone in tones):
        return "Favor shadow drift, subtle camera pressure, and held character expressions"
    return "Use light parallax, expression holds, and calm panel transitions"


def _build_panel_flow(layout_roles: list[str], transitions: list[str]) -> str:
    steps: list[str] = []
    for index, role in enumerate(layout_roles, start=1):
        transition = transitions[index - 1] if index - 1 < len(transitions) else "continuous right-to-left flow"
        steps.append(f"P{index}: {role} -> {transition}")
    return " | ".join(steps) if steps else "Single-panel beat"


def _dominant_value(values: list[str], default: str) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda item: item[1])[0]


def _estimate_duration(cleaned_dialogue: list[str], caption_texts: list[str]) -> int:
    word_count = sum(len(line.split()) for line in cleaned_dialogue)
    beat_count = max(len(caption_texts), 1)
    return max(4, min(10, 4 + beat_count + (word_count // 8)))


def _detect_bubble_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes, _ = _detect_bubble_boxes_with_debug(gray)
    return boxes


def _detect_bubble_boxes_with_debug(gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    area = gray.shape[0] * gray.shape[1]
    boxes: list[tuple[int, int, int, int]] = []
    candidate_count = 0
    for threshold in (180, 190, 200):
        bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=2,
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        for label in range(1, component_count):
            x, y, w, h, component_area = stats[label]
            candidate_count += 1
            if component_area < area * 0.005 or component_area > area * 0.22:
                continue
            if x <= 1 or y <= 1 or x + w >= gray.shape[1] - 1 or y + h >= gray.shape[0] - 1:
                continue
            aspect = w / max(h, 1)
            if not (0.35 <= aspect <= 3.5):
                continue
            fill_ratio = component_area / max(w * h, 1)
            if fill_ratio < 0.35:
                continue
            mean_brightness = float(np.mean(gray[y : y + h, x : x + w]))
            if mean_brightness < 130:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(_box_iou(box, kept) < 0.35 for kept in deduped):
            deduped.append(box)
    deduped.sort(key=lambda box: (box[1] // max(gray.shape[0] // 6, 1), -(box[0] + box[2])))
    return deduped[:6], candidate_count


def _estimate_bubble_count(gray: np.ndarray) -> int:
    return len(_detect_bubble_boxes(gray))


def _describe_bubble_sequence(
    bubble_boxes: list[tuple[int, int, int, int]],
    panel_width: int,
    panel_height: int,
) -> list[str]:
    sequence: list[str] = []
    for x, y, w, h in bubble_boxes[:4]:
        x_center = x + (w / 2)
        y_center = y + (h / 2)
        horizontal = "right" if x_center >= panel_width * 0.6 else "left" if x_center <= panel_width * 0.4 else "center"
        vertical = "top" if y_center <= panel_height * 0.35 else "bottom" if y_center >= panel_height * 0.68 else "middle"
        sequence.append(f"{vertical}-{horizontal}")
    return sequence


def _promote_bubbles_from_text_regions(
    bubble_boxes: list[tuple[int, int, int, int]],
    gray: np.ndarray,
    panel_width: int,
    panel_height: int,
    bubble_candidate_count: int,
) -> list[tuple[int, int, int, int]]:
    accepted = list(bubble_boxes)
    text_regions = _detect_text_like_regions(gray)
    if not text_regions:
        return accepted

    max_promotions = 2
    if bubble_candidate_count >= 10:
        max_promotions = 3
    if bubble_candidate_count >= 25 or panel_height > panel_width * 1.4:
        max_promotions = 4

    promoted = 0
    for x, y, w, h in text_regions[:10]:
        if promoted >= max_promotions:
            break
        expanded = _expand_box(x, y, w, h, panel_width, panel_height, x_pad=18, y_pad=20)
        if any(_box_iou(expanded, existing) >= 0.30 for existing in accepted):
            continue
        brightness = _region_brightness(gray, expanded)
        min_brightness = 150
        if bubble_candidate_count >= 25:
            min_brightness = 138
        elif bubble_candidate_count >= 10:
            min_brightness = 144
        if brightness < min_brightness:
            continue
        accepted.append(expanded)
        promoted += 1

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(accepted, key=lambda item: item[2] * item[3], reverse=True):
        if all(_box_iou(box, kept) < 0.35 for kept in deduped):
            deduped.append(box)
    deduped.sort(key=lambda box: (box[1] // max(panel_height // 6, 1), -(box[0] + box[2])))
    return deduped[:6]


def _classify_text_regions(
    gray: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    panel_width: int,
    panel_height: int,
    speech_seed_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[dict[str, object]]:
    classified: list[dict[str, object]] = []
    panel_area = max(panel_width * panel_height, 1)
    edges = cv2.Canny(gray, 80, 160)
    speech_seed_boxes = speech_seed_boxes or []
    for box in regions:
        x, y, w, h = box
        region = gray[y : y + h, x : x + w]
        edge_region = edges[y : y + h, x : x + w]
        if region.size == 0:
            continue
        brightness = float(np.mean(region))
        dark_ratio = float(np.mean(region < 120))
        edge_density = float(np.mean(edge_region > 0))
        area_ratio = (w * h) / panel_area
        aspect_ratio = w / max(h, 1)
        border_dark = _border_dark_ratio(region)

        if any(_box_iou(box, seed) >= 0.20 for seed in speech_seed_boxes):
            kind = "speech"
        elif dark_ratio >= 0.35 and border_dark >= 0.22:
            kind = "narration"
        elif brightness >= 165 and border_dark >= 0.03 and 0.45 <= aspect_ratio <= 2.8:
            kind = "speech"
        elif area_ratio >= 0.10 or edge_density >= 0.16 or aspect_ratio >= 3.2:
            kind = "sfx"
        elif brightness >= 150 and dark_ratio <= 0.22:
            kind = "speech"
        elif dark_ratio >= 0.25:
            kind = "narration"
        else:
            kind = "sfx"

        classified.append({"box": box, "kind": kind})

    classified.sort(key=lambda item: _region_sort_key(item["box"], panel_height))
    return classified


def _detect_text_like_regions(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        iterations=2,
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = gray.shape[0] * gray.shape[1]
    regions: list[tuple[int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, w, h, component_area = stats[label]
        if component_area < area * 0.002 or component_area > area * 0.10:
            continue
        if w < 24 or h < 18:
            continue
        aspect = w / max(h, 1)
        if not (0.45 <= aspect <= 5.5):
            continue
        regions.append((int(x), int(y), int(w), int(h)))
    regions.sort(key=lambda box: (box[1] // max(gray.shape[0] // 6, 1), -(box[0] + box[2])))
    return regions


def _expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    panel_width: int,
    panel_height: int,
    x_pad: int,
    y_pad: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, x - x_pad)
    y1 = max(0, y - y_pad)
    x2 = min(panel_width, x + w + x_pad)
    y2 = min(panel_height, y + h + y_pad)
    return (x1, y1, x2 - x1, y2 - y1)


def _region_brightness(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    region = gray[y : y + h, x : x + w]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))


def _border_dark_ratio(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    band = max(1, min(region.shape[0], region.shape[1]) // 12)
    top = region[:band, :]
    bottom = region[-band:, :]
    left = region[:, :band]
    right = region[:, -band:]
    border = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
    return float(np.mean(border < 120))


def _region_sort_key(box: tuple[int, int, int, int], panel_height: int) -> tuple[int, int]:
    return (box[1] // max(panel_height // 6, 1), -(box[0] + box[2]))


def _build_reading_graph(layout_roles: list[str], bubble_sequences: list[str], transitions: list[str]) -> str:
    nodes: list[str] = []
    for index, role in enumerate(layout_roles, start=1):
        bubbles = bubble_sequences[index - 1] if index - 1 < len(bubble_sequences) else "none"
        transition = transitions[index - 1] if index - 1 < len(transitions) else "continuous right-to-left flow"
        nodes.append(f"P{index}[{role}; bubbles={bubbles}] -> {transition}")
    return " || ".join(nodes) if nodes else "No reading graph available"


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_detection_boxes(
    heuristic_boxes: list[tuple[int, int, int, int]],
    model_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged = list(heuristic_boxes)
    for model_box in model_boxes:
        if all(_box_iou(model_box, existing) < 0.30 for existing in merged):
            merged.append(model_box)
    merged.sort(key=lambda box: (box[1], box[0]))
    return merged
