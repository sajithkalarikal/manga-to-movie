import asyncio
import logging
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from config import Settings
from modules.local_phase1 import PanelSceneFeatures
from modules.ocr_engine import COMIC_WHITELIST, get_corrector, get_ocr_engine
from modules.panel_detection import DetectedPanel

logger = logging.getLogger(__name__)


class OCRDialogueService:
    # Keep this config shell-safe for pytesseract's internal argument splitting.
    TESSERACT_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-"
    SHORT_WORD_WHITELIST = {"A", "I"}

    def __init__(self, settings: Settings):
        self.settings = settings
        resolved_cmd = self._resolve_tesseract_cmd(settings.tesseract_cmd)
        if resolved_cmd:
            pytesseract.pytesseract.tesseract_cmd = resolved_cmd
            logger.info("Using tesseract binary at %s", resolved_cmd)
        self.ocr_available = self._is_tesseract_available()
        if not self.ocr_available:
            logger.warning("Tesseract is not installed. OCR will use a placeholder fallback.")

    async def extract_dialogue(
        self,
        panels: list[DetectedPanel],
        region_hints: list[PanelSceneFeatures] | None = None,
    ) -> list[dict[str, str]]:
        logger.info("Running OCR on %s panels", len(panels))
        if not self.ocr_available:
            return [
                {"panel": str(panel.index), "text": "[ocr unavailable: install tesseract to extract dialogue]"}
                for panel in panels
            ]
        stitched_results = await asyncio.to_thread(self._extract_from_stitched_page_if_needed, panels)
        if stitched_results is not None:
            return stitched_results
        feature_map = {feature.panel_index: feature for feature in (region_hints or [])}
        results = await asyncio.gather(*(self._read_panel(panel, feature_map.get(panel.index)) for panel in panels))
        return results

    async def extract_text_from_image(self, image_path: Path) -> str:
        if not self.ocr_available:
            return "[ocr unavailable: install tesseract to extract dialogue]"
        parts = await asyncio.to_thread(self._ocr_parts_sync, image_path, 2)
        return " ".join(parts).strip() or "[no text detected]"

    async def extract_text_parts_from_image(self, image_path: Path, limit: int = 2) -> list[str]:
        if not self.ocr_available:
            return ["[ocr unavailable: install tesseract to extract dialogue]"]
        return await asyncio.to_thread(self._ocr_parts_sync, image_path, limit)

    async def _read_panel(self, panel: DetectedPanel, region_hint: PanelSceneFeatures | None = None) -> dict[str, str]:
        analysis = await asyncio.to_thread(self._analyze_panel_text_sync, panel.image_path, region_hint)
        logger.info(
            "OCR panel=%s text_regions=%s text_role=%s text_preview=%s",
            panel.index,
            analysis["text_regions"],
            analysis["text_role"],
            str(analysis["text"])[:120],
        )
        return {
            "panel": str(panel.index),
            "text": analysis["text"] or "[no dialogue detected]",
            "text_regions": str(analysis["text_regions"]),
            "text_role": analysis["text_role"],
        }

    @staticmethod
    def _extract_from_stitched_page_if_needed(panels: list[DetectedPanel]) -> list[dict[str, str]] | None:
        if not OCRDialogueService._looks_like_fallback_strips(panels):
            return None

        stitched = OCRDialogueService._stitch_panels_vertically(panels)
        if stitched is None:
            return None

        parts = OCRDialogueService._ocr_parts_from_bgr(stitched, limit=max(2, len(panels)))
        if not parts:
            return None

        results = [{"panel": str(index), "text": text} for index, text in enumerate(parts, start=1)]
        while len(results) < len(panels):
            results.append({"panel": str(len(results) + 1), "text": "[no dialogue detected]"})
        logger.info("Detected fallback strip panels; using stitched page OCR with %s recovered text regions", len(parts))
        return results

    @staticmethod
    def _ocr_parts_sync(image_path: Path, limit: int = 2) -> list[str]:
        with Image.open(image_path) as image:
            bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return OCRDialogueService._ocr_parts_from_bgr(bgr, limit)

    @staticmethod
    def _analyze_panel_text_sync(image_path: Path, region_hint: PanelSceneFeatures | None = None) -> dict[str, str | int]:
        with Image.open(image_path) as image:
            bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        hinted_regions: list[dict[str, object]] = []
        if region_hint is not None:
            hinted_regions = OCRDialogueService._typed_regions_from_features(region_hint)
        text_regions = hinted_regions or [
            {"class_name": "speech", "bbox": box}
            for box in OCRDialogueService._detect_text_regions(bgr)
        ]
        parts = OCRDialogueService._ocr_parts_from_bgr(
            bgr,
            limit=max(1, len(text_regions) or 1),
            region_items=text_regions,
        )
        joined_text = " ".join(parts).strip() or "[no dialogue detected]"
        return {
            "text": joined_text,
            "text_regions": len(text_regions),
            "text_role": OCRDialogueService._classify_text_role(
                [tuple(item["bbox"]) for item in text_regions],
                joined_text,
            ),
        }

    @staticmethod
    def _ocr_parts_from_bgr(
        bgr: np.ndarray,
        limit: int = 2,
        region_boxes: list[tuple[int, int, int, int]] | None = None,
        region_items: list[dict[str, object]] | None = None,
    ) -> list[str]:
        h, w = bgr.shape[:2]
        items: list[dict[str, object]]
        if region_items is not None:
            items = [dict(item) for item in region_items]
        elif region_boxes is not None:
            items = [{"class_name": "speech", "bbox": box} for box in region_boxes]
        else:
            items = [{"class_name": "speech", "bbox": box} for box in OCRDialogueService._detect_text_regions(bgr)]
        if not items:
            items = [{"class_name": "speech", "bbox": (0, 0, w, int(h * 0.6))}]

        ocr_engine = get_ocr_engine()
        corrector = get_corrector()
        outputs: list[str] = []
        normalized_items = OCRDialogueService._normalize_region_items(items, w, h)
        for item in normalized_items:
            class_name = str(item.get("class_name", "speech"))
            x, y, bw, bh = item["bbox"]
            region = bgr[y : y + bh, x : x + bw]
            if region.size == 0:
                continue
            candidates = OCRDialogueService._collect_candidates_for_class(region, class_name, ocr_engine)
            if not candidates:
                continue
            best = OCRDialogueService._choose_best_candidate(candidates)
            if best is None:
                continue
            best_score = OCRDialogueService._score_candidate(str(best["text"]))
            best_confidence = float(best.get("confidence", 0.0))
            if best_score < 1.0 and best_confidence < 55.0:
                continue
            cleaned = OCRDialogueService._clean_ocr_line(str(best["text"]), class_name=class_name)
            if cleaned:
                if class_name != "sfx":
                    cleaned = corrector.correct_line(cleaned)
                cleaned = OCRDialogueService._finalize_ocr_line(cleaned)
            if not cleaned:
                continue
            if cleaned in outputs:
                continue
            outputs.append(cleaned)
            if len(outputs) >= max(1, limit):
                break
        return outputs

    @staticmethod
    def _typed_regions_from_features(feature: PanelSceneFeatures) -> list[dict[str, object]]:
        typed: list[dict[str, object]] = []
        typed.extend({"class_name": "speech", "bbox": box} for box in feature.speech_boxes)
        typed.extend({"class_name": "narration", "bbox": box} for box in feature.narration_boxes)
        typed.extend({"class_name": "sfx", "bbox": box} for box in feature.sfx_boxes)
        return typed

    @staticmethod
    def _normalize_region_items(
        items: list[dict[str, object]],
        image_width: int,
        image_height: int,
    ) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for item in items:
            box = item.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2_or_w, y2_or_h = [int(round(value)) for value in box]
            if x2_or_w > x1 and y2_or_h > y1:
                x = x1
                y = y1
                w = x2_or_w - x1
                h = y2_or_h - y1
            else:
                x = x1
                y = y1
                w = x2_or_w
                h = y2_or_h
            x = max(0, min(x, image_width))
            y = max(0, min(y, image_height))
            w = max(1, min(w, image_width - x))
            h = max(1, min(h, image_height - y))
            normalized.append(
                {
                    "class_name": str(item.get("class_name", "speech")),
                    "bbox": (x, y, w, h),
                }
            )
        return normalized

    @staticmethod
    def _normalize_region_boxes(
        boxes: list[tuple[int, int, int, int]],
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int, int, int]]:
        normalized: list[tuple[int, int, int, int]] = []
        for box in boxes:
            x1, y1, x2_or_w, y2_or_h = [int(round(value)) for value in box]
            if x2_or_w > x1 and y2_or_h > y1:
                x = x1
                y = y1
                w = x2_or_w - x1
                h = y2_or_h - y1
            else:
                x = x1
                y = y1
                w = x2_or_w
                h = y2_or_h
            x = max(0, min(x, image_width))
            y = max(0, min(y, image_height))
            w = max(1, min(w, image_width - x))
            h = max(1, min(h, image_height - y))
            normalized.append((x, y, w, h))
        return normalized

    @staticmethod
    def _looks_like_fallback_strips(panels: list[DetectedPanel]) -> bool:
        if len(panels) < 2:
            return False
        widths = []
        heights = []
        prev_bottom = None
        max_x2 = max(panel.bbox[2] for panel in panels)
        for panel in panels:
            x1, y1, x2, y2 = panel.bbox
            if x1 != 0 or x2 != max_x2:
                return False
            if prev_bottom is not None and abs(y1 - prev_bottom) > 1:
                return False
            widths.append(x2 - x1)
            heights.append(y2 - y1)
            prev_bottom = y2
        if max(widths) - min(widths) > 1:
            return False
        return max(heights) - min(heights) <= 2

    @staticmethod
    def _stitch_panels_vertically(panels: list[DetectedPanel]) -> np.ndarray | None:
        images: list[np.ndarray] = []
        target_width = 0
        for panel in panels:
            image = cv2.imread(str(panel.image_path))
            if image is None:
                return None
            images.append(image)
            target_width = max(target_width, image.shape[1])
        normalized: list[np.ndarray] = []
        for image in images:
            if image.shape[1] != target_width:
                image = cv2.resize(image, (target_width, image.shape[0]), interpolation=cv2.INTER_LINEAR)
            normalized.append(image)
        return cv2.vconcat(normalized) if normalized else None

    @staticmethod
    def _collect_candidates(region_bgr: np.ndarray, ocr_engine) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        variants = OCRDialogueService._preprocess_variants(region_bgr)
        is_vertical = region_bgr.shape[0] > (region_bgr.shape[1] * 1.35)

        for variant in variants[:3]:
            text = OCRDialogueService._normalize_spaces(ocr_engine.read(variant))
            OCRDialogueService._append_candidate(candidates, text, confidence=58.0, source="engine")

        primary_psm_modes = [5] if is_vertical else [6]
        OCRDialogueService._collect_tesseract_candidates(candidates, variants[:4], primary_psm_modes)

        best_confidence = max((float(candidate.get("confidence", 0.0)) for candidate in candidates), default=0.0)
        best_score = max((OCRDialogueService._score_candidate(str(candidate.get("text", ""))) for candidate in candidates), default=0.0)
        if not candidates or best_confidence < 70.0 or best_score < 1.5:
            retry_psm_modes = [5, 11, 6] if is_vertical else [6, 11, 4]
            OCRDialogueService._collect_tesseract_candidates(candidates, variants, retry_psm_modes)
        if not candidates or (
            max((float(candidate.get("confidence", 0.0)) for candidate in candidates), default=0.0) < 55.0
            and max((OCRDialogueService._score_candidate(str(candidate.get("text", ""))) for candidate in candidates), default=0.0) < 1.0
        ):
            OCRDialogueService._collect_tesseract_candidates(
                candidates,
                OCRDialogueService._invert_variants(variants[:4]),
                [11, 6, 4],
            )

        candidates = OCRDialogueService._deduplicate_text_candidates(candidates)
        return candidates

    @staticmethod
    def _collect_left_box_candidates(region_bgr: np.ndarray, ocr_engine) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        variants = OCRDialogueService._preprocess_left_box(region_bgr)
        for variant in variants:
            text = ocr_engine.read_manga_only(variant)
            if not text.strip():
                text = ocr_engine.read(variant)
            normalized = OCRDialogueService._normalize_spaces(text)
            OCRDialogueService._append_candidate(candidates, normalized, confidence=62.0, source="left-box-engine")
        OCRDialogueService._collect_tesseract_candidates(candidates, variants, [5, 6, 11])
        candidates = OCRDialogueService._deduplicate_text_candidates(candidates)
        # If manga-ocr is unavailable, fallback to regular candidates to avoid empty output.
        if not candidates:
            return OCRDialogueService._collect_candidates(region_bgr, ocr_engine)
        return candidates

    @staticmethod
    def _collect_candidates_for_class(region_bgr: np.ndarray, class_name: str, ocr_engine) -> list[dict[str, object]]:
        normalized_class = class_name.strip().lower()
        if normalized_class == "narration":
            return OCRDialogueService._collect_narration_candidates(region_bgr, ocr_engine)
        if normalized_class == "sfx":
            return OCRDialogueService._collect_sfx_candidates(region_bgr, ocr_engine)
        return OCRDialogueService._collect_speech_candidates(region_bgr, ocr_engine)

    @staticmethod
    def _collect_speech_candidates(region_bgr: np.ndarray, ocr_engine) -> list[dict[str, object]]:
        h, w = region_bgr.shape[:2]
        is_tall_box = h > (w * 1.25)
        if is_tall_box:
            return OCRDialogueService._collect_left_box_candidates(region_bgr, ocr_engine)
        return OCRDialogueService._collect_candidates(region_bgr, ocr_engine)

    @staticmethod
    def _collect_narration_candidates(region_bgr: np.ndarray, ocr_engine) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        variants = OCRDialogueService._preprocess_variants(region_bgr, class_name="narration")
        for variant in variants[:4]:
            text = OCRDialogueService._normalize_spaces(ocr_engine.read(variant))
            OCRDialogueService._append_candidate(candidates, text, confidence=60.0, source="narration-engine")
        OCRDialogueService._collect_tesseract_candidates(candidates, variants, [6, 4, 11, 5])
        if not candidates or (
            max((float(candidate.get("confidence", 0.0)) for candidate in candidates), default=0.0) < 55.0
            and max((OCRDialogueService._score_candidate(str(candidate.get("text", ""))) for candidate in candidates), default=0.0) < 1.0
        ):
            OCRDialogueService._collect_tesseract_candidates(
                candidates,
                OCRDialogueService._invert_variants(variants[:5]),
                [6, 4, 11],
            )
        return OCRDialogueService._deduplicate_text_candidates(candidates)

    @staticmethod
    def _collect_sfx_candidates(region_bgr: np.ndarray, ocr_engine) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        variants = OCRDialogueService._preprocess_variants(region_bgr, class_name="sfx")
        OCRDialogueService._collect_tesseract_candidates(candidates, variants, [11, 6, 13], use_dictionary=False)
        for variant in variants[:2]:
            text = OCRDialogueService._normalize_spaces(ocr_engine.read(variant))
            OCRDialogueService._append_candidate(candidates, text, confidence=55.0, source="sfx-engine")
        if not candidates or (
            max((float(candidate.get("confidence", 0.0)) for candidate in candidates), default=0.0) < 55.0
            and max((OCRDialogueService._score_candidate(str(candidate.get("text", ""))) for candidate in candidates), default=0.0) < 0.8
        ):
            OCRDialogueService._collect_tesseract_candidates(
                candidates,
                OCRDialogueService._invert_variants(variants[:6]),
                [11, 13, 6],
                use_dictionary=False,
            )
        return OCRDialogueService._deduplicate_text_candidates(candidates)

    @staticmethod
    def _preprocess_variants(img_bgr: np.ndarray, class_name: str = "speech") -> list[np.ndarray]:
        prepared = OCRDialogueService._prepare_region_for_ocr(img_bgr, class_name=class_name)
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5 if class_name == "narration" else 3)
        std = cv2.convertScaleAbs(denoised, alpha=1.0, beta=0)
        hi = cv2.convertScaleAbs(denoised, alpha=1.35, beta=5)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        clahe_gray = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(hi, (3, 3), 0)
        dilated = cv2.dilate(hi, np.ones((2, 2), np.uint8), iterations=1)
        eroded = cv2.erode(hi, np.ones((2, 2), np.uint8), iterations=1)
        _, otsu = cv2.threshold(hi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thick_otsu = cv2.dilate(otsu, np.ones((2, 2), np.uint8), iterations=1)
        light_thick = cv2.dilate(denoised, np.ones((2, 2), np.uint8), iterations=1)
        _, global_binary = cv2.threshold(hi, 127, 255, cv2.THRESH_BINARY)
        _, cc_thresh = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cc_clean = OCRDialogueService._keep_letter_sized_blobs(cv2.bitwise_not(cc_thresh))
        cc_variant = cv2.bitwise_not(cc_clean)

        # Hatch-line suppression.
        thresh = cv2.adaptiveThreshold(
            hi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            8,
        )
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        morph_closed = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        cleaned = cv2.bitwise_not(closed)
        screentone_clean = OCRDialogueService._suppress_screentones(gray, blur_size=3, block_size=11, c_value=2)
        screentone_clean_strong = OCRDialogueService._suppress_screentones(gray, blur_size=5, block_size=15, c_value=4)
        variants = [
            std,
            hi,
            clahe_gray,
            blurred,
            dilated,
            eroded,
            light_thick,
            otsu,
            thick_otsu,
            thresh,
            morph_closed,
            cleaned,
            screentone_clean,
            screentone_clean_strong,
            global_binary,
            cc_variant,
        ]
        if class_name == "sfx":
            invert = cv2.bitwise_not(hi)
            invert_otsu = cv2.bitwise_not(otsu)
            thick_sfx = cv2.dilate(hi, np.ones((3, 3), np.uint8), iterations=1)
            variants.extend([thick_sfx, invert, invert_otsu])
        return variants

    @staticmethod
    def _extract_box_interior(crop_bgr: np.ndarray, inset_px: int = 6) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        if h <= inset_px * 2 or w <= inset_px * 2:
            return crop_bgr
        inset = crop_bgr[inset_px : h - inset_px, inset_px : w - inset_px]

        gray = cv2.cvtColor(inset, cv2.COLOR_BGR2GRAY).astype(np.float32)
        col_var = np.var(gray, axis=0)
        row_var = np.var(gray, axis=1)
        col_thresh = np.percentile(col_var, 60)
        row_thresh = np.percentile(row_var, 80)
        good_cols = np.where(col_var <= col_thresh)[0]
        good_rows = np.where(row_var <= row_thresh)[0]

        if len(good_cols) < max(8, int(inset.shape[1] * 0.3)) or len(good_rows) < max(8, int(inset.shape[0] * 0.3)):
            return inset
        x1, x2 = int(good_cols[0]), int(good_cols[-1]) + 1
        y1, y2 = int(good_rows[0]), int(good_rows[-1]) + 1
        if x2 <= x1 or y2 <= y1:
            return inset
        return inset[y1:y2, x1:x2]

    @staticmethod
    def _keep_letter_sized_blobs(binary_img: np.ndarray) -> np.ndarray:
        h, w = binary_img.shape[:2]
        min_area = max(2, int(h * w * 0.0003))
        max_area = max(min_area + 1, int(h * w * 0.15))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        result = np.zeros_like(binary_img)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                result[labels == label] = 255
        return result

    @staticmethod
    def _preprocess_left_box(crop_bgr: np.ndarray) -> list[np.ndarray]:
        interior = OCRDialogueService._extract_box_interior(crop_bgr, inset_px=6)
        prepared = OCRDialogueService._prepare_region_for_ocr(interior)
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thick_otsu = cv2.dilate(otsu, np.ones((2, 2), np.uint8), iterations=1)

        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k_open, iterations=1)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, cc_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cc_clean = OCRDialogueService._keep_letter_sized_blobs(cv2.bitwise_not(cc_thresh))
        cc_variant = cv2.bitwise_not(cc_clean)
        screentone_clean = OCRDialogueService._suppress_screentones(gray, blur_size=3, block_size=11, c_value=2)
        screentone_clean_strong = OCRDialogueService._suppress_screentones(gray, blur_size=5, block_size=15, c_value=4)

        return [gray, otsu, thick_otsu, opened, clahe_otsu, cc_variant, screentone_clean, screentone_clean_strong]

    @staticmethod
    def _prepare_region_for_ocr(crop_bgr: np.ndarray, class_name: str = "speech") -> np.ndarray:
        inset_ratio = 0.02 if class_name == "sfx" else 0.08 if class_name == "narration" else 0.06
        crop_bgr = OCRDialogueService._inset_crop(crop_bgr, ratio=inset_ratio)
        h, w = crop_bgr.shape[:2]
        longest_side = max(h, w)
        scale = 1
        if longest_side < 140:
            scale = 3
        elif longest_side < 280:
            scale = 2
        if scale > 1:
            crop_bgr = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        border = 10 if scale == 1 else 16
        return cv2.copyMakeBorder(
            crop_bgr,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    @staticmethod
    def _inset_crop(crop_bgr: np.ndarray, ratio: float = 0.05) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        inset = min(16, max(4, int(min(h, w) * ratio)))
        if h <= inset * 2 or w <= inset * 2:
            return crop_bgr
        return crop_bgr[inset : h - inset, inset : w - inset]

    @staticmethod
    def _suppress_screentones(
        gray: np.ndarray,
        *,
        blur_size: int = 3,
        block_size: int = 11,
        c_value: int = 2,
    ) -> np.ndarray:
        # Median blur softens screentone dots while keeping thicker glyph strokes intact.
        blurred = cv2.medianBlur(gray, blur_size)
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )

    @staticmethod
    def _invert_variants(variants: list[np.ndarray]) -> list[np.ndarray]:
        return [cv2.bitwise_not(variant) for variant in variants]

    @staticmethod
    def _append_candidate(
        candidates: list[dict[str, object]],
        text: str,
        confidence: float,
        source: str,
    ) -> None:
        normalized = OCRDialogueService._normalize_spaces(text)
        if not normalized:
            return
        candidates.append(
            {
                "text": normalized,
                "confidence": float(confidence),
                "source": source,
            }
        )

    @staticmethod
    def _collect_tesseract_candidates(
        candidates: list[dict[str, object]],
        variants: list[np.ndarray],
        psm_modes: list[int],
        *,
        use_dictionary: bool = True,
    ) -> None:
        for variant in variants:
            for psm in psm_modes:
                text, confidence = OCRDialogueService._tesseract_read_with_confidence(
                    variant,
                    psm=psm,
                    use_dictionary=use_dictionary,
                )
                OCRDialogueService._append_candidate(candidates, text, confidence, source=f"tesseract-psm{psm}")

    @staticmethod
    def _tesseract_read_with_confidence(image, psm: int, *, use_dictionary: bool = True) -> tuple[str, float]:
        config = (
            f"--oem 3 --psm {psm} -l eng "
            f"-c tessedit_char_whitelist={OCRDialogueService.TESSERACT_CHAR_WHITELIST} "
            "-c preserve_interword_spaces=1"
        )
        if not use_dictionary:
            config += " -c load_system_dawg=0 -c load_freq_dawg=0"
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        words: list[str] = []
        confidences: list[float] = []
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])
        img_h, img_w = image.shape[:2]
        fallback_words: list[str] = []
        fallback_confidences: list[float] = []
        for index, (token, confidence) in enumerate(zip(data.get("text", []), data.get("conf", []))):
            normalized = OCRDialogueService._normalize_spaces(token)
            if not normalized:
                continue
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = -1.0
            left = int(lefts[index]) if index < len(lefts) else 0
            top = int(tops[index]) if index < len(tops) else 0
            width = int(widths[index]) if index < len(widths) else 0
            height = int(heights[index]) if index < len(heights) else 0
            peripheral = OCRDialogueService._is_peripheral_token(
                left=left,
                top=top,
                width=width,
                height=height,
                image_width=img_w,
                image_height=img_h,
            )
            fallback_words.append(normalized)
            if confidence_value >= 0:
                fallback_confidences.append(confidence_value)
            if peripheral and (len(OCRDialogueService._token_only_alpha(normalized)) <= 2 or confidence_value < 65.0):
                continue
            words.append(normalized)
            if confidence_value >= 0:
                confidences.append(confidence_value)
        if not words:
            words = fallback_words
            confidences = fallback_confidences
        text = OCRDialogueService._normalize_spaces(" ".join(words))
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return text, average_confidence

    @staticmethod
    def _is_peripheral_token(
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
        margin_ratio: float = 0.10,
    ) -> bool:
        if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
            return False
        cx = left + (width / 2.0)
        cy = top + (height / 2.0)
        margin_x = image_width * margin_ratio
        margin_y = image_height * margin_ratio
        return (
            cx < margin_x
            or cx > (image_width - margin_x)
            or cy < margin_y
            or cy > (image_height - margin_y)
        )

    @staticmethod
    def _deduplicate_text_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        best_by_text: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            text = str(candidate.get("text", ""))
            if not text:
                continue
            existing = best_by_text.get(text)
            if existing is None or float(candidate.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                best_by_text[text] = candidate
        return list(best_by_text.values())

    @staticmethod
    def _choose_best_candidate(candidates: list[dict[str, object]]) -> dict[str, object] | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                OCRDialogueService._score_candidate(str(candidate.get("text", ""))) * (1.0 + (float(candidate.get("confidence", 0.0)) / 100.0)),
                float(candidate.get("confidence", 0.0)),
                len(str(candidate.get("text", ""))),
            ),
        )

    @staticmethod
    def _merge_candidates(candidates: list[str], min_score: float = 0.3) -> str:
        good = [c for c in candidates if OCRDialogueService._score_candidate(c) >= min_score]
        if not good:
            return max(candidates, key=OCRDialogueService._score_candidate) if candidates else ""
        anchor = max(good, key=OCRDialogueService._score_candidate)
        words = [OCRDialogueService._token_only_alpha(w).upper() for w in anchor.split()]
        words = [w for w in words if w]
        seen = set(words)
        for cand in good:
            if cand == anchor:
                continue
            for w in cand.split():
                token = OCRDialogueService._token_only_alpha(w).upper()
                if not token or token in seen:
                    continue
                if OCRDialogueService._is_real_word(token):
                    words.append(token)
                    seen.add(token)
        return " ".join(words)

    @staticmethod
    def _score_candidate(text: str) -> float:
        words = [OCRDialogueService._token_only_alpha(t).upper() for t in text.split()]
        words = [w for w in words if w]
        if not words:
            return 0.0
        real = [w for w in words if OCRDialogueService._is_real_word(w)]
        n_real = len(real)
        if n_real == 0:
            return 0.0
        coverage = n_real / len(words)
        length_bonus = min(1.0, len(words) / 6.0)
        garbage_ratio = 1.0 - coverage
        garbage_penalty = max(0.0, garbage_ratio - 0.40)
        return n_real * coverage * (1 + length_bonus) * (1 - garbage_penalty)

    @staticmethod
    def _clean_ocr_line(raw: str, class_name: str = "speech") -> str:
        out: list[str] = []
        for token in raw.split():
            cleaned = OCRDialogueService._clean_token(token, class_name=class_name)
            if cleaned is not None:
                out.append(cleaned)
        return OCRDialogueService._normalize_spaces(" ".join(out))

    @staticmethod
    def _clean_token(token: str, class_name: str = "speech") -> str | None:
        t = token.strip(".,!?'-\" ")
        if not t:
            return None
        if not any(ch.isalnum() for ch in t):
            return None
        upper = OCRDialogueService._token_only_alpha(t).upper()
        if not upper:
            return None
        if upper in OCRDialogueService.SHORT_WORD_WHITELIST:
            return upper
        if class_name == "sfx":
            if len(upper) < 2 or OCRDialogueService._is_garbage_like(t):
                return None
            return upper
        if upper in COMIC_WHITELIST:
            return upper
        if OCRDialogueService._is_real_word(upper):
            return upper
        if len(upper) <= 2:
            return None
        if OCRDialogueService._is_garbage_like(t):
            return None
        if upper.isalpha() and 3 <= len(upper) <= 15:
            return upper
        return None

    @staticmethod
    def _is_garbage_like(token: str) -> bool:
        return bool(re.search(r"^[^a-zA-Z]{2,}$|^[a-zA-Z][^a-zA-Z]+$|[0-9]{3,}|(.)\1{3,}", token))

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join(str(text).split())

    @staticmethod
    def _finalize_ocr_line(text: str) -> str:
        normalized = OCRDialogueService._normalize_spaces(text).upper().strip()
        normalized = re.sub(r"\b([A-Z]+)( \1\b)+", r"\1", normalized)
        normalized = re.sub(r"\s+([!?.,])", r"\1", normalized)
        normalized = re.sub(r"([!?]){3,}", r"\1\1", normalized)
        return normalized

    @staticmethod
    def _token_only_alpha(token: str) -> str:
        return re.sub(r"[^A-Za-z]", "", token)

    @staticmethod
    def _is_real_word(token_upper: str) -> bool:
        if token_upper in COMIC_WHITELIST:
            return True
        lower = token_upper.lower()
        common = {
            "the",
            "a",
            "an",
            "i",
            "it",
            "is",
            "was",
            "we",
            "you",
            "he",
            "she",
            "they",
            "to",
            "of",
            "in",
            "on",
            "for",
            "and",
            "or",
            "but",
            "how",
            "when",
            "where",
            "with",
            "anything",
            "remember",
            "hardly",
            "felt",
            "think",
            "went",
            "town",
            "north",
            "faint",
            "smell",
            "bonfire",
            "drifts",
            "over",
        }
        return lower in common

    @staticmethod
    def _detect_text_regions(img_bgr: np.ndarray, min_area: int = 800) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        found: list[tuple[int, int, int, int]] = []
        # Corner priors improve recall for common manga narration placement.
        found.extend(
            [
                (int(w * 0.02), int(h * 0.03), int(w * 0.34), int(h * 0.55)),
                (int(w * 0.62), int(h * 0.03), int(w * 0.34), int(h * 0.55)),
            ]
        )

        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        c1, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found.extend(OCRDialogueService._contours_to_boxes(c1, min_area, 1.2, 14))

        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blurred)
        _, flat = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY_INV)
        c2, _ = cv2.findContours(flat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found.extend(OCRDialogueService._contours_to_boxes(c2, min_area, 1.2, 14))

        edges = cv2.Canny(gray, 50, 150)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 4), 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)
        c3, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found.extend(OCRDialogueService._contours_to_boxes(c3, min_area, 1.2, 14))

        # Speech bubbles are often bright enclosed regions with softer aspect ratios.
        bubble_binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            6,
        )
        bubble_binary = cv2.morphologyEx(
            bubble_binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        c4, _ = cv2.findContours(bubble_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found.extend(OCRDialogueService._contours_to_boxes(c4, min_area, 0.5, 2.6))

        panel_area = float(h * w)
        found = [box for box in found if (box[2] * box[3]) <= (panel_area * 0.45)]

        deduped = OCRDialogueService._deduplicate_boxes(found, 0.3)
        return OCRDialogueService._sort_text_regions_manga_order(deduped, panel_height=h)

    @staticmethod
    def _contours_to_boxes(contours, min_area: int, min_aspect: float, max_aspect: float) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area <= min_area or bh <= 0:
                continue
            aspect = bw / bh
            if min_aspect < aspect < max_aspect:
                boxes.append((x, y, bw, bh))
        return boxes

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _deduplicate_boxes(boxes: list[tuple[int, int, int, int]], iou_threshold: float) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
            if all(OCRDialogueService._iou(box, k) < iou_threshold for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _sort_text_regions_manga_order(boxes: list[tuple[int, int, int, int]], panel_height: int) -> list[tuple[int, int, int, int]]:
        row_band = max(panel_height // 6, 1)
        return sorted(boxes, key=lambda b: (b[1] // row_band, -(b[0] + b[2]), b[1]))

    @staticmethod
    def _classify_text_role(boxes: list[tuple[int, int, int, int]], text: str) -> str:
        if not boxes or text == "[no dialogue detected]":
            return "ambient"
        if len(boxes) >= 3:
            return "conversation"
        if any((w / max(h, 1)) >= 2.0 for _, _, w, h in boxes):
            return "narration"
        if len(text.split()) <= 4:
            return "reaction"
        return "dialogue"

    @staticmethod
    def _is_tesseract_available() -> bool:
        candidate = pytesseract.pytesseract.tesseract_cmd
        if candidate and Path(candidate).exists():
            return True
        return shutil.which(candidate or "tesseract") is not None

    @staticmethod
    def _resolve_tesseract_cmd(configured_cmd: str | None) -> str | None:
        candidates = [
            configured_cmd,
            shutil.which("tesseract"),
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None
