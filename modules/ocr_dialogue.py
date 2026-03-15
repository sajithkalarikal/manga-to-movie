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
from modules.ocr_engine import COMIC_WHITELIST, get_corrector, get_ocr_engine
from modules.panel_detection import DetectedPanel

logger = logging.getLogger(__name__)


class OCRDialogueService:
    def __init__(self, settings: Settings):
        self.settings = settings
        resolved_cmd = self._resolve_tesseract_cmd(settings.tesseract_cmd)
        if resolved_cmd:
            pytesseract.pytesseract.tesseract_cmd = resolved_cmd
            logger.info("Using tesseract binary at %s", resolved_cmd)
        self.ocr_available = self._is_tesseract_available()
        if not self.ocr_available:
            logger.warning("Tesseract is not installed. OCR will use a placeholder fallback.")

        # Pre-warm optional models (manga-ocr / easyocr / symspell) once.
        get_ocr_engine()
        get_corrector()

    async def extract_dialogue(self, panels: list[DetectedPanel]) -> list[dict[str, str]]:
        logger.info("Running OCR on %s panels", len(panels))
        if not self.ocr_available:
            return [
                {"panel": str(panel.index), "text": "[ocr unavailable: install tesseract to extract dialogue]"}
                for panel in panels
            ]
        stitched_results = await asyncio.to_thread(self._extract_from_stitched_page_if_needed, panels)
        if stitched_results is not None:
            return stitched_results
        results = await asyncio.gather(*(self._read_panel(panel) for panel in panels))
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

    async def _read_panel(self, panel: DetectedPanel) -> dict[str, str]:
        text = await asyncio.to_thread(self._ocr_parts_sync, panel.image_path, 1)
        return {
            "panel": str(panel.index),
            "text": (text[0] if text else "") or "[no dialogue detected]",
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
    def _ocr_parts_from_bgr(bgr: np.ndarray, limit: int = 2) -> list[str]:
        h, w = bgr.shape[:2]
        boxes = OCRDialogueService._detect_all_narration_boxes(bgr)
        if not boxes:
            h, w = bgr.shape[:2]
            boxes = [(0, 0, w, int(h * 0.6))]

        ocr_engine = get_ocr_engine()
        corrector = get_corrector()
        outputs: list[str] = []
        for x, y, bw, bh in boxes:
            region = bgr[y : y + bh, x : x + bw]
            if region.size == 0:
                continue
            center_x = x + (bw // 2)
            is_left_box = center_x < int(w * 0.45)
            if is_left_box:
                candidates = OCRDialogueService._collect_left_box_candidates(region, ocr_engine)
            else:
                candidates = OCRDialogueService._collect_candidates(region, ocr_engine)
            if not candidates:
                continue
            best = max(candidates, key=OCRDialogueService._score_candidate)
            best_score = OCRDialogueService._score_candidate(best)
            if best_score < 1.0:
                continue
            cleaned = OCRDialogueService._clean_ocr_line(best)
            if cleaned:
                cleaned = corrector.correct_line(cleaned)
            if not cleaned:
                continue
            if cleaned in outputs:
                continue
            outputs.append(cleaned)
            if len(outputs) >= max(1, limit):
                break
        return outputs

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
    def _collect_candidates(region_bgr: np.ndarray, ocr_engine) -> list[str]:
        candidates: list[str] = []
        for variant in OCRDialogueService._preprocess_variants(region_bgr):
            text = ocr_engine.read(variant)
            if not text.strip():
                # Fallback to tesseract only if model engines returned nothing.
                text = pytesseract.image_to_string(variant, config="--oem 3 --psm 4 -l eng")
            normalized = OCRDialogueService._normalize_spaces(text)
            if normalized:
                candidates.append(normalized)
        return candidates

    @staticmethod
    def _collect_left_box_candidates(region_bgr: np.ndarray, ocr_engine) -> list[str]:
        candidates: list[str] = []
        variants = OCRDialogueService._preprocess_left_box(region_bgr)
        for variant in variants:
            text = ocr_engine.read_manga_only(variant)
            if not text.strip():
                text = ocr_engine.read(variant)
            if not text.strip():
                continue
            normalized = OCRDialogueService._normalize_spaces(text)
            if normalized:
                candidates.append(normalized)
        # If manga-ocr is unavailable, fallback to regular candidates to avoid empty output.
        if not candidates:
            return OCRDialogueService._collect_candidates(region_bgr, ocr_engine)
        return candidates

    @staticmethod
    def _preprocess_variants(img_bgr: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        std = cv2.convertScaleAbs(gray, alpha=1.0, beta=0)
        hi = cv2.convertScaleAbs(gray, alpha=1.35, beta=5)
        dilated = cv2.dilate(hi, np.ones((2, 2), np.uint8), iterations=1)
        eroded = cv2.erode(hi, np.ones((2, 2), np.uint8), iterations=1)

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
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        cleaned = cv2.bitwise_not(closed)
        return [std, hi, dilated, eroded, cleaned]

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
        gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k_open, iterations=1)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, cc_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cc_clean = OCRDialogueService._keep_letter_sized_blobs(cv2.bitwise_not(cc_thresh))
        cc_variant = cv2.bitwise_not(cc_clean)

        return [otsu, opened, clahe_otsu, cc_variant]

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
    def _clean_ocr_line(raw: str) -> str:
        out: list[str] = []
        for token in raw.split():
            cleaned = OCRDialogueService._clean_token(token)
            if cleaned is not None:
                out.append(cleaned)
        return " ".join(out)

    @staticmethod
    def _clean_token(token: str) -> str | None:
        t = token.strip(".,!?'-\" ")
        if not t:
            return None
        upper = OCRDialogueService._token_only_alpha(t).upper()
        if not upper:
            return None
        if upper in COMIC_WHITELIST:
            return upper
        if OCRDialogueService._is_real_word(upper):
            return upper
        if OCRDialogueService._is_garbage_like(t):
            return None
        if upper.isalpha() and 2 <= len(upper) <= 15:
            return upper
        return None

    @staticmethod
    def _is_garbage_like(token: str) -> bool:
        return bool(re.search(r"^[^a-zA-Z]{2,}$|^[a-zA-Z][^a-zA-Z]+$|[0-9]{3,}|(.)\1{3,}", token))

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join(str(text).split())

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
    def _detect_all_narration_boxes(img_bgr: np.ndarray, min_area: int = 800) -> list[tuple[int, int, int, int]]:
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

        panel_area = float(h * w)
        found = [box for box in found if (box[2] * box[3]) <= (panel_area * 0.45)]

        deduped = OCRDialogueService._deduplicate_boxes(found, 0.3)
        return deduped

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
        return sorted(kept, key=lambda b: (b[1], b[0]))

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
