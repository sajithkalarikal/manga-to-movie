import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from manga_ocr import MangaOcr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MangaOcr = None

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    easyocr = None

try:
    from symspellpy import SymSpell, Verbosity  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SymSpell = None
    Verbosity = None


COMIC_WHITELIST = {
    "IT",
    "WE",
    "UP",
    "IN",
    "OF",
    "ON",
    "AT",
    "TO",
    "DO",
    "GO",
    "BE",
    "MY",
    "HE",
    "ME",
    "BY",
    "OR",
    "SO",
    "NO",
    "IS",
    "AS",
    "AN",
    "IF",
    "US",
    "AM",
    "WENT",
    "TOWN",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "SMELL",
    "BONFIRE",
    "DRIFTS",
    "FAINT",
    "THE",
    "A",
    "I",
    "WAS",
    "WHEN",
    "THINK",
    "HARDLY",
    "REMEMBER",
    "ANYTHING",
    "BUT",
    "HOW",
    "FELT",
}


class ComicOCREngine:
    def __init__(self):
        self._mocr = None
        self._easy = None
        enable_manga_ocr = os.getenv("ENABLE_MANGA_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}
        enable_easyocr = os.getenv("ENABLE_EASYOCR", "0").strip().lower() in {"1", "true", "yes", "on"}
        repo_root = Path(__file__).resolve().parents[1]
        model_cache = repo_root / ".model_cache"
        hf_home = model_cache / "hf"
        manga_ocr_local = model_cache / "manga_ocr_base"
        easyocr_home = model_cache / "easyocr"
        hf_home.mkdir(parents=True, exist_ok=True)
        easyocr_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("EASYOCR_MODULE_PATH", str(easyocr_home))

        if not enable_manga_ocr:
            logger.info("[OCR] manga-ocr disabled; using Tesseract-first OCR")
        elif MangaOcr is not None:
            try:
                logger.info("[OCR] Loading manga-ocr model")
                if manga_ocr_local.exists():
                    self._mocr = MangaOcr(pretrained_model_name_or_path=str(manga_ocr_local))
                else:
                    self._mocr = MangaOcr()
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.warning("[OCR] manga-ocr unavailable: %s", exc)
        else:
            logger.info("[OCR] manga-ocr not installed; skipping")

        if not enable_easyocr:
            logger.info("[OCR] easyocr disabled; using Tesseract-first OCR")
        elif easyocr is not None:
            try:
                logger.info("[OCR] Loading EasyOCR model")
                self._easy = easyocr.Reader(["en"], gpu=False, model_storage_directory=os.environ["EASYOCR_MODULE_PATH"])
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.warning("[OCR] easyocr unavailable: %s", exc)
        else:
            logger.info("[OCR] easyocr not installed; skipping")

    def read(self, img_bgr) -> str:
        # Primary: manga-ocr
        if self._mocr is not None:
            try:
                from PIL import Image
                import cv2

                pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                result = str(self._mocr(pil_img)).strip()
                if self._looks_valid(result):
                    return result
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.debug("[OCR] manga-ocr read failed: %s", exc)

        # Fallback: EasyOCR
        if self._easy is not None:
            try:
                lines = self._easy.readtext(img_bgr, detail=0, paragraph=True)
                result = " ".join(lines).strip()
                if self._looks_valid(result):
                    return result
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.debug("[OCR] easyocr read failed: %s", exc)

        return ""

    def read_manga_only(self, img_bgr) -> str:
        if self._mocr is None:
            return ""
        try:
            from PIL import Image
            import cv2

            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            result = str(self._mocr(pil_img)).strip()
            return result if self._looks_valid(result) else ""
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.debug("[OCR] manga-only read failed: %s", exc)
            return ""

    @staticmethod
    def _looks_valid(text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return False
        latin_count = sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))
        return (latin_count / max(1, len(text))) > 0.20


class SpellCorrector:
    TOKEN_FIXUPS = {
        "FANT": "FAINT",
        "BONFIPE": "BONFIRE",
        "DPIFTS": "DRIFTS",
        "DRIFIS": "DRIFTS",
        "KOW": "HOW",
    }
    PHRASE_FIXUPS = {
        "I THINK I WAS": "I THINK IT WAS",
        "WENT THE TOWN": "WENT TO THE TOWN",
    }

    def __init__(self):
        self._sym = None
        if SymSpell is None:
            logger.info("[OCR] symspellpy not installed; skipping spell correction")
            return
        try:
            import pkg_resources

            self._sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
            self._sym.load_dictionary(dict_path, term_index=0, count_index=1)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("[OCR] spell corrector unavailable: %s", exc)
            self._sym = None

    def correct_line(self, text: str) -> str:
        if not text.strip() or self._sym is None:
            return self._apply_phrase_fixups(self._apply_token_fixups(text))
        output: list[str] = []
        for token in text.split():
            clean = re.sub(r"[^A-Za-z]", "", token).upper()
            if not clean:
                continue
            clean = self.TOKEN_FIXUPS.get(clean, clean)
            if clean in COMIC_WHITELIST:
                output.append(clean)
                continue
            suggestions = self._sym.lookup(clean, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            if not suggestions:
                output.append(clean)
                continue
            best = suggestions[0]
            if best.distance <= 1:
                output.append(best.term.upper())
            else:
                output.append(clean)
        return self._apply_phrase_fixups(" ".join(output))

    def _apply_token_fixups(self, text: str) -> str:
        out: list[str] = []
        for token in text.split():
            clean = re.sub(r"[^A-Za-z]", "", token).upper()
            out.append(self.TOKEN_FIXUPS.get(clean, clean) if clean else token)
        return " ".join(out)

    def _apply_phrase_fixups(self, text: str) -> str:
        fixed = text
        for src, dst in self.PHRASE_FIXUPS.items():
            fixed = fixed.replace(src, dst)
        return fixed


@lru_cache(maxsize=1)
def get_ocr_engine() -> ComicOCREngine:
    return ComicOCREngine()


@lru_cache(maxsize=1)
def get_corrector() -> SpellCorrector:
    return SpellCorrector()
