import json
import logging
import re

import aiohttp

from config import Settings

logger = logging.getLogger(__name__)

FALLBACK_CAPTION = "Dynamic manga scene with dramatic motion"


class ScriptGeneratorService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_script(self, dialogue: list[dict[str, str]], captions: list[dict[str, str]]) -> dict[str, object]:
        logger.info("Generating cinematic scene script")
        if not self.settings.llm_api_key:
            return self._build_fallback_script(dialogue=dialogue, captions=captions)

        prompt = self._build_prompt(dialogue=dialogue, captions=captions)
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cinematic storyboard writer. Return JSON with keys "
                        "scene_description, camera_motion, animation_action, dialogue, duration."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
            async with session.post(self.settings.llm_api_url, headers=headers, json=payload) as response:
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    raw = await response.text()
                    raise RuntimeError(f"LLM API returned non-JSON (status={response.status}): {raw[:400]}")  # noqa: TRY003
                if response.status >= 400:
                    raise RuntimeError(f"LLM API error {response.status}: {data}")

        content = data["choices"][0]["message"]["content"]
        return self._parse_script(content)

    def _build_prompt(self, dialogue: list[dict[str, str]], captions: list[dict[str, str]]) -> str:
        return (
            "Create a cinematic anime-video scene plan from manga annotations. "
            "Keep it concise and production-ready.\n\n"
            f"Dialogue:\n{json.dumps(dialogue, indent=2)}\n\n"
            f"Captions:\n{json.dumps(captions, indent=2)}"
        )

    @staticmethod
    def _parse_script(content: str) -> dict[str, object]:
        raw = (content or "").strip()
        # Tolerate code fences or extra text by extracting the outermost JSON object.
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw).strip()
            raw = re.sub(r"\n?```$", "", raw).strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{") : raw.rfind("}") + 1]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {content}") from exc
        duration_raw = parsed.get("duration", 5)
        try:
            duration = int(duration_raw)
        except Exception:
            duration = 5
        return {
            "scene_description": parsed.get("scene_description", "Manga scene with dramatic action."),
            "camera_motion": parsed.get("camera_motion", "Slow push-in"),
            "animation_action": parsed.get("animation_action", "Subtle panel parallax and hair movement"),
            "dialogue": parsed.get("dialogue", ""),
            "duration": duration,
        }

    @staticmethod
    def _build_fallback_script(dialogue: list[dict[str, str]], captions: list[dict[str, str]]) -> dict[str, object]:
        caption_texts = [
            str(item.get("caption", "")).strip()
            for item in captions
            if str(item.get("caption", "")).strip()
        ]
        # If BLIP was unavailable, captions will often be the same generic fallback; ignore it.
        non_generic = [c for c in caption_texts if c != FALLBACK_CAPTION]
        captions_for_scene = non_generic[:3] if non_generic else []

        dialogue_lines = [
            str(item.get("text", "")).strip()
            for item in dialogue
            if str(item.get("text", "")).strip() and str(item.get("text", "")).strip() != "[no dialogue detected]"
        ]
        dialogue_text = " ".join(dialogue_lines).strip()

        if captions_for_scene:
            scene_description = " ".join(captions_for_scene)
        elif dialogue_text:
            # Use dialogue as the anchor when captions are generic/unavailable.
            scene_description = f"A cinematic manga scene narrated as: {dialogue_text}"
        else:
            scene_description = "A cinematic manga scene with dramatic motion."

        # Heuristic duration: scale with dialogue density and number of panels, but keep it short.
        words = len(dialogue_text.split()) if dialogue_text else 0
        duration = max(4, min(10, max(len(captions) * 2, 4 + (words // 3))))
        return {
            "scene_description": scene_description,
            "camera_motion": "Slow cinematic zoom with subtle left-to-right pan",
            "animation_action": "Animate layered panel depth, eye blinks, and speed lines",
            "dialogue": dialogue_text,
            "duration": duration,
        }
