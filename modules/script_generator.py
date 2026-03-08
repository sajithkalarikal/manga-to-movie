import json
import logging

import aiohttp

from config import Settings

logger = logging.getLogger(__name__)


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
                data = await response.json(content_type=None)
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
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {content}") from exc
        return {
            "scene_description": parsed.get("scene_description", "Manga scene with dramatic action."),
            "camera_motion": parsed.get("camera_motion", "Slow push-in"),
            "animation_action": parsed.get("animation_action", "Subtle panel parallax and hair movement"),
            "dialogue": parsed.get("dialogue", ""),
            "duration": int(parsed.get("duration", 5)),
        }

    @staticmethod
    def _build_fallback_script(dialogue: list[dict[str, str]], captions: list[dict[str, str]]) -> dict[str, object]:
        scene_description = " ".join(item["caption"] for item in captions[:3]) or "Manga hero in a dramatic scene."
        dialogue_text = " ".join(item["text"] for item in dialogue if item["text"] != "[no dialogue detected]")
        return {
            "scene_description": scene_description,
            "camera_motion": "Slow cinematic zoom with subtle left-to-right pan",
            "animation_action": "Animate layered panel depth, eye blinks, and speed lines",
            "dialogue": dialogue_text,
            "duration": max(4, min(10, len(captions) * 2)),
        }
