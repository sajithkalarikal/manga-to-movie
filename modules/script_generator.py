import asyncio
import logging

from config import Settings
from modules.local_phase1 import build_rule_based_scene_script


class ScriptGeneratorService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_script(self, dialogue: list[dict[str, str]], captions: list[dict[str, str]]) -> dict[str, object]:
        await asyncio.sleep(0)
        return build_rule_based_scene_script(dialogue=dialogue, captions=captions)
