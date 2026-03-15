import asyncio
import base64
import logging
import mimetypes
from pathlib import Path

import aiohttp

from config import Settings
from utils.ffmpeg_utils import create_placeholder_video

logger = logging.getLogger(__name__)


class VideoGeneratorService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_video(self, scene_script: dict[str, object], output_path: Path, scene_image_path: Path) -> Path:
        provider = self.settings.video_provider.strip().lower()
        logger.info("Generating video with provider=%s", provider)

        if provider == "runway":
            if not self.settings.runway_api_key:
                await create_placeholder_video(
                    output_path,
                    int(scene_script.get("duration", 5)),
                    str(scene_script.get("scene_description", "Manga scene")),
                    self.settings.ffmpeg_binary,
                    image_path=scene_image_path,
                )
                return output_path
            return await self._generate_with_runway(scene_script, output_path, scene_image_path)

        if provider == "pika":
            if not self.settings.fal_api_key:
                await create_placeholder_video(
                    output_path,
                    int(scene_script.get("duration", 5)),
                    str(scene_script.get("scene_description", "Manga scene")),
                    self.settings.ffmpeg_binary,
                    image_path=scene_image_path,
                )
                return output_path
            return await self._generate_with_pika(scene_script, output_path, scene_image_path)

        raise ValueError(f"Unsupported VIDEO_PROVIDER '{self.settings.video_provider}'")

    async def _generate_with_runway(self, scene_script: dict[str, object], output_path: Path, scene_image_path: Path) -> Path:
        headers = {
            "Authorization": f"Bearer {self.settings.runway_api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": self.settings.runway_api_version,
        }
        payload = {
            "model": self.settings.runway_model,
            "promptImage": self._to_data_uri(scene_image_path),
            "promptText": self._build_video_prompt(scene_script),
            "ratio": self.settings.runway_ratio,
            "duration": int(scene_script.get("duration", self.settings.runway_duration)),
        }

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.settings.runway_base_url}/v1/image_to_video", headers=headers, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(f"Runway submit error {response.status}: {data}")
                task_id = data.get("id")
                if not task_id:
                    raise RuntimeError("Runway did not return a task id")

            video_url = await self._poll_runway_task(session, task_id, headers)
            await self._download_video(session, video_url, output_path)
        return output_path

    async def _generate_with_pika(self, scene_script: dict[str, object], output_path: Path, scene_image_path: Path) -> Path:
        headers = {
            "Authorization": f"Key {self.settings.fal_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": self._build_video_prompt(scene_script),
            "image_url": self._to_data_uri(scene_image_path),
            "duration": int(scene_script.get("duration", self.settings.pika_duration)),
            "resolution": self.settings.pika_resolution,
        }
        submit_url = f"https://queue.fal.run/{self.settings.pika_model}"

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(submit_url, headers=headers, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(f"Pika submit error {response.status}: {data}")
                request_id = data.get("request_id")
                if not request_id:
                    raise RuntimeError("Pika did not return a request id")

            video_url = await self._poll_fal_result(session, request_id, headers)
            await self._download_video(session, video_url, output_path)
        return output_path

    async def _poll_runway_task(self, session: aiohttp.ClientSession, task_id: str, headers: dict[str, str]) -> str:
        status_url = f"{self.settings.runway_base_url}/v1/tasks/{task_id}"
        for _ in range(60):
            async with session.get(status_url, headers=headers) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(f"Runway poll error {response.status}: {data}")
                status = str(data.get("status", "")).upper()
                if status == "SUCCEEDED":
                    output = data.get("output") or []
                    if output:
                        return output[0]
                    raise RuntimeError("Runway task succeeded without an output URL")
                if status in {"FAILED", "CANCELLED"}:
                    raise RuntimeError(f"Runway task ended with status {status}")
            await asyncio.sleep(5)
        raise TimeoutError("Timed out waiting for Runway video generation")

    async def _poll_fal_result(self, session: aiohttp.ClientSession, request_id: str, headers: dict[str, str]) -> str:
        status_url = f"https://queue.fal.run/{self.settings.pika_model}/requests/{request_id}"
        result_url = f"{status_url}/result"
        for _ in range(60):
            async with session.get(status_url, headers=headers) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(f"Pika poll error {response.status}: {data}")
                status = str(data.get("status", "")).upper()
                if status == "COMPLETED":
                    async with session.get(result_url, headers=headers) as result_response:
                        result = await result_response.json(content_type=None)
                        if result_response.status >= 400:
                            raise RuntimeError(f"Pika result error {result_response.status}: {result}")
                        video = result.get("video") or {}
                        if video.get("url"):
                            return str(video["url"])
                        raise RuntimeError("Pika result completed without a video URL")
                if status in {"FAILED", "CANCELLED"}:
                    raise RuntimeError(f"Pika request ended with status {status}")
            await asyncio.sleep(self.settings.pika_poll_seconds)
        raise TimeoutError("Timed out waiting for Pika video generation")

    @staticmethod
    async def _download_video(session: aiohttp.ClientSession, video_url: str, output_path: Path) -> None:
        async with session.get(video_url) as response:
            data = await response.read()
            if response.status >= 400:
                raise RuntimeError(f"Video download failed with status {response.status}")
        await asyncio.to_thread(output_path.write_bytes, data)

    @staticmethod
    def _build_video_prompt(scene_script: dict[str, object]) -> str:
        return (
            f"Scene: {scene_script.get('scene_description', '')}. "
            f"Camera: {scene_script.get('camera_motion', '')}. "
            f"Animation: {scene_script.get('animation_action', '')}."
        )

    @staticmethod
    def _to_data_uri(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        payload = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type or 'image/png'};base64,{payload}"
