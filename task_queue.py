import asyncio
import json
from datetime import datetime, timezone

from arq.connections import ArqRedis, RedisSettings, create_pool

from config import get_settings
from modules.database import upsert_job


async def get_redis_pool() -> ArqRedis:
    settings = get_settings()
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    return await create_pool(redis_settings, default_queue_name=settings.redis_queue_name)


def build_status_payload(request_id: str, status: str, **extra) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": request_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra)
    return payload


async def set_task_status(redis: ArqRedis, request_id: str, payload: dict[str, object]) -> None:
    settings = get_settings()
    key = f"task:{request_id}"
    await redis.set(key, json.dumps(payload), ex=settings.task_result_ttl_seconds)
    await asyncio.to_thread(
        upsert_job,
        settings,
        payload,
        image_id=int(payload["image_id"]) if payload.get("image_id") is not None else None,
        job_type=str(payload.get("job_type", "generate_video")),
    )


async def get_task_status(redis: ArqRedis, request_id: str) -> dict[str, object] | None:
    key = f"task:{request_id}"
    value = await redis.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


async def enqueue_manga_job(redis: ArqRedis, request_id: str, upload_path: str):
    settings = get_settings()
    return await redis.enqueue_job(
        "process_manga_job",
        request_id,
        upload_path,
        _job_id=request_id,
        _expires=settings.job_timeout_seconds + settings.task_result_ttl_seconds,
    )
