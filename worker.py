import logging
from pathlib import Path

from arq import Retry
from arq.connections import RedisSettings
from arq.worker import func

from config import Settings, get_settings
from pipeline import MangaVideoPipeline
from task_queue import build_status_payload, get_redis_pool, set_task_status

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    settings = get_settings()
    ctx["settings"] = settings
    ctx["pipeline"] = MangaVideoPipeline(settings)
    ctx["redis"] = await get_redis_pool()


async def shutdown(ctx: dict) -> None:
    redis = ctx.get("redis")
    if redis is not None:
        await redis.close()


async def process_manga_job(ctx: dict, request_id: str, upload_path: str) -> dict[str, object]:
    settings: Settings = ctx["settings"]
    redis = ctx["redis"]
    pipeline: MangaVideoPipeline = ctx["pipeline"]
    attempt = int(ctx.get("job_try", 1))

    await set_task_status(
        redis,
        request_id,
        build_status_payload(request_id, "processing", attempt=attempt, upload_path=upload_path),
    )

    try:
        result = await pipeline.run(request_id=request_id, upload_path=Path(upload_path))
    except Exception as exc:
        logger.exception("Worker failed request_id=%s attempt=%s", request_id, attempt)
        if attempt < settings.job_retry_count:
            await set_task_status(
                redis,
                request_id,
                build_status_payload(
                    request_id,
                    "retrying",
                    attempt=attempt,
                    max_attempts=settings.job_retry_count,
                    error=str(exc),
                ),
            )
            raise Retry(defer=settings.retry_delay_seconds) from exc

        failure = build_status_payload(
            request_id,
            "failed",
            attempt=attempt,
            max_attempts=settings.job_retry_count,
            error=str(exc),
        )
        await set_task_status(redis, request_id, failure)
        return failure

    success = build_status_payload(
        request_id,
        "completed",
        attempt=attempt,
        max_attempts=settings.job_retry_count,
        **result,
    )
    await set_task_status(redis, request_id, success)
    return success


class WorkerSettings:
    functions = [func(process_manga_job, keep_result=0)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    queue_name = get_settings().redis_queue_name
    max_tries = get_settings().job_retry_count
    job_timeout = get_settings().job_timeout_seconds
    max_jobs = get_settings().worker_concurrency
