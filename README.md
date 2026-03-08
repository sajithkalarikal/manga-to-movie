# manga-to-video-ai

FastAPI project that converts a manga image or manga page into a short animated video using OCR, image captioning, an LLM, video generation APIs, ElevenLabs voice, and FFmpeg composition.

## Scalability Upgrades

- Redis-backed job queue with `arq`
- Background worker process for long-running generation tasks
- Async task submission and status polling
- Automatic retry logic for failed jobs
- File and console logging in `logs/app.log`

## Features

- Manga image upload via `POST /generate-video`
- Panel detection with OpenCV and top-left to bottom-right ordering
- OCR dialogue extraction with Tesseract
- BLIP image captioning via API
- Structured scene JSON generation with an LLM
- Video generation with Runway or Pika
- Voice generation with ElevenLabs
- FFmpeg composition with subtitles and optional background music
- Scene metadata and intermediate artifacts saved in `outputs/<request_id>/`

## Project Structure

```text
manga-to-video-ai/
+-- app.py
+-- config.py
+-- pipeline.py
+-- task_queue.py
+-- worker.py
+-- requirements.txt
+-- modules/
+-- utils/
+-- uploads/
+-- outputs/
+-- logs/
+-- docker/
```

## Local Run

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure system tools are installed:
   - `ffmpeg`
   - `tesseract`
   - `redis`

4. Start Redis.

5. Optional environment variables:

```bash
set REDIS_URL=redis://localhost:6379/0
set OPENAI_API_KEY=your_key
set HUGGINGFACE_API_TOKEN=your_key
set RUNWAY_API_KEY=your_key
set FAL_KEY=your_key
set ELEVENLABS_API_KEY=your_key
set VIDEO_PROVIDER=runway
```

6. Start the API:

```bash
uvicorn app:app --reload
```

7. Start the worker in a second terminal:

```bash
arq worker.WorkerSettings
```

8. Open Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)

## Docker Run

```bash
docker compose -f docker/compose.yml up --build
```

## API Flow

1. `POST /generate-video` uploads the manga image and enqueues a Redis job.
2. The API returns `202 Accepted` with a `request_id`.
3. The worker processes the job asynchronously.
4. `GET /tasks/{request_id}` returns queued, processing, retrying, completed, or failed status.

## Example Queue Request

```bash
curl -X POST "http://localhost:8000/generate-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_manga_page.png"
```

## Example Queue Response

```json
{
  "status": "queued",
  "request_id": "2c6d2ec7f2b84a408a49d0c9b4675022",
  "task_status_url": "/tasks/2c6d2ec7f2b84a408a49d0c9b4675022"
}
```

## Example Task Status Response

```json
{
  "request_id": "2c6d2ec7f2b84a408a49d0c9b4675022",
  "status": "completed",
  "updated_at": "2026-03-08T13:20:00.000000+00:00",
  "attempt": 1,
  "max_attempts": 3,
  "video_url": "/outputs/2c6d2ec7f2b84a408a49d0c9b4675022/final_video.mp4",
  "metadata_url": "/outputs/2c6d2ec7f2b84a408a49d0c9b4675022/scene_metadata.json",
  "subtitles_url": "/outputs/2c6d2ec7f2b84a408a49d0c9b4675022/subtitles.srt",
  "error": null
}
```

## Retry Behavior

- Jobs retry automatically up to `JOB_RETRY_COUNT`
- Delay between retries is controlled by `RETRY_DELAY_SECONDS`
- Task state moves through `queued` -> `processing` -> `retrying` -> `completed` or `failed`

## Notes

- If API keys are missing, the project falls back to placeholder video and silent audio so the pipeline can still run.
- Generated artifacts are served from `/outputs`.
- Job state is stored in Redis with TTL controlled by `TASK_RESULT_TTL_SECONDS`.
