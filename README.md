# manga-to-video-ai

FastAPI project that converts a manga image or manga page into a short animated video using OCR, image captioning, an LLM, video generation APIs, ElevenLabs voice, and FFmpeg composition.

The current Phase 1 path is increasingly local-first:
- panel detection
- OCR
- heuristic scene analysis
- optional local bubble detector trained from COCO annotations
- manual correction in the frontend

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
- Manual panel override saving for speech, narration, SFX, and bubble counts

## Architecture Notes

- Database architecture and SQLite migration notes:
  - [DATABASE_README.md](/Users/sajith/Documents/New%20project/manga-to-movie/DATABASE_README.md)
- Current live state:
  - legacy SQLite still handles image/job/training metadata
  - new SQLite v2 now stores request rows and override-derived annotation history
  - override loading is table-first with JSON fallback

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

### Quick Start

Use the helper script once for setup, then run the API and web watcher separately:

```bash
./scripts/local.sh setup
./load.sh
./init.md
```

Setup and startup behavior:
- creates `.venv` if needed
- installs `requirements.txt`
- installs web dependencies when Node is available
- creates `local.keys.json` from the example file if missing
- `./load.sh` starts the API, starts Redis if it is not already running, and shuts down the Redis instance it started when you exit
- `./init.md` runs `npx vite build --watch` from `web/` so `web/dist` stays updated on file changes

If you want `./load.sh` to always stop Redis on exit, even when Redis was already running before startup, use:

```bash
REDIS_EXIT_MODE=always ./load.sh
```

After startup, you can use:
- Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)
- React UI at [http://localhost:8000/ui_v2/home](http://localhost:8000/ui_v2/home)
- Health UI at [http://localhost:8000/ui_v2/health](http://localhost:8000/ui_v2/health)

You can also start pieces individually:

```bash
./scripts/local.sh api
./scripts/local.sh worker
```

`./scripts/local.sh api` now starts Redis automatically if needed.

The legacy static UI is still available at [http://localhost:8000/ui](http://localhost:8000/ui), but current frontend work should go through `ui_v2`.

### Manual Run

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

5. Configure API keys.

Create `local.keys.json` in the project root, for example:

```json
{
  "OPENAI_API_KEY": "",
  "HUGGINGFACE_API_TOKEN": "",
  "RUNWAY_API_KEY": "",
  "RUNWAY_BASE_URL": "https://api.dev.runwayml.com",
  "FAL_KEY": "",
  "ELEVENLABS_API_KEY": ""
}
```

Environment variables still override file values if you need that for deployment.

Optional environment variables:

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
9. Build the React frontend in a second terminal:

```bash
./init.md
```

10. Open the frontend app at [http://localhost:8000/ui_v2/home](http://localhost:8000/ui_v2/home)

## Local Bubble Detector

You can train a local speech-bubble detector from a COCO dataset and plug it into Phase 1.

Expected dataset structure:

```text
Manga Bubble.v4i.coco/
+-- train/
|   +-- _annotations.coco.json
+-- valid/
|   +-- _annotations.coco.json
+-- test/
    +-- _annotations.coco.json
```

Train a first model:

```bash
./.venv/bin/python scripts/train_bubble_detector.py \
  --dataset-root "/Users/sajith/Documents/New project/Manga Bubble.v4i.coco" \
  --output models/bubble_detector.pt \
  --epochs 5 \
  --batch-size 2
```

Quick smoke run:

```bash
./.venv/bin/python scripts/train_bubble_detector.py \
  --dataset-root "/Users/sajith/Documents/New project/Manga Bubble.v4i.coco" \
  --output models/bubble_detector_smoke.pt \
  --epochs 1 \
  --batch-size 1 \
  --limit-train 2 \
  --limit-valid 1 \
  --device cpu
```

Enable the trained detector in the app:

```bash
./load.sh
```

By default, `./load.sh` now auto-loads:

- [models/bubble_detector_v2_new_only.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector_v2_new_only.pt) if it exists
- otherwise falls back to [models/bubble_detector.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector.pt)
- `BUBBLE_DETECTOR_SCORE_THRESHOLD=0.45`

Optional override:

```bash
BUBBLE_DETECTOR_WEIGHTS="/custom/path/bubble_detector.pt" \
BUBBLE_DETECTOR_SCORE_THRESHOLD="0.50" \
./load.sh
```

When weights are configured, Phase 1 uses:
- trained bubble detector first
- heuristic bubble logic as fallback and merge layer
- manual overrides in the frontend for correction and future dataset expansion

### Two-Stage Training

For the current multi-class detector flow, the recommended path is:

1. speech-focused pretraining or reuse of the older bubble checkpoint
2. new-dataset-only multi-class fine-tuning

Current recommended fine-tune command:

```bash
./.venv/bin/python scripts/train_bubble_detector.py \
  --dataset-root "/Users/sajith/Documents/New project/manga-to-movie/outputs/dataset_annotations/exported_coco/new object training data.v1.coco" \
  --output models/bubble_detector_v2_new_only.pt \
  --init-weights models/bubble_detector.pt \
  --epochs 3 \
  --batch-size 2
```

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
