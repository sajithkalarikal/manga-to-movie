#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
BUBBLE_DETECTOR_DEFAULT_WEIGHTS="$ROOT_DIR/models/bubble_detector.pt"
ENABLE_MANGA_OCR="${ENABLE_MANGA_OCR:-0}"
ENABLE_EASYOCR="${ENABLE_EASYOCR:-0}"

cd "$ROOT_DIR"

ensure_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    exit 1
  fi
}

ensure_venv() {
  ensure_cmd "$PYTHON_BIN"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

ensure_keys_file() {
  if [[ ! -f "$ROOT_DIR/local.keys.json" && -f "$ROOT_DIR/local.keys.json.example" ]]; then
    cp "$ROOT_DIR/local.keys.json.example" "$ROOT_DIR/local.keys.json"
  fi
}

configure_bubble_detector_env() {
  if [[ -z "${BUBBLE_DETECTOR_WEIGHTS:-}" && -f "$BUBBLE_DETECTOR_DEFAULT_WEIGHTS" ]]; then
    export BUBBLE_DETECTOR_WEIGHTS="$BUBBLE_DETECTOR_DEFAULT_WEIGHTS"
    echo "Using local bubble detector weights at $BUBBLE_DETECTOR_WEIGHTS"
  fi

  if [[ -n "${BUBBLE_DETECTOR_WEIGHTS:-}" ]]; then
    export BUBBLE_DETECTOR_SCORE_THRESHOLD="${BUBBLE_DETECTOR_SCORE_THRESHOLD:-0.45}"
    echo "Bubble detector score threshold: $BUBBLE_DETECTOR_SCORE_THRESHOLD"
  fi
}

configure_fast_ocr_env() {
  export ENABLE_MANGA_OCR
  export ENABLE_EASYOCR
  if [[ "$ENABLE_MANGA_OCR" != "0" || "$ENABLE_EASYOCR" != "0" ]]; then
    echo "Optional OCR engines enabled: manga-ocr=$ENABLE_MANGA_OCR easyocr=$ENABLE_EASYOCR"
  else
    echo "Fast OCR mode enabled: using Tesseract-first OCR"
  fi
}

install_deps() {
  ensure_venv
  "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"
}

ensure_system_deps() {
  ensure_cmd ffmpeg
  ensure_cmd tesseract
  ensure_cmd redis-server
  ensure_cmd redis-cli
}

redis_running() {
  redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1
}

start_redis() {
  ensure_system_deps
  if redis_running; then
    echo "Redis already running at $REDIS_URL"
    return
  fi

  local redis_host redis_port
  redis_host="$(printf '%s' "$REDIS_URL" | sed -E 's#redis://([^:/]+).*#\1#')"
  redis_port="$(printf '%s' "$REDIS_URL" | sed -E 's#redis://[^:/]+:([0-9]+).*#\1#')"
  echo "Starting Redis on ${redis_host}:${redis_port}"
  redis-server --bind "$redis_host" --port "$redis_port" --daemonize yes
}

run_api() {
  ensure_system_deps
  ensure_venv
  ensure_keys_file
  configure_fast_ocr_env
  configure_bubble_detector_env
  exec "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" --reload
}

run_worker() {
  ensure_system_deps
  ensure_venv
  ensure_keys_file
  configure_fast_ocr_env
  configure_bubble_detector_env
  exec "$VENV_DIR/bin/arq" worker.WorkerSettings
}

run_dev() {
  start_redis
  ensure_venv
  ensure_keys_file
  configure_fast_ocr_env
  configure_bubble_detector_env

  "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" --reload &
  local api_pid=$!
  "$VENV_DIR/bin/arq" worker.WorkerSettings &
  local worker_pid=$!

  trap 'kill "$api_pid" "$worker_pid" 2>/dev/null || true' EXIT INT TERM
  wait "$api_pid" "$worker_pid"
}

setup_local() {
  ensure_system_deps
  install_deps
  ensure_keys_file
  start_redis
  echo "Local setup complete."
  echo "Run ./scripts/local.sh dev to start the API and worker."
}

usage() {
  cat <<'EOF'
Usage: ./scripts/local.sh <command>

Commands:
  setup   Create .venv, install Python deps, ensure local.keys.json, start Redis
  redis   Start Redis if it is not already running
  api     Run the FastAPI app
  worker  Run the background worker
  dev     Start Redis, the API, and the worker together
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    setup)
      setup_local
      ;;
    redis)
      start_redis
      ;;
    api)
      run_api
      ;;
    worker)
      run_worker
      ;;
    dev)
      run_dev
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
