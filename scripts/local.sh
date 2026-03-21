#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
WEB_DIR="$ROOT_DIR/web"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-4173}"
BUBBLE_DETECTOR_DEFAULT_WEIGHTS="$ROOT_DIR/models/bubble_detector_v2_new_only.pt"
PANEL_DETECTOR_DEFAULT_WEIGHTS="$ROOT_DIR/models/panel_detector.pt"
ENABLE_MANGA_OCR="${ENABLE_MANGA_OCR:-0}"
ENABLE_EASYOCR="${ENABLE_EASYOCR:-0}"
API_RELOAD="${API_RELOAD:-0}"
BUBBLE_DETECTOR_DEVICE="${BUBBLE_DETECTOR_DEVICE:-cpu}"
BUBBLE_DETECTOR_MAX_SIDE="${BUBBLE_DETECTOR_MAX_SIDE:-960}"
PANEL_DETECTOR_DEVICE="${PANEL_DETECTOR_DEVICE:-cpu}"
PANEL_DETECTOR_MAX_SIDE="${PANEL_DETECTOR_MAX_SIDE:-960}"

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
  export BUBBLE_DETECTOR_DEVICE
  export BUBBLE_DETECTOR_MAX_SIDE
  if [[ -z "${BUBBLE_DETECTOR_WEIGHTS:-}" && -f "$BUBBLE_DETECTOR_DEFAULT_WEIGHTS" ]]; then
    export BUBBLE_DETECTOR_WEIGHTS="$BUBBLE_DETECTOR_DEFAULT_WEIGHTS"
    echo "Using local bubble detector weights at $BUBBLE_DETECTOR_WEIGHTS"
  fi

  if [[ -n "${BUBBLE_DETECTOR_WEIGHTS:-}" ]]; then
    export BUBBLE_DETECTOR_SCORE_THRESHOLD="${BUBBLE_DETECTOR_SCORE_THRESHOLD:-0.45}"
    echo "Bubble detector score threshold: $BUBBLE_DETECTOR_SCORE_THRESHOLD"
    echo "Bubble detector device: $BUBBLE_DETECTOR_DEVICE"
    echo "Bubble detector max side: $BUBBLE_DETECTOR_MAX_SIDE"
  fi
}

configure_panel_detector_env() {
  export PANEL_DETECTOR_DEVICE
  export PANEL_DETECTOR_MAX_SIDE
  if [[ -z "${PANEL_DETECTOR_WEIGHTS:-}" && -f "$PANEL_DETECTOR_DEFAULT_WEIGHTS" ]]; then
    export PANEL_DETECTOR_WEIGHTS="$PANEL_DETECTOR_DEFAULT_WEIGHTS"
    echo "Using local panel detector weights at $PANEL_DETECTOR_WEIGHTS"
  fi

  if [[ -n "${PANEL_DETECTOR_WEIGHTS:-}" ]]; then
    export PANEL_DETECTOR_SCORE_THRESHOLD="${PANEL_DETECTOR_SCORE_THRESHOLD:-0.45}"
    export PANEL_DETECTOR_MAX_DETECTIONS="${PANEL_DETECTOR_MAX_DETECTIONS:-16}"
    echo "Panel detector score threshold: $PANEL_DETECTOR_SCORE_THRESHOLD"
    echo "Panel detector device: $PANEL_DETECTOR_DEVICE"
    echo "Panel detector max side: $PANEL_DETECTOR_MAX_SIDE"
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

ensure_web_tooling() {
  ensure_cmd node
  ensure_cmd npm
  if [[ ! -f "$WEB_DIR/package.json" ]]; then
    echo "Missing web/package.json" >&2
    exit 1
  fi
}

web_tooling_available() {
  command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 && [[ -f "$WEB_DIR/package.json" ]]
}

install_web_deps() {
  ensure_web_tooling
  (
    cd "$WEB_DIR"
    npm install
  )
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
  configure_panel_detector_env
  if [[ "$API_RELOAD" == "1" ]]; then
    echo "API reload enabled"
    exec "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" --reload
  else
    echo "API reload disabled for lower laptop load"
    exec "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT"
  fi
}

run_worker() {
  ensure_system_deps
  ensure_venv
  ensure_keys_file
  configure_fast_ocr_env
  configure_bubble_detector_env
  configure_panel_detector_env
  exec "$VENV_DIR/bin/arq" worker.WorkerSettings
}

run_dev() {
  start_redis
  ensure_venv
  ensure_keys_file
  if web_tooling_available; then
    run_dev_with_web
    return
  fi

  configure_fast_ocr_env
  configure_bubble_detector_env
  configure_panel_detector_env

  if [[ "$API_RELOAD" == "1" ]]; then
    echo "API reload enabled"
    "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" --reload &
  else
    echo "API reload disabled for lower laptop load"
    "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" &
  fi
  local api_pid=$!
  "$VENV_DIR/bin/arq" worker.WorkerSettings &
  local worker_pid=$!

  echo "Node/npm not found or web/package.json missing. Running API and worker only."
  trap 'kill "$api_pid" "$worker_pid" 2>/dev/null || true' EXIT INT TERM
  wait "$api_pid" "$worker_pid"
}

run_web() {
  ensure_web_tooling
  (
    cd "$WEB_DIR"
    npm run run -- --host "$WEB_HOST" --port "$WEB_PORT"
  )
}

build_web() {
  ensure_web_tooling
  (
    cd "$WEB_DIR"
    npm run build
  )
}

run_dev_with_web() {
  start_redis
  ensure_venv
  ensure_keys_file
  ensure_web_tooling
  configure_fast_ocr_env
  configure_bubble_detector_env
  configure_panel_detector_env

  if [[ "$API_RELOAD" == "1" ]]; then
    echo "API reload enabled"
    "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" --reload &
  else
    echo "API reload disabled for lower laptop load"
    "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" &
  fi
  local api_pid=$!
  "$VENV_DIR/bin/arq" worker.WorkerSettings &
  local worker_pid=$!
  (
    cd "$WEB_DIR"
    npm run run -- --host "$WEB_HOST" --port "$WEB_PORT"
  ) &
  local web_pid=$!

  trap 'kill "$api_pid" "$worker_pid" "$web_pid" 2>/dev/null || true' EXIT INT TERM
  wait "$api_pid" "$worker_pid" "$web_pid"
}

setup_local() {
  ensure_system_deps
  install_deps
  if web_tooling_available; then
    install_web_deps
  else
    echo "Skipping web dependency install because node/npm is not available."
  fi
  ensure_keys_file
  start_redis
  echo "Local setup complete."
  echo "Run ./scripts/local.sh dev to start the API, worker, and web app when available."
}

usage() {
  cat <<'EOF'
Usage: ./scripts/local.sh <command>

Commands:
  setup   Create .venv, install Python deps, install web deps when available, ensure local.keys.json, start Redis
  redis   Start Redis if it is not already running
  api     Run the FastAPI app
  worker  Run the background worker
  dev     Start Redis, the API, the worker, and the web app when available
  web-install  Install Node dependencies for web/
  web     Run the UI v2 Vite dev server from web/
  web-build  Build the UI v2 app into web/dist
  dev-web  Start Redis, the API, the worker, and the UI v2 dev server
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
    web-install)
      install_web_deps
      ;;
    web)
      run_web
      ;;
    web-build)
      build_web
      ;;
    dev-web)
      run_dev_with_web
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
