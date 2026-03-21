#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/scripts/local.sh"

REDIS_STARTED_BY_SCRIPT=0
API_PID=""
CLEANED_UP=0
REDIS_EXIT_MODE="${REDIS_EXIT_MODE:-auto}"

api_port() {
  printf '%s' "$API_PORT"
}

redis_port_from_url() {
  printf '%s' "$REDIS_URL" | sed -E 's#redis://[^:/]+:([0-9]+).*#\1#'
}

kill_listener_on_port() {
  local port="$1"
  local label="$2"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  fi

  if [[ -z "$pids" ]]; then
    return
  fi

  echo "Stopping existing $label process on port $port"
  kill $pids >/dev/null 2>&1 || true
  sleep 1

  local remaining=""
  if command -v lsof >/dev/null 2>&1; then
    remaining="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if [[ -n "$remaining" ]]; then
    kill -9 $remaining >/dev/null 2>&1 || true
  fi
}

cleanup_existing_instances() {
  kill_listener_on_port "$(api_port)" "API"
  kill_listener_on_port "$(redis_port_from_url)" "Redis"
}

stop_started_redis() {
  case "$REDIS_EXIT_MODE" in
    never)
      return
      ;;
    always)
      ;;
    auto)
      if [[ "$REDIS_STARTED_BY_SCRIPT" != "1" ]]; then
        return
      fi
      ;;
    *)
      echo "Unsupported REDIS_EXIT_MODE=$REDIS_EXIT_MODE (expected auto, always, or never)" >&2
      return
      ;;
  esac

  if [[ "$REDIS_STARTED_BY_SCRIPT" == "1" ]]; then
    echo "Stopping Redis started by load.sh"
  else
    echo "Stopping Redis because REDIS_EXIT_MODE=always"
  fi
  redis-cli -u "$REDIS_URL" shutdown nosave >/dev/null 2>&1 || true
  pkill -f "redis-server.*--port $(redis_port_from_url)" >/dev/null 2>&1 || true
}

cleanup() {
  if [[ "$CLEANED_UP" == "1" ]]; then
    return
  fi
  CLEANED_UP=1

  if [[ -n "$API_PID" ]]; then
    kill "$API_PID" >/dev/null 2>&1 || true
    wait "$API_PID" 2>/dev/null || true
  fi

  stop_started_redis
}

trap cleanup EXIT INT TERM

ensure_system_deps
ensure_venv
ensure_keys_file
configure_fast_ocr_env
configure_bubble_detector_env
configure_panel_detector_env
cleanup_existing_instances

if redis_running; then
  echo "Redis already running at $REDIS_URL"
else
  start_redis
  REDIS_STARTED_BY_SCRIPT=1
fi

if [[ "$API_RELOAD" == "1" ]]; then
  echo "API reload enabled"
  "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" --reload &
else
  echo "API reload disabled for lower laptop load"
  "$VENV_DIR/bin/uvicorn" app:app --host "$API_HOST" --port "$API_PORT" &
fi

API_PID=$!
wait "$API_PID"
