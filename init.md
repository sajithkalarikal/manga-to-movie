#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$ROOT_DIR/web"

ensure_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    exit 1
  fi
}

ensure_cmd node
ensure_cmd npx

if [[ ! -f "$WEB_DIR/package.json" ]]; then
  echo "Missing web/package.json" >&2
  exit 1
fi

cd "$WEB_DIR"

echo "Watching web sources and rebuilding web/dist with npx vite build --watch"
exec npx vite build --watch
