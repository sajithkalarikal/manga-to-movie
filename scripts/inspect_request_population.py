from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DB = ROOT / "outputs" / "app_state.sqlite3"
V2_DB = ROOT / "outputs" / "app_state_v2.sqlite3"
OUTPUTS_DIR = ROOT / "outputs"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_all(conn: sqlite3.Connection, query: str, params: tuple[object, ...]) -> list[dict[str, object]]:
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _find_override_path(request_id: str) -> Path | None:
    direct = OUTPUTS_DIR / request_id / "panel_overrides.json"
    analyze = OUTPUTS_DIR / f"analyze-{request_id}" / "panel_overrides.json"
    if direct.exists():
        return direct
    if analyze.exists():
        return analyze
    return None


def inspect_request(request_id: str) -> dict[str, object]:
    report: dict[str, object] = {"request_id": request_id}

    override_path = _find_override_path(request_id)
    if override_path and override_path.exists():
        payload = json.loads(override_path.read_text("utf-8"))
        report["override_json"] = {
            "path": str(override_path),
            "override_groups": len(payload.get("overrides") or {}),
            "panel_boxes": len(payload.get("panel_boxes") or []),
            "panel_region_groups": len(payload.get("panel_regions") or {}),
            "panel_region_total": sum(len(items) for items in (payload.get("panel_regions") or {}).values()),
        }
    else:
        report["override_json"] = None

    if CURRENT_DB.exists():
        with _connect(CURRENT_DB) as conn:
            current: dict[str, object] = {}
            current["jobs"] = _fetch_all(
                conn,
                "SELECT request_id, job_type, status, filename, input_path, output_path, updated_at FROM jobs WHERE request_id = ?",
                (request_id,),
            )
            current["job_events"] = _fetch_all(
                conn,
                "SELECT request_id, event_type, status, created_at FROM job_events WHERE request_id = ? ORDER BY id",
                (request_id,),
            )
            report["current_db"] = current
    else:
        report["current_db"] = None

    if V2_DB.exists():
        with _connect(V2_DB) as conn:
            v2: dict[str, object] = {}
            v2["requests"] = _fetch_all(
                conn,
                "SELECT request_id, status, fs_root_path, panel_mode, bubble_mode, metadata_json, created_at, updated_at FROM requests WHERE request_id = ?",
                (request_id,),
            )
            v2["request_assets"] = _fetch_all(
                conn,
                "SELECT asset_id, request_id, asset_type, rel_path, panel_index, metadata_json, created_at FROM request_assets WHERE request_id = ? ORDER BY asset_id",
                (request_id,),
            )
            v2["annotations"] = _fetch_all(
                conn,
                "SELECT annotation_id, request_id, entity_type, label_type, role, current_version_id, is_active, created_at FROM annotations WHERE request_id = ? ORDER BY annotation_id",
                (request_id,),
            )
            v2["annotation_versions"] = _fetch_all(
                conn,
                """
                SELECT av.version_id, av.annotation_id, av.created_by, av.source_type,
                       av.is_manual_override, av.is_active, av.notes, av.created_at
                FROM annotation_versions av
                JOIN annotations a ON a.annotation_id = av.annotation_id
                WHERE a.request_id = ?
                ORDER BY av.version_id
                """,
                (request_id,),
            )
            report["v2_db"] = v2
    else:
        report["v2_db"] = None

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect what storage tables/files were populated for a request_id.")
    parser.add_argument("request_id", help="Pipeline request_id to inspect.")
    args = parser.parse_args()

    report = inspect_request(args.request_id)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
