from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
import json
from typing import Any


V2_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        fs_root_path TEXT NOT NULL,
        panel_mode TEXT,
        bubble_mode TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS request_assets (
        asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        asset_type TEXT NOT NULL,
        rel_path TEXT NOT NULL,
        panel_index INTEGER,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(request_id) REFERENCES requests(request_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_request_assets_request
    ON request_assets(request_id, asset_type)
    """,
    """
    CREATE TABLE IF NOT EXISTS annotations (
        annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        asset_id INTEGER,
        entity_type TEXT NOT NULL,
        label_type TEXT NOT NULL,
        role TEXT,
        current_version_id INTEGER,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(request_id) REFERENCES requests(request_id) ON DELETE CASCADE,
        FOREIGN KEY(asset_id) REFERENCES request_assets(asset_id) ON DELETE SET NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_annotations_request
    ON annotations(request_id, entity_type, label_type, is_active)
    """,
    """
    CREATE TABLE IF NOT EXISTS annotation_versions (
        version_id INTEGER PRIMARY KEY AUTOINCREMENT,
        annotation_id INTEGER NOT NULL,
        data_json TEXT NOT NULL,
        created_by TEXT,
        source_type TEXT NOT NULL DEFAULT 'manual_override',
        is_manual_override INTEGER NOT NULL DEFAULT 0,
        is_active INTEGER NOT NULL DEFAULT 1,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(annotation_id) REFERENCES annotations(annotation_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_annotation_versions_annotation
    ON annotation_versions(annotation_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS exports (
        export_id TEXT PRIMARY KEY,
        format TEXT NOT NULL,
        dataset_name TEXT,
        fs_export_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'created',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS export_annotations (
        export_id TEXT NOT NULL,
        version_id INTEGER NOT NULL,
        PRIMARY KEY(export_id, version_id),
        FOREIGN KEY(export_id) REFERENCES exports(export_id) ON DELETE CASCADE,
        FOREIGN KEY(version_id) REFERENCES annotation_versions(version_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_export_annotations_version
    ON export_annotations(version_id)
    """,
)


def default_v2_database_path() -> Path:
    return Path("outputs/app_state_v2.sqlite3").expanduser().resolve()


@contextmanager
def connect_v2(database_path: Path | str | None = None):
    resolved = Path(database_path or default_v2_database_path()).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_v2_database(database_path: Path | str | None = None) -> Path:
    resolved = Path(database_path or default_v2_database_path()).expanduser().resolve()
    with connect_v2(resolved) as connection:
        for statement in V2_SCHEMA_STATEMENTS:
            connection.execute(statement)
    return resolved


def list_v2_tables(database_path: Path | str | None = None) -> list[str]:
    resolved = Path(database_path or default_v2_database_path()).expanduser().resolve()
    with connect_v2(resolved) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [str(row["name"]) for row in rows]


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def upsert_request_v2(
    *,
    request_id: str,
    status: str,
    fs_root_path: Path | str,
    panel_mode: str | None = None,
    bubble_mode: str | None = None,
    metadata: dict[str, object] | None = None,
    database_path: Path | str | None = None,
) -> None:
    resolved_root = Path(fs_root_path).expanduser().resolve()
    with connect_v2(database_path) as connection:
        existing = connection.execute(
            "SELECT metadata_json FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        existing_metadata: dict[str, Any] = {}
        if existing is not None and existing["metadata_json"]:
            try:
                existing_metadata = json.loads(str(existing["metadata_json"]))
            except json.JSONDecodeError:
                existing_metadata = {}
        merged_metadata = dict(existing_metadata)
        merged_metadata.update(metadata or {})
        connection.execute(
            """
            INSERT INTO requests (
                request_id, status, fs_root_path, panel_mode, bubble_mode,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(request_id) DO UPDATE SET
                status = excluded.status,
                fs_root_path = excluded.fs_root_path,
                panel_mode = COALESCE(excluded.panel_mode, requests.panel_mode),
                bubble_mode = COALESCE(excluded.bubble_mode, requests.bubble_mode),
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                request_id,
                status,
                str(resolved_root),
                panel_mode,
                bubble_mode,
                _json_dumps(merged_metadata),
            ),
        )


def insert_request_asset_v2(
    *,
    request_id: str,
    asset_type: str,
    rel_path: Path | str,
    panel_index: int | None = None,
    metadata: dict[str, object] | None = None,
    database_path: Path | str | None = None,
) -> int:
    relative_path = Path(rel_path)
    with connect_v2(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO request_assets (
                request_id, asset_type, rel_path, panel_index, metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                request_id,
                asset_type,
                str(relative_path),
                panel_index,
                _json_dumps(metadata or {}),
            ),
        )
        return int(cursor.lastrowid)


def replace_request_annotations_v2(
    *,
    request_id: str,
    panel_boxes: list[dict[str, object]] | None = None,
    panel_regions: dict[str, list[dict[str, object]]] | None = None,
    created_by: str = "override_ui",
    database_path: Path | str | None = None,
) -> dict[str, int]:
    panel_boxes = panel_boxes or []
    panel_regions = panel_regions or {}
    created_annotations = 0
    created_versions = 0

    with connect_v2(database_path) as connection:
        connection.execute(
            "UPDATE annotations SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE request_id = ?",
            (request_id,),
        )

        for item in panel_boxes:
            cursor = connection.execute(
                """
                INSERT INTO annotations (
                    request_id, asset_id, entity_type, label_type, role, current_version_id,
                    is_active, created_at, updated_at
                ) VALUES (?, NULL, 'panel', 'panel', ?, NULL, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (request_id, str(item.get("role") or "panel")),
            )
            annotation_id = int(cursor.lastrowid)
            version_cursor = connection.execute(
                """
                INSERT INTO annotation_versions (
                    annotation_id, data_json, created_by, source_type, is_manual_override,
                    is_active, notes, created_at
                ) VALUES (?, ?, ?, 'manual_override', 1, 1, ?, CURRENT_TIMESTAMP)
                """,
                (
                    annotation_id,
                    _json_dumps(
                        {
                            "index": item.get("index"),
                            "bbox": item.get("bbox"),
                            "points": item.get("points"),
                            "role": item.get("role") or "panel",
                        }
                    ),
                    created_by,
                    "panel override save",
                ),
            )
            version_id = int(version_cursor.lastrowid)
            connection.execute(
                "UPDATE annotations SET current_version_id = ?, updated_at = CURRENT_TIMESTAMP WHERE annotation_id = ?",
                (version_id, annotation_id),
            )
            created_annotations += 1
            created_versions += 1

        for panel_key, regions in panel_regions.items():
            for item in regions:
                cursor = connection.execute(
                    """
                    INSERT INTO annotations (
                        request_id, asset_id, entity_type, label_type, role, current_version_id,
                        is_active, created_at, updated_at
                    ) VALUES (?, NULL, 'region', ?, NULL, NULL, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (request_id, str(item.get("class_name") or "unknown")),
                )
                annotation_id = int(cursor.lastrowid)
                version_cursor = connection.execute(
                    """
                    INSERT INTO annotation_versions (
                        annotation_id, data_json, created_by, source_type, is_manual_override,
                        is_active, notes, created_at
                    ) VALUES (?, ?, ?, 'manual_override', 1, 1, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        annotation_id,
                        _json_dumps(
                            {
                                "panel_key": panel_key,
                                "class_name": item.get("class_name"),
                                "bbox": item.get("bbox"),
                            }
                        ),
                        created_by,
                        f"region override save for panel {panel_key}",
                    ),
                )
                version_id = int(version_cursor.lastrowid)
                connection.execute(
                    "UPDATE annotations SET current_version_id = ?, updated_at = CURRENT_TIMESTAMP WHERE annotation_id = ?",
                    (version_id, annotation_id),
                )
                created_annotations += 1
                created_versions += 1

    return {
        "annotation_count": created_annotations,
        "version_count": created_versions,
        "panel_box_count": len(panel_boxes),
        "panel_region_count": sum(len(items) for items in panel_regions.values()),
    }


def load_request_override_v2(
    request_id: str,
    *,
    database_path: Path | str | None = None,
) -> dict[str, Any] | None:
    with connect_v2(database_path) as connection:
        request_row = connection.execute(
            "SELECT metadata_json FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if request_row is None:
            return None

        metadata: dict[str, Any] = {}
        raw_metadata = request_row["metadata_json"]
        if raw_metadata:
            try:
                metadata = json.loads(str(raw_metadata))
            except json.JSONDecodeError:
                metadata = {}

        rows = connection.execute(
            """
            SELECT a.annotation_id, a.entity_type, a.label_type, a.role, av.data_json
            FROM annotations a
            JOIN annotation_versions av ON av.version_id = a.current_version_id
            WHERE a.request_id = ? AND a.is_active = 1
            ORDER BY a.annotation_id
            """,
            (request_id,),
        ).fetchall()

    panel_boxes: list[dict[str, Any]] = []
    panel_regions: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        try:
            payload = json.loads(str(row["data_json"]))
        except json.JSONDecodeError:
            continue

        if str(row["entity_type"]) == "panel":
            panel_boxes.append(
                {
                    "index": payload.get("index"),
                    "bbox": payload.get("bbox") or [],
                    "points": payload.get("points"),
                    "role": payload.get("role") or row["role"] or "panel",
                }
            )
        elif str(row["entity_type"]) == "region":
            panel_key = str(payload.get("panel_key") or "0")
            panel_regions.setdefault(panel_key, []).append(
                {
                    "class_name": payload.get("class_name") or row["label_type"],
                    "bbox": payload.get("bbox") or [],
                }
            )

    panel_boxes.sort(key=lambda item: int(item.get("index") or 0))

    return {
        "request_id": request_id,
        "overrides": metadata.get("overrides") or {},
        "panel_boxes": panel_boxes,
        "panel_regions": panel_regions,
    }
