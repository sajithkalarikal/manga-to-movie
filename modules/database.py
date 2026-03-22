from __future__ import annotations

import json
import mimetypes
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import Settings


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS image_path (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_key TEXT NOT NULL UNIQUE,
        sha256 TEXT NOT NULL UNIQUE,
        canonical_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        width INTEGER NOT NULL DEFAULT 0,
        height INTEGER NOT NULL DEFAULT 0,
        byte_size INTEGER NOT NULL DEFAULT 0,
        mime_type TEXT,
        source_type TEXT NOT NULL DEFAULT 'dataset',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets_split (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        dataset_name TEXT NOT NULL,
        split_name TEXT NOT NULL,
        split_version TEXT,
        annotation_status TEXT NOT NULL DEFAULT 'pending',
        review_status TEXT NOT NULL DEFAULT 'unreviewed',
        is_current INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_path(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_datasets_split_current
    ON datasets_split(image_id, dataset_name)
    WHERE is_current = 1
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_datasets_split_lookup
    ON datasets_split(dataset_name, split_name, is_current)
    """,
    """
    CREATE TABLE IF NOT EXISTS panels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        panel_index INTEGER NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        width REAL NOT NULL,
        height REAL NOT NULL,
        polygon_json TEXT,
        source TEXT NOT NULL,
        confidence REAL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_by TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_path(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_panels_image_active
    ON panels(image_id, is_active)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_panels_image_order
    ON panels(image_id, panel_index, is_active)
    """,
    """
    CREATE TABLE IF NOT EXISTS bubbles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        panel_id INTEGER,
        bubble_type TEXT NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        width REAL NOT NULL,
        height REAL NOT NULL,
        polygon_json TEXT,
        ocr_text TEXT,
        source TEXT NOT NULL,
        confidence REAL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_by TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_path(id) ON DELETE CASCADE,
        FOREIGN KEY(panel_id) REFERENCES panels(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_bubbles_image_active
    ON bubbles(image_id, is_active)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_bubbles_panel_active
    ON bubbles(panel_id, is_active)
    """,
    """
    CREATE TABLE IF NOT EXISTS training_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_key TEXT NOT NULL UNIQUE,
        model_type TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        train_split_version TEXT,
        epochs INTEGER NOT NULL DEFAULT 0,
        batch_size INTEGER NOT NULL DEFAULT 0,
        learning_rate REAL,
        train_image_count INTEGER NOT NULL DEFAULT 0,
        train_annotation_count INTEGER NOT NULL DEFAULT 0,
        best_checkpoint_path TEXT,
        final_checkpoint_path TEXT,
        best_train_loss REAL,
        status TEXT NOT NULL,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT NOT NULL,
        finished_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_result_id INTEGER,
        dataset_name TEXT NOT NULL,
        split_name TEXT NOT NULL,
        image_count INTEGER NOT NULL DEFAULT 0,
        annotation_count INTEGER NOT NULL DEFAULT 0,
        loss REAL,
        map_50 REAL,
        map_50_95 REAL,
        precision_score REAL,
        recall_score REAL,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        evaluated_at TEXT NOT NULL,
        FOREIGN KEY(training_result_id) REFERENCES training_results(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS derived_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        panel_id INTEGER,
        asset_type TEXT NOT NULL,
        derived_path TEXT NOT NULL,
        sha256 TEXT,
        byte_size INTEGER NOT NULL DEFAULT 0,
        generator TEXT,
        is_stale INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_path(id) ON DELETE CASCADE,
        FOREIGN KEY(panel_id) REFERENCES panels(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_derived_assets_unique
    ON derived_assets(image_id, panel_id, asset_type, derived_path)
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL UNIQUE,
        job_type TEXT NOT NULL,
        image_id INTEGER,
        status TEXT NOT NULL,
        filename TEXT,
        input_path TEXT,
        output_path TEXT,
        error_message TEXT,
        attempt INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 0,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_path(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        status TEXT,
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(request_id) REFERENCES jobs(request_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_job_events_request_id
    ON job_events(request_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS annotation_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,
        dataset_name TEXT NOT NULL,
        split_name TEXT NOT NULL,
        event_type TEXT NOT NULL,
        actor TEXT,
        file_name TEXT,
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_path(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_annotation_events_lookup
    ON annotation_events(dataset_name, split_name, created_at)
    """,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _normalize_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _guess_mime_type(path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type


@contextmanager
def connect(settings: Settings):
    database_path = settings.sql_database_path.expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(settings: Settings) -> None:
    with connect(settings) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)


def register_image_path(
    settings: Settings,
    image_path: Path | str,
    *,
    asset_key: str | None = None,
    source_type: str = "dataset",
    width: int = 0,
    height: int = 0,
) -> int:
    initialize_database(settings)
    resolved_path = _normalize_path(image_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {resolved_path}")

    sha_value, byte_size = _hash_file(resolved_path)
    now = _utcnow()
    normalized_asset_key = asset_key or sha_value

    with connect(settings) as connection:
        existing = connection.execute(
            "SELECT id, width, height FROM image_path WHERE sha256 = ?",
            (sha_value,),
        ).fetchone()
        existing_width = int(existing["width"]) if existing is not None else 0
        existing_height = int(existing["height"]) if existing is not None else 0
        connection.execute(
            """
            INSERT INTO image_path (
                asset_key, sha256, canonical_path, file_name, width, height, byte_size,
                mime_type, source_type, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                asset_key = excluded.asset_key,
                canonical_path = excluded.canonical_path,
                file_name = excluded.file_name,
                width = CASE WHEN excluded.width > 0 THEN excluded.width ELSE image_path.width END,
                height = CASE WHEN excluded.height > 0 THEN excluded.height ELSE image_path.height END,
                byte_size = excluded.byte_size,
                mime_type = excluded.mime_type,
                source_type = excluded.source_type,
                is_active = 1,
                updated_at = excluded.updated_at
            """,
            (
                normalized_asset_key,
                sha_value,
                str(resolved_path),
                resolved_path.name,
                width or existing_width,
                height or existing_height,
                byte_size,
                _guess_mime_type(resolved_path),
                source_type,
                now,
                now,
            ),
        )
        row = connection.execute(
            "SELECT id FROM image_path WHERE sha256 = ?",
            (sha_value,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to register image path: {resolved_path}")
        return int(row["id"])


def assign_dataset_split(
    settings: Settings,
    *,
    image_id: int,
    dataset_name: str,
    split_name: str,
    split_version: str | None = None,
    annotation_status: str = "pending",
    review_status: str = "unreviewed",
) -> None:
    initialize_database(settings)
    now = _utcnow()
    with connect(settings) as connection:
        connection.execute(
            """
            UPDATE datasets_split
            SET is_current = 0, updated_at = ?
            WHERE image_id = ? AND dataset_name = ? AND is_current = 1
            """,
            (now, image_id, dataset_name),
        )
        connection.execute(
            """
            INSERT INTO datasets_split (
                image_id, dataset_name, split_name, split_version, annotation_status,
                review_status, is_current, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                image_id,
                dataset_name,
                split_name,
                split_version,
                annotation_status,
                review_status,
                now,
                now,
            ),
        )


def _round_number(value: Any) -> float:
    return round(float(value), 3)


def _normalize_polygon(points: Any) -> str | None:
    if not isinstance(points, list) or len(points) < 3:
        return None
    normalized: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            continue
        normalized.append([_round_number(point[0]), _round_number(point[1])])
    return _json_dumps(normalized) if len(normalized) >= 3 else None


def replace_panels(
    settings: Settings,
    *,
    image_id: int,
    panels: Sequence[dict[str, Any]],
    source: str,
    created_by: str | None = None,
) -> list[int]:
    initialize_database(settings)
    now = _utcnow()
    created_ids: list[int] = []
    with connect(settings) as connection:
        connection.execute(
            "UPDATE panels SET is_active = 0, updated_at = ? WHERE image_id = ? AND is_active = 1",
            (now, image_id),
        )
        connection.execute(
            "UPDATE derived_assets SET is_stale = 1, updated_at = ? WHERE image_id = ?",
            (now, image_id),
        )
        for index, panel in enumerate(panels, start=1):
            bbox = panel.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("Panel bbox must be a list of four numbers.")
            cursor = connection.execute(
                """
                INSERT INTO panels (
                    image_id, panel_index, x, y, width, height, polygon_json, source,
                    confidence, is_active, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    image_id,
                    int(panel.get("panel_index", index)),
                    _round_number(bbox[0]),
                    _round_number(bbox[1]),
                    _round_number(bbox[2]),
                    _round_number(bbox[3]),
                    _normalize_polygon(panel.get("points") or panel.get("polygon")),
                    source,
                    float(panel["confidence"]) if panel.get("confidence") is not None else None,
                    created_by,
                    now,
                    now,
                ),
            )
            created_ids.append(int(cursor.lastrowid))
    return created_ids


def replace_bubbles(
    settings: Settings,
    *,
    image_id: int,
    bubbles: Sequence[dict[str, Any]],
    source: str,
    created_by: str | None = None,
) -> list[int]:
    initialize_database(settings)
    now = _utcnow()
    created_ids: list[int] = []
    with connect(settings) as connection:
        connection.execute(
            "UPDATE bubbles SET is_active = 0, updated_at = ? WHERE image_id = ? AND is_active = 1",
            (now, image_id),
        )
        for bubble in bubbles:
            bbox = bubble.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("Bubble bbox must be a list of four numbers.")
            cursor = connection.execute(
                """
                INSERT INTO bubbles (
                    image_id, panel_id, bubble_type, x, y, width, height, polygon_json,
                    ocr_text, source, confidence, is_active, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    image_id,
                    bubble.get("panel_id"),
                    str(bubble.get("bubble_type")),
                    _round_number(bbox[0]),
                    _round_number(bbox[1]),
                    _round_number(bbox[2]),
                    _round_number(bbox[3]),
                    _normalize_polygon(bubble.get("points") or bubble.get("polygon")),
                    bubble.get("ocr_text"),
                    source,
                    float(bubble["confidence"]) if bubble.get("confidence") is not None else None,
                    created_by,
                    now,
                    now,
                ),
            )
            created_ids.append(int(cursor.lastrowid))
    return created_ids


def register_detected_panels(
    settings: Settings,
    *,
    image_id: int,
    panels: Sequence[Any],
    generator: str,
    created_by: str | None = None,
) -> list[int]:
    panel_rows = []
    for index, panel in enumerate(panels, start=1):
        bbox = getattr(panel, "bbox", None)
        if bbox is None and isinstance(panel, dict):
            bbox = panel.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Detected panel bbox must be a list of four numbers.")
        x1, y1, x2, y2 = [float(value) for value in bbox]
        panel_rows.append(
            {
                "panel_index": getattr(panel, "index", None) or (panel.get("panel_index") if isinstance(panel, dict) else index) or index,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "confidence": getattr(panel, "confidence", None) if not isinstance(panel, dict) else panel.get("confidence"),
            }
        )

    panel_ids = replace_panels(
        settings,
        image_id=image_id,
        panels=panel_rows,
        source="detected",
        created_by=created_by,
    )
    for panel_id, panel in zip(panel_ids, panels):
        image_path = getattr(panel, "image_path", None)
        if image_path is None and isinstance(panel, dict):
            image_path = panel.get("image_path")
        if image_path is None:
            continue
        upsert_derived_asset(
            settings,
            image_id=image_id,
            panel_id=panel_id,
            asset_type="panel_crop",
            derived_path=image_path,
            generator=generator,
            is_stale=False,
        )
    return panel_ids


def upsert_derived_asset(
    settings: Settings,
    *,
    image_id: int,
    panel_id: int | None,
    asset_type: str,
    derived_path: Path | str,
    generator: str | None = None,
    is_stale: bool = False,
) -> None:
    initialize_database(settings)
    now = _utcnow()
    resolved_path = _normalize_path(derived_path)
    sha_value = None
    byte_size = 0
    if resolved_path.exists():
        sha_value, byte_size = _hash_file(resolved_path)
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO derived_assets (
                image_id, panel_id, asset_type, derived_path, sha256, byte_size,
                generator, is_stale, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id, panel_id, asset_type, derived_path) DO UPDATE SET
                sha256 = excluded.sha256,
                byte_size = excluded.byte_size,
                generator = excluded.generator,
                is_stale = excluded.is_stale,
                updated_at = excluded.updated_at
            """,
            (
                image_id,
                panel_id,
                asset_type,
                str(resolved_path),
                sha_value,
                byte_size,
                generator,
                1 if is_stale else 0,
                now,
                now,
            ),
        )


def upsert_job(
    settings: Settings,
    payload: dict[str, Any],
    *,
    image_id: int | None = None,
    job_type: str = "generate_video",
) -> None:
    initialize_database(settings)
    request_id = str(payload["request_id"])
    now = _utcnow()

    with connect(settings) as connection:
        existing = connection.execute(
            "SELECT id, created_at, image_id, job_type FROM jobs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing is not None else now
        persisted_image_id = image_id if image_id is not None else (int(existing["image_id"]) if existing and existing["image_id"] is not None else None)
        persisted_job_type = job_type or (str(existing["job_type"]) if existing is not None else "generate_video")
        connection.execute(
            """
            INSERT INTO jobs (
                request_id, job_type, image_id, status, filename, input_path, output_path,
                error_message, attempt, max_attempts, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                job_type = excluded.job_type,
                image_id = COALESCE(excluded.image_id, jobs.image_id),
                status = excluded.status,
                filename = COALESCE(excluded.filename, jobs.filename),
                input_path = COALESCE(excluded.input_path, jobs.input_path),
                output_path = COALESCE(excluded.output_path, jobs.output_path),
                error_message = excluded.error_message,
                attempt = excluded.attempt,
                max_attempts = excluded.max_attempts,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (
                request_id,
                persisted_job_type,
                persisted_image_id,
                str(payload.get("status", "unknown")),
                payload.get("filename"),
                payload.get("upload_path") or payload.get("input_path"),
                payload.get("output_path") or payload.get("video_url") or payload.get("metadata_url"),
                payload.get("error"),
                int(payload.get("attempt", 0) or 0),
                int(payload.get("max_attempts", 0) or 0),
                _json_dumps(payload),
                created_at,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO job_events (request_id, event_type, status, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                "status_update",
                str(payload.get("status", "unknown")),
                _json_dumps(payload),
                now,
            ),
        )


def append_job_event(
    settings: Settings,
    *,
    request_id: str,
    event_type: str,
    status: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    initialize_database(settings)
    now = _utcnow()
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO job_events (request_id, event_type, status, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                event_type,
                status,
                _json_dumps(details or {}),
                now,
            ),
        )


def record_training_result(
    settings: Settings,
    *,
    run_key: str,
    model_type: str,
    dataset_name: str,
    train_split_version: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float | None,
    train_image_count: int,
    train_annotation_count: int,
    best_checkpoint_path: Path | str | None,
    final_checkpoint_path: Path | str | None,
    best_train_loss: float | None,
    status: str,
    metrics: dict[str, Any],
    started_at: str,
    finished_at: str | None,
) -> int:
    initialize_database(settings)
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO training_results (
                run_key, model_type, dataset_name, train_split_version, epochs, batch_size,
                learning_rate, train_image_count, train_annotation_count, best_checkpoint_path,
                final_checkpoint_path, best_train_loss, status, metrics_json, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_key) DO UPDATE SET
                model_type = excluded.model_type,
                dataset_name = excluded.dataset_name,
                train_split_version = excluded.train_split_version,
                epochs = excluded.epochs,
                batch_size = excluded.batch_size,
                learning_rate = excluded.learning_rate,
                train_image_count = excluded.train_image_count,
                train_annotation_count = excluded.train_annotation_count,
                best_checkpoint_path = excluded.best_checkpoint_path,
                final_checkpoint_path = excluded.final_checkpoint_path,
                best_train_loss = excluded.best_train_loss,
                status = excluded.status,
                metrics_json = excluded.metrics_json,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at
            """,
            (
                run_key,
                model_type,
                dataset_name,
                train_split_version,
                epochs,
                batch_size,
                learning_rate,
                train_image_count,
                train_annotation_count,
                str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                str(final_checkpoint_path) if final_checkpoint_path is not None else None,
                best_train_loss,
                status,
                _json_dumps(metrics),
                started_at,
                finished_at,
            ),
        )
        row = connection.execute(
            "SELECT id FROM training_results WHERE run_key = ?",
            (run_key,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to record training result for run_key={run_key}")
        return int(row["id"])


def record_validation_result(
    settings: Settings,
    *,
    training_result_id: int | None,
    dataset_name: str,
    split_name: str,
    image_count: int,
    annotation_count: int,
    loss: float | None,
    map_50: float | None,
    map_50_95: float | None,
    precision_score: float | None,
    recall_score: float | None,
    metrics: dict[str, Any],
    evaluated_at: str | None = None,
) -> None:
    initialize_database(settings)
    now = evaluated_at or _utcnow()
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO validation_results (
                training_result_id, dataset_name, split_name, image_count, annotation_count,
                loss, map_50, map_50_95, precision_score, recall_score, metrics_json, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                training_result_id,
                dataset_name,
                split_name,
                image_count,
                annotation_count,
                loss,
                map_50,
                map_50_95,
                precision_score,
                recall_score,
                _json_dumps(metrics),
                now,
            ),
        )


def append_annotation_event(
    settings: Settings,
    *,
    image_id: int | None,
    dataset_name: str,
    split_name: str,
    event_type: str,
    actor: str | None = None,
    file_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    initialize_database(settings)
    now = _utcnow()
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO annotation_events (
                image_id, dataset_name, split_name, event_type, actor, file_name, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                dataset_name,
                split_name,
                event_type,
                actor,
                file_name,
                _json_dumps(details or {}),
                now,
            ),
        )


def sync_dataset_split_from_coco(settings: Settings, dataset_root: Path | str) -> None:
    initialize_database(settings)
    dataset_root = _normalize_path(dataset_root)
    for split_name in ("train", "valid", "test"):
        annotation_path = dataset_root / split_name / "_annotations.coco.json"
        if not annotation_path.exists():
            continue
        payload = json.loads(annotation_path.read_text("utf-8"))
        split_dir = annotation_path.parent
        for image in payload.get("images", []):
            file_name = str(image["file_name"])
            image_id = register_image_path(
                settings,
                split_dir / file_name,
                asset_key=f"{dataset_root.name}:{file_name}",
                source_type="dataset",
                width=int(image.get("width", 0) or 0),
                height=int(image.get("height", 0) or 0),
            )
            assign_dataset_split(
                settings,
                image_id=image_id,
                dataset_name=dataset_root.name,
                split_name="validation" if split_name == "valid" else split_name,
                split_version=None,
                annotation_status="pending",
                review_status="unreviewed",
            )


def store_annotation_snapshot(
    settings: Settings,
    *,
    dataset_root: Path | str,
    split_name: str,
    file_name: str,
    width: int,
    height: int,
    annotations: Sequence[dict[str, Any]],
    created_by: str | None = None,
) -> int:
    normalized_split = "validation" if split_name == "valid" else split_name
    dataset_root = _normalize_path(dataset_root)
    image_id = register_image_path(
        settings,
        dataset_root / split_name / file_name,
        asset_key=f"{dataset_root.name}:{file_name}",
        source_type="dataset",
        width=width,
        height=height,
    )
    assign_dataset_split(
        settings,
        image_id=image_id,
        dataset_name=dataset_root.name,
        split_name=normalized_split,
        annotation_status="submitted",
        review_status="approved",
    )

    panel_items = [
        {
            "panel_index": index,
            "bbox": item.get("bbox"),
            "points": item.get("points"),
        }
        for index, item in enumerate(annotations, start=1)
        if str(item.get("class_name", "")).strip() == "panel"
    ]
    bubble_items = [
        {
            "bubble_type": str(item.get("class_name", "")).strip(),
            "bbox": item.get("bbox"),
            "points": item.get("points"),
        }
        for item in annotations
        if str(item.get("class_name", "")).strip() in {"speech_bubble", "narration_box", "sfx"}
    ]
    replace_panels(settings, image_id=image_id, panels=panel_items, source="manual", created_by=created_by)
    replace_bubbles(settings, image_id=image_id, bubbles=bubble_items, source="manual", created_by=created_by)
    return image_id
