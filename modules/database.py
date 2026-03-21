from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from config import Settings


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_key TEXT NOT NULL UNIQUE,
        dataset_name TEXT NOT NULL,
        root_path TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_splits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        split_name TEXT NOT NULL,
        annotation_path TEXT NOT NULL,
        image_count INTEGER NOT NULL DEFAULT 0,
        annotation_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(dataset_id, split_name),
        FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS image_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sha256 TEXT NOT NULL UNIQUE,
        canonical_path TEXT NOT NULL,
        byte_size INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS split_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        split_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        file_name TEXT NOT NULL,
        width INTEGER NOT NULL DEFAULT 0,
        height INTEGER NOT NULL DEFAULT 0,
        asset_id INTEGER,
        source_path TEXT NOT NULL,
        reference_mode TEXT NOT NULL DEFAULT 'copy',
        split_index INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(split_id, image_id),
        UNIQUE(split_id, file_name),
        FOREIGN KEY(split_id) REFERENCES dataset_splits(id) ON DELETE CASCADE,
        FOREIGN KEY(asset_id) REFERENCES image_assets(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        request_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        filename TEXT,
        upload_path TEXT,
        attempt INTEGER,
        max_attempts INTEGER,
        video_url TEXT,
        metadata_url TEXT,
        subtitles_url TEXT,
        error TEXT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(request_id) REFERENCES jobs(request_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detector_type TEXT NOT NULL,
        output_path TEXT NOT NULL,
        status TEXT NOT NULL,
        dataset_roots_json TEXT NOT NULL,
        train_size INTEGER NOT NULL DEFAULT 0,
        valid_size INTEGER NOT NULL DEFAULT 0,
        test_size INTEGER NOT NULL DEFAULT 0,
        best_valid_loss REAL,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def upsert_job(settings: Settings, payload: dict[str, Any]) -> None:
    initialize_database(settings)
    request_id = str(payload["request_id"])
    now = _utcnow()
    status = str(payload.get("status", "unknown"))
    serialized = _json_dumps(payload)

    with connect(settings) as connection:
        existing = connection.execute(
            "SELECT created_at FROM jobs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO jobs (
                request_id, status, filename, upload_path, attempt, max_attempts,
                video_url, metadata_url, subtitles_url, error, payload_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                status = excluded.status,
                filename = COALESCE(excluded.filename, jobs.filename),
                upload_path = COALESCE(excluded.upload_path, jobs.upload_path),
                attempt = excluded.attempt,
                max_attempts = excluded.max_attempts,
                video_url = excluded.video_url,
                metadata_url = excluded.metadata_url,
                subtitles_url = excluded.subtitles_url,
                error = excluded.error,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (
                request_id,
                status,
                payload.get("filename"),
                payload.get("upload_path"),
                payload.get("attempt"),
                payload.get("max_attempts"),
                payload.get("video_url"),
                payload.get("metadata_url"),
                payload.get("subtitles_url"),
                payload.get("error"),
                serialized,
                created_at,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO job_events (request_id, status, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (request_id, status, serialized, now),
        )


def register_coco_dataset(settings: Settings, dataset_root: Path) -> None:
    initialize_database(settings)
    dataset_root = dataset_root.expanduser().resolve()
    now = _utcnow()

    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO datasets (dataset_key, dataset_name, root_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_key) DO UPDATE SET
                dataset_name = excluded.dataset_name,
                root_path = excluded.root_path,
                updated_at = excluded.updated_at
            """,
            (dataset_root.name, dataset_root.name, str(dataset_root), now, now),
        )
        dataset_row = connection.execute(
            "SELECT id FROM datasets WHERE dataset_key = ?",
            (dataset_root.name,),
        ).fetchone()
        if dataset_row is None:
            raise RuntimeError(f"Failed to upsert dataset metadata for {dataset_root}")
        dataset_id = int(dataset_row["id"])

        for split in ("train", "valid", "test"):
            annotation_path = dataset_root / split / "_annotations.coco.json"
            if not annotation_path.exists():
                continue
            payload = json.loads(annotation_path.read_text("utf-8"))
            images = payload.get("images", [])
            annotations = payload.get("annotations", [])
            connection.execute(
                """
                INSERT INTO dataset_splits (
                    dataset_id, split_name, annotation_path, image_count, annotation_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, split_name) DO UPDATE SET
                    annotation_path = excluded.annotation_path,
                    image_count = excluded.image_count,
                    annotation_count = excluded.annotation_count,
                    updated_at = excluded.updated_at
                """,
                (
                    dataset_id,
                    split,
                    str(annotation_path),
                    len(images),
                    len(annotations),
                    now,
                    now,
                ),
            )
            split_row = connection.execute(
                "SELECT id FROM dataset_splits WHERE dataset_id = ? AND split_name = ?",
                (dataset_id, split),
            ).fetchone()
            if split_row is None:
                raise RuntimeError(f"Failed to upsert split metadata for {dataset_root}/{split}")
            split_id = int(split_row["id"])

            source_dir = dataset_root / split
            indexed_image_ids: set[int] = set()
            for split_index, image in enumerate(images):
                image_id = int(image["id"])
                indexed_image_ids.add(image_id)
                file_name = str(image["file_name"])
                source_path = source_dir / file_name
                width = int(image.get("width", 0) or 0)
                height = int(image.get("height", 0) or 0)
                asset_id = None
                reference_mode = "missing"

                if source_path.exists():
                    resolved = source_path.resolve()
                    sha_value, byte_size = _hash_file(resolved)
                    reference_mode = "symlink" if source_path.is_symlink() else "copy"
                    connection.execute(
                        """
                        INSERT INTO image_assets (sha256, canonical_path, byte_size, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(sha256) DO UPDATE SET
                            canonical_path = excluded.canonical_path,
                            byte_size = excluded.byte_size,
                            updated_at = excluded.updated_at
                        """,
                        (sha_value, str(resolved), byte_size, now, now),
                    )
                    asset_row = connection.execute(
                        "SELECT id FROM image_assets WHERE sha256 = ?",
                        (sha_value,),
                    ).fetchone()
                    asset_id = int(asset_row["id"]) if asset_row is not None else None

                connection.execute(
                    """
                    INSERT INTO split_images (
                        split_id, image_id, file_name, width, height, asset_id, source_path,
                        reference_mode, split_index, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(split_id, image_id) DO UPDATE SET
                        file_name = excluded.file_name,
                        width = excluded.width,
                        height = excluded.height,
                        asset_id = excluded.asset_id,
                        source_path = excluded.source_path,
                        reference_mode = excluded.reference_mode,
                        split_index = excluded.split_index,
                        updated_at = excluded.updated_at
                    """,
                    (
                        split_id,
                        image_id,
                        file_name,
                        width,
                        height,
                        asset_id,
                        str(source_path),
                        reference_mode,
                        split_index,
                        now,
                        now,
                    ),
                )

            stale_rows = connection.execute(
                "SELECT image_id FROM split_images WHERE split_id = ?",
                (split_id,),
            ).fetchall()
            stale_ids = [int(row["image_id"]) for row in stale_rows if int(row["image_id"]) not in indexed_image_ids]
            if stale_ids:
                connection.executemany(
                    "DELETE FROM split_images WHERE split_id = ? AND image_id = ?",
                    [(split_id, stale_id) for stale_id in stale_ids],
                )


def record_training_run(
    settings: Settings,
    *,
    detector_type: str,
    output_path: Path,
    status: str,
    dataset_roots: Iterable[Path],
    train_size: int,
    valid_size: int,
    test_size: int,
    best_valid_loss: float | None,
    metrics: dict[str, Any],
) -> None:
    initialize_database(settings)
    now = _utcnow()
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT INTO training_runs (
                detector_type, output_path, status, dataset_roots_json, train_size,
                valid_size, test_size, best_valid_loss, metrics_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detector_type,
                str(output_path),
                status,
                _json_dumps([str(path) for path in dataset_roots]),
                train_size,
                valid_size,
                test_size,
                best_valid_loss,
                _json_dumps(metrics),
                now,
                now,
            ),
        )
