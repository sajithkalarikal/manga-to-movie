from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import Settings

TARGET_CLASSES = ["panel", "speech_bubble", "narration_box", "sfx"]
SOURCE_TO_TARGET_CLASS = {
    "Bubble with Text": "speech_bubble",
    "objects": "speech_bubble",
}
VALID_SPLITS = {"train", "valid", "test"}


def _ensure_split(split: str) -> str:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    return split


def _normalize_dataset_key(value: str) -> str:
    return value.strip()


def available_datasets(settings: Settings) -> list[dict[str, str]]:
    roots: dict[str, Path] = {}
    default_root = settings.annotation_dataset_root.resolve()
    roots[default_root.name] = default_root

    search_roots = [
        default_root.parent,
        (settings.annotation_output_dir / "exported_coco").resolve(),
    ]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in sorted(search_root.iterdir()):
            if not candidate.is_dir():
                continue
            if not ((candidate / "train").is_dir() and (candidate / "valid").is_dir() and (candidate / "test").is_dir()):
                continue
            key = candidate.name
            if key in roots and roots[key] != candidate.resolve():
                key = str(candidate.resolve())
            roots.setdefault(key, candidate.resolve())

    return [
        {
            "key": key,
            "name": key,
            "path": str(path),
        }
        for key, path in sorted(roots.items(), key=lambda item: item[0].lower())
    ]


def resolve_dataset_root(settings: Settings, dataset_key: str | None = None) -> Path:
    datasets = {item["key"]: Path(item["path"]) for item in available_datasets(settings)}
    if dataset_key:
        key = _normalize_dataset_key(dataset_key)
        if key not in datasets:
            raise ValueError(f"Unsupported dataset: {dataset_key}")
        return datasets[key]
    return settings.annotation_dataset_root.resolve()


def _dataset_storage_slug(dataset_root: Path) -> str:
    return str(dataset_root.resolve()).replace("/", "_")


def _legacy_dataset_storage_slug(dataset_root: Path) -> str:
    return dataset_root.name.replace("/", "_")


def _annotation_json_path(settings: Settings, split: str, dataset_key: str | None = None) -> Path:
    split = _ensure_split(split)
    return resolve_dataset_root(settings, dataset_key) / split / "_annotations.coco.json"


def _split_dir(settings: Settings, split: str, dataset_key: str | None = None) -> Path:
    split = _ensure_split(split)
    return resolve_dataset_root(settings, dataset_key) / split


def _overrides_dir(settings: Settings, split: str, dataset_key: str | None = None) -> Path:
    split = _ensure_split(split)
    dataset_root = resolve_dataset_root(settings, dataset_key)
    path = settings.annotation_output_dir / "overrides" / _dataset_storage_slug(dataset_root) / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def _override_path(settings: Settings, split: str, file_name: str, dataset_key: str | None = None) -> Path:
    safe_name = file_name.replace("/", "_")
    return _overrides_dir(settings, split, dataset_key) / f"{safe_name}.json"


def _existing_override_path(settings: Settings, split: str, file_name: str, dataset_key: str | None = None) -> Path | None:
    dataset_root = resolve_dataset_root(settings, dataset_key)
    safe_name = file_name.replace("/", "_")
    candidates = [
        settings.annotation_output_dir / "overrides" / _dataset_storage_slug(dataset_root) / split / f"{safe_name}.json",
        settings.annotation_output_dir / "overrides" / _legacy_dataset_storage_slug(dataset_root) / split / f"{safe_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_coco(settings: Settings, split: str, dataset_key: str | None = None) -> dict[str, Any]:
    annotation_path = _annotation_json_path(settings, split, dataset_key)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    return json.loads(annotation_path.read_text("utf-8"))


def _effective_annotations(
    settings: Settings,
    *,
    dataset_key: str | None,
    split: str,
    file_name: str,
    image_id: int,
    source_annotations: list[dict[str, Any]],
    categories: dict[int, str],
) -> tuple[list[dict[str, Any]], str]:
    override_path = _existing_override_path(settings, split, file_name, dataset_key)
    if override_path is not None:
        override_payload = json.loads(override_path.read_text("utf-8"))
        return list(override_payload.get("annotations", [])), "override"

    return (
        [
            {
                "id": str(item["id"]),
                "class_name": SOURCE_TO_TARGET_CLASS.get(categories.get(int(item["category_id"]), ""), "speech_bubble"),
                "bbox": [float(v) for v in item["bbox"]],
                "points": None,
            }
            for item in source_annotations
            if int(item["image_id"]) == image_id
        ],
        "source",
    )


def list_dataset_images(settings: Settings, split: str, offset: int = 0, limit: int = 50, dataset_key: str | None = None) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(settings, dataset_key)
    payload = _load_coco(settings, split, dataset_key)
    images = payload.get("images", [])
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for item in payload.get("annotations", []):
        annotations_by_image.setdefault(int(item["image_id"]), []).append(item)

    total = len(images)
    page = images[offset : offset + limit]
    items: list[dict[str, Any]] = []
    for index, image in enumerate(page, start=offset):
        image_id = int(image["id"])
        source_annotations = annotations_by_image.get(image_id, [])
        override_exists = _existing_override_path(settings, split, str(image["file_name"]), dataset_root.name) is not None
        items.append(
            {
                "index": index,
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "source_annotation_count": len(source_annotations),
                "source_categories": sorted(
                    {
                        SOURCE_TO_TARGET_CLASS.get(categories.get(int(ann["category_id"]), ""), "speech_bubble")
                        for ann in source_annotations
                    }
                ),
                "override_exists": override_exists,
                "image_url": f"/dataset-images/{dataset_root.name}/{split}/{image['file_name']}",
            }
        )
    return {
        "dataset": dataset_root.name,
        "split": split,
        "offset": offset,
        "limit": limit,
        "total": total,
        "items": items,
        "classes": TARGET_CLASSES,
        "available_datasets": available_datasets(settings),
    }


def load_image_annotations(settings: Settings, split: str, index: int, dataset_key: str | None = None) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(settings, dataset_key)
    payload = _load_coco(settings, split, dataset_key)
    images = payload.get("images", [])
    if index < 0 or index >= len(images):
        raise IndexError(f"Image index out of range: {index}")

    image = images[index]
    image_id = int(image["id"])
    file_name = str(image["file_name"])
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    source_annotations = [
        item for item in payload.get("annotations", []) if int(item["image_id"]) == image_id
    ]

    annotations, annotation_source = _effective_annotations(
        settings,
        dataset_key=dataset_key,
        split=split,
        file_name=file_name,
        image_id=image_id,
        source_annotations=source_annotations,
        categories=categories,
    )

    return {
        "dataset": dataset_root.name,
        "split": split,
        "index": index,
        "image_id": image_id,
        "file_name": file_name,
        "width": int(image.get("width", 0)),
        "height": int(image.get("height", 0)),
        "image_url": f"/dataset-images/{dataset_root.name}/{split}/{file_name}",
        "classes": TARGET_CLASSES,
        "annotation_source": annotation_source,
        "annotations": annotations,
        "available_datasets": available_datasets(settings),
    }


def build_review_queue(settings: Settings, dataset_key: str | None = None) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(settings, dataset_key)
    split_rank = {"valid": 0, "test": 1, "train": 2}
    queue_items: list[dict[str, Any]] = []

    for split in ("valid", "test", "train"):
        payload = _load_coco(settings, split, dataset_root.name)
        images = payload.get("images", [])
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for item in payload.get("annotations", []):
            annotations_by_image.setdefault(int(item["image_id"]), []).append(item)

        for index, image in enumerate(images):
            image_id = int(image["id"])
            file_name = str(image["file_name"])
            source_annotations = annotations_by_image.get(image_id, [])
            effective_annotations, annotation_source = _effective_annotations(
                settings,
                dataset_key=dataset_root.name,
                split=split,
                file_name=file_name,
                image_id=image_id,
                source_annotations=source_annotations,
                categories=categories,
            )
            class_names = sorted({str(item.get("class_name", "")).strip() for item in effective_annotations if item.get("class_name")})
            annotation_count = len(effective_annotations)
            override_exists = annotation_source == "override"

            priority = 0
            reasons: list[str] = []

            if split in {"valid", "test"}:
                priority += 300
                reasons.append(f"{split} split")
            if not override_exists:
                priority += 160
                reasons.append("no saved manual correction yet")
            if "narration_box" in class_names:
                priority += 120
                reasons.append("contains narration_box")
            if len(class_names) >= 2:
                priority += 90
                reasons.append("multi-class page")
            if "sfx" in class_names:
                priority += 50
                reasons.append("contains sfx")
            if annotation_count == 0:
                priority += 40
                reasons.append("no effective annotations")
            if not reasons:
                reasons.append("general verification pass")

            queue_items.append(
                {
                    "split": split,
                    "index": index,
                    "image_id": image_id,
                    "file_name": file_name,
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                    "annotation_count": annotation_count,
                    "class_names": class_names,
                    "override_exists": override_exists,
                    "annotation_source": annotation_source,
                    "priority": priority,
                    "reasons": reasons,
                    "image_url": f"/dataset-images/{dataset_root.name}/{split}/{file_name}",
                }
            )

    queue_items.sort(
        key=lambda item: (
            -int(item["priority"]),
            split_rank.get(str(item["split"]), 99),
            0 if not bool(item["override_exists"]) else 1,
            str(item["file_name"]).lower(),
        )
    )

    for queue_index, item in enumerate(queue_items):
        item["queue_index"] = queue_index

    return {
        "dataset": dataset_root.name,
        "total": len(queue_items),
        "items": queue_items,
    }


def save_image_annotations(
    settings: Settings,
    *,
    dataset_key: str | None,
    split: str,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
) -> Path:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(annotations, start=1):
        class_name = str(item.get("class_name", "")).strip()
        if class_name not in TARGET_CLASSES:
            raise ValueError(f"Unsupported annotation class: {class_name}")
        points = item.get("points")
        normalized_points: list[list[float]] | None = None
        if points is not None:
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError("Polygon annotations must include at least three [x, y] points.")
            normalized_points = []
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each polygon point must be a list of two numbers.")
                px, py = [max(0.0, float(value)) for value in point]
                normalized_points.append([round(px, 3), round(py, 3)])
            xs = [point[0] for point in normalized_points]
            ys = [point[1] for point in normalized_points]
            x = min(xs)
            y = min(ys)
            w = max(xs) - x
            h = max(ys) - y
        else:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Each annotation bbox must be a list of four numbers.")
            x, y, w, h = [max(0.0, float(value)) for value in bbox]
        normalized.append(
            {
                "id": str(item.get("id") or index),
                "class_name": class_name,
                "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                "points": normalized_points,
            }
        )

    payload = {
        "dataset": resolve_dataset_root(settings, dataset_key).name,
        "split": split,
        "image_id": int(image_id),
        "file_name": file_name,
        "width": int(width),
        "height": int(height),
        "annotations": normalized,
    }
    output_path = _override_path(settings, split, file_name, dataset_key)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def export_annotated_dataset(
    settings: Settings,
    destination_root: Path,
    dataset_key: str | None = None,
    *,
    validated_only: bool = False,
    bubble_only: bool = False,
) -> list[Path]:
    dataset_root = resolve_dataset_root(settings, dataset_key)
    destination_root.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    target_categories = (
        [{"id": 1, "name": "speech_bubble", "supercategory": "text"}]
        if bubble_only
        else [
            {"id": 1, "name": "speech_bubble", "supercategory": "text"},
            {"id": 2, "name": "narration_box", "supercategory": "text"},
            {"id": 3, "name": "sfx", "supercategory": "text"},
        ]
    )
    class_to_id = {item["name"]: item["id"] for item in target_categories}

    for split in sorted(VALID_SPLITS):
        source = _load_coco(settings, split, dataset_root.name)
        split_destination = destination_root / split
        split_destination.mkdir(parents=True, exist_ok=True)

        source_dir = _split_dir(settings, split, dataset_root.name)
        annotations: list[dict[str, Any]] = []
        categories = {int(item["id"]): str(item["name"]) for item in source.get("categories", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for item in source.get("annotations", []):
            annotations_by_image.setdefault(int(item["image_id"]), []).append(item)

        annotation_id = 1
        exported_images: list[dict[str, Any]] = []
        for image in source.get("images", []):
            image_id = int(image["id"])
            file_name = str(image["file_name"])
            source_annotations = annotations_by_image.get(image_id, [])
            effective_annotations, annotation_source = _effective_annotations(
                settings,
                dataset_key=dataset_root.name,
                split=split,
                file_name=file_name,
                image_id=image_id,
                source_annotations=source_annotations,
                categories=categories,
            )
            if validated_only and annotation_source != "override":
                continue

            if bubble_only:
                effective_annotations = [
                    item for item in effective_annotations if str(item.get("class_name", "")).strip() == "speech_bubble"
                ]
            if bubble_only and not effective_annotations:
                continue

            src = source_dir / file_name
            dst = split_destination / file_name
            if not dst.exists():
                dst.symlink_to(src)
            exported_images.append(image)

            for item in effective_annotations:
                class_name = str(item.get("class_name", "")).strip()
                if class_name not in class_to_id:
                    continue
                x, y, w, h = [float(v) for v in item["bbox"]]
                segmentation = []
                points = item.get("points")
                if isinstance(points, list) and len(points) >= 3:
                    flattened: list[float] = []
                    for point in points:
                        if not isinstance(point, list) or len(point) != 2:
                            continue
                        flattened.extend([round(float(point[0]), 3), round(float(point[1]), 3)])
                    if len(flattened) >= 6:
                        segmentation = [flattened]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_to_id[class_name],
                        "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                        "area": round(w * h, 3),
                        "segmentation": segmentation,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        coco_payload = {
            "licenses": source.get("licenses", []),
            "info": {
                "description": (
                    "Validated-only speech bubble training dataset exported from local annotation workspace"
                    if validated_only and bubble_only
                    else "Annotated manga bubble dataset exported from local annotation workspace"
                ),
            },
            "images": exported_images,
            "annotations": annotations,
            "categories": target_categories,
        }
        annotation_path = split_destination / "_annotations.coco.json"
        annotation_path.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")
        exported_paths.append(annotation_path)

    return exported_paths
