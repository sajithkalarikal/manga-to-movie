from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F

from config import get_settings
from modules.database import record_training_result, sync_dataset_split_from_coco


DEFAULT_CLASS_NAME = "Bubble with Text"
CLASS_ALIASES = {
    "Bubble with Text": "speech_bubble",
    "objects": "speech_bubble",
}


class COCODetectionDataset(Dataset):
    def __init__(self, split_dir: Path, class_names: list[str]):
        self.split_dir = split_dir
        self.class_names = class_names
        self.class_to_label = {name: index for index, name in enumerate(class_names, start=1)}
        annotation_path = split_dir / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text("utf-8"))
        self.images = {int(item["id"]): item for item in payload["images"]}
        self.category_name_by_id = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
        self.annotations_by_image: dict[int, list[dict[str, object]]] = {}
        for item in payload["annotations"]:
            self.annotations_by_image.setdefault(int(item["image_id"]), []).append(item)

        valid_ids: list[int] = []
        for image_id, image_info in self.images.items():
            anns = self.annotations_by_image.get(image_id, [])
            if any(self._normalize_category_name(self.category_name_by_id.get(int(ann["category_id"]), "")) in self.class_to_label for ann in anns):
                valid_ids.append(image_id)
        self.image_ids = sorted(valid_ids)

    def _normalize_category_name(self, name: str) -> str:
        return CLASS_ALIASES.get(name, name)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_info = self.images[image_id]
        image_path = self.split_dir / str(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for ann in self.annotations_by_image.get(image_id, []):
            category_name = self._normalize_category_name(self.category_name_by_id.get(int(ann["category_id"]), ""))
            if category_name not in self.class_to_label:
                continue
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(width), x1 + float(w))
            y2 = min(float(height), y1 + float(h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_label[category_name])
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int = 2) -> torch.nn.Module:
    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )


def train_one_epoch(model, loader, optimizer, device, epoch: int, print_every: int = 20) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0
    start = time.time()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        batch_count += 1
        if batch_index % print_every == 0:
            avg = running_loss / batch_count
            print(f"[train] epoch={epoch} batch={batch_index}/{len(loader)} loss={avg:.4f}")

    avg_loss = running_loss / max(batch_count, 1)
    elapsed = time.time() - start
    print(f"[train] epoch={epoch} avg_loss={avg_loss:.4f} time={elapsed:.1f}s")
    return avg_loss


def validate(model, loader, device) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            running_loss += float(loss.item())
            batch_count += 1
    avg_loss = running_loss / max(batch_count, 1)
    print(f"[valid] avg_loss={avg_loss:.4f}")
    return avg_loss


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_limit_dataset(dataset: Dataset, limit: int, seed: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:limit])


def detect_class_names(dataset_roots: list[Path]) -> list[str]:
    names: list[str] = []
    for dataset_root in dataset_roots:
        annotation_path = dataset_root / "train" / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text("utf-8"))
        for item in payload.get("categories", []):
            normalized = CLASS_ALIASES.get(str(item["name"]), str(item["name"]))
            if normalized not in names:
                names.append(normalized)
    return names or [DEFAULT_CLASS_NAME]


def build_split_dataset(dataset_roots: list[Path], split: str, class_names: list[str]) -> Dataset:
    datasets = [COCODetectionDataset(dataset_root / split, class_names) for dataset_root in dataset_roots]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def count_split_annotations(dataset_roots: list[Path], split: str, class_names: list[str]) -> int:
    total = 0
    for dataset_root in dataset_roots:
        annotation_path = dataset_root / split / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text("utf-8"))
        allowed = set(class_names)
        categories = {
            int(item["id"]): CLASS_ALIASES.get(str(item["name"]), str(item["name"]))
            for item in payload.get("categories", [])
        }
        for item in payload.get("annotations", []):
            if categories.get(int(item["category_id"])) in allowed:
                total += 1
    return total


def load_init_weights(model: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("roi_heads.box_predictor.")
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Initialized from {weights_path}")
    print(f"Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local manga bubble detector from a COCO dataset.")
    parser.add_argument("--dataset-root", type=Path, nargs="+", required=True, help="One or more COCO dataset roots.")
    parser.add_argument("--output", type=Path, default=Path("models/bubble_detector.pt"), help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--limit-train", type=int, default=0, help="Limit training images for quick smoke tests.")
    parser.add_argument("--limit-valid", type=int, default=0, help="Limit validation images for quick smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume training from <output stem>.latest.pt if present.")
    parser.add_argument("--init-weights", type=Path, default=None, help="Optional checkpoint to initialize from without reusing the final predictor head.")
    parser.add_argument("--class-names", nargs="*", default=None, help="Optional explicit class names. Defaults to categories discovered in the dataset(s).")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    settings = get_settings()
    dataset_roots = [path.expanduser().resolve() for path in args.dataset_root]
    for dataset_root in dataset_roots:
        sync_dataset_split_from_coco(settings, dataset_root)
    class_names = args.class_names or detect_class_names(dataset_roots)
    print(f"Training classes: {class_names}")

    train_dataset = build_split_dataset(dataset_roots, "train", class_names)
    valid_dataset = build_split_dataset(dataset_roots, "valid", class_names)
    train_dataset = maybe_limit_dataset(train_dataset, args.limit_train, args.seed)
    valid_dataset = maybe_limit_dataset(valid_dataset, args.limit_valid, args.seed)
    print(f"Train images: {len(train_dataset)}")
    print(f"Valid images: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=len(class_names) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest_path = args.output.with_name(f"{args.output.stem}.latest{args.output.suffix}")
    start_epoch = 1
    best_valid_loss = float("inf")
    history: list[dict[str, float | int]] = []

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("valid_loss", float("inf")))
        saved_history = checkpoint.get("history")
        if isinstance(saved_history, list):
            history = [
                {
                    "epoch": int(item["epoch"]),
                    "train_loss": float(item["train_loss"]),
                    "valid_loss": float(item["valid_loss"]),
                }
                for item in saved_history
                if isinstance(item, dict)
                and item.get("epoch") is not None
                and item.get("train_loss") is not None
                and item.get("valid_loss") is not None
            ]
        print(
            f"Resuming from {latest_path} at epoch {start_epoch} "
            f"(last train_loss={checkpoint.get('train_loss')}, last valid_loss={checkpoint.get('valid_loss')})"
        )
    elif args.init_weights is not None:
        load_init_weights(model, args.init_weights.expanduser().resolve(), device)

    started_at = datetime.now(timezone.utc).isoformat()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        valid_loss = validate(model, valid_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "class_names": ["background", *class_names],
            "dataset_roots": [str(path) for path in dataset_roots],
            "history": history,
        }
        torch.save(checkpoint, latest_path)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, args.output)
            print(f"Saved best checkpoint to {args.output}")

    finished_at = datetime.now(timezone.utc).isoformat()
    train_annotation_count = count_split_annotations(dataset_roots, "train", class_names)
    record_training_result(
        settings,
        run_key=f"{args.output.resolve()}",
        model_type="bubble_detector",
        dataset_name=",".join(path.name for path in dataset_roots),
        train_split_version=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_image_count=len(train_dataset),
        train_annotation_count=train_annotation_count,
        best_checkpoint_path=args.output if args.output.exists() else None,
        final_checkpoint_path=latest_path if latest_path.exists() else None,
        best_train_loss=best_valid_loss if best_valid_loss != float("inf") else None,
        status="completed",
        metrics={
            "history": history,
            "class_names": class_names,
            "dataset_roots": [str(path) for path in dataset_roots],
            "device": str(device),
        },
        started_at=started_at,
        finished_at=finished_at,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
