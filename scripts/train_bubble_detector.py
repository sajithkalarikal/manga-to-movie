from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F


CLASS_NAME = "Bubble with Text"


class COCODetectionDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
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
            if any(self.category_name_by_id.get(int(ann["category_id"])) == CLASS_NAME for ann in anns):
                valid_ids.append(image_id)
        self.image_ids = sorted(valid_ids)

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
            if self.category_name_by_id.get(int(ann["category_id"])) != CLASS_NAME:
                continue
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(width), x1 + float(w))
            y2 = min(float(height), y1 + float(h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(1)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local manga bubble detector from a COCO dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the COCO dataset root.")
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
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    train_dataset = COCODetectionDataset(args.dataset_root / "train")
    valid_dataset = COCODetectionDataset(args.dataset_root / "valid")
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

    model = build_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        valid_loss = validate(model, valid_loader, device)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "class_names": ["background", CLASS_NAME],
        }
        latest_path = args.output.with_name(f"{args.output.stem}.latest{args.output.suffix}")
        torch.save(checkpoint, latest_path)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, args.output)
            print(f"Saved best checkpoint to {args.output}")

    print("Training complete.")


if __name__ == "__main__":
    main()
