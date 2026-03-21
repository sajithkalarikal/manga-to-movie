from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_bubble_detector import (
    COCODetectionDataset,
    build_model,
    build_split_dataset,
    collate_fn,
    detect_class_names,
    load_init_weights,
    maybe_limit_dataset,
    select_device,
    train_one_epoch,
    validate,
)


DEFAULT_PANEL_CLASS = "panel"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local manga panel detector from a COCO dataset.")
    parser.add_argument("--dataset-root", type=Path, nargs="+", required=True, help="One or more COCO dataset roots.")
    parser.add_argument("--output", type=Path, default=Path("models/panel_detector.pt"), help="Checkpoint output path.")
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
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional explicit panel class names. Defaults to dataset categories, normalized to at least ['panel'].",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    dataset_roots = [path.expanduser().resolve() for path in args.dataset_root]
    detected = args.class_names or detect_class_names(dataset_roots)
    class_names = [name for name in detected if name != "background"] or [DEFAULT_PANEL_CLASS]
    if DEFAULT_PANEL_CLASS not in class_names:
        class_names = [DEFAULT_PANEL_CLASS, *class_names]
    # Preserve order while removing duplicates such as repeated "panel" categories.
    class_names = list(dict.fromkeys(class_names))
    print(f"Training panel classes: {class_names}")

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
            "detector_type": "panel",
            "history": history,
        }
        torch.save(checkpoint, latest_path)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, args.output)
            print(f"Saved best checkpoint to {args.output}")

    print("Panel detector training complete.")


if __name__ == "__main__":
    main()
