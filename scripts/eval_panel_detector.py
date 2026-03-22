from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from config import get_settings
from modules.database import record_validation_result, sync_dataset_split_from_coco
from eval_bubble_detector import evaluate_coco, load_checkpoint, run_predictions, update_current_metrics_snapshot
from train_bubble_detector import COCODetectionDataset, collate_fn, select_device

from torch.utils.data import DataLoader
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local panel detector with precision, recall, and mAP.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the COCO dataset root.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained panel detector checkpoint.")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--run-name", type=str, default="", help="Optional suffix used to keep evaluation artifacts separate.")
    args = parser.parse_args()

    split_dir = args.dataset_root / args.split
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing detector weights: {args.weights}")

    settings = get_settings()
    sync_dataset_split_from_coco(settings, args.dataset_root)
    device = select_device(args.device)
    model, checkpoint, class_names = load_checkpoint(args.weights, device)
    dataset = COCODetectionDataset(split_dir, class_names[1:])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    predictions = run_predictions(model, loader, device, args.score_threshold, args.max_detections, class_names)
    default_run_name = f"{args.weights.stem}_{args.dataset_root.name}_{args.split}"
    run_name = (args.run_name.strip() or default_run_name).replace(" ", "_")
    predictions_path = args.output_dir / f"predictions_{run_name}.json"
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    metrics = evaluate_coco(annotation_path, predictions_path)
    elapsed_seconds = time.time() - started_at

    metrics_payload = {
        "run_name": run_name,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "weights": str(args.weights),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else None,
        "class_names": class_names,
        "score_threshold": args.score_threshold,
        "max_detections": args.max_detections,
        "device": str(device),
        "images": len(dataset),
        "predictions": len(predictions),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "metrics": metrics,
    }
    metrics_path = args.output_dir / f"metrics_{run_name}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    snapshot_path = update_current_metrics_snapshot(args.output_dir)
    record_validation_result(
        settings,
        training_result_id=None,
        dataset_name=args.dataset_root.name,
        split_name="validation" if args.split == "valid" else args.split,
        image_count=len(dataset),
        annotation_count=sum(len(items) for items in dataset.annotations_by_image.values()),
        loss=None,
        map_50=metrics.get("mAP_50"),
        map_50_95=metrics.get("mAP_50_95"),
        precision_score=metrics.get("precision"),
        recall_score=metrics.get("recall"),
        metrics=metrics_payload,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )

    print(json.dumps(metrics_payload, indent=2))
    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Snapshot updated at {snapshot_path}")


if __name__ == "__main__":
    main()
