from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from train_bubble_detector import COCODetectionDataset, build_model, collate_fn, select_device

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    COCO = None
    COCOeval = None
    PYCOCO_IMPORT_ERROR = exc
else:
    PYCOCO_IMPORT_ERROR = None


def _safe_read_metrics(metrics_path: Path) -> dict | None:
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_latest_metrics(output_dir: Path, detector_prefix: str) -> dict | None:
    latest_payload: dict | None = None
    latest_mtime = -1.0
    for metrics_path in output_dir.glob("metrics_*.json"):
        payload = _safe_read_metrics(metrics_path)
        if not payload:
            continue
        weights = str(payload.get("weights", ""))
        if detector_prefix not in Path(weights).name:
            continue
        mtime = metrics_path.stat().st_mtime
        if mtime > latest_mtime:
            latest_payload = payload
            latest_mtime = mtime
    return latest_payload


def _format_metric_percent(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _relative_or_abs(path_str: str, root: Path) -> str:
    path = Path(path_str)
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _render_detector_section(title: str, payload: dict | None, root: Path) -> list[str]:
    lines = [f"## {title}", ""]
    if not payload:
        lines.extend(["No evaluation snapshot available yet.", ""])
        return lines

    metrics = payload.get("metrics", {})
    weights = str(payload.get("weights", ""))
    dataset_root = str(payload.get("dataset_root", ""))
    split = str(payload.get("split", ""))
    run_name = str(payload.get("run_name", ""))
    metrics_source = root / "outputs" / "eval" / f"metrics_{run_name}.json"

    lines.extend(
        [
            "Model:",
            f"- `{_relative_or_abs(weights, root)}`",
            "",
            "Dataset:",
            f"- `{_relative_or_abs(dataset_root, root)}`",
            "",
            f"Latest split: `{split}`",
            "",
            "Metrics:",
            f"- precision: `{_format_metric_percent(metrics.get('precision'))}`",
            f"- recall: `{_format_metric_percent(metrics.get('recall'))}`",
            f"- mAP@0.50: `{_format_metric_percent(metrics.get('mAP_50'))}`",
            f"- mAP@0.50:0.95: `{_format_metric_percent(metrics.get('mAP_50_95'))}`",
            "",
            "Source:",
            f"- `{_relative_or_abs(str(metrics_source), root)}`",
            "",
        ]
    )
    return lines


def update_current_metrics_snapshot(output_dir: Path) -> Path:
    root = output_dir.resolve().parents[1]
    panel_payload = _find_latest_metrics(output_dir, "panel_detector")
    bubble_payload = _find_latest_metrics(output_dir, "bubble_detector")

    lines = [
        "# Current Model Metrics",
        "",
        f"Date: {datetime.now().date().isoformat()}",
        "",
        "This file is the current benchmark snapshot for the latest evaluation runs we are using in the pipeline.",
        "",
    ]
    lines.extend(_render_detector_section("Panel Detector", panel_payload, root))
    lines.extend(_render_detector_section("Bubble Detector", bubble_payload, root))
    lines.extend(
        [
            "## Reading",
            "",
            "- This file updates automatically after panel or bubble evaluation runs.",
            "- Treat panel and bubble sections as the latest available snapshot, not a historical log.",
        ]
    )

    snapshot_path = output_dir / "CURRENT_MODEL_METRICS.md"
    snapshot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snapshot_path


def load_checkpoint(weights_path: Path, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    class_names = checkpoint.get("class_names", ["background", "Bubble with Text"]) if isinstance(checkpoint, dict) else ["background", "Bubble with Text"]
    model = build_model(num_classes=len(class_names)).to(device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, class_names


def run_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    max_detections: int,
    class_names: list[str],
) -> list[dict[str, float | int | list[float]]]:
    predictions: list[dict[str, float | int | list[float]]] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].item())
                boxes = output.get("boxes")
                scores = output.get("scores")
                labels = output.get("labels")
                if boxes is None or scores is None or labels is None:
                    continue
                kept = 0
                for box_tensor, score_tensor, label_tensor in zip(boxes, scores, labels):
                    score = float(score_tensor.item())
                    label = int(label_tensor.item())
                    if label <= 0 or label >= len(class_names) or score < score_threshold:
                        continue
                    x1, y1, x2, y2 = box_tensor.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w < 1.0 or h < 1.0:
                        continue
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": label,
                            "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(w), 3), round(float(h), 3)],
                            "score": round(score, 6),
                        }
                    )
                    kept += 1
                    if kept >= max_detections:
                        break
    return predictions


def evaluate_coco(annotation_path: Path, predictions_path: Path) -> dict[str, float]:
    if COCO is None or COCOeval is None:
        raise RuntimeError(
            "pycocotools is required for evaluation. Install requirements first. "
            f"Import error: {PYCOCO_IMPORT_ERROR}"
        )
    coco_gt = COCO(str(annotation_path))
    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    if not predictions:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
        }
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "precision": float(stats[1]),
        "recall": float(stats[8]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local bubble detector with precision, recall, and mAP.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the COCO dataset root.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained detector checkpoint.")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=8)
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

    print(json.dumps(metrics_payload, indent=2))
    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Snapshot updated at {snapshot_path}")


if __name__ == "__main__":
    main()
