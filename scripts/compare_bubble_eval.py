from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = ("precision", "recall", "mAP_50", "mAP_50_95")


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def metric_value(payload: dict, key: str) -> float:
    return float(payload.get("metrics", {}).get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two bubble detector evaluation runs.")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline metrics JSON.")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate metrics JSON.")
    args = parser.parse_args()

    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)

    print("Bubble Detector Evaluation Comparison")
    print("=" * 40)
    print(f"Baseline : {baseline.get('run_name', args.baseline.stem)}")
    print(f"Candidate: {candidate.get('run_name', args.candidate.stem)}")
    print(f"Split    : {baseline.get('split')} -> {candidate.get('split')}")
    print(f"Dataset  : {baseline.get('dataset_root')} -> {candidate.get('dataset_root')}")
    print()
    print(f"{'Metric':<12} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}")
    print("-" * 46)
    for key in METRIC_KEYS:
        base = metric_value(baseline, key)
        cand = metric_value(candidate, key)
        delta = cand - base
        print(f"{key:<12} {base:>10.4f} {cand:>10.4f} {delta:>+10.4f}")
    print()
    print(f"Baseline weights : {baseline.get('weights')}")
    print(f"Candidate weights: {candidate.get('weights')}")


if __name__ == "__main__":
    main()
