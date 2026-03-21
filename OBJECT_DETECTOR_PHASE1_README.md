# Object Detector Phase 1

This document describes the current object-detector-based Phase 1 path for manga bubble detection in this project.

## Goal

Phase 1 with the object detector is used to improve speech-bubble detection over the heuristic-only path.

Current high-level flow:

```text
upload manga page
-> panel detection
-> panel crop generation
-> OCR extraction
-> bubble detection mode selection
   -> heuristic
   -> object detector
-> local scene/caption analysis
-> manual correction in override UI
```

## Current Implementation

### Main entrypoint

- API endpoint: [app.py](/Users/sajith/Documents/New%20project/manga-to-movie/app.py)
  - `POST /analyze-panels`
  - accepts `bubble_mode` with:
    - `heuristic`
    - `detector`

### Detector integration

- detector wrapper: [modules/bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/bubble_detector.py)
- phase 1 analysis: [modules/local_phase1.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/local_phase1.py)
- caption pipeline: [modules/scene_caption.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/scene_caption.py)

When `bubble_mode=detector`:

1. the app loads the local bubble detector weights
2. each panel image is passed through the detector
3. detector boxes become the primary bubble boxes
4. OCR/text-region classification still runs on top of those boxes
5. speech/narration/SFX counts are derived from the classified regions
6. the override UI is used to correct results when needed

When `bubble_mode=heuristic`:

1. no model inference is used
2. bright-region and text-region heuristics generate bubble candidates
3. OCR/text-region classification refines them

## Model Architecture

Current detector model:

- framework: `torchvision`
- detector: `fasterrcnn_mobilenet_v3_large_fpn`
- classes:
  - `background`
  - `speech_bubble`
  - `narration_box`
  - `sfx`

Model definition and load path:

- training model builder: [scripts/train_bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/scripts/train_bubble_detector.py)
- inference loader: [modules/bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/bubble_detector.py)

## Dataset

Current best multi-class experiment:

- checkpoint: [models/bubble_detector_v2_new_only.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector_v2_new_only.pt)
- training data: [outputs/dataset_annotations/exported_coco/new object training data.v1.coco](/Users/sajith/Documents/New%20project/manga-to-movie/outputs/dataset_annotations/exported_coco/new%20object%20training%20data.v1.coco)

Split counts for the exported multi-class dataset:

- train: `144` images
- valid: `18` images
- test: `18` images

Labeled classes:

- `speech_bubble`
- `narration_box`
- `sfx`

## Training Configuration

Current train script:

- [scripts/train_bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/scripts/train_bubble_detector.py)

Current important values:

- epochs: `5`
- batch size: `2`
- workers: `0`
- learning rate: `1e-4`
- weight decay: `1e-4`
- device selection:
  - `cuda` if available
  - else `mps` if available
  - else `cpu`

Optimizer:

- `AdamW`

Checkpoint outputs:

- best model: [models/bubble_detector_v2_new_only.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector_v2_new_only.pt)
- latest checkpoint: [models/bubble_detector_v2_new_only.latest.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector_v2_new_only.latest.pt)

## Inference Configuration

Current detector runtime settings from [config.py](/Users/sajith/Documents/New%20project/manga-to-movie/config.py):

- `BUBBLE_DETECTOR_SCORE_THRESHOLD = 0.45`
- `BUBBLE_DETECTOR_MAX_DETECTIONS = 8`

Post-processing behavior from [modules/bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/bubble_detector.py):

- class names are loaded from the checkpoint
- `Bubble with Text` and `objects` are normalized to `speech_bubble`
- detections below score threshold are discarded
- boxes with width or height under `8` pixels are discarded
- detections are capped to the configured max detections

## UI Behavior

Main review page:

- [frontend/index.html](/Users/sajith/Documents/New%20project/manga-to-movie/frontend/index.html)

User can choose:

- `Heuristic`
- `Object Detector`

Override workspace:

- [frontend/override.html](/Users/sajith/Documents/New%20project/manga-to-movie/frontend/override.html)

The override UI lets the user correct:

- `Speech`
- `Narration`
- `SFX`
- `Bubbles`
- `Bubble Sequence`

Saved override payloads are written to:

- `outputs/<request_id>/panel_overrides.json`

## Phase 1 Timing Logs

For `bubble_mode=detector`, the backend now logs:

- start time
- end time
- total duration in seconds
- total duration in minutes

These are logged in [app.py](/Users/sajith/Documents/New%20project/manga-to-movie/app.py) with lines like:

- `Phase1 detector run started ...`
- `Phase1 detector run finished ... duration_minutes=...`

Check runtime logs here:

- [logs/app.log](/Users/sajith/Documents/New%20project/manga-to-movie/logs/app.log)

## Current Metrics

Current multi-class metrics:

- valid:
  - `mAP@0.50:0.95 = 0.1846`
  - `mAP@0.50 = 0.2750`
  - `recall = 0.2166`
- test:
  - `mAP@0.50:0.95 = 0.1852`
  - `mAP@0.50 = 0.2799`
  - `recall = 0.2124`

Evaluation support is now set up with:

- [scripts/eval_bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/scripts/eval_bubble_detector.py)

This script is designed to report:

- precision
- recall
- mAP@0.50
- mAP@0.50:0.95

Metrics are written to:

- `outputs/eval/predictions_<split>.json`
- `outputs/eval/metrics_<split>.json`

Example command:

```bash
./.venv/bin/python scripts/eval_bubble_detector.py \
  --dataset-root "/Users/sajith/Documents/New project/Manga Bubble.v4i.coco" \
  --weights models/bubble_detector.pt \
  --split valid \
  --run-name baseline_valid \
  --score-threshold 0.45 \
  --max-detections 8
```

Comparison command after the new model is trained:

```bash
./.venv/bin/python scripts/compare_bubble_eval.py \
  --baseline outputs/eval/metrics_baseline_valid.json \
  --candidate outputs/eval/metrics_new_model_valid.json
```

Current note:

- `pycocotools` is required for the evaluation script
- if it is not installed yet, install dependencies first with `pip install -r requirements.txt`

## Current Strengths

- uses your local COCO dataset directly
- fully local inference
- works with the existing Phase 1 UI and override flow
- explicit switch between heuristic and detector mode
- detector timing is now logged

## Current Limitations

- multi-class quality is still modest and needs more annotations
- narration and SFX still rely partly on downstream heuristics around detector output
- CPU training and inference are slower than GPU

## Recommended Next Improvements

1. Run the new evaluation script on `valid` and `test`.
2. Expand the dataset to separate:
   - `speech_bubble`
   - `narration_box`
   - `sfx`
3. Add detector overlay debugging to the override page.
4. Export corrected override data back into future training annotations.
