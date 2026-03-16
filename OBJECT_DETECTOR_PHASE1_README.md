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
  - `Bubble with Text`

Model definition and load path:

- training model builder: [scripts/train_bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/scripts/train_bubble_detector.py)
- inference loader: [modules/bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/bubble_detector.py)

## Dataset

Training dataset used:

- [Manga Bubble.v4i.coco](/Users/sajith/Documents/New%20project/Manga%20Bubble.v4i.coco)

Verified split counts:

- train: `1905` images
- valid: `244` images
- test: `250` images

Used annotation class:

- `Bubble with Text`

Note:

- the COCO file contains another category named `objects`
- current training data only uses `Bubble with Text`

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

- best model: [models/bubble_detector.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector.pt)
- latest checkpoint: [models/bubble_detector.latest.pt](/Users/sajith/Documents/New%20project/manga-to-movie/models/bubble_detector.latest.pt)

## Inference Configuration

Current detector runtime settings from [config.py](/Users/sajith/Documents/New%20project/manga-to-movie/config.py):

- `BUBBLE_DETECTOR_SCORE_THRESHOLD = 0.45`
- `BUBBLE_DETECTOR_MAX_DETECTIONS = 8`

Post-processing behavior from [modules/bubble_detector.py](/Users/sajith/Documents/New%20project/manga-to-movie/modules/bubble_detector.py):

- only label `1` is accepted
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

What we have currently from training:

- epoch 1 train loss: `0.7407`
- epoch 1 valid loss: `0.5312`
- epoch 2 train loss: `0.4626`
- epoch 2 valid loss: `0.3730`

What we do not yet have in the current trainer:

- precision
- recall
- mAP
- F1

So the current measurable training status is based on:

- training loss
- validation loss

If precision/recall/mAP are needed, the next improvement is to add a proper object-detection evaluation pass against the validation or test split.

## Current Strengths

- uses your local COCO dataset directly
- fully local inference
- works with the existing Phase 1 UI and override flow
- explicit switch between heuristic and detector mode
- detector timing is now logged

## Current Limitations

- precision/recall/mAP are not yet computed
- detector only predicts one class: `Bubble with Text`
- narration and SFX are still derived from downstream heuristics, not from the detector
- CPU training and inference are slower than GPU

## Recommended Next Improvements

1. Add validation/test evaluation with precision, recall, and mAP.
2. Expand the dataset to separate:
   - `speech_bubble`
   - `narration_box`
   - `sfx`
3. Add detector overlay debugging to the override page.
4. Export corrected override data back into future training annotations.
