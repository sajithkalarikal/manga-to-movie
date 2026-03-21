export interface HealthRedisStatus {
  ok: boolean;
  detail: string;
}

export interface HealthWorkerItem {
  worker_key: string;
  hostname: string;
  pid: number;
  queue_name: string;
  updated_at: string;
  age_seconds: number;
  status: string;
}

export interface HealthWorkerStatus {
  ok: boolean;
  live_workers: number;
  stale_workers: number;
  workers: HealthWorkerItem[];
}

export interface HealthModelStatus {
  name: string;
  ok: boolean;
  available: boolean;
  weights_path?: string | null;
  device?: string | null;
  classes: string[];
  load_error?: string | null;
}

export interface HealthOCRStatus {
  ok: boolean;
  tesseract_available: boolean;
  tesseract_cmd?: string | null;
  manga_ocr_loaded: boolean;
  easyocr_loaded: boolean;
}

export interface HealthDiskStatus {
  ok: boolean;
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
  free_percent: number;
  path: string;
}

export interface HealthQueueStatus {
  ok: boolean;
  queue_name: string;
  backlog: number;
}

export interface HealthArtifactStatus {
  ok: boolean;
  path?: string | null;
  updated_at?: string | null;
  seconds_since_update?: number | null;
  details: Record<string, unknown>;
}

export interface TrainingRunDetails {
  epoch?: number | null;
  train_loss?: number | null;
  valid_loss?: number | null;
  class_names?: string[] | null;
  dataset_roots?: string[] | null;
  detector_type?: string | null;
  history?: Array<{
    epoch?: number | null;
    train_loss?: number | null;
    valid_loss?: number | null;
  }> | null;
  history_source?: string | null;
}

export interface EvalMetrics {
  mAP_50_95?: number | null;
  mAP_50?: number | null;
  recall?: number | null;
}

export interface EvalRunDetails {
  run_name?: string | null;
  weights?: string | null;
  dataset_root?: string | null;
  split?: string | null;
  images?: number | null;
  predictions?: number | null;
  elapsed_seconds?: number | null;
  score_threshold?: number | null;
  max_detections?: number | null;
  metrics?: EvalMetrics | null;
  raw_metrics?: Record<string, unknown> | null;
  metric_notes?: Record<string, string> | null;
}

export interface SystemHealthReport {
  status: string;
  environment: string;
  checked_at: string;
  redis: HealthRedisStatus;
  worker: HealthWorkerStatus;
  models: HealthModelStatus[];
  ocr: HealthOCRStatus;
  disk: HealthDiskStatus;
  queue: HealthQueueStatus;
  last_training_run: HealthArtifactStatus;
  last_eval_run: HealthArtifactStatus;
  evaluation_runs: HealthArtifactStatus[];
}

export interface HealthSuggestion {
  title: string;
  detail: string;
}

export function formatBytes(value: number | undefined) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = Number(value) || 0;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatAge(seconds: number | null | undefined) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  if (value < 60) {
    return `${Math.round(value)}s ago`;
  }
  if (value < 3600) {
    return `${Math.round(value / 60)}m ago`;
  }
  if (value < 86400) {
    return `${Math.round(value / 3600)}h ago`;
  }
  return `${Math.round(value / 86400)}d ago`;
}

export function formatMetricPercent(value: number | null | undefined) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 'n/a';
  }
  return `${(numericValue * 100).toFixed(1)}%`;
}

export function formatMetricNumber(value: number | null | undefined, digits = 3) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 'n/a';
  }
  return numericValue.toFixed(digits);
}

export function formatDurationSeconds(value: number | null | undefined) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 'n/a';
  }
  if (numericValue < 60) {
    return `${numericValue.toFixed(1)}s`;
  }
  const minutes = Math.floor(numericValue / 60);
  const seconds = Math.round(numericValue % 60);
  return `${minutes}m ${seconds}s`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function getTrainingRunDetails(artifact: HealthArtifactStatus | null | undefined): TrainingRunDetails | null {
  if (!artifact || !isRecord(artifact.details)) {
    return null;
  }
  return artifact.details as TrainingRunDetails;
}

export function getEvalRunDetails(artifact: HealthArtifactStatus | null | undefined): EvalRunDetails | null {
  if (!artifact || !isRecord(artifact.details)) {
    return null;
  }
  return artifact.details as EvalRunDetails;
}

export async function fetchHealthReport(): Promise<SystemHealthReport> {
  const response = await fetch('/health');
  if (!response.ok) {
    throw new Error(`/health failed (${response.status})`);
  }
  return (await response.json()) as SystemHealthReport;
}

export function buildHealthSuggestions(report: SystemHealthReport): HealthSuggestion[] {
  const suggestions: HealthSuggestion[] = [];

  if (!report.redis.ok) {
    suggestions.push({
      title: 'Restore Redis',
      detail: 'Redis is the queue backbone. Check `REDIS_URL` and restart Redis so jobs and heartbeats resume.',
    });
  }

  if (report.worker.live_workers === 0) {
    suggestions.push({
      title: 'Restart a worker',
      detail: 'No live worker heartbeat means queued jobs will not progress. Restart the ARQ worker process.',
    });
  }

  const failingModels = report.models.filter((model) => !model.ok);
  if (failingModels.length > 0) {
    suggestions.push({
      title: 'Fix model weights',
      detail: `Reload weights for ${failingModels.map((model) => model.name).join(', ')} so detector inference can start cleanly.`,
    });
  }

  if (!report.ocr.ok) {
    suggestions.push({
      title: 'Recover OCR readiness',
      detail: 'Tesseract is unavailable. Reinstall or repoint the OCR binary before running extraction-heavy flows.',
    });
  }

  if (report.disk.free_percent < 15) {
    suggestions.push({
      title: 'Free disk space',
      detail: 'Low free space can break exports, checkpoints, and OCR temp files. Clear old outputs or archives soon.',
    });
  }

  if (report.queue.backlog > 10) {
    suggestions.push({
      title: 'Reduce queue lag',
      detail: 'Backlog is rising. Add worker capacity or reduce heavy jobs so interactive runs return faster.',
    });
  }

  if ((report.last_eval_run.seconds_since_update ?? Number.POSITIVE_INFINITY) > 7 * 24 * 3600) {
    suggestions.push({
      title: 'Refresh evaluation',
      detail: 'Eval metrics are stale. Re-run detector eval after the latest annotation and training changes.',
    });
  }

  if ((report.last_training_run.seconds_since_update ?? Number.POSITIVE_INFINITY) > 14 * 24 * 3600) {
    suggestions.push({
      title: 'Retrain after validation',
      detail: 'Training artifacts are old compared with recent annotation work. A fresh fine-tune is likely worthwhile.',
    });
  }

  if (suggestions.length === 0) {
    suggestions.push({
      title: 'System looks stable',
      detail: 'No urgent health issues were detected. Next gains will likely come from detector tuning, not infrastructure repair.',
    });
  }

  return suggestions.slice(0, 3);
}

export function buildModelPerformanceSuggestions(
  training: TrainingRunDetails | null,
  evaluation: EvalRunDetails | null,
): HealthSuggestion[] {
  const suggestions: HealthSuggestion[] = [];

  const trainLoss = Number(training?.train_loss);
  const validLoss = Number(training?.valid_loss);
  const recall = Number(evaluation?.metrics?.recall);
  const map50 = Number(evaluation?.metrics?.mAP_50);
  const map5095 = Number(evaluation?.metrics?.mAP_50_95);

  if (Number.isFinite(trainLoss) && Number.isFinite(validLoss) && validLoss > trainLoss * 1.1) {
    suggestions.push({
      title: 'Reduce overfit pressure',
      detail: 'Validation loss sits above train loss, so add more reviewed pages or shorten the fine-tune before the model memorizes noise.',
    });
  }

  if (Number.isFinite(recall) && recall < 0.3) {
    suggestions.push({
      title: 'Raise recall on missed regions',
      detail: 'The detector is still missing many targets. Prioritize hard false negatives in `/annotate` and consider a slightly lower score threshold.',
    });
  }

  if (Number.isFinite(map50) && Number.isFinite(map5095) && map50 - map5095 > 0.15) {
    suggestions.push({
      title: 'Tighten box quality',
      detail: 'The model finds regions more often than it localizes them tightly. Cleaner box edits and more crowded-page examples should help.',
    });
  }

  if (
    Number.isFinite(evaluation?.images) &&
    Number.isFinite(evaluation?.predictions) &&
    Number(evaluation?.images) > 0 &&
    Number(evaluation?.predictions) / Number(evaluation?.images) > 8
  ) {
    suggestions.push({
      title: 'Trim noisy detections',
      detail: 'Predictions per page are high, which often creates OCR clutter. Tighten the threshold or add more negative examples in dense scenes.',
    });
  }

  if (suggestions.length === 0) {
    suggestions.push({
      title: 'Performance is stable',
      detail: 'The latest training and eval snapshot looks internally consistent. Keep iterating on targeted annotation rather than changing many knobs at once.',
    });
  }

  return suggestions.slice(0, 3);
}
