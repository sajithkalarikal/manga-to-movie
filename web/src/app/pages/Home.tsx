import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  FileImage,
  Play,
  RefreshCw,
  Upload,
} from 'lucide-react';
import { FlashBanner, useFlashBanner } from '../components/FlashBanner';
import { RegionOverlayBox } from '../components/RegionOverlayBox';
import { SegmentedControl } from '../components/SegmentedControl';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../components/ui/accordion';
import { cn } from '../components/ui/utils';
import { DETECTION_MODE_OPTIONS, getRegionAppearance, type DetectionMode } from '../lib/editor-config';

type ViewMode = 'source' | 'panels' | 'bubbles' | 'captions';

interface PanelBox {
  index: number;
  bbox: [number, number, number, number];
  image_path: string;
  image_url: string;
}

interface DetectPanelsResponse {
  request_id: string;
  filename: string;
  panels: number;
  panel_images_dir: string;
  panel_image_urls: string[];
  panel_boxes: PanelBox[];
  source_image_url?: string | null;
}

interface DialogueItem {
  index?: number;
  text?: string;
  text_regions?: number | string;
  text_role?: string;
}

interface CaptionItem {
  panel: number;
  caption: string;
  shot_type: string;
  motion_level: string;
  tone: string;
  emotion: string;
  action_level: string;
  bubble_count: number;
  bubble_candidates: number;
  bubble_sequence: string;
  speech_count: number;
  narration_count: number;
  sfx_count: number;
  speech_boxes: number[][];
  narration_boxes: number[][];
  sfx_boxes: number[][];
  transition_hint: string;
  layout_role: string;
}

interface AnalyzePanelsResponse {
  request_id: string;
  filename: string;
  panels: number;
  panel_mode: DetectionMode;
  bubble_mode: DetectionMode;
  dialogue: DialogueItem[];
  captions: CaptionItem[];
}

interface PersistedOverridePayload {
  panelResult: DetectPanelsResponse & { request_id: string };
  dialogue: DialogueItem[];
  captions: CaptionItem[];
  bubbleMode: DetectionMode;
  panelMode: DetectionMode;
  sourceImageUrl: string;
  savedAt: string;
}

interface BubbleOverlay {
  id: string;
  type: 'speech' | 'narration' | 'sfx';
  panelIndex: number;
  panelLabel: string;
  text?: string;
  left: number;
  top: number;
  width: number;
  height: number;
}

interface HealthRedisStatus {
  ok: boolean;
  detail: string;
}

interface HealthWorkerItem {
  worker_key: string;
  hostname: string;
  pid: number;
  queue_name: string;
  updated_at: string;
  age_seconds: number;
  status: string;
}

interface HealthWorkerStatus {
  ok: boolean;
  live_workers: number;
  stale_workers: number;
  workers: HealthWorkerItem[];
}

interface HealthModelStatus {
  name: string;
  ok: boolean;
  available: boolean;
  weights_path?: string | null;
  device?: string | null;
  classes: string[];
  load_error?: string | null;
}

interface HealthOCRStatus {
  ok: boolean;
  tesseract_available: boolean;
  tesseract_cmd?: string | null;
  manga_ocr_loaded: boolean;
  easyocr_loaded: boolean;
}

interface HealthDiskStatus {
  ok: boolean;
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
  free_percent: number;
  path: string;
}

interface HealthQueueStatus {
  ok: boolean;
  queue_name: string;
  backlog: number;
}

interface HealthArtifactStatus {
  ok: boolean;
  path?: string | null;
  updated_at?: string | null;
  seconds_since_update?: number | null;
  details: Record<string, unknown>;
}

interface RestoredImageMeta {
  filename: string;
  format: string;
  sizeLabel: string;
}

interface SystemHealthReport {
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
}

function normalizeMode(value: string): DetectionMode {
  return value === 'detector' ? 'detector' : 'heuristic';
}

function fileSizeLabel(file: File | null) {
  if (!file) {
    return 'Unknown';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = file.size;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatBytes(value: number | undefined) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = Number(value) || 0;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatAge(seconds: number | null | undefined) {
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
  return `${Math.round(value / 3600)}h ago`;
}

function inferImageFormat(filename: string) {
  const extension = filename.split('.').pop()?.trim().toUpperCase();
  return extension || 'IMAGE';
}

function readLatestOverridePayload() {
  const raw = sessionStorage.getItem('phase1_override_payload');
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as PersistedOverridePayload;
  } catch {
    return null;
  }
}

function notifyOverrideNavigationSync() {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new Event('phase1-override-updated'));
  }
}

function clearStoredPhase1Payloads() {
  const keysToRemove: string[] = [];
  for (let index = 0; index < sessionStorage.length; index += 1) {
    const key = sessionStorage.key(index);
    if (!key) {
      continue;
    }
    if (key === 'phase1_override_payload' || key.startsWith('phase1_override_payload:')) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => sessionStorage.removeItem(key));
  notifyOverrideNavigationSync();
}

function restoreAnalysisResult(payload: PersistedOverridePayload): AnalyzePanelsResponse {
  return {
    request_id: payload.panelResult.request_id,
    filename: payload.panelResult.filename,
    panels: payload.panelResult.panels,
    panel_mode: normalizeMode(payload.panelMode),
    bubble_mode: normalizeMode(payload.bubbleMode),
    dialogue: payload.dialogue || [],
    captions: payload.captions || [],
  };
}

async function uploadTo<T>(endpoint: string, file: File, extraFields: Record<string, string> = {}): Promise<T> {
  const formData = new FormData();
  formData.append('file', file, file.name);
  Object.entries(extraFields).forEach(([key, value]) => {
    formData.append(key, value);
  });

  const response = await fetch(endpoint, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let message = `Request failed for ${endpoint}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      message = `${message} (${response.status})`;
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function Home() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { banner, showBanner } = useFlashBanner();
  const [file, setFile] = useState<File | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [panelMode, setPanelMode] = useState<DetectionMode>('detector');
  const [bubbleMode, setBubbleMode] = useState<DetectionMode>('detector');
  const [viewMode, setViewMode] = useState<ViewMode>('source');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusText, setStatusText] = useState('Upload a manga page to begin Phase 1 analysis.');
  const [errorText, setErrorText] = useState('');
  const [panelResult, setPanelResult] = useState<(DetectPanelsResponse & { request_id?: string }) | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzePanelsResponse | null>(null);
  const [healthReport, setHealthReport] = useState<SystemHealthReport | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState('');
  const [restoredImageMeta, setRestoredImageMeta] = useState<RestoredImageMeta | null>(null);

  useEffect(() => {
    if (!file) {
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setUploadedImage(objectUrl);
    setRestoredImageMeta(null);

    const image = new Image();
    image.onload = () => {
      setImageDimensions({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };
    image.src = objectUrl;

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  useEffect(() => {
    if (file || uploadedImage || panelResult || analysisResult) {
      return;
    }

    const navigationEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
    if (navigationEntry?.type === 'reload') {
      clearStoredPhase1Payloads();
      return;
    }

    const payload = readLatestOverridePayload();
    if (!payload?.sourceImageUrl || !payload.panelResult?.request_id) {
      return;
    }

    setUploadedImage(payload.sourceImageUrl);
    setPanelResult(payload.panelResult);
    setAnalysisResult(restoreAnalysisResult(payload));
    setPanelMode(normalizeMode(payload.panelMode));
    setBubbleMode(normalizeMode(payload.bubbleMode));
    setViewMode(payload.captions?.length ? 'panels' : 'source');
    setStatusText(`Restored ${payload.panelResult.filename} from your latest Phase 1 session.`);
    setRestoredImageMeta({
      filename: payload.panelResult.filename,
      format: inferImageFormat(payload.panelResult.filename),
      sizeLabel: 'Restored session',
    });

    const image = new Image();
    image.onload = () => {
      setImageDimensions({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };
    image.src = payload.sourceImageUrl;

    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch(payload.sourceImageUrl);
        if (!response.ok) {
          return;
        }
        const blob = await response.blob();
        if (cancelled) {
          return;
        }
        setFile(
          new File([blob], payload.panelResult.filename, {
            type: blob.type || `image/${inferImageFormat(payload.panelResult.filename).toLowerCase()}`,
          }),
        );
      } catch {
        // The preview/result restore above is enough for a smooth back-navigation experience.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [analysisResult, file, panelResult, uploadedImage]);

  const loadHealthReport = async () => {
    setHealthLoading(true);
    setHealthError('');
    try {
      const response = await fetch('/health');
      if (!response.ok) {
        throw new Error(`/health failed (${response.status})`);
      }
      const payload = (await response.json()) as SystemHealthReport;
      setHealthReport(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load system health.';
      setHealthError(message);
      showBanner(message, 'error');
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    void loadHealthReport();
  }, []);

  const effectiveRequestId = analysisResult?.request_id || panelResult?.request_id || '';
  const captionItems = analysisResult?.captions || [];
  const dialogueItems = analysisResult?.dialogue || [];
  const panelBoxes = panelResult?.panel_boxes || [];

  const bubbleOverlays = useMemo<BubbleOverlay[]>(() => {
    if (!analysisResult || !panelResult) {
      return [];
    }

    return panelResult.panel_boxes.flatMap((panel, index) => {
      const caption = analysisResult.captions[index];
      if (!caption) {
        return [];
      }
      const [panelLeft, panelTop] = panel.bbox;
      const text = analysisResult.dialogue[index]?.text || '';
      const overlayGroups = [
        { type: 'speech' as const, boxes: caption.speech_boxes },
        { type: 'narration' as const, boxes: caption.narration_boxes },
        { type: 'sfx' as const, boxes: caption.sfx_boxes },
      ];

      return overlayGroups.flatMap((group) =>
        group.boxes.map((box, boxIndex) => ({
          id: `${panel.index}-${group.type}-${boxIndex}`,
          type: group.type,
          panelIndex: panel.index,
          panelLabel: `Panel ${panel.index}`,
          text,
          left: panelLeft + (box[0] || 0),
          top: panelTop + (box[1] || 0),
          width: box[2] || 0,
          height: box[3] || 0,
        })),
      );
    });
  }, [analysisResult, panelResult]);

  const bubbleCounts = useMemo(
    () => ({
      speech: bubbleOverlays.filter((bubble) => bubble.type === 'speech').length,
      narration: bubbleOverlays.filter((bubble) => bubble.type === 'narration').length,
      sfx: bubbleOverlays.filter((bubble) => bubble.type === 'sfx').length,
    }),
    [bubbleOverlays],
  );

  const analysisComplete = Boolean(analysisResult && panelResult);
  const currentFilename = file?.name || restoredImageMeta?.filename || panelResult?.filename || 'Uploaded image';
  const currentFormat = file?.type.split('/')[1]?.toUpperCase() || restoredImageMeta?.format || 'IMAGE';
  const currentSizeLabel = file ? fileSizeLabel(file) : restoredImageMeta?.sizeLabel || 'Unknown';

  const renderHealthSection = () => (
    <div className="pt-4 border-t border-border space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-xs font-medium text-foreground">System Health</h3>
          <p className="text-[11px] text-muted-foreground">
            Redis, worker, model, OCR, disk, queue, training and eval state.
          </p>
        </div>
        <button
          onClick={() => {
            void loadHealthReport();
          }}
          disabled={healthLoading}
          className={cn(
            'h-8 px-3 rounded-lg border text-xs flex items-center gap-2 transition-colors',
            healthLoading
              ? 'bg-muted text-muted-foreground border-border cursor-not-allowed'
              : 'bg-surface-container hover:bg-surface-container-high text-foreground border-border',
          )}
        >
          <RefreshCw className={cn('w-3.5 h-3.5', healthLoading && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {healthError ? (
        <div className="p-3 rounded-lg border bg-crimson/10 border-crimson/20">
          <p className="text-xs text-foreground">{healthError}</p>
        </div>
      ) : healthReport ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className={cn('p-3 rounded-lg border', healthReport.redis.ok ? 'bg-cyan/10 border-cyan/20' : 'bg-crimson/10 border-crimson/20')}>
              <div className="font-medium text-foreground">Redis</div>
              <div className="text-muted-foreground mt-1">{healthReport.redis.ok ? 'Connected' : healthReport.redis.detail}</div>
            </div>
            <div className={cn('p-3 rounded-lg border', healthReport.worker.ok ? 'bg-cyan/10 border-cyan/20' : 'bg-crimson/10 border-crimson/20')}>
              <div className="font-medium text-foreground">Workers</div>
              <div className="text-muted-foreground mt-1">{healthReport.worker.live_workers} live / {healthReport.worker.stale_workers} stale</div>
            </div>
            <div className={cn('p-3 rounded-lg border', healthReport.ocr.ok ? 'bg-cyan/10 border-cyan/20' : 'bg-crimson/10 border-crimson/20')}>
              <div className="font-medium text-foreground">OCR</div>
              <div className="text-muted-foreground mt-1">{healthReport.ocr.tesseract_available ? 'Tesseract ready' : 'Unavailable'}</div>
            </div>
            <div className={cn('p-3 rounded-lg border', healthReport.disk.ok ? 'bg-cyan/10 border-cyan/20' : 'bg-crimson/10 border-crimson/20')}>
              <div className="font-medium text-foreground">Disk</div>
              <div className="text-muted-foreground mt-1">{healthReport.disk.free_percent}% free</div>
            </div>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Overall</span>
              <span className="text-foreground font-medium">{healthReport.status}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Queue backlog</span>
              <span className="text-foreground">{healthReport.queue.backlog}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Free space</span>
              <span className="text-foreground">{formatBytes(healthReport.disk.free_bytes)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Last training</span>
              <span className="text-foreground">{formatAge(healthReport.last_training_run.seconds_since_update)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Last eval</span>
              <span className="text-foreground">{formatAge(healthReport.last_eval_run.seconds_since_update)}</span>
            </div>
          </div>

          <div className="space-y-2">
            {healthReport.models.map((model) => (
              <div key={model.name} className="p-3 bg-surface-container rounded-lg text-xs space-y-1">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-foreground">{model.name}</span>
                  <span className={cn('px-2 py-0.5 rounded border text-[10px]', model.ok ? 'bg-cyan/10 text-cyan border-cyan/20' : 'bg-crimson/10 text-crimson border-crimson/20')}>
                    {model.ok ? 'ok' : 'degraded'}
                  </span>
                </div>
                <p className="text-muted-foreground break-all">{model.weights_path || 'not configured'}</p>
                <p className="text-muted-foreground">device: {model.device || 'n/a'}</p>
                <p className="text-muted-foreground">classes: {(model.classes || []).join(', ') || 'none'}</p>
                {model.load_error ? <p className="text-crimson">error: {model.load_error}</p> : null}
              </div>
            ))}
          </div>

          <div className="space-y-2">
            {(healthReport.worker.workers || []).length ? (
              healthReport.worker.workers.map((worker) => (
                <div key={worker.worker_key} className="p-3 bg-surface-container rounded-lg text-xs">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-foreground">{worker.hostname}:{worker.pid}</span>
                    <span className={cn('px-2 py-0.5 rounded border text-[10px]', worker.status === 'alive' ? 'bg-cyan/10 text-cyan border-cyan/20' : 'bg-crimson/10 text-crimson border-crimson/20')}>
                      {worker.status}
                    </span>
                  </div>
                  <p className="text-muted-foreground mt-1">{worker.queue_name}</p>
                  <p className="text-muted-foreground">{worker.updated_at} ({formatAge(worker.age_seconds)})</p>
                </div>
              ))
            ) : (
              <div className="p-3 bg-surface-container rounded-lg text-xs text-muted-foreground">
                No worker heartbeats found yet. Restart the worker to publish health heartbeats.
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="p-3 rounded-lg border bg-surface-container text-xs text-muted-foreground">
          Loading health report...
        </div>
      )}
    </div>
  );

  const resetResults = () => {
    setPanelResult(null);
    setAnalysisResult(null);
    setViewMode('source');
    setErrorText('');
    setRestoredImageMeta(null);
    clearStoredPhase1Payloads();
  };

  const persistOverridePayload = (
    detectResult: DetectPanelsResponse,
    analyzeResult: AnalyzePanelsResponse,
    sourceImageUrl: string,
    activePanelMode: DetectionMode,
    activeBubbleMode: DetectionMode,
  ) => {
    const payload: PersistedOverridePayload = {
      panelResult: {
        ...detectResult,
        request_id: analyzeResult.request_id,
      },
      dialogue: analyzeResult.dialogue,
      captions: analyzeResult.captions,
      bubbleMode: activeBubbleMode,
      panelMode: activePanelMode,
      sourceImageUrl,
      savedAt: new Date().toISOString(),
    };

    sessionStorage.setItem('phase1_override_payload', JSON.stringify(payload));
    sessionStorage.setItem(`phase1_override_payload:${analyzeResult.request_id}`, JSON.stringify(payload));
    notifyOverrideNavigationSync();
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] || null;
    if (!nextFile) {
      return;
    }

    setFile(nextFile);
    resetResults();
    setStatusText(`Ready to analyze ${nextFile.name}.`);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async () => {
    if (!file) {
      const message = 'Choose an image before running the pipeline.';
      setErrorText(message);
      showBanner(message, 'error');
      return;
    }

    const activePanelMode = normalizeMode(panelMode);
    const activeBubbleMode = normalizeMode(bubbleMode);
    clearStoredPhase1Payloads();
    setIsAnalyzing(true);
    setErrorText('');
    setStatusText('Detecting panels...');
    setPanelResult(null);
    setAnalysisResult(null);

    try {
      const detectResult = await uploadTo<DetectPanelsResponse>('/detect-panels', file, {
        panel_mode: activePanelMode,
      });
      setPanelResult(detectResult);
      setStatusText('Running panel analysis...');

      const analyzeResult = await uploadTo<AnalyzePanelsResponse>('/analyze-panels', file, {
        bubble_mode: activeBubbleMode,
        panel_mode: activePanelMode,
      });

      const mergedPanelResult = {
        ...detectResult,
        request_id: analyzeResult.request_id,
      };
      setPanelResult(mergedPanelResult);
      setAnalysisResult(analyzeResult);
      setViewMode('panels');
      persistOverridePayload(
        detectResult,
        analyzeResult,
        detectResult.source_image_url || uploadedImage || '',
        activePanelMode,
        activeBubbleMode,
      );
      const message = `Phase 1 completed using panel=${activePanelMode} and bubble=${activeBubbleMode}.`;
      setStatusText(message);
      showBanner(message, 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Something went wrong while running the pipeline.';
      setErrorText(message);
      setStatusText(message);
      showBanner(message, 'error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleOpenOverride = () => {
    if (!effectiveRequestId) {
      const message = 'Run Phase 1 first, then open the override workspace.';
      setErrorText(message);
      showBanner(message, 'error');
      return;
    }
    navigate(`/ui_v2/${effectiveRequestId}/override`);
  };

  return (
    <div className="relative flex-1 flex flex-col h-full">
      <FlashBanner banner={banner} />
      <div className="h-14 bg-surface-container-low border-b border-border flex items-center px-4 gap-4">
        <h1 className="text-sm font-medium text-foreground">Phase 1 Analysis</h1>
        <div className="flex-1" />
        <div className="text-xs text-muted-foreground hidden xl:block">
          {uploadedImage ? statusText : 'Desktop story editor workspace'}
        </div>
        <button
          onClick={handleUploadClick}
          className="sparkle-upload-button h-9 px-4 flex items-center gap-2 bg-surface-container hover:bg-surface-container-high rounded-lg text-sm text-foreground transition-colors border border-border"
        >
          <span className="sparkle-upload-button__glint" aria-hidden="true" />
          <span className="sparkle-upload-button__twinkle sparkle-upload-button__twinkle--one" aria-hidden="true" />
          <span className="sparkle-upload-button__twinkle sparkle-upload-button__twinkle--two" aria-hidden="true" />
          <Upload className="w-4 h-4" />
          {uploadedImage ? 'Change Image' : 'Upload Image'}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 bg-surface p-6 overflow-auto">
          {!uploadedImage ? (
            <div className="h-full flex items-center justify-center">
              <div onClick={handleUploadClick} className="w-full max-w-2xl cursor-pointer">
                <div className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-foreground/20 transition-colors group bg-surface-variant/30">
                  <div className="w-20 h-20 mx-auto mb-6 rounded-lg bg-surface-container flex items-center justify-center group-hover:bg-surface-container-high transition-colors">
                    <Upload className="w-10 h-10 text-muted-foreground" />
                  </div>
                  <h3 className="text-base font-medium text-foreground mb-2">Upload manga page</h3>
                  <p className="text-sm text-muted-foreground mb-6">
                    Drop your image here or click to browse
                  </p>
                  <div className="inline-flex items-center gap-2 px-4 py-2 bg-crimson text-crimson-foreground rounded-lg text-sm">
                    <Upload className="w-4 h-4" />
                    Select Image
                  </div>
                  <p className="text-xs text-muted-foreground mt-6">
                    Supported formats: PNG, JPG, JPEG, WebP
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <SegmentedControl
                  options={[
                    { value: 'source', label: 'Source' },
                    {
                      value: 'panels',
                      label: 'Panels',
                      icon: analysisComplete ? <CheckCircle2 className="w-3 h-3" /> : undefined,
                    },
                    {
                      value: 'bubbles',
                      label: 'Bubbles',
                      icon: analysisComplete ? <CheckCircle2 className="w-3 h-3" /> : undefined,
                    },
                    {
                      value: 'captions',
                      label: 'Captions',
                      icon: analysisComplete ? <CheckCircle2 className="w-3 h-3" /> : undefined,
                    },
                  ]}
                  value={viewMode}
                  onChange={(value) => setViewMode(value as ViewMode)}
                />

                {analysisComplete && (
                  <div className="flex items-center gap-2 ml-auto text-xs text-muted-foreground">
                    <CheckCircle2 className="w-4 h-4 text-cyan" />
                    <span>Analysis complete</span>
                  </div>
                )}
              </div>

              <div className="flex-1 flex items-center justify-center bg-surface-variant/30 rounded-lg p-6 overflow-auto">
                <div className="w-full h-full">
                  {viewMode === 'captions' ? (
                    <div className="h-full overflow-auto rounded-lg border border-border bg-surface-container">
                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 p-4">
                        {captionItems.length ? (
                          captionItems.map((caption, index) => (
                            <article
                              key={`${caption.panel}-${index}`}
                              className="rounded-lg border border-border bg-surface-container-high p-4 space-y-3"
                            >
                              <div className="flex items-center justify-between gap-3">
                                <div>
                                  <h3 className="text-sm font-medium text-foreground">
                                    Panel {caption.panel}
                                  </h3>
                                  <p className="text-xs text-muted-foreground">
                                    {caption.shot_type} • {caption.layout_role} • {caption.transition_hint}
                                  </p>
                                </div>
                                <span className="text-xs px-2 py-1 rounded bg-crimson/10 text-foreground border border-crimson/20">
                                  {caption.bubble_count} speech
                                </span>
                              </div>
                              <p className="text-sm text-foreground leading-6">{caption.caption}</p>
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                <div className="rounded bg-surface p-2">
                                  <div className="text-muted-foreground">OCR</div>
                                  <div className="text-foreground mt-1">
                                    {dialogueItems[index]?.text || '[no dialogue detected]'}
                                  </div>
                                </div>
                                <div className="rounded bg-surface p-2">
                                  <div className="text-muted-foreground">Mood</div>
                                  <div className="text-foreground mt-1">
                                    {caption.tone} • {caption.emotion}
                                  </div>
                                </div>
                              </div>
                            </article>
                          ))
                        ) : (
                          <div className="col-span-full h-full min-h-64 flex items-center justify-center text-sm text-muted-foreground">
                            Run analysis to populate panel captions.
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div
                      className="relative mx-auto bg-surface-container border border-border rounded-lg shadow-lg overflow-hidden"
                      style={{
                        aspectRatio: imageDimensions ? `${imageDimensions.width} / ${imageDimensions.height}` : '2 / 3',
                        maxHeight: '100%',
                        maxWidth: '100%',
                      }}
                    >
                      {uploadedImage ? (
                        <img
                          src={uploadedImage}
                          alt={file?.name || 'Uploaded manga page'}
                          className="block w-full h-full object-contain"
                        />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-muted/10 to-muted/5 flex items-center justify-center">
                          <FileImage className="w-16 h-16 text-muted-foreground/30" />
                        </div>
                      )}

                      {analysisComplete &&
                        viewMode === 'panels' &&
                        imageDimensions &&
                        panelBoxes.map((panel) => {
                          const [left, top, right, bottom] = panel.bbox;
                          const width = right - left;
                          const height = bottom - top;
                          return (
                            <RegionOverlayBox
                              key={panel.index}
                              regionType="panel"
                              label={`Panel ${panel.index}`}
                              rect={{ x: left, y: top, width, height }}
                              canvasWidth={imageDimensions.width}
                              canvasHeight={imageDimensions.height}
                            />
                          );
                        })}

                      {analysisComplete &&
                        viewMode === 'bubbles' &&
                        imageDimensions &&
                        bubbleOverlays.map((bubble) => (
                          <RegionOverlayBox
                            key={bubble.id}
                            regionType={bubble.type}
                            label={bubble.type}
                            rect={{
                              x: bubble.left,
                              y: bubble.top,
                              width: bubble.width,
                              height: bubble.height,
                            }}
                            canvasWidth={imageDimensions.width}
                            canvasHeight={imageDimensions.height}
                            title={`${bubble.panelLabel}${bubble.text ? `: ${bubble.text}` : ''}`}
                          />
                        ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="w-96 bg-surface-container-low border-l border-border flex flex-col overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-sm font-medium text-foreground">Controls</h2>
          </div>

          <div className="flex-1 overflow-auto p-4 space-y-6">
            {!uploadedImage ? (
              <div className="flex flex-col items-center justify-center text-center px-6 py-10">
                <div className="w-16 h-16 mb-4 rounded-full bg-surface-container flex items-center justify-center">
                  <AlertCircle className="w-8 h-8 text-muted-foreground" />
                </div>
                <h3 className="text-sm font-medium text-foreground mb-2">No image uploaded</h3>
                <p className="text-xs text-muted-foreground">
                  Upload a manga page to begin Phase 1 analysis
                </p>
              </div>
            ) : (
              <>
                <div className="space-y-4">
                  <div>
                    <label className="text-xs font-medium text-foreground mb-2 block">
                      Panel Detection
                    </label>
                    <SegmentedControl
                      options={DETECTION_MODE_OPTIONS}
                      value={panelMode}
                      onChange={(value) => setPanelMode(value as DetectionMode)}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground mt-1.5">
                      {panelMode === 'detector' ? 'AI-powered detection (recommended)' : 'Rule-based detection'}
                    </p>
                  </div>

                  <div>
                    <label className="text-xs font-medium text-foreground mb-2 block">
                      Bubble Detection
                    </label>
                    <SegmentedControl
                      options={DETECTION_MODE_OPTIONS}
                      value={bubbleMode}
                      onChange={(value) => setBubbleMode(value as DetectionMode)}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground mt-1.5">
                      {bubbleMode === 'detector' ? 'AI-powered detection (recommended)' : 'Rule-based detection'}
                    </p>
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className={cn(
                      'w-full py-3 px-4 flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-all',
                      isAnalyzing
                        ? 'bg-muted text-muted-foreground cursor-not-allowed'
                        : 'bg-crimson text-crimson-foreground hover:opacity-90 shadow-sm',
                    )}
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="w-4 h-4 border-2 border-crimson-foreground/30 border-t-crimson-foreground rounded-full animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Run Analysis
                      </>
                    )}
                  </button>
                </div>

                {analysisComplete && effectiveRequestId ? (
                  <div className="pt-2">
                    <button
                      onClick={handleOpenOverride}
                      className="w-full py-2.5 px-3 bg-surface-container-high hover:bg-surface-container-highest text-foreground rounded-lg text-sm transition-colors flex items-center justify-between group"
                    >
                      <span>Go to Override</span>
                      <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                    </button>
                  </div>
                ) : null}

                {(isAnalyzing || analysisComplete || errorText) && (
                  <div
                    className={cn(
                      'p-3 rounded-lg border',
                      errorText
                        ? 'bg-crimson/10 border-crimson/20'
                        : 'bg-cyan/10 border-cyan/20',
                    )}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {errorText ? (
                        <AlertCircle className="w-4 h-4 text-crimson" />
                      ) : analysisComplete ? (
                        <CheckCircle2 className="w-4 h-4 text-cyan" />
                      ) : (
                        <div className="w-2 h-2 bg-cyan rounded-full animate-pulse" />
                      )}
                      <span className="text-xs font-medium text-foreground">
                        {errorText ? 'Request Failed' : analysisComplete ? 'Analysis Complete' : 'Processing...'}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">{statusText}</p>
                  </div>
                )}

                <div className="pt-4 border-t border-border space-y-3">
                  <h3 className="text-xs font-medium text-foreground">Image Metadata</h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Filename</span>
                      <span className="text-foreground font-mono truncate text-right" title={currentFilename}>
                        {currentFilename}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Resolution</span>
                      <span className="text-foreground">
                        {imageDimensions ? `${imageDimensions.width}×${imageDimensions.height}` : 'Loading...'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Format</span>
                      <span className="text-foreground">{currentFormat}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Size</span>
                      <span className="text-foreground">{currentSizeLabel}</span>
                    </div>
                  </div>
                </div>

                {analysisComplete && (
                  <div className="pt-4 border-t border-border space-y-3">
                    <Accordion type="single" collapsible defaultValue="detection-summary" className="w-full">
                      <AccordionItem value="detection-summary" className="border-none">
                        <AccordionTrigger className="py-0 hover:no-underline">
                          <div className="flex w-full items-center justify-between gap-3 pr-2">
                            <div>
                              <h3 className="text-xs font-medium text-foreground">Detection Summary</h3>
                              <p className="mt-1 text-[11px] text-muted-foreground">
                                Panels, bubble types, and generated panel captions.
                              </p>
                            </div>
                            <div className="flex items-center gap-2 text-[11px]">
                              <span className="rounded-full border border-border bg-surface-container px-2 py-1 text-muted-foreground">
                                {panelBoxes.length} panels
                              </span>
                              <span className="rounded-full border border-cyan/20 bg-cyan/10 px-2 py-1 text-muted-foreground">
                                {bubbleOverlays.length} bubbles
                              </span>
                            </div>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="pt-3">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">Detected Panels</span>
                              <span className="text-foreground font-medium">{panelBoxes.length}</span>
                            </div>
                            <div className="space-y-1">
                              {panelBoxes.slice(0, 3).map((panel, index) => (
                                <div
                                  key={panel.index}
                                  className="p-2 bg-surface-container rounded text-xs hover:bg-surface-container-high transition-colors"
                                >
                                  <div className="font-medium text-foreground mb-1">Panel {panel.index}</div>
                                  <div className="text-muted-foreground line-clamp-2">
                                    {captionItems[index]?.caption || '[no caption generated]'}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="space-y-2 pt-3 border-t border-border">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">Detected Bubbles</span>
                              <span className="text-foreground font-medium">{bubbleOverlays.length}</span>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              <div className={cn('rounded border p-2 text-center', getRegionAppearance('speech').summaryClassName)}>
                                <div className="font-medium text-foreground">{bubbleCounts.speech}</div>
                                <div className="text-muted-foreground">Speech</div>
                              </div>
                              <div className={cn('rounded border p-2 text-center', getRegionAppearance('narration').summaryClassName)}>
                                <div className="font-medium text-foreground">{bubbleCounts.narration}</div>
                                <div className="text-muted-foreground">Narration</div>
                              </div>
                              <div className={cn('rounded border p-2 text-center', getRegionAppearance('sfx').summaryClassName)}>
                                <div className="font-medium text-foreground">{bubbleCounts.sfx}</div>
                                <div className="text-muted-foreground">SFX</div>
                              </div>
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  </div>
                )}
              </>
            )}

            {renderHealthSection()}
          </div>
        </div>
      </div>
    </div>
  );
}
