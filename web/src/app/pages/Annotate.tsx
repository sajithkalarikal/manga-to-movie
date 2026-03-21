import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Eraser,
  ImageOff,
  PencilRuler,
  Save,
  SquareDashedMousePointer,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { FlashBanner, useFlashBanner } from '../components/FlashBanner';
import { RegionOverlayBox } from '../components/RegionOverlayBox';
import { RegionPropertyCard } from '../components/RegionPropertyCard';
import { SegmentedControl } from '../components/SegmentedControl';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { cn } from '../components/ui/utils';
import {
  ANNOTATION_BUBBLE_TYPE_OPTIONS,
  getRegionAppearance,
  getRegionLabel,
} from '../lib/editor-config';
import { formatBboxLabel, rectToStyle } from '../lib/region-utils';

type AnnotationTab = 'panels' | 'bubbles' | 'captions';
type AnnotationViewMode = 'all' | 'review';
type SplitMode = 'train' | 'valid' | 'test';
type AnnotationClass = 'panel' | 'speech_bubble' | 'narration_box' | 'sfx';
type BubbleClass = Exclude<AnnotationClass, 'panel'>;
type ExportMode = 'full' | 'validated_bubble_only';

interface DatasetOption {
  key: string;
  name: string;
}

interface DatasetAnnotation {
  id: string;
  class_name: AnnotationClass;
  bbox: [number, number, number, number];
  points: number[][] | null;
}

interface DatasetImageListResponse {
  dataset: string;
  split: SplitMode;
  offset: number;
  limit: number;
  total: number;
  items: Array<{
    index: number;
    image_id: number;
    file_name: string;
    width: number;
    height: number;
    source_annotation_count: number;
    source_categories: string[];
    override_exists: boolean;
    image_url: string;
  }>;
  classes: string[];
  available_datasets: DatasetOption[];
}

interface DatasetAnnotationItemResponse {
  dataset: string;
  split: SplitMode;
  index: number;
  image_id: number;
  file_name: string;
  width: number;
  height: number;
  image_url: string;
  classes: string[];
  annotation_source: string;
  annotations: DatasetAnnotation[];
  available_datasets: DatasetOption[];
}

interface AnnotationReviewQueueItem {
  queue_index: number;
  split: SplitMode;
  index: number;
  image_id: number;
  file_name: string;
  width: number;
  height: number;
  annotation_count: number;
  class_names: string[];
  override_exists: boolean;
  annotation_source: string;
  priority: number;
  reasons: string[];
  image_url: string;
}

interface AnnotationReviewQueueResponse {
  dataset: string;
  total: number;
  items: AnnotationReviewQueueItem[];
}

interface SaveAnnotationsResponse {
  dataset: string;
  split: SplitMode;
  image_id: number;
  file_name: string;
  saved: boolean;
  annotation_path: string;
}

interface ExportDatasetResponse {
  dataset: string;
  export_mode: ExportMode;
  exported: boolean;
  output_dir: string;
  annotation_files: string[];
}

interface DraftRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface EditInteraction {
  mode: 'move' | 'resize';
  target: {
    kind: 'panel' | 'bubble';
    annotationId: string;
  };
  startPointer: { x: number; y: number };
  startBbox: [number, number, number, number];
}

const VIEW_OPTIONS = [
  { value: 'all', label: 'All Images' },
  { value: 'review', label: 'Validate Images' },
];

const TAB_OPTIONS = [
  { value: 'panels', label: 'Panels' },
  { value: 'bubbles', label: 'Bubbles' },
  { value: 'captions', label: 'Captions' },
];

function cloneAnnotation(annotation: DatasetAnnotation): DatasetAnnotation {
  return {
    id: String(annotation.id),
    class_name: annotation.class_name,
    bbox: [
      Number(annotation.bbox[0] || 0),
      Number(annotation.bbox[1] || 0),
      Number(annotation.bbox[2] || 0),
      Number(annotation.bbox[3] || 0),
    ],
    points: Array.isArray(annotation.points)
      ? annotation.points.map((point) => [Number(point[0] || 0), Number(point[1] || 0)])
      : null,
  };
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `Request failed for ${url}`;
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

function clampBbox(
  bbox: [number, number, number, number],
  width: number,
  height: number,
): [number, number, number, number] {
  const maxWidth = Math.max(1, Math.round(width));
  const maxHeight = Math.max(1, Math.round(height));
  let [x, y, w, h] = bbox.map((value) => Math.round(Number(value))) as [number, number, number, number];
  x = Math.min(Math.max(0, x), maxWidth - 1);
  y = Math.min(Math.max(0, y), maxHeight - 1);
  w = Math.min(Math.max(1, w), maxWidth - x);
  h = Math.min(Math.max(1, h), maxHeight - y);
  return [x, y, w, h];
}

function clampZoom(nextValue: number) {
  return Math.min(2.5, Math.max(0.4, Number(nextValue.toFixed(2))));
}

export function Annotate() {
  const drawingSurfaceRef = useRef<HTMLDivElement | null>(null);
  const drawingStartRef = useRef<{ x: number; y: number } | null>(null);
  const editInteractionRef = useRef<EditInteraction | null>(null);
  const { banner, showBanner } = useFlashBanner();

  const [datasetOptions, setDatasetOptions] = useState<DatasetOption[]>([]);
  const [dataset, setDataset] = useState('');
  const [viewMode, setViewMode] = useState<AnnotationViewMode>('all');
  const [split, setSplit] = useState<SplitMode>('train');
  const [queueItems, setQueueItems] = useState<AnnotationReviewQueueItem[]>([]);
  const [queueIndex, setQueueIndex] = useState(0);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [totalItems, setTotalItems] = useState(0);
  const [item, setItem] = useState<DatasetAnnotationItemResponse | null>(null);
  const [originalAnnotations, setOriginalAnnotations] = useState<DatasetAnnotation[]>([]);
  const [annotations, setAnnotations] = useState<DatasetAnnotation[]>([]);
  const [tab, setTab] = useState<AnnotationTab>('panels');
  const [bubbleType, setBubbleType] = useState<BubbleClass>('speech_bubble');
  const [selectionEnabled, setSelectionEnabled] = useState(false);
  const [draftRect, setDraftRect] = useState<DraftRect | null>(null);
  const [editingPanelId, setEditingPanelId] = useState<string | null>(null);
  const [editingBubbleId, setEditingBubbleId] = useState<string | null>(null);
  const [panelEditBaselines, setPanelEditBaselines] = useState<Record<string, [number, number, number, number]>>({});
  const [bubbleEditBaselines, setBubbleEditBaselines] = useState<
    Record<string, { bbox: [number, number, number, number]; class_name: BubbleClass }>
  >({});
  const [panelDraftBoxes, setPanelDraftBoxes] = useState<Record<string, [number, number, number, number]>>({});
  const [bubbleDrafts, setBubbleDrafts] = useState<
    Record<string, { bbox: [number, number, number, number]; class_name: BubbleClass }>
  >({});
  const [hoveredAnnotationId, setHoveredAnnotationId] = useState<string | null>(null);
  const [statusText, setStatusText] = useState('Loading annotation workspace...');
  const [errorText, setErrorText] = useState('');
  const [missingImageMessage, setMissingImageMessage] = useState('');
  const [savedPath, setSavedPath] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportMode, setExportMode] = useState<ExportMode>('full');
  const [zoomLevel, setZoomLevel] = useState(0.8);
  const [zoomInput, setZoomInput] = useState('80');

  const imageWidth = item?.width || 1;
  const imageHeight = item?.height || 1;

  const updateZoomLevel = (nextValue: number) => {
    const clamped = clampZoom(nextValue);
    setZoomLevel(clamped);
    setZoomInput(String(Math.round(clamped * 100)));
  };

  const panelAnnotations = useMemo(
    () => annotations.filter((annotation) => annotation.class_name === 'panel'),
    [annotations],
  );
  const bubbleAnnotations = useMemo(
    () => annotations.filter((annotation) => annotation.class_name !== 'panel'),
    [annotations],
  );
  const displayedPanelAnnotations = useMemo(
    () =>
      panelAnnotations.map((annotation) =>
        panelDraftBoxes[annotation.id]
          ? { ...annotation, bbox: panelDraftBoxes[annotation.id] }
          : annotation,
      ),
    [panelAnnotations, panelDraftBoxes],
  );
  const displayedBubbleAnnotations = useMemo(
    () =>
      bubbleAnnotations.map((annotation) => {
        const draft = bubbleDrafts[annotation.id];
        return draft
          ? {
              ...annotation,
              bbox: draft.bbox,
              class_name: draft.class_name,
            }
          : annotation;
      }),
    [bubbleAnnotations, bubbleDrafts],
  );
  const currentOverlayAnnotations = tab === 'panels' ? displayedPanelAnnotations : displayedBubbleAnnotations;
  const editingPanel = displayedPanelAnnotations.find((annotation) => annotation.id === editingPanelId) || null;
  const editingBubble = displayedBubbleAnnotations.find((annotation) => annotation.id === editingBubbleId) || null;
  const navigationDisabled = Boolean(missingImageMessage);

  const resetInteractions = () => {
    setSelectionEnabled(false);
    setDraftRect(null);
    setEditingPanelId(null);
    setEditingBubbleId(null);
    setPanelEditBaselines({});
    setBubbleEditBaselines({});
    setPanelDraftBoxes({});
    setBubbleDrafts({});
    setHoveredAnnotationId(null);
    editInteractionRef.current = null;
  };

  const clearPanelDraft = (annotationId: string | null) => {
    if (!annotationId) {
      return;
    }
    setPanelDraftBoxes((current) => {
      if (!current[annotationId]) {
        return current;
      }
      const next = { ...current };
      delete next[annotationId];
      return next;
    });
  };

  const clearBubbleDraft = (annotationId: string | null) => {
    if (!annotationId) {
      return;
    }
    setBubbleDrafts((current) => {
      if (!current[annotationId]) {
        return current;
      }
      const next = { ...current };
      delete next[annotationId];
      return next;
    });
  };

  const loadDatasets = async () => {
    const payload = await fetchJson<{ datasets: DatasetOption[]; default_dataset: string }>(
      '/api/annotation/datasets',
    );
    setDatasetOptions(payload.datasets || []);
    setDataset((current) => current || payload.default_dataset || payload.datasets?.[0]?.key || '');
  };

  useEffect(() => {
    void loadDatasets().catch((error) => {
      const message = error instanceof Error ? error.message : 'Failed to load datasets.';
      setErrorText(message);
      showBanner(message, 'error');
    });
  }, [showBanner]);

  useEffect(() => {
    if (!dataset) {
      return;
    }

    let cancelled = false;

    const run = async () => {
      setErrorText('');
      setMissingImageMessage('');
      setStatusText('Loading annotation item...');

      if (viewMode === 'review') {
        const queuePayload = await fetchJson<AnnotationReviewQueueResponse>(
          `/api/annotation/review-queue?dataset=${encodeURIComponent(dataset)}`,
        );
        if (cancelled) {
          return;
        }
        setQueueItems(queuePayload.items || []);
        const safeQueueIndex = Math.max(0, Math.min(queueIndex, Math.max(0, queuePayload.items.length - 1)));
        if (safeQueueIndex !== queueIndex) {
          setQueueIndex(safeQueueIndex);
        }
        const queueItem = queuePayload.items[safeQueueIndex];
        if (!queueItem) {
          setItem(null);
          setAnnotations([]);
          setOriginalAnnotations([]);
          setTotalItems(0);
          const message = 'No validation queue items available for this dataset.';
          setMissingImageMessage(message);
          setStatusText(message);
          return;
        }
        const itemPayload = await fetchJson<DatasetAnnotationItemResponse>(
          `/api/annotation/item?dataset=${encodeURIComponent(dataset)}&split=${queueItem.split}&index=${queueItem.index}`,
        );
        if (cancelled) {
          return;
        }
        setItem(itemPayload);
        setSplit(queueItem.split);
        setAnnotations((itemPayload.annotations || []).map(cloneAnnotation));
        setOriginalAnnotations((itemPayload.annotations || []).map(cloneAnnotation));
        setDatasetOptions(itemPayload.available_datasets || []);
        setTotalItems(queuePayload.total);
        resetInteractions();
        setMissingImageMessage('');
        setStatusText(
          `Loaded queue item ${safeQueueIndex + 1}/${queuePayload.total}: ${itemPayload.file_name}.`,
        );
        return;
      }

      const listPayload = await fetchJson<DatasetImageListResponse>(
        `/api/annotation/images?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}&offset=0&limit=1`,
      );
      if (cancelled) {
        return;
      }
      setTotalItems(listPayload.total);
      setDatasetOptions(listPayload.available_datasets || []);
      setQueueItems([]);

      if (!listPayload.total) {
        setItem(null);
        setAnnotations([]);
        setOriginalAnnotations([]);
        resetInteractions();
        const message = 'No image available for this split.';
        setMissingImageMessage(message);
        setStatusText(message);
        return;
      }

      const safeIndex = Math.max(0, Math.min(currentIndex, Math.max(0, listPayload.total - 1)));
      if (safeIndex !== currentIndex) {
        setCurrentIndex(safeIndex);
      }

      const itemPayload = await fetchJson<DatasetAnnotationItemResponse>(
        `/api/annotation/item?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}&index=${safeIndex}`,
      );
      if (cancelled) {
        return;
      }
      setItem(itemPayload);
      setAnnotations((itemPayload.annotations || []).map(cloneAnnotation));
      setOriginalAnnotations((itemPayload.annotations || []).map(cloneAnnotation));
      resetInteractions();
      setMissingImageMessage('');
      setStatusText(`Loaded ${itemPayload.file_name} from ${itemPayload.annotation_source} annotations.`);
    };

    void run().catch((error) => {
      if (cancelled) {
        return;
      }
      const message = error instanceof Error ? error.message : 'Failed to load annotation workspace.';
      if (message.toLowerCase().includes('image index out of range')) {
        setItem(null);
        setAnnotations([]);
        setOriginalAnnotations([]);
        resetInteractions();
        setMissingImageMessage('No image available for this dataset selection.');
        setStatusText('No image available for this dataset selection.');
        return;
      }
      setErrorText(message);
      setStatusText(message);
      showBanner(message, 'error');
    });

    return () => {
      cancelled = true;
    };
  }, [currentIndex, dataset, queueIndex, showBanner, split, viewMode]);

  const updateAnnotationBbox = (annotationId: string, bbox: [number, number, number, number]) => {
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.id === annotationId ? { ...annotation, bbox: clampBbox(bbox, imageWidth, imageHeight), points: null } : annotation,
      ),
    );
  };

  const removeAnnotation = (annotationId: string) => {
    const removedAnnotation = annotations.find((annotation) => annotation.id === annotationId) || null;
    setAnnotations((current) => current.filter((annotation) => annotation.id !== annotationId));
    clearPanelDraft(annotationId);
    clearBubbleDraft(annotationId);
    setPanelEditBaselines((current) => {
      if (!current[annotationId]) {
        return current;
      }
      const next = { ...current };
      delete next[annotationId];
      return next;
    });
    setBubbleEditBaselines((current) => {
      if (!current[annotationId]) {
        return current;
      }
      const next = { ...current };
      delete next[annotationId];
      return next;
    });
    setEditingPanelId((current) => (current === annotationId ? null : current));
    setEditingBubbleId((current) => (current === annotationId ? null : current));
    setHoveredAnnotationId((current) => (current === annotationId ? null : current));
    if (removedAnnotation) {
      const label =
        removedAnnotation.class_name === 'panel'
          ? 'Deleted panel region.'
          : `Deleted ${getRegionLabel(removedAnnotation.class_name)} region.`;
      setStatusText(label);
      showBanner(label, 'success');
    }
  };

  const togglePanelEditor = (annotation: DatasetAnnotation) => {
    if (editingPanelId === annotation.id) {
      clearPanelDraft(annotation.id);
      setEditingPanelId(null);
      editInteractionRef.current = null;
      return;
    }

    if (editingBubbleId) {
      clearBubbleDraft(editingBubbleId);
      setEditingBubbleId(null);
    }
    if (editingPanelId && editingPanelId !== annotation.id) {
      clearPanelDraft(editingPanelId);
    }

    setPanelDraftBoxes((current) => ({
      ...current,
      [annotation.id]:
        current[annotation.id] || ([...annotation.bbox] as [number, number, number, number]),
    }));
    setPanelEditBaselines((existing) => ({
      ...existing,
      [annotation.id]:
        existing[annotation.id] || ([...annotation.bbox] as [number, number, number, number]),
    }));
    setEditingPanelId(annotation.id);
    setHoveredAnnotationId(annotation.id);
    setSelectionEnabled(false);
    setDraftRect(null);
    editInteractionRef.current = null;
  };

  const toggleBubbleEditor = (annotation: DatasetAnnotation) => {
    if (annotation.class_name === 'panel') {
      return;
    }
    if (editingBubbleId === annotation.id) {
      clearBubbleDraft(annotation.id);
      setEditingBubbleId(null);
      editInteractionRef.current = null;
      return;
    }

    if (editingPanelId) {
      clearPanelDraft(editingPanelId);
      setEditingPanelId(null);
    }
    if (editingBubbleId && editingBubbleId !== annotation.id) {
      clearBubbleDraft(editingBubbleId);
    }

    setBubbleDrafts((current) => ({
      ...current,
      [annotation.id]:
        current[annotation.id] || {
          bbox: [...annotation.bbox] as [number, number, number, number],
          class_name: annotation.class_name,
        },
    }));
    setBubbleEditBaselines((existing) => ({
      ...existing,
      [annotation.id]:
        existing[annotation.id] || {
          bbox: [...annotation.bbox] as [number, number, number, number],
          class_name: annotation.class_name,
        },
    }));
    setEditingBubbleId(annotation.id);
    setHoveredAnnotationId(annotation.id);
    setSelectionEnabled(false);
    setDraftRect(null);
    editInteractionRef.current = null;
  };

  const updateBubbleType = (annotationId: string, nextType: BubbleClass) => {
    const currentDraft =
      bubbleDrafts[annotationId] || bubbleAnnotations.find((annotation) => annotation.id === annotationId);
    if (!currentDraft || currentDraft.class_name === 'panel') {
      return;
    }
    setBubbleDrafts((current) => ({
      ...current,
      [annotationId]: {
        bbox: currentDraft.bbox,
        class_name: nextType,
      },
    }));
  };

  const beginEditInteraction = (
    event: React.PointerEvent<HTMLDivElement>,
    target: EditInteraction['target'],
    startBbox: [number, number, number, number],
  ) => {
    if (!drawingSurfaceRef.current) {
      return;
    }
    event.stopPropagation();
    event.preventDefault();
    const surfaceBounds = drawingSurfaceRef.current.getBoundingClientRect();
    const pointerX = ((event.clientX - surfaceBounds.left) / surfaceBounds.width) * imageWidth;
    const pointerY = ((event.clientY - surfaceBounds.top) / surfaceBounds.height) * imageHeight;
    const overlayBounds = event.currentTarget.getBoundingClientRect();
    const resizeThreshold = Math.min(20, Math.max(12, Math.min(overlayBounds.width, overlayBounds.height) * 0.24));
    const mode =
      event.clientX >= overlayBounds.right - resizeThreshold &&
      event.clientY >= overlayBounds.bottom - resizeThreshold
        ? 'resize'
        : 'move';
    editInteractionRef.current = {
      mode,
      target,
      startPointer: { x: pointerX, y: pointerY },
      startBbox,
    };
    setSelectionEnabled(false);
    setDraftRect(null);
    setErrorText('');
  };

  const handleSurfacePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!selectionEnabled) {
      return;
    }
    const target = drawingSurfaceRef.current;
    if (!target) {
      return;
    }
    const bounds = target.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * imageWidth;
    const y = ((event.clientY - bounds.top) / bounds.height) * imageHeight;
    drawingStartRef.current = { x, y };
    setDraftRect({ x, y, width: 0, height: 0 });
    setErrorText('');
  };

  const handleSurfacePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (editInteractionRef.current && drawingSurfaceRef.current) {
      const interaction = editInteractionRef.current;
      const bounds = drawingSurfaceRef.current.getBoundingClientRect();
      const pointerX = ((event.clientX - bounds.left) / bounds.width) * imageWidth;
      const pointerY = ((event.clientY - bounds.top) / bounds.height) * imageHeight;
      const deltaX = pointerX - interaction.startPointer.x;
      const deltaY = pointerY - interaction.startPointer.y;
      let [x, y, width, height] = interaction.startBbox;
      if (interaction.mode === 'move') {
        x = interaction.startBbox[0] + deltaX;
        y = interaction.startBbox[1] + deltaY;
      } else {
        width = interaction.startBbox[2] + deltaX;
        height = interaction.startBbox[3] + deltaY;
      }
      const clamped = clampBbox([x, y, width, height], imageWidth, imageHeight);
      if (interaction.target.kind === 'panel') {
        setPanelDraftBoxes((current) => ({
          ...current,
          [interaction.target.annotationId]: clamped,
        }));
      } else {
        const activeDraft =
          bubbleDrafts[interaction.target.annotationId] ||
          bubbleAnnotations.find((annotation) => annotation.id === interaction.target.annotationId);
        if (!activeDraft || activeDraft.class_name === 'panel') {
          return;
        }
        setBubbleDrafts((current) => ({
          ...current,
          [interaction.target.annotationId]: {
            bbox: clamped,
            class_name: activeDraft.class_name,
          },
        }));
      }
      return;
    }

    if (!selectionEnabled || !drawingStartRef.current || !drawingSurfaceRef.current) {
      return;
    }
    const bounds = drawingSurfaceRef.current.getBoundingClientRect();
    const currentX = ((event.clientX - bounds.left) / bounds.width) * imageWidth;
    const currentY = ((event.clientY - bounds.top) / bounds.height) * imageHeight;
    const start = drawingStartRef.current;
    setDraftRect({
      x: Math.min(start.x, currentX),
      y: Math.min(start.y, currentY),
      width: Math.abs(currentX - start.x),
      height: Math.abs(currentY - start.y),
    });
  };

  const handleSurfacePointerUp = () => {
    if (editInteractionRef.current) {
      editInteractionRef.current = null;
    }
    drawingStartRef.current = null;
  };

  const applyPanelEditorChanges = () => {
    if (!editingPanelId) {
      return;
    }
    const draft = panelDraftBoxes[editingPanelId];
    if (draft) {
      updateAnnotationBbox(editingPanelId, draft);
      const message = 'Applied panel region changes.';
      setStatusText(message);
      showBanner(message, 'success');
    }
    clearPanelDraft(editingPanelId);
    setEditingPanelId(null);
    editInteractionRef.current = null;
  };

  const applyBubbleEditorChanges = (annotationId: string) => {
    const draft = bubbleDrafts[annotationId];
    if (!draft) {
      setEditingBubbleId(null);
      return;
    }
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.id === annotationId
          ? {
              ...annotation,
              bbox: clampBbox(draft.bbox, imageWidth, imageHeight),
              class_name: draft.class_name,
              points: null,
            }
          : annotation,
      ),
    );
    const message = 'Applied bubble region changes.';
    setStatusText(message);
    showBanner(message, 'success');
    clearBubbleDraft(annotationId);
    setEditingBubbleId(null);
    editInteractionRef.current = null;
  };

  const applyDraftSelection = () => {
    if (!draftRect || draftRect.width < 2 || draftRect.height < 2) {
      const message = 'Draw a rectangle before applying the selection.';
      setErrorText(message);
      showBanner(message, 'error');
      return;
    }
    if (tab === 'captions') {
      return;
    }
    const nextAnnotation: DatasetAnnotation = {
      id: `${Date.now()}`,
      class_name: tab === 'panels' ? 'panel' : bubbleType,
      bbox: clampBbox(
        [draftRect.x, draftRect.y, draftRect.width, draftRect.height],
        imageWidth,
        imageHeight,
      ),
      points: null,
    };
    setAnnotations((current) => [nextAnnotation, ...current]);
    setDraftRect(null);
    setSelectionEnabled(false);
    const message = tab === 'panels' ? 'Added panel region.' : `Added ${bubbleType} region.`;
    setStatusText(message);
    showBanner(message, 'success');
  };

  const restoreDefaultForTab = () => {
    if (tab === 'captions') {
      return;
    }
    setAnnotations((current) => {
      const originalPanelAnnotations = originalAnnotations.filter((annotation) => annotation.class_name === 'panel');
      const originalBubbleAnnotations = originalAnnotations.filter((annotation) => annotation.class_name !== 'panel');
      if (tab === 'panels') {
        return [...originalPanelAnnotations.map(cloneAnnotation), ...current.filter((annotation) => annotation.class_name !== 'panel')];
      }
      return [...current.filter((annotation) => annotation.class_name === 'panel'), ...originalBubbleAnnotations.map(cloneAnnotation)];
    });
    if (tab === 'panels') {
      setEditingPanelId(null);
      setPanelEditBaselines({});
      setPanelDraftBoxes({});
      const message = 'Restored default panel regions.';
      setStatusText(message);
      showBanner(message, 'success');
    } else {
      setEditingBubbleId(null);
      setBubbleEditBaselines({});
      setBubbleDrafts({});
      const message = 'Restored default bubble regions.';
      setStatusText(message);
      showBanner(message, 'success');
    }
  };

  const handleSave = async () => {
    if (!item) {
      return;
    }
    setIsSaving(true);
    setErrorText('');
    try {
      const payload = {
        dataset: item.dataset,
        split: item.split,
        image_id: item.image_id,
        file_name: item.file_name,
        width: item.width,
        height: item.height,
        annotations: annotations.map((annotation) => ({
          id: annotation.id,
          class_name: annotation.class_name,
          bbox: annotation.bbox,
          points: annotation.points,
        })),
      };
      const response = await fetchJson<SaveAnnotationsResponse>('/api/annotation/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      setSavedPath(response.annotation_path);
      setOriginalAnnotations(annotations.map(cloneAnnotation));
      const message = `Saved annotations to ${response.annotation_path}`;
      setStatusText(message);
      showBanner(message, 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save annotations.';
      setErrorText(message);
      setStatusText(message);
      showBanner(message, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const handleExport = async () => {
    if (!dataset) {
      return;
    }
    setIsExporting(true);
    setErrorText('');
    try {
      const formData = new FormData();
      formData.set('dataset', dataset);
      formData.set('export_mode', exportMode);
      const response = await fetchJson<ExportDatasetResponse>('/api/annotation/export', {
        method: 'POST',
        body: formData,
      });
      const message = `Exported dataset to ${response.output_dir}`;
      setStatusText(message);
      showBanner(message, 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to export dataset.';
      setErrorText(message);
      setStatusText(message);
      showBanner(message, 'error');
    } finally {
      setIsExporting(false);
    }
  };

  const handlePrev = () => {
    if (viewMode === 'review') {
      setQueueIndex((current) => Math.max(0, current - 1));
      return;
    }
    setCurrentIndex((current) => Math.max(0, current - 1));
  };

  const handleNext = () => {
    if (viewMode === 'review') {
      setQueueIndex((current) => Math.min(queueItems.length - 1, current + 1));
      return;
    }
    setCurrentIndex((current) => Math.min(Math.max(0, totalItems - 1), current + 1));
  };

  const summaryCounts = useMemo(
    () => ({
      panels: panelAnnotations.length,
      speech: bubbleAnnotations.filter((annotation) => annotation.class_name === 'speech_bubble').length,
      narration: bubbleAnnotations.filter((annotation) => annotation.class_name === 'narration_box').length,
      sfx: bubbleAnnotations.filter((annotation) => annotation.class_name === 'sfx').length,
    }),
    [bubbleAnnotations, panelAnnotations],
  );

  const originalBubbleMap = useMemo(
    () =>
      Object.fromEntries(
        originalAnnotations
          .filter((annotation) => annotation.class_name !== 'panel')
          .map((annotation) => [annotation.id, annotation]),
      ),
    [originalAnnotations],
  );
  const originalPanelMap = useMemo(
    () =>
      Object.fromEntries(
        originalAnnotations
          .filter((annotation) => annotation.class_name === 'panel')
          .map((annotation) => [annotation.id, annotation]),
      ),
    [originalAnnotations],
  );
  const headerStatus = useMemo(() => {
    if (!item) {
      return null;
    }
    if (viewMode === 'review') {
      return {
        prefix: `Loaded queue item ${Math.min(queueIndex + 1, Math.max(queueItems.length, 1))}/${queueItems.length || 0}:`,
        filename: item.file_name,
      };
    }
    return {
      prefix: `Loaded ${split} image ${Math.min(currentIndex + 1, Math.max(totalItems, 1))}/${totalItems || 0}:`,
      filename: item.file_name,
    };
  }, [currentIndex, item, queueIndex, queueItems.length, split, totalItems, viewMode]);

  return (
    <div className="relative flex-1 flex flex-col h-full">
      <FlashBanner banner={banner} />
      <div className="h-14 bg-surface-container-low border-b border-border flex items-center px-4 gap-4">
        <h1 className="text-sm font-medium text-foreground">Annotation Workspace</h1>
        <div className="min-w-0 flex-1 hidden xl:flex justify-center">
          {headerStatus ? (
            <div className="flex min-w-0 max-w-[26rem] items-center gap-1 text-xs text-muted-foreground">
              <span className="shrink-0">{headerStatus.prefix}</span>
              <span className="max-w-[100px] truncate font-semibold text-foreground" title={headerStatus.filename}>
                {headerStatus.filename}
              </span>
            </div>
          ) : (
            <div className="truncate text-xs text-muted-foreground">{statusText}</div>
          )}
        </div>
        <Select value={exportMode} onValueChange={(value) => setExportMode(value as ExportMode)}>
          <SelectTrigger className="h-9 w-[240px] rounded-lg border-border bg-surface-container">
            <SelectValue placeholder="Select export mode" />
          </SelectTrigger>
          <SelectContent className="max-h-72">
            <SelectItem value="full">Export to COCO Dataset</SelectItem>
            <SelectItem value="validated_bubble_only">Export to Validated COCO Dataset</SelectItem>
          </SelectContent>
        </Select>
        <button
          onClick={handleExport}
          disabled={isExporting || !dataset}
          className={cn(
            'h-9 px-3 flex items-center gap-2 rounded-lg text-sm transition-colors',
            isExporting
              ? 'bg-muted text-muted-foreground cursor-not-allowed'
              : 'bg-surface-container hover:bg-surface-container-high text-foreground border border-border',
          )}
        >
          <Download className="w-4 h-4" />
          {isExporting ? 'Exporting...' : 'Export'}
        </button>
        <button
          onClick={handleSave}
          disabled={isSaving || !item}
          className={cn(
            'h-9 px-4 flex items-center gap-2 rounded-lg text-sm transition-colors',
            isSaving
              ? 'bg-muted text-muted-foreground cursor-not-allowed'
              : 'bg-crimson text-crimson-foreground hover:opacity-90',
          )}
        >
          <Save className="w-4 h-4" />
          {isSaving ? 'Saving...' : 'Save Annotations'}
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 bg-surface p-6 overflow-auto">
          <div className="h-full flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <SegmentedControl
                options={TAB_OPTIONS}
                value={tab}
                onChange={(value) => {
                  setTab(value as AnnotationTab);
                  resetInteractions();
                }}
              />
              {tab !== 'captions' ? (
                <div className="flex items-center gap-1 ml-2 rounded-lg border border-border bg-surface-container p-1">
                  <button
                    onClick={() => updateZoomLevel(zoomLevel - 0.1)}
                    className="w-8 h-8 rounded-md hover:bg-surface-container-high text-muted-foreground hover:text-foreground flex items-center justify-center transition-colors"
                    title="Zoom out"
                  >
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  <div className="flex items-center gap-1 pr-2">
                    <input
                      value={zoomInput}
                      onChange={(event) =>
                        setZoomInput(event.target.value.replace(/[^\d]/g, '').slice(0, 3))
                      }
                      onBlur={() => updateZoomLevel((Number(zoomInput) || 80) / 100)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault();
                          updateZoomLevel((Number(zoomInput) || 80) / 100);
                        }
                      }}
                      inputMode="numeric"
                      className="w-12 h-8 rounded-md bg-transparent border border-transparent focus:border-border focus:outline-none text-right text-xs text-foreground"
                      aria-label="Zoom percentage"
                    />
                    <span className="text-xs text-foreground">%</span>
                  </div>
                  <button
                    onClick={() => updateZoomLevel(zoomLevel + 0.1)}
                    className="w-8 h-8 rounded-md hover:bg-surface-container-high text-muted-foreground hover:text-foreground flex items-center justify-center transition-colors"
                    title="Zoom in"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                </div>
              ) : null}
              <div className="ml-auto flex items-center gap-2">
                <button
                  onClick={handlePrev}
                  disabled={navigationDisabled || (viewMode === 'review' ? queueIndex <= 0 : currentIndex <= 0)}
                  className="h-9 px-3 rounded-lg border border-border bg-surface-container text-sm text-foreground disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <div className="rounded-lg border border-border bg-surface-container px-3 py-2 text-xs text-muted-foreground">
                  {viewMode === 'review'
                    ? `Queue ${Math.min(queueIndex + 1, Math.max(queueItems.length, 1))} / ${queueItems.length || 0}`
                    : `Image ${Math.min(currentIndex + 1, Math.max(totalItems, 1))} / ${totalItems || 0}`}
                </div>
                <button
                  onClick={handleNext}
                  disabled={navigationDisabled || (viewMode === 'review' ? queueIndex >= queueItems.length - 1 : currentIndex >= totalItems - 1)}
                  className="h-9 px-3 rounded-lg border border-border bg-surface-container text-sm text-foreground disabled:opacity-40"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 flex items-center justify-center bg-surface-variant/30 rounded-lg p-6 overflow-auto">
              {!item && !missingImageMessage ? (
                <div className="text-sm text-muted-foreground">Loading current image…</div>
              ) : tab === 'captions' && item ? (
                <div className="w-full h-full overflow-auto rounded-lg border border-border bg-surface-container">
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 p-4">
                    <article className="rounded-lg border border-border bg-surface-container-high p-4 space-y-3">
                      <h3 className="text-sm font-medium text-foreground">Image Summary</h3>
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Dataset</span>
                          <span className="text-foreground">{item.dataset}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Split</span>
                          <span className="text-foreground">{item.split}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Source</span>
                          <span className="text-foreground">{item.annotation_source}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Panels</span>
                          <span className="text-foreground">{summaryCounts.panels}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Speech</span>
                          <span className="text-foreground">{summaryCounts.speech}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Narration</span>
                          <span className="text-foreground">{summaryCounts.narration}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">SFX</span>
                          <span className="text-foreground">{summaryCounts.sfx}</span>
                        </div>
                      </div>
                    </article>

                    <article className="rounded-lg border border-border bg-surface-container-high p-4 space-y-3">
                      <h3 className="text-sm font-medium text-foreground">Caption Notes</h3>
                      <p className="text-sm text-muted-foreground leading-6">
                        The current annotation dataset does not provide generated caption text fields. This tab is
                        reserved as a read-only summary view while panel and bubble regions are edited in the other tabs.
                      </p>
                      {viewMode === 'review' && queueItems[queueIndex] ? (
                        <div className="rounded-lg border border-border bg-surface p-3 text-xs text-muted-foreground">
                          Review reasons: {queueItems[queueIndex].reasons.join(', ')}
                        </div>
                      ) : null}
                    </article>
                  </div>
                </div>
              ) : missingImageMessage ? (
                <div className="flex h-full w-full items-center justify-center">
                  <div className="w-full max-w-lg rounded-2xl border border-border bg-surface-container p-10 text-center shadow-sm">
                    <ImageOff className="mx-auto h-10 w-10 text-muted-foreground" />
                    <h3 className="mt-4 text-lg font-medium text-foreground">No Image Available</h3>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">{missingImageMessage}</p>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full overflow-auto">
                  <div className="min-h-full flex items-start justify-center">
                    <div
                      ref={drawingSurfaceRef}
                      className={cn(
                        'relative mx-auto bg-surface-container border border-border rounded-lg shadow-lg overflow-hidden touch-none select-none',
                        selectionEnabled ? 'cursor-crosshair' : 'cursor-default',
                      )}
                      style={{
                        aspectRatio: `${imageWidth} / ${imageHeight}`,
                        maxHeight: '100%',
                        maxWidth: 'none',
                        width: `${zoomLevel * 100}%`,
                        minWidth: '640px',
                        touchAction: 'none',
                      }}
                      onPointerDown={handleSurfacePointerDown}
                      onPointerMove={handleSurfacePointerMove}
                      onPointerUp={handleSurfacePointerUp}
                      onPointerLeave={handleSurfacePointerUp}
                    >
                      <img
                        src={item.image_url}
                        alt={item.file_name}
                        className="block w-full h-full object-contain pointer-events-none select-none"
                        draggable={false}
                        onLoad={() => setMissingImageMessage('')}
                        onError={() => {
                          const message = 'Image file is unavailable for this dataset item.';
                          setMissingImageMessage(message);
                          setStatusText(message);
                        }}
                      />

                      {currentOverlayAnnotations.map((annotation) => {
                        const isEditing =
                          (annotation.class_name === 'panel' && editingPanelId === annotation.id) ||
                          (annotation.class_name !== 'panel' && editingBubbleId === annotation.id);
                        return (
                          <RegionOverlayBox
                            key={annotation.id}
                            regionType={annotation.class_name}
                            label={
                              annotation.class_name === 'panel'
                                ? 'panel'
                                : getRegionAppearance(annotation.class_name).badgeLabel
                            }
                            rect={{
                              x: annotation.bbox[0],
                              y: annotation.bbox[1],
                              width: annotation.bbox[2],
                              height: annotation.bbox[3],
                            }}
                            canvasWidth={imageWidth}
                            canvasHeight={imageHeight}
                            isHovered={hoveredAnnotationId === annotation.id}
                            isEditing={isEditing}
                            onMouseEnter={() => setHoveredAnnotationId(annotation.id)}
                            onMouseLeave={() =>
                              setHoveredAnnotationId((current) => (current === annotation.id ? null : current))
                            }
                            onClick={() => setHoveredAnnotationId(annotation.id)}
                            onPointerDown={(event) => {
                              if (selectionEnabled || !isEditing) {
                                return;
                              }
                              beginEditInteraction(
                                event,
                                { kind: annotation.class_name === 'panel' ? 'panel' : 'bubble', annotationId: annotation.id },
                                annotation.bbox,
                              );
                            }}
                          />
                        );
                      })}

                      {draftRect ? (
                        <div
                          className="absolute border-2 border-dashed border-cyan bg-cyan/10"
                          style={rectToStyle(draftRect, imageWidth, imageHeight)}
                        />
                      ) : null}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="w-96 bg-surface-container-low border-l border-border flex flex-col overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-sm font-medium text-foreground">Inspector</h2>
          </div>

          <div className="flex-1 overflow-auto p-4 space-y-6">
            <div className="space-y-4 min-h-[210px]">
              <div>
                <label className="text-xs font-medium text-foreground mb-2 block">Dataset</label>
                <Select
                  value={dataset || undefined}
                  onValueChange={(value) => {
                    setDataset(value);
                    setCurrentIndex(0);
                    setQueueIndex(0);
                  }}
                >
                  <SelectTrigger className="w-full rounded-lg border-border bg-surface-container">
                    <SelectValue placeholder="Select dataset" />
                  </SelectTrigger>
                  <SelectContent className="max-h-72">
                    {datasetOptions.map((option) => (
                      <SelectItem key={option.key} value={option.key}>
                        {option.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium text-foreground mb-2 block">View Mode</label>
                <SegmentedControl
                  options={VIEW_OPTIONS}
                  value={viewMode}
                  onChange={(value) => {
                    setViewMode(value as AnnotationViewMode);
                    setCurrentIndex(0);
                    setQueueIndex(0);
                    resetInteractions();
                  }}
                  className="w-full"
                />
              </div>

              <div className="min-h-[88px]">
                {viewMode === 'all' ? (
                  <div>
                    <label className="text-xs font-medium text-foreground mb-2 block">Split</label>
                    <Select
                      value={split}
                      onValueChange={(value) => {
                        setSplit(value as SplitMode);
                        setCurrentIndex(0);
                      }}
                    >
                      <SelectTrigger className="w-full rounded-lg border-border bg-surface-container">
                        <SelectValue placeholder="Select split" />
                      </SelectTrigger>
                      <SelectContent className="max-h-72">
                        <SelectItem value="train">Train</SelectItem>
                        <SelectItem value="valid">Valid</SelectItem>
                        <SelectItem value="test">Test</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                ) : (
                  <div>
                    <label className="text-xs font-medium text-foreground mb-2 block">Queue Item</label>
                    <Select
                      value={String(queueIndex)}
                      onValueChange={(value) => setQueueIndex(Number(value || 0))}
                      disabled={!queueItems.length}
                    >
                      <SelectTrigger className="w-full rounded-lg border-border bg-surface-container">
                        <SelectValue placeholder="Select queue item" />
                      </SelectTrigger>
                      <SelectContent className="max-h-72">
                        {queueItems.length ? (
                          queueItems.map((queueItem) => (
                            <SelectItem key={queueItem.queue_index} value={String(queueItem.queue_index)}>
                              #{queueItem.queue_index + 1} {queueItem.split} / {queueItem.file_name}
                            </SelectItem>
                          ))
                        ) : (
                          <SelectItem value="0">No queued files</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </div>

            {tab !== 'captions' ? (
              <>
                <div className="pt-4 border-t border-border space-y-3">
                  <label className="text-xs font-medium text-foreground block">Selection Controls</label>
                  <div className="grid grid-cols-[1.2fr_1fr_auto] gap-2 items-center">
                    <button
                      onClick={() => {
                        setSelectionEnabled((current) => !current);
                        setDraftRect(null);
                        clearPanelDraft(editingPanelId);
                        clearBubbleDraft(editingBubbleId);
                        setEditingPanelId(null);
                        setEditingBubbleId(null);
                        editInteractionRef.current = null;
                      }}
                      className={cn(
                        'w-full py-2.5 px-3 rounded-lg text-sm transition-colors flex items-center justify-center gap-2',
                        selectionEnabled
                          ? 'bg-cyan text-cyan-foreground'
                          : 'bg-surface-container-high hover:bg-surface-container-highest text-foreground',
                      )}
                    >
                      <SquareDashedMousePointer className="w-4 h-4" />
                      {selectionEnabled ? 'Select On' : 'Select'}
                    </button>
                    <button
                      onClick={applyDraftSelection}
                      disabled={!draftRect}
                      className={cn(
                        'w-full py-2.5 px-3 rounded-lg text-sm flex items-center justify-center gap-2 transition-colors',
                        draftRect
                          ? 'bg-crimson text-crimson-foreground hover:opacity-90'
                          : 'bg-muted text-muted-foreground cursor-not-allowed',
                      )}
                    >
                      <PencilRuler className="w-4 h-4" />
                      Apply
                    </button>
                    <button
                      onClick={restoreDefaultForTab}
                      className="py-2.5 px-3 rounded-lg text-sm border border-border bg-transparent hover:bg-surface-container text-muted-foreground hover:text-foreground flex items-center justify-center gap-2 transition-colors"
                    >
                      <Eraser className="w-4 h-4" />
                      Reset
                    </button>
                  </div>
                </div>

                {tab === 'bubbles' ? (
                  <div>
                    <label className="text-xs font-medium text-foreground mb-2 block">Bubble Tagging</label>
                    <SegmentedControl
                      options={ANNOTATION_BUBBLE_TYPE_OPTIONS}
                      value={bubbleType}
                      onChange={(value) => setBubbleType(value as BubbleClass)}
                      className="w-full"
                    />
                  </div>
                ) : null}
              </>
            ) : null}

            {tab === 'panels' ? (
              <div className="pt-4 border-t border-border space-y-3">
                <h3 className="text-xs font-medium text-foreground">Panel Regions</h3>
                <div className="space-y-2">
                  {panelAnnotations.length ? (
                    displayedPanelAnnotations.map((annotation) => (
                      <RegionPropertyCard
                        key={annotation.id}
                        title="Panel Region"
                        regionType="panel"
                        badge={`Region ${displayedPanelAnnotations.findIndex((item) => item.id === annotation.id) + 1}`}
                        isEditing={editingPanelId === annotation.id}
                        currentBbox={annotation.bbox}
                        previousBbox={panelEditBaselines[annotation.id] || originalPanelMap[annotation.id]?.bbox}
                        updatedBbox={editingPanel?.id === annotation.id ? editingPanel.bbox : annotation.bbox}
                        highlighted={hoveredAnnotationId === annotation.id}
                        onMouseEnter={() => setHoveredAnnotationId(annotation.id)}
                        onMouseLeave={() => setHoveredAnnotationId((current) => (current === annotation.id ? null : current))}
                        onToggleEdit={() => togglePanelEditor(annotation)}
                        onApply={applyPanelEditorChanges}
                        onDelete={() => removeAnnotation(annotation.id)}
                        helperText="Move the panel outline or drag the corner handle on the image, then click the tick to apply."
                      />
                    ))
                  ) : (
                    <div className="rounded-lg border border-border bg-surface-container p-3 text-xs text-muted-foreground">
                      No panel regions yet.
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            {tab === 'bubbles' ? (
              <div className="pt-4 border-t border-border space-y-3">
                <h3 className="text-xs font-medium text-foreground">Bubble Regions</h3>
                <div className="space-y-2">
                  {bubbleAnnotations.length ? (
                    displayedBubbleAnnotations.map((annotation) => (
                      <RegionPropertyCard
                        key={annotation.id}
                        title={getRegionLabel(annotation.class_name)}
                        regionType={annotation.class_name}
                        badge={`Region ${displayedBubbleAnnotations.findIndex((item) => item.id === annotation.id) + 1}`}
                        isEditing={editingBubbleId === annotation.id}
                        currentBbox={annotation.bbox}
                        previousBbox={bubbleEditBaselines[annotation.id]?.bbox || originalBubbleMap[annotation.id]?.bbox}
                        updatedBbox={editingBubble?.id === annotation.id ? editingBubble.bbox : annotation.bbox}
                        highlighted={hoveredAnnotationId === annotation.id}
                        onMouseEnter={() => setHoveredAnnotationId(annotation.id)}
                        onMouseLeave={() => setHoveredAnnotationId((current) => (current === annotation.id ? null : current))}
                        onToggleEdit={() => toggleBubbleEditor(annotation)}
                        onApply={() => applyBubbleEditorChanges(annotation.id)}
                        onDelete={() => removeAnnotation(annotation.id)}
                        typeValue={annotation.class_name}
                        typeOptions={ANNOTATION_BUBBLE_TYPE_OPTIONS}
                        onTypeChange={(value) => updateBubbleType(annotation.id, value as BubbleClass)}
                        helperText="Move the bubble box or drag the corner handle on the image, then click the tick to apply."
                      />
                    ))
                  ) : (
                    <div className="rounded-lg border border-border bg-surface-container p-3 text-xs text-muted-foreground">
                      No bubble regions yet.
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            <div className="pt-4 border-t border-border space-y-3">
              <h3 className="text-xs font-medium text-foreground">Image Metadata</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Filename</span>
                  <span className="truncate text-right ml-3 font-semibold text-foreground" title={item?.file_name || '—'}>
                    {item?.file_name || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Resolution</span>
                  <span className="text-foreground">{item ? `${item.width}×${item.height}` : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Annotation Source</span>
                  <span className="text-foreground">{item?.annotation_source || '—'}</span>
                </div>
                {savedPath ? (
                  <div className="flex justify-between gap-4">
                    <span className="text-muted-foreground">Last Save</span>
                    <span className="text-foreground truncate text-right ml-3">{savedPath}</span>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="pt-4 border-t border-border space-y-3">
              <h3 className="text-xs font-medium text-foreground">Summary</h3>
              <div className="grid grid-cols-4 gap-2 text-xs">
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('panel').summaryClassName)}>
                  <div className="font-medium text-foreground">{summaryCounts.panels}</div>
                  <div className="text-muted-foreground">Panels</div>
                </div>
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('speech_bubble').summaryClassName)}>
                  <div className="font-medium text-foreground">{summaryCounts.speech}</div>
                  <div className="text-muted-foreground">Speech</div>
                </div>
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('narration_box').summaryClassName)}>
                  <div className="font-medium text-foreground">{summaryCounts.narration}</div>
                  <div className="text-muted-foreground">Narration</div>
                </div>
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('sfx').summaryClassName)}>
                  <div className="font-medium text-foreground">{summaryCounts.sfx}</div>
                  <div className="text-muted-foreground">SFX</div>
                </div>
              </div>
              <div className="rounded-lg border border-border bg-surface-container p-3 text-xs text-muted-foreground">
                Draw one rectangle at a time. Use the edit icons to move or resize existing regions directly on the image.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
