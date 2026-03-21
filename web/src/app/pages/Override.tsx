import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router';
import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Eraser,
  PencilRuler,
  Save,
  SquareDashedMousePointer,
  Tag,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { FlashBanner, useFlashBanner } from '../components/FlashBanner';
import { RegionOverlayBox } from '../components/RegionOverlayBox';
import { RegionPropertyCard } from '../components/RegionPropertyCard';
import { SegmentedControl } from '../components/SegmentedControl';
import { cn } from '../components/ui/utils';
import {
  getRegionAppearance,
  OVERRIDE_BUBBLE_TYPE_OPTIONS,
  getRegionLabel,
  type OverrideBubbleType as BubbleType,
} from '../lib/editor-config';
import { formatBboxLabel, rectToStyle } from '../lib/region-utils';

type OverrideTab = 'panels' | 'bubbles' | 'captions';

interface PanelBox {
  index: number;
  bbox: [number, number, number, number];
  image_path: string;
  image_url: string;
}

interface DialogueItem {
  panel?: number;
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

interface PanelResultPayload {
  request_id: string;
  filename: string;
  panels: number;
  panel_boxes: PanelBox[];
  source_image_url?: string;
}

interface OverrideSessionPayload {
  panelResult: PanelResultPayload;
  dialogue: DialogueItem[];
  captions: CaptionItem[];
  bubbleMode: string;
  panelMode: string;
  sourceImageUrl: string;
  savedAt: string;
}

interface PanelBoxOverrideItem {
  index?: number | null;
  bbox: number[];
  role?: string | null;
}

interface PanelRegionOverrideItem {
  class_name: BubbleType;
  bbox: number[];
}

interface LoadOverridesResponse {
  request_id: string;
  exists: boolean;
  overrides_path?: string | null;
  overrides?: Record<
    string,
    {
      speech_count?: string | null;
      narration_count?: string | null;
      sfx_count?: string | null;
      bubble_count?: string | null;
      bubble_sequence?: string | null;
    }
  > | null;
  panel_boxes?: PanelBoxOverrideItem[] | null;
  panel_regions?: Record<string, PanelRegionOverrideItem[]> | null;
}

interface SaveOverridesResponse {
  request_id: string;
  saved: boolean;
  overrides_path: string;
}

interface RegionItem {
  id: string;
  class_name: BubbleType;
  bbox: [number, number, number, number];
}

interface DraftRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface PanelDiffContent {
  panelText: string;
  ocrText: string;
  speechCount: number;
  narrationCount: number;
  sfxCount: number;
  bubbleSequenceText: string;
  bubbleSequenceList: string[];
}

interface PanelDiffField {
  key: string;
  label: string;
  original: string;
  updated: string;
  changed: boolean;
}

type EditTarget =
  | {
      kind: 'panel';
      panelIndex: number;
    }
  | {
      kind: 'bubble';
      panelIndex: number;
      regionId: string;
    };

interface EditInteraction {
  mode: 'move' | 'resize';
  target: EditTarget;
  startPointer: { x: number; y: number };
  startBbox: [number, number, number, number];
}

async function cropPanelFromSource(sourceUrl: string, bbox: [number, number, number, number]) {
  const image = new Image();
  image.crossOrigin = 'anonymous';

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error(`Failed to load source image: ${sourceUrl}`));
    image.src = sourceUrl;
  });

  const [x1, y1, x2, y2] = bbox;
  const width = Math.max(1, Math.round(x2 - x1));
  const height = Math.max(1, Math.round(y2 - y1));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Failed to prepare panel crop canvas.');
  }
  context.drawImage(
    image,
    Math.round(x1),
    Math.round(y1),
    width,
    height,
    0,
    0,
    width,
    height,
  );
  return canvas.toDataURL('image/png');
}

function readOverridePayload(requestId: string) {
  const candidates = [
    sessionStorage.getItem(`phase1_override_payload:${requestId}`),
    sessionStorage.getItem('phase1_override_payload'),
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    try {
      const payload = JSON.parse(candidate) as OverrideSessionPayload;
      if (payload?.panelResult?.request_id === requestId) {
        return payload;
      }
    } catch {
      continue;
    }
  }

  return null;
}

function regionsFromCaption(caption?: CaptionItem | null): RegionItem[] {
  if (!caption) {
    return [];
  }
  return [
    ...(caption.speech_boxes || []).map((bbox, index) => ({
      id: `speech-${index}`,
      class_name: 'speech' as const,
      bbox: [bbox[0], bbox[1], bbox[2], bbox[3]] as [number, number, number, number],
    })),
    ...(caption.narration_boxes || []).map((bbox, index) => ({
      id: `narration-${index}`,
      class_name: 'narration' as const,
      bbox: [bbox[0], bbox[1], bbox[2], bbox[3]] as [number, number, number, number],
    })),
    ...(caption.sfx_boxes || []).map((bbox, index) => ({
      id: `sfx-${index}`,
      class_name: 'sfx' as const,
      bbox: [bbox[0], bbox[1], bbox[2], bbox[3]] as [number, number, number, number],
    })),
  ];
}

function describeBubbleSequence(boxes: RegionItem[], panelWidth: number, panelHeight: number) {
  return boxes
    .filter((item) => item.class_name === 'speech')
    .slice()
    .sort((a, b) => {
      const rowA = (a.bbox[1] + a.bbox[3] / 2) / Math.max(panelHeight / 6, 1);
      const rowB = (b.bbox[1] + b.bbox[3] / 2) / Math.max(panelHeight / 6, 1);
      if (Math.floor(rowA) !== Math.floor(rowB)) {
        return rowA - rowB;
      }
      return b.bbox[0] + b.bbox[2] - (a.bbox[0] + a.bbox[2]);
    })
    .map((item) => {
      const [x, y, w, h] = item.bbox;
      const xCenter = x + w / 2;
      const yCenter = y + h / 2;
      const horizontal = xCenter >= panelWidth * 0.6 ? 'right' : xCenter <= panelWidth * 0.4 ? 'left' : 'center';
      const vertical = yCenter <= panelHeight * 0.35 ? 'top' : yCenter >= panelHeight * 0.68 ? 'bottom' : 'middle';
      return `${vertical}-${horizontal}`;
    });
}

function clampZoom(value: number) {
  return Math.min(2.5, Math.max(0.4, Number(value.toFixed(2))));
}

function bboxEquals(a?: number[] | null, b?: number[] | null) {
  if (!a || !b || a.length !== 4 || b.length !== 4) {
    return false;
  }
  return a.every((value, index) => Number(value) === Number(b[index]));
}

function normalizeRegionForCompare(region: RegionItem) {
  return `${region.class_name}:${region.bbox.map((value) => Math.round(Number(value))).join(',')}`;
}

function regionListEquals(a: RegionItem[], b: RegionItem[]) {
  if (a.length !== b.length) {
    return false;
  }
  const left = a.map(normalizeRegionForCompare).sort();
  const right = b.map(normalizeRegionForCompare).sort();
  return left.every((value, index) => value === right[index]);
}

function summarizeRegions(regions: RegionItem[], panelWidth: number, panelHeight: number) {
  const speechCount = regions.filter((item) => item.class_name === 'speech').length;
  const narrationCount = regions.filter((item) => item.class_name === 'narration').length;
  const sfxCount = regions.filter((item) => item.class_name === 'sfx').length;
  const bubbleSequenceList = describeBubbleSequence(regions, panelWidth, panelHeight);

  return {
    speechCount,
    narrationCount,
    sfxCount,
    bubbleSequenceList,
    bubbleSequenceText: bubbleSequenceList.join(' | ') || 'none',
  };
}

function buildPanelDiffContent(
  caption: CaptionItem | null,
  dialogue: DialogueItem | null,
  regions: RegionItem[],
  panelWidth: number,
  panelHeight: number,
): PanelDiffContent {
  const summary = summarizeRegions(regions, panelWidth, panelHeight);
  return {
    panelText: caption?.caption || '[no panel text generated]',
    ocrText: dialogue?.text || '[no dialogue detected]',
    speechCount: summary.speechCount,
    narrationCount: summary.narrationCount,
    sfxCount: summary.sfxCount,
    bubbleSequenceList: summary.bubbleSequenceList,
    bubbleSequenceText: summary.bubbleSequenceText,
  };
}

function getPanelDimensions(panel: PanelBox | null) {
  if (!panel) {
    return { width: 1, height: 1 };
  }
  return {
    width: Math.max(1, panel.bbox[2] - panel.bbox[0]),
    height: Math.max(1, panel.bbox[3] - panel.bbox[1]),
  };
}

function normalizePanelBoxOverride(
  item: PanelBoxOverrideItem,
  fallbackPanel: PanelBox | null,
  fallbackIndex: number,
): PanelBox {
  const index = Number.isFinite(Number(item.index)) ? Number(item.index) : fallbackIndex;
  return {
    index,
    bbox: [
      Number(item.bbox[0] || 0),
      Number(item.bbox[1] || 0),
      Number(item.bbox[2] || 0),
      Number(item.bbox[3] || 0),
    ] as [number, number, number, number],
    image_path: fallbackPanel?.image_path || '',
    image_url: fallbackPanel?.image_url || '',
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

function preventNativeDrag(event: React.DragEvent<HTMLImageElement>) {
  event.preventDefault();
}

function getCaptionForPanel(payload: OverrideSessionPayload | null, panelIndex: number, fallbackIndex: number) {
  if (!payload) {
    return null;
  }
  return payload.captions.find((caption) => caption.panel === panelIndex) || payload.captions[fallbackIndex] || null;
}

function getDialogueForPanel(payload: OverrideSessionPayload | null, panelIndex: number, fallbackIndex: number) {
  if (!payload) {
    return null;
  }
  return (
    payload.dialogue.find((dialogue) => Number(dialogue.panel) === panelIndex) ||
    payload.dialogue[fallbackIndex] ||
    null
  );
}

function clampRegionBbox(
  bbox: [number, number, number, number],
  panelWidth: number,
  panelHeight: number,
): [number, number, number, number] {
  const widthLimit = Math.max(1, Math.round(panelWidth));
  const heightLimit = Math.max(1, Math.round(panelHeight));

  let [x, y, w, h] = bbox.map((value) => Math.round(Number(value))) as [number, number, number, number];
  x = Math.min(Math.max(0, x), widthLimit - 1);
  y = Math.min(Math.max(0, y), heightLimit - 1);
  w = Math.min(Math.max(1, w), widthLimit - x);
  h = Math.min(Math.max(1, h), heightLimit - y);

  return [x, y, w, h];
}

function panelToEditableBbox(panel: PanelBox | null): [number, number, number, number] | null {
  if (!panel) {
    return null;
  }
  return [
    panel.bbox[0],
    panel.bbox[1],
    Math.max(1, panel.bbox[2] - panel.bbox[0]),
    Math.max(1, panel.bbox[3] - panel.bbox[1]),
  ];
}

function editableToPanelBbox(bbox: [number, number, number, number]): [number, number, number, number] {
  return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]];
}

function buildPanelDiffFields(
  originalContent: PanelDiffContent,
  updatedContent: PanelDiffContent,
): { fields: PanelDiffField[]; updatedFieldKeys: string[] } {
  const fields = [
    {
      key: 'panelText',
      label: 'Panel Text',
      original: originalContent.panelText,
      updated: updatedContent.panelText,
    },
    {
      key: 'speechCount',
      label: 'Speech Count',
      original: String(originalContent.speechCount),
      updated: String(updatedContent.speechCount),
    },
    {
      key: 'narrationCount',
      label: 'Narration Count',
      original: String(originalContent.narrationCount),
      updated: String(updatedContent.narrationCount),
    },
    {
      key: 'sfxCount',
      label: 'SFX Count',
      original: String(originalContent.sfxCount),
      updated: String(updatedContent.sfxCount),
    },
    {
      key: 'bubbleSequence',
      label: 'Bubble Sequence',
      original: originalContent.bubbleSequenceText,
      updated: updatedContent.bubbleSequenceText,
    },
    {
      key: 'ocrText',
      label: 'OCR Text',
      original: originalContent.ocrText,
      updated: updatedContent.ocrText,
    },
  ];

  return fields.reduce<{ fields: PanelDiffField[]; updatedFieldKeys: string[] }>(
    (accumulator, field) => {
      const nextField = {
        ...field,
        changed: field.original !== field.updated,
      };
      accumulator.fields.push(nextField);
      if (nextField.changed) {
        accumulator.updatedFieldKeys.push(nextField.key);
      }
      return accumulator;
    },
    { fields: [], updatedFieldKeys: [] },
  );
}

export function Override() {
  const { requestID } = useParams();
  const navigate = useNavigate();
  const { banner, showBanner } = useFlashBanner();
  const requestId = requestID || '';
  const initialPayload = requestId ? readOverridePayload(requestId) : null;

  const [payload, setPayload] = useState<OverrideSessionPayload | null>(initialPayload);
  const [sourceDimensions, setSourceDimensions] = useState<{ width: number; height: number } | null>(null);
  const [tab, setTab] = useState<OverrideTab>('panels');
  const [selectionEnabled, setSelectionEnabled] = useState(false);
  const [bubbleType, setBubbleType] = useState<BubbleType>('speech');
  const [selectedPanelIndex, setSelectedPanelIndex] = useState<number>(
    initialPayload?.panelResult?.panel_boxes?.[0]?.index || 1,
  );
  const [panelBoxes, setPanelBoxes] = useState<PanelBox[]>(initialPayload?.panelResult?.panel_boxes || []);
  const [panelRegions, setPanelRegions] = useState<Record<string, RegionItem[]>>({});
  const [draftRect, setDraftRect] = useState<DraftRect | null>(null);
  const [statusText, setStatusText] = useState('Load a request from Home to begin refining detections.');
  const [errorText, setErrorText] = useState('');
  const [savedPath, setSavedPath] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [selectedPanelPreview, setSelectedPanelPreview] = useState('');
  const [zoomLevel, setZoomLevel] = useState(0.8);
  const [zoomInput, setZoomInput] = useState('80');
  const [hoveredRegionId, setHoveredRegionId] = useState<string | null>(null);
  const [pendingRegionIds, setPendingRegionIds] = useState<string[]>([]);
  const [editingPanelIndex, setEditingPanelIndex] = useState<number | null>(null);
  const [editingBubbleId, setEditingBubbleId] = useState<string | null>(null);
  const [panelEditBaseline, setPanelEditBaseline] = useState<[number, number, number, number] | null>(null);
  const [panelDraftBoxes, setPanelDraftBoxes] = useState<Record<number, [number, number, number, number]>>({});
  const [bubbleEditBaselines, setBubbleEditBaselines] = useState<
    Record<string, { bbox: [number, number, number, number]; class_name: BubbleType }>
  >({});
  const [bubbleDrafts, setBubbleDrafts] = useState<
    Record<string, { bbox: [number, number, number, number]; class_name: BubbleType }>
  >({});

  const drawingSurfaceRef = useRef<HTMLDivElement | null>(null);
  const drawingStartRef = useRef<{ x: number; y: number } | null>(null);
  const editInteractionRef = useRef<EditInteraction | null>(null);

  const updateZoomLevel = (nextValue: number) => {
    const clamped = clampZoom(nextValue);
    setZoomLevel(clamped);
    setZoomInput(String(Math.round(clamped * 100)));
  };

  useEffect(() => {
    if (!payload?.sourceImageUrl) {
      setSourceDimensions(null);
      return;
    }
    const image = new Image();
    image.onload = () => {
      setSourceDimensions({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };
    image.src = payload.sourceImageUrl;
  }, [payload?.sourceImageUrl]);

  useEffect(() => {
    if (!requestId) {
      return;
    }

    let cancelled = false;

    const loadSaved = async () => {
      try {
        const saved = await fetchJson<LoadOverridesResponse>(`/panel-overrides/${encodeURIComponent(requestId)}`);
        if (cancelled || !payload) {
          return;
        }

        if (saved.exists) {
          setSavedPath(saved.overrides_path || '');

          const validIndexedPanelBoxes = (saved.panel_boxes || []).filter(
            (item) =>
              Number.isFinite(item.index) &&
              Array.isArray(item.bbox) &&
              item.bbox.length === 4,
          );

          if (validIndexedPanelBoxes.length) {
            const originalByIndex = new Map(
              payload.panelResult.panel_boxes.map((panel) => [panel.index, panel]),
            );
            const restoredPanelBoxes = validIndexedPanelBoxes.map((item, position) =>
              normalizePanelBoxOverride(
                item,
                originalByIndex.get(Number(item.index)),
                position + 1,
              ),
            );

            setPanelBoxes(restoredPanelBoxes);
            setSelectedPanelIndex((current) =>
              restoredPanelBoxes.some((panel) => panel.index === current)
                ? current
                : restoredPanelBoxes[0]?.index || 1,
            );
          }

          if (saved.panel_regions) {
            const normalized: Record<string, RegionItem[]> = {};
            Object.entries(saved.panel_regions).forEach(([panelIndex, regions]) => {
              normalized[panelIndex] = (regions || []).map((region, index) => ({
                id: `${panelIndex}-${region.class_name}-${index}`,
                class_name: region.class_name,
                bbox: [
                  Number(region.bbox[0] || 0),
                  Number(region.bbox[1] || 0),
                  Number(region.bbox[2] || 0),
                  Number(region.bbox[3] || 0),
                ] as [number, number, number, number],
              }));
            });
            setPanelRegions((current) => ({ ...current, ...normalized }));
          }

          setStatusText(`Loaded saved overrides for request ${requestId}.`);
        } else {
          setStatusText(`Loaded request ${requestId}. No saved overrides yet.`);
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Failed to load saved overrides.';
          setErrorText(message);
          showBanner(message, 'error');
        }
      }
    };

    loadSaved();
    return () => {
      cancelled = true;
    };
  }, [payload, requestId]);

  useEffect(() => {
    if (!payload) {
      return;
    }

    const initialRegions: Record<string, RegionItem[]> = {};
    payload.panelResult.panel_boxes.forEach((panel, index) => {
      initialRegions[String(panel.index)] = regionsFromCaption(getCaptionForPanel(payload, panel.index, index));
    });
    setPanelRegions((current) => {
      const merged = { ...initialRegions, ...current };
      return merged;
    });
    setPanelBoxes(payload.panelResult.panel_boxes);
    if (payload.panelResult.panel_boxes.length) {
      setSelectedPanelIndex(payload.panelResult.panel_boxes[0].index);
    }
  }, [payload]);

  const selectedPanel = useMemo(
    () => panelBoxes.find((panel) => panel.index === selectedPanelIndex) || panelBoxes[0] || null,
    [panelBoxes, selectedPanelIndex],
  );

  const selectedPanelDimensions = getPanelDimensions(selectedPanel);
  const selectedPanelRegions = panelRegions[String(selectedPanelIndex)] || [];
  const selectedPanelCaptionIndex = Math.max(
    0,
    payload?.panelResult.panel_boxes.findIndex((panel) => panel.index === selectedPanelIndex) ?? 0,
  );
  const selectedCaption = getCaptionForPanel(payload, selectedPanelIndex, selectedPanelCaptionIndex);
  const selectedDialogue = getDialogueForPanel(payload, selectedPanelIndex, selectedPanelCaptionIndex);
  const editableSelectedPanelBbox = panelToEditableBbox(selectedPanel);
  const originalSelectedPanelBbox = panelToEditableBbox(
    payload?.panelResult.panel_boxes.find((panel) => panel.index === selectedPanelIndex) || null,
  );
  const displayedSelectedPanelRegions = useMemo(
    () =>
      selectedPanelRegions.map((region) => {
        const draft = bubbleDrafts[region.id];
        return draft
          ? {
              ...region,
              bbox: draft.bbox,
              class_name: draft.class_name,
            }
          : region;
      }),
    [bubbleDrafts, selectedPanelRegions],
  );
  const editingBubbleRegion =
    displayedSelectedPanelRegions.find((region) => region.id === editingBubbleId) || null;
  const editingBubbleBaseline = editingBubbleId ? bubbleEditBaselines[editingBubbleId] || null : null;
  const selectedPanelDraftBbox =
    editingPanelIndex === selectedPanelIndex ? panelDraftBoxes[selectedPanelIndex] || null : null;

  useEffect(() => {
    if (!payload?.sourceImageUrl || !selectedPanel) {
      setSelectedPanelPreview('');
      return;
    }

    let cancelled = false;
    cropPanelFromSource(payload.sourceImageUrl, selectedPanel.bbox)
      .then((dataUrl) => {
        if (!cancelled) {
          setSelectedPanelPreview(dataUrl);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSelectedPanelPreview(selectedPanel.image_url || '');
        }
      });

    return () => {
      cancelled = true;
    };
  }, [payload?.sourceImageUrl, selectedPanel]);

  const perPanelSummaries = useMemo(() => {
    return panelBoxes.map((panel, index) => {
      const originalCaption = getCaptionForPanel(payload, panel.index, index);
      const originalDialogue = getDialogueForPanel(payload, panel.index, index);
      const originalRegions = regionsFromCaption(originalCaption);
      const regions = panelRegions[String(panel.index)] || originalRegions;
      const originalPanel = payload?.panelResult.panel_boxes.find((item) => item.index === panel.index) || null;
      const originalDimensions = getPanelDimensions(originalPanel || panel);
      const currentDimensions = getPanelDimensions(panel);
      const hasPanelChange = !bboxEquals(panel.bbox, originalPanel?.bbox || null);
      const hasRegionChange = !regionListEquals(regions, originalRegions);
      const hasDiff = hasPanelChange || hasRegionChange;
      const originalContent = buildPanelDiffContent(
        originalCaption,
        originalDialogue,
        originalRegions,
        originalDimensions.width,
        originalDimensions.height,
      );
      const updatedContent = hasDiff
        ? buildPanelDiffContent(
            originalCaption,
            originalDialogue,
            regions,
            currentDimensions.width,
            currentDimensions.height,
          )
        : originalContent;

      return {
        panel,
        originalCaption,
        originalContent,
        updatedContent,
        speechCount: updatedContent.speechCount,
        narrationCount: updatedContent.narrationCount,
        sfxCount: updatedContent.sfxCount,
        bubbleSequence: updatedContent.bubbleSequenceText,
        regions,
        hasDiff,
      };
    });
  }, [panelBoxes, panelRegions, payload]);
  const canEdit = tab !== 'captions';
  const canApplySelection = canEdit && Boolean(draftRect);

  const clearPanelDraft = (panelIndex: number | null) => {
    if (panelIndex === null) {
      return;
    }
    setPanelDraftBoxes((current) => {
      if (!current[panelIndex]) {
        return current;
      }
      const next = { ...current };
      delete next[panelIndex];
      return next;
    });
  };

  const clearBubbleDraft = (regionId: string | null) => {
    if (!regionId) {
      return;
    }
    setBubbleDrafts((current) => {
      if (!current[regionId]) {
        return current;
      }
      const next = { ...current };
      delete next[regionId];
      return next;
    });
  };

  const updatePanelEditableBbox = (panelIndex: number, bbox: [number, number, number, number]) => {
    setPanelBoxes((current) =>
      current.map((panel) =>
        panel.index === panelIndex ? { ...panel, bbox: editableToPanelBbox(bbox) } : panel,
      ),
    );
  };

  const updateRegionBbox = (
    panelIndex: number,
    regionId: string,
    bbox: [number, number, number, number],
  ) => {
    updateRegion(panelIndex, regionId, (region) => ({
      ...region,
      bbox,
    }));
  };

  const beginEditInteraction = (
    event: React.PointerEvent<HTMLDivElement>,
    target: EditTarget,
    startBbox: [number, number, number, number],
    boundsWidth: number,
    boundsHeight: number,
  ) => {
    if (!drawingSurfaceRef.current) {
      return;
    }
    event.stopPropagation();
    event.preventDefault();
    const surfaceBounds = drawingSurfaceRef.current.getBoundingClientRect();
    const pointerX = ((event.clientX - surfaceBounds.left) / surfaceBounds.width) * boundsWidth;
    const pointerY = ((event.clientY - surfaceBounds.top) / surfaceBounds.height) * boundsHeight;
    const overlayBounds = event.currentTarget.getBoundingClientRect();
    const resizeThreshold = Math.min(
      20,
      Math.max(12, Math.min(overlayBounds.width, overlayBounds.height) * 0.24),
    );
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
    const dimensions = tab === 'panels' ? sourceDimensions : selectedPanelDimensions;
    if (!dimensions) {
      return;
    }
    const x = ((event.clientX - bounds.left) / bounds.width) * dimensions.width;
    const y = ((event.clientY - bounds.top) / bounds.height) * dimensions.height;
    drawingStartRef.current = { x, y };
    setDraftRect({ x, y, width: 0, height: 0 });
    setErrorText('');
  };

  const handleSurfacePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (editInteractionRef.current && drawingSurfaceRef.current) {
      const interaction = editInteractionRef.current;
      const dimensions =
        interaction.target.kind === 'panel' ? sourceDimensions : selectedPanelDimensions;
      if (!dimensions) {
        return;
      }
      const bounds = drawingSurfaceRef.current.getBoundingClientRect();
      const pointerX = ((event.clientX - bounds.left) / bounds.width) * dimensions.width;
      const pointerY = ((event.clientY - bounds.top) / bounds.height) * dimensions.height;
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

      const clamped = clampRegionBbox([x, y, width, height], dimensions.width, dimensions.height);
      if (interaction.target.kind === 'panel') {
        setPanelDraftBoxes((current) => ({
          ...current,
          [interaction.target.panelIndex]: clamped,
        }));
      } else {
        const activeRegion =
          bubbleDrafts[interaction.target.regionId] ||
          selectedPanelRegions.find((region) => region.id === interaction.target.regionId);
        if (!activeRegion) {
          return;
        }
        setBubbleDrafts((current) => ({
          ...current,
          [interaction.target.regionId]: {
            bbox: clamped,
            class_name: activeRegion.class_name,
          },
        }));
      }
      return;
    }

    if (!selectionEnabled || !drawingStartRef.current || !drawingSurfaceRef.current) {
      return;
    }
    const target = drawingSurfaceRef.current;
    const bounds = target.getBoundingClientRect();
    const dimensions = tab === 'panels' ? sourceDimensions : selectedPanelDimensions;
    if (!dimensions) {
      return;
    }
    const currentX = ((event.clientX - bounds.left) / bounds.width) * dimensions.width;
    const currentY = ((event.clientY - bounds.top) / bounds.height) * dimensions.height;
    const start = drawingStartRef.current;
    const x = Math.min(start.x, currentX);
    const y = Math.min(start.y, currentY);
    const width = Math.abs(currentX - start.x);
    const height = Math.abs(currentY - start.y);
    setDraftRect({ x, y, width, height });
  };

  const handleSurfacePointerUp = () => {
    if (editInteractionRef.current) {
      editInteractionRef.current = null;
    }
    drawingStartRef.current = null;
  };

  const restoreCurrentTab = () => {
    if (!payload || !selectedPanel) {
      return;
    }

    if (tab === 'panels') {
      const original = payload.panelResult.panel_boxes.find((panel) => panel.index === selectedPanel.index);
      if (!original) {
        return;
      }
      setPanelBoxes((current) =>
        current.map((panel) => (panel.index === selectedPanel.index ? { ...panel, bbox: original.bbox } : panel)),
      );
      const message = `Restored original panel box for Panel ${selectedPanel.index}.`;
      setStatusText(message);
      showBanner(message, 'success');
      setEditingPanelIndex(null);
      setPanelEditBaseline(null);
      clearPanelDraft(selectedPanel.index);
    }

    if (tab === 'bubbles') {
      const currentRegionIds = new Set((panelRegions[String(selectedPanel.index)] || []).map((region) => region.id));
      const originalCaption = getCaptionForPanel(payload, selectedPanel.index, selectedPanelCaptionIndex);
      setPanelRegions((current) => ({
        ...current,
        [String(selectedPanel.index)]: regionsFromCaption(originalCaption),
      }));
      setPendingRegionIds((current) => current.filter((id) => !currentRegionIds.has(id)));
      setHoveredRegionId(null);
      const message = `Restored original bubble regions for Panel ${selectedPanel.index}.`;
      setStatusText(message);
      showBanner(message, 'success');
      clearBubbleDraft(editingBubbleId);
      setEditingBubbleId(null);
    }

    setDraftRect(null);
  };

  const applyDraftSelection = () => {
    if (!draftRect || draftRect.width < 2 || draftRect.height < 2) {
      const message = 'Draw a rectangle before applying the selection.';
      setErrorText(message);
      showBanner(message, 'error');
      return;
    }

    if (tab === 'panels') {
      const nextBbox: [number, number, number, number] = [
        Math.round(draftRect.x),
        Math.round(draftRect.y),
        Math.round(draftRect.x + draftRect.width),
        Math.round(draftRect.y + draftRect.height),
      ];
      const nextIndex = panelBoxes.reduce((maxValue, panel) => Math.max(maxValue, panel.index), 0) + 1;
      const nextPanel: PanelBox = {
        index: nextIndex,
        bbox: nextBbox,
        image_path: '',
        image_url: '',
      };
      setPanelBoxes((current) => [...current, nextPanel]);
      setPanelRegions((current) => ({
        ...current,
        [String(nextIndex)]: current[String(nextIndex)] || [],
      }));
      setSelectedPanelIndex(nextIndex);
      const message = `Added Panel ${nextIndex}.`;
      setStatusText(message);
      showBanner(message, 'success');
    }

    if (tab === 'bubbles') {
      if (!selectedPanel) {
        const message = 'Select a panel before adding a bubble region.';
        setErrorText(message);
        showBanner(message, 'error');
        return;
      }
      const region: RegionItem = {
        id: `${selectedPanel.index}-${bubbleType}-${Date.now()}`,
        class_name: bubbleType,
        bbox: [
          Math.round(draftRect.x),
          Math.round(draftRect.y),
          Math.round(draftRect.width),
          Math.round(draftRect.height),
        ],
      };
      setPanelRegions((current) => ({
        ...current,
        [String(selectedPanel.index)]: [region, ...(current[String(selectedPanel.index)] || [])],
      }));
      setPendingRegionIds((current) => [region.id, ...current.filter((id) => id !== region.id)]);
      setHoveredRegionId(region.id);
      const message = `Added ${bubbleType} region to Panel ${selectedPanel.index}.`;
      setStatusText(message);
      showBanner(message, 'success');
    }

    setDraftRect(null);
    setSelectionEnabled(false);
  };

  const removeRegion = (panelIndex: number, regionId: string) => {
    setPanelRegions((current) => ({
      ...current,
      [String(panelIndex)]: (current[String(panelIndex)] || []).filter((item) => item.id !== regionId),
    }));
    setPendingRegionIds((current) => current.filter((id) => id !== regionId));
    setHoveredRegionId((current) => (current === regionId ? null : current));
    setBubbleEditBaselines((current) => {
      if (!current[regionId]) {
        return current;
      }
      const next = { ...current };
      delete next[regionId];
      return next;
    });
    clearBubbleDraft(regionId);
    setEditingBubbleId((current) => (current === regionId ? null : current));
  };

  const updateRegion = (panelIndex: number, regionId: string, updater: (region: RegionItem) => RegionItem) => {
    const panel = panelBoxes.find((item) => item.index === panelIndex) || null;
    const dimensions = getPanelDimensions(panel);

    setPanelRegions((current) => ({
      ...current,
      [String(panelIndex)]: (current[String(panelIndex)] || []).map((region) => {
        if (region.id !== regionId) {
          return region;
        }
        const nextRegion = updater(region);
        return {
          ...nextRegion,
          bbox: clampRegionBbox(nextRegion.bbox, dimensions.width, dimensions.height),
        };
      }),
    }));
    setPendingRegionIds((current) => (current.includes(regionId) ? current : [regionId, ...current]));
    setHoveredRegionId(regionId);
  };

  const removePanel = (panelIndex: number) => {
    const removedRegionIds = (panelRegions[String(panelIndex)] || []).map((region) => region.id);
    const remainingPanels = panelBoxes.filter((panel) => panel.index !== panelIndex);
    const nextSelectedPanel =
      remainingPanels.find((panel) => panel.index > panelIndex) ||
      remainingPanels[remainingPanels.length - 1] ||
      null;

    setPanelBoxes(remainingPanels);
    setPanelRegions((current) => {
      if (!current[String(panelIndex)]) {
        return current;
      }
      const next = { ...current };
      delete next[String(panelIndex)];
      return next;
    });
    setPendingRegionIds((current) => current.filter((id) => !removedRegionIds.includes(id)));
    setSelectedPanelIndex(nextSelectedPanel?.index || 1);
    clearPanelDraft(panelIndex);
    if (editingPanelIndex === panelIndex) {
      setEditingPanelIndex(null);
      setPanelEditBaseline(null);
    }
    if (selectedPanelIndex === panelIndex) {
      setHoveredRegionId(null);
    }
    if (editingBubbleId && removedRegionIds.includes(editingBubbleId)) {
      clearBubbleDraft(editingBubbleId);
      setEditingBubbleId(null);
    }
    editInteractionRef.current = null;
    const message = remainingPanels.length
      ? `Deleted Panel ${panelIndex}.`
      : `Deleted Panel ${panelIndex}. No panels remain.`;
    setStatusText(message);
    showBanner(message, 'success');
  };

  const updateRegionType = (panelIndex: number, regionId: string, nextType: BubbleType) => {
    const currentDraft =
      bubbleDrafts[regionId] || selectedPanelRegions.find((region) => region.id === regionId);
    if (!currentDraft) {
      return;
    }
    setBubbleDrafts((current) => ({
      ...current,
      [regionId]: {
        bbox: currentDraft.bbox,
        class_name: nextType,
      },
    }));
  };

  const togglePanelEditor = () => {
    if (!editableSelectedPanelBbox || !selectedPanel) {
      return;
    }
    if (editingPanelIndex === selectedPanel.index) {
      clearPanelDraft(selectedPanel.index);
      setEditingPanelIndex(null);
      setPanelEditBaseline(null);
      editInteractionRef.current = null;
      return;
    }

    if (editingPanelIndex !== null) {
      clearPanelDraft(editingPanelIndex);
    }
    if (editingBubbleId) {
      clearBubbleDraft(editingBubbleId);
      setEditingBubbleId(null);
    }

    setPanelDraftBoxes((current) => ({
      ...current,
      [selectedPanel.index]:
        current[selectedPanel.index] || ([...editableSelectedPanelBbox] as [number, number, number, number]),
    }));
    setEditingPanelIndex(selectedPanel.index);
    setPanelEditBaseline([...editableSelectedPanelBbox] as [number, number, number, number]);
    setSelectionEnabled(false);
    setDraftRect(null);
    editInteractionRef.current = null;
  };

  const toggleBubbleEditor = (region: RegionItem) => {
    if (editingBubbleId === region.id) {
      clearBubbleDraft(region.id);
      setEditingBubbleId(null);
      editInteractionRef.current = null;
      return;
    }

    if (editingBubbleId) {
      clearBubbleDraft(editingBubbleId);
    }
    if (editingPanelIndex !== null) {
      clearPanelDraft(editingPanelIndex);
      setEditingPanelIndex(null);
      setPanelEditBaseline(null);
    }

    setBubbleDrafts((current) => ({
      ...current,
      [region.id]:
        current[region.id] || {
          bbox: [...region.bbox] as [number, number, number, number],
          class_name: region.class_name,
        },
    }));
    setBubbleEditBaselines((existing) => ({
      ...existing,
      [region.id]: existing[region.id] || {
        bbox: [...region.bbox] as [number, number, number, number],
        class_name: region.class_name,
      },
    }));
    setEditingBubbleId(region.id);
    setSelectionEnabled(false);
    setDraftRect(null);
    setHoveredRegionId(region.id);
    editInteractionRef.current = null;
  };

  const applyPanelEditorChanges = () => {
    if (editingPanelIndex === null) {
      return;
    }
    const draft = panelDraftBoxes[editingPanelIndex];
    if (draft) {
      updatePanelEditableBbox(editingPanelIndex, draft);
      const message = `Applied panel edit for Panel ${editingPanelIndex}.`;
      setStatusText(message);
      showBanner(message, 'success');
    }
    clearPanelDraft(editingPanelIndex);
    setEditingPanelIndex(null);
    setPanelEditBaseline(null);
    editInteractionRef.current = null;
  };

  const applyBubbleEditorChanges = (regionId: string) => {
    const draft = bubbleDrafts[regionId];
    if (!draft) {
      setEditingBubbleId(null);
      return;
    }

    updateRegion(selectedPanelIndex, regionId, (region) => ({
      ...region,
      bbox: draft.bbox,
      class_name: draft.class_name,
    }));
    const message = `Applied bubble edit for Panel ${selectedPanelIndex}.`;
    setStatusText(message);
    showBanner(message, 'success');
    clearBubbleDraft(regionId);
    setEditingBubbleId(null);
    editInteractionRef.current = null;
  };

  const saveOverrides = async () => {
    if (!payload || !requestId) {
      return;
    }
    setIsSaving(true);
    setErrorText('');

    const overrides = Object.fromEntries(
      perPanelSummaries.map((summary) => [
        String(summary.panel.index),
        {
          speech_count: String(summary.speechCount),
          narration_count: String(summary.narrationCount),
          sfx_count: String(summary.sfxCount),
          bubble_count: String(summary.speechCount),
          bubble_sequence: summary.bubbleSequence,
        },
      ]),
    );

    const requestBody = {
      request_id: requestId,
      overrides,
      panel_boxes: panelBoxes.map((panel) => ({
        index: panel.index,
        bbox: panel.bbox,
        role: 'panel',
      })),
      panel_regions: Object.fromEntries(
        Object.entries(panelRegions).map(([panelIndex, regions]) => [
          panelIndex,
          regions.map((region) => ({
            class_name: region.class_name,
            bbox: region.bbox,
          })),
        ]),
      ),
    };

    try {
      const saved = await fetchJson<SaveOverridesResponse>('/panel-overrides', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      setSavedPath(saved.overrides_path);
      setPendingRegionIds([]);
      const message = `Overrides saved to ${saved.overrides_path}`;
      setStatusText(message);
      showBanner(message, 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save overrides.';
      setErrorText(message);
      showBanner(message, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const renderDiffPreview = (summary: (typeof perPanelSummaries)[number]) => {
    const { fields, updatedFieldKeys } = buildPanelDiffFields(summary.originalContent, summary.updatedContent);
    return (
      <div className="rounded-lg border border-border bg-surface-container-high p-4 space-y-3">
        {fields.map((field) => (
          <div key={`inline-${summary.panel.index}-${field.key}`} className="space-y-1.5">
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{field.label}</div>
            {updatedFieldKeys.includes(field.key) ? (
              <>
                <div className="rounded-md border border-crimson/20 bg-crimson/10 px-3 py-2 text-sm leading-6 text-foreground whitespace-pre-wrap">
                  {field.original}
                </div>
                <div className="rounded-md border border-cyan/20 bg-cyan/10 px-3 py-2 text-sm leading-6 text-foreground whitespace-pre-wrap">
                  {field.updated}
                </div>
              </>
            ) : (
              <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm leading-6 text-foreground whitespace-pre-wrap">
                {field.original}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  if (!requestId) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface">
        <div className="text-center">
          <h1 className="text-lg font-medium text-foreground">Missing request id</h1>
          <p className="text-sm text-muted-foreground mt-2">Open the override page from the Home workflow.</p>
        </div>
      </div>
    );
  }

  if (!payload) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface">
        <div className="max-w-lg text-center space-y-3 px-6">
          <AlertCircle className="w-10 h-10 text-crimson mx-auto" />
          <h1 className="text-lg font-medium text-foreground">Request context unavailable</h1>
          <p className="text-sm text-muted-foreground">
            The override page needs the request payload created from Home. Open the image in the
            Home workspace, run Phase 1, then come back to this request.
          </p>
          <button
            onClick={() => navigate('/ui_v2/home')}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-crimson text-crimson-foreground"
          >
            Go to Home
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex-1 flex flex-col h-full">
      <FlashBanner banner={banner} />
      <div className="h-14 bg-surface-container-low border-b border-border flex items-center px-4 gap-4">
        <h1 className="text-sm font-medium text-foreground">Override Workspace</h1>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Request</span>
          <span className="font-mono text-foreground">{requestId}</span>
        </div>
        <div className="flex-1" />
        <div className="text-xs text-muted-foreground hidden xl:block">{statusText}</div>
        <button
          onClick={saveOverrides}
          disabled={isSaving}
          className={cn(
            'h-9 px-4 flex items-center gap-2 rounded-lg text-sm transition-colors',
            isSaving
              ? 'bg-muted text-muted-foreground cursor-not-allowed'
              : 'bg-crimson text-crimson-foreground hover:opacity-90',
          )}
        >
          <Save className="w-4 h-4" />
          {isSaving ? 'Saving...' : 'Save Overrides'}
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 bg-surface p-6 overflow-auto">
          <div className="h-full flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <SegmentedControl
                options={[
                  { value: 'panels', label: 'Panels' },
                  { value: 'bubbles', label: 'Bubbles' },
                  { value: 'captions', label: 'Diff' },
                ]}
                value={tab}
                onChange={(value) => {
                  setTab(value as OverrideTab);
                  setDraftRect(null);
                  setSelectionEnabled(false);
                  setHoveredRegionId(null);
                  clearPanelDraft(editingPanelIndex);
                  clearBubbleDraft(editingBubbleId);
                  setEditingPanelIndex(null);
                  setEditingBubbleId(null);
                  setPanelEditBaseline(null);
                  editInteractionRef.current = null;
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
                      onChange={(event) => setZoomInput(event.target.value.replace(/[^\d]/g, '').slice(0, 3))}
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

              <div className="ml-auto text-xs text-muted-foreground">
                {savedPath ? `Saved override: ${savedPath}` : 'Unsaved local edits'}
              </div>
            </div>

            <div className="flex-1 flex items-center justify-center bg-surface-variant/30 rounded-lg p-6 overflow-auto">
              {tab === 'captions' ? (
                <div className="w-full h-full overflow-auto rounded-lg border border-border bg-surface-container">
                  <div className="sticky top-0 z-10 border-b border-border bg-surface-container/95 px-4 py-3 backdrop-blur">
                    <div className="text-sm font-medium text-foreground">Caption Diff Preview</div>
                    <div className="text-xs text-muted-foreground">
                      Changed fields show the original value in red and the updated value directly beneath in green.
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-4 p-4">
                    {perPanelSummaries.map((summary) => (
                      <article
                        key={summary.panel.index}
                        className="rounded-lg border border-border bg-surface-container-high p-4"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <h3 className="text-sm font-medium text-foreground">Panel {summary.panel.index}</h3>
                            <p className="text-xs text-muted-foreground">
                              Read-only diff based on applied panel or bubble overrides
                            </p>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {summary.speechCount} speech • {summary.narrationCount} narration • {summary.sfxCount} sfx
                          </div>
                        </div>

                        {renderDiffPreview(summary)}

                        {!summary.hasDiff ? (
                          <div className="mt-3 rounded bg-surface p-3 text-xs text-muted-foreground">
                            No applied override for this panel yet. Updated content is intentionally matching the
                            original content.
                          </div>
                        ) : null}
                      </article>
                    ))}
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
                        aspectRatio:
                          tab === 'panels'
                            ? sourceDimensions
                              ? `${sourceDimensions.width} / ${sourceDimensions.height}`
                              : '2 / 3'
                            : `${selectedPanelDimensions.width} / ${selectedPanelDimensions.height}`,
                        maxWidth: 'none',
                        width: `${zoomLevel * 100}%`,
                        minWidth: tab === 'bubbles' ? '480px' : '640px',
                        touchAction: 'none',
                      }}
                      onPointerDown={handleSurfacePointerDown}
                      onPointerMove={handleSurfacePointerMove}
                      onPointerUp={handleSurfacePointerUp}
                      onPointerLeave={handleSurfacePointerUp}
                    >
                      {tab === 'panels' ? (
                        <>
                          {payload.sourceImageUrl ? (
                            <img
                              src={payload.sourceImageUrl}
                              alt={payload.panelResult.filename}
                              className="block w-full h-full object-fill pointer-events-none select-none"
                              draggable={false}
                              onDragStart={preventNativeDrag}
                            />
                          ) : null}

                          {sourceDimensions &&
                            panelBoxes.map((panel) => {
                              const editablePanelBbox =
                                panelDraftBoxes[panel.index] || panelToEditableBbox(panel);
                              if (!editablePanelBbox) {
                                return null;
                              }
                              const [x1, y1, width, height] = editablePanelBbox;
                              return (
                                <RegionOverlayBox
                                  key={panel.index}
                                  regionType="panel"
                                  label={`Panel ${panel.index}`}
                                  rect={{ x: x1, y: y1, width, height }}
                                  canvasWidth={sourceDimensions.width}
                                  canvasHeight={sourceDimensions.height}
                                  className={cn(
                                    panel.index === selectedPanelIndex
                                      ? undefined
                                      : 'border-foreground/60 bg-foreground/5',
                                  )}
                                  badgeClassName={cn(
                                    panel.index === selectedPanelIndex
                                      ? undefined
                                      : 'bg-foreground text-background',
                                  )}
                                  isEditing={editingPanelIndex === panel.index}
                                  onClick={() => {
                                    if (editingPanelIndex === panel.index) {
                                      return;
                                    }
                                    setSelectedPanelIndex(panel.index);
                                    setHoveredRegionId(null);
                                    clearPanelDraft(editingPanelIndex);
                                    clearBubbleDraft(editingBubbleId);
                                    setEditingPanelIndex(null);
                                    setPanelEditBaseline(null);
                                    setEditingBubbleId(null);
                                  }}
                                  onPointerDown={(event) => {
                                    if (
                                      selectionEnabled ||
                                      editingPanelIndex !== panel.index ||
                                      !sourceDimensions
                                    ) {
                                      return;
                                    }
                                    beginEditInteraction(
                                      event,
                                      { kind: 'panel', panelIndex: panel.index },
                                      [x1, y1, width, height],
                                      sourceDimensions.width,
                                      sourceDimensions.height,
                                    );
                                  }}
                                />
                              );
                            })}

                          {sourceDimensions && draftRect ? (
                            <div
                              className="absolute border-2 border-dashed border-cyan bg-cyan/10"
                              style={rectToStyle(draftRect, sourceDimensions.width, sourceDimensions.height)}
                            />
                          ) : null}
                        </>
                      ) : (
                        <>
                          {selectedPanel ? (
                            <img
                              src={selectedPanelPreview || selectedPanel.image_url}
                              alt={`Panel ${selectedPanel.index}`}
                              className="block w-full h-full object-fill pointer-events-none select-none"
                              draggable={false}
                              onDragStart={preventNativeDrag}
                            />
                          ) : null}

                          {displayedSelectedPanelRegions.map((region) => (
                            <RegionOverlayBox
                              key={region.id}
                              regionType={region.class_name}
                              label={region.class_name}
                              rect={{
                                x: region.bbox[0],
                                y: region.bbox[1],
                                width: region.bbox[2],
                                height: region.bbox[3],
                              }}
                              canvasWidth={selectedPanelDimensions.width}
                              canvasHeight={selectedPanelDimensions.height}
                              isHovered={hoveredRegionId === region.id}
                              isEditing={editingBubbleId === region.id}
                              onMouseEnter={() => setHoveredRegionId(region.id)}
                              onMouseLeave={() => setHoveredRegionId((current) => (current === region.id ? null : current))}
                              onClick={() => setHoveredRegionId(region.id)}
                              onPointerDown={(event) => {
                                if (selectionEnabled || editingBubbleId !== region.id) {
                                  return;
                                }
                                beginEditInteraction(
                                  event,
                                  {
                                    kind: 'bubble',
                                    panelIndex: selectedPanelIndex,
                                    regionId: region.id,
                                  },
                                  [...region.bbox] as [number, number, number, number],
                                  selectedPanelDimensions.width,
                                  selectedPanelDimensions.height,
                                );
                              }}
                            />
                          ))}

                          {draftRect ? (
                            <div
                              className="absolute border-2 border-dashed border-cyan bg-cyan/10"
                              style={rectToStyle(draftRect, selectedPanelDimensions.width, selectedPanelDimensions.height)}
                            />
                          ) : null}
                        </>
                      )}
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
            {(errorText || statusText) && (
              <div
                className={cn(
                  'p-3 rounded-lg border',
                  errorText ? 'bg-crimson/10 border-crimson/20' : 'bg-cyan/10 border-cyan/20',
                )}
              >
                <div className="flex items-center gap-2 mb-2">
                  {errorText ? (
                    <AlertCircle className="w-4 h-4 text-crimson" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 text-cyan" />
                  )}
                  <span className="text-xs font-medium text-foreground">
                    {errorText ? 'Needs Attention' : 'Workspace Status'}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">{errorText || statusText}</p>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-foreground mb-2 block">Active Panel</label>
                <div className="grid grid-cols-3 gap-2">
                  {panelBoxes.map((panel) => (
                    <button
                      key={panel.index}
                      onClick={() => {
                        setSelectedPanelIndex(panel.index);
                        setHoveredRegionId(null);
                        clearPanelDraft(editingPanelIndex);
                        clearBubbleDraft(editingBubbleId);
                        setEditingPanelIndex(null);
                        setPanelEditBaseline(null);
                        setEditingBubbleId(null);
                      }}
                      className={cn(
                        'rounded-lg border px-3 py-2 text-xs transition-colors',
                        selectedPanelIndex === panel.index
                          ? 'border-crimson bg-crimson/10 text-foreground'
                          : 'border-border bg-surface-container text-muted-foreground hover:text-foreground',
                      )}
                    >
                      Panel {panel.index}
                    </button>
                  ))}
                </div>
              </div>

              {tab === 'panels' && selectedPanel && editableSelectedPanelBbox ? (
                <RegionPropertyCard
                  title="Selected Panel Region"
                  regionType="panel"
                  badge={`Panel ${selectedPanel.index}`}
                  isEditing={editingPanelIndex === selectedPanel.index}
                  currentBbox={selectedPanelDraftBbox || editableSelectedPanelBbox}
                  previousBbox={panelEditBaseline || originalSelectedPanelBbox}
                  updatedBbox={selectedPanelDraftBbox || editableSelectedPanelBbox}
                  onToggleEdit={togglePanelEditor}
                  onApply={applyPanelEditorChanges}
                  onDelete={() => removePanel(selectedPanel.index)}
                  helperText="Click edit, then move the panel box or drag its corner handle. The tick applies the draft."
                />
              ) : null}

              {tab !== 'captions' && (
                <>
                  <div>
                    <label className="text-xs font-medium text-foreground mb-2 block">
                      Selection Controls
                    </label>
                    <div className="grid grid-cols-[1.2fr_1fr_auto] gap-2 items-center">
                      <button
                        onClick={() => {
                          if (!canEdit) {
                            return;
                          }
                          setSelectionEnabled((current) => !current);
                          setDraftRect(null);
                          editInteractionRef.current = null;
                          clearPanelDraft(editingPanelIndex);
                          clearBubbleDraft(editingBubbleId);
                          setEditingPanelIndex(null);
                          setPanelEditBaseline(null);
                          setEditingBubbleId(null);
                        }}
                        disabled={!canEdit}
                        className={cn(
                          'w-full py-2.5 px-3 rounded-lg text-sm transition-colors flex items-center justify-center gap-2',
                          selectionEnabled
                            ? 'bg-cyan text-cyan-foreground'
                            : 'bg-surface-container-high hover:bg-surface-container-highest text-foreground',
                          !canEdit && 'opacity-50 cursor-not-allowed',
                        )}
                        title={selectionEnabled ? 'Selection enabled' : 'Enable selection'}
                      >
                        <SquareDashedMousePointer className="w-4 h-4" />
                        {selectionEnabled ? 'Select On' : 'Select'}
                      </button>
                      <button
                        onClick={applyDraftSelection}
                        disabled={!canApplySelection}
                        className={cn(
                          'w-full py-2.5 px-3 rounded-lg text-sm flex items-center justify-center gap-2 transition-colors',
                          canApplySelection
                            ? 'bg-crimson text-crimson-foreground hover:opacity-90'
                            : 'bg-muted text-muted-foreground cursor-not-allowed',
                        )}
                        title="Apply selection"
                      >
                        <PencilRuler className="w-4 h-4" />
                        Apply
                      </button>
                      <button
                        onClick={restoreCurrentTab}
                        className="py-2.5 px-3 rounded-lg text-sm border border-border bg-transparent hover:bg-surface-container text-muted-foreground hover:text-foreground flex items-center justify-center gap-2 transition-colors"
                        title="Restore original"
                      >
                        <Eraser className="w-4 h-4" />
                        Reset
                      </button>
                    </div>
                  </div>

                  {tab === 'bubbles' && (
                    <div>
                      <label className="text-xs font-medium text-foreground mb-2 block">Bubble Tagging</label>
                      <SegmentedControl
                        options={OVERRIDE_BUBBLE_TYPE_OPTIONS}
                        value={bubbleType}
                        onChange={(value) => setBubbleType(value as BubbleType)}
                        className="w-full"
                      />
                    </div>
                  )}
                </>
              )}
            </div>

            {tab === 'bubbles' && (
              <div className="pt-4 border-t border-border space-y-3">
                <h3 className="text-xs font-medium text-foreground">Bubble Regions</h3>
                <div className="space-y-2">
                  {selectedPanelRegions.length ? (
                    displayedSelectedPanelRegions.map((region) => (
                      <RegionPropertyCard
                        key={region.id}
                        title={pendingRegionIds.includes(region.id) ? `+ ${getRegionLabel(region.class_name)}` : getRegionLabel(region.class_name)}
                        regionType={region.class_name}
                        badge={`Region ${displayedSelectedPanelRegions.findIndex((item) => item.id === region.id) + 1}`}
                        isEditing={editingBubbleId === region.id}
                        currentBbox={region.bbox}
                        previousBbox={editingBubbleBaseline?.bbox}
                        updatedBbox={editingBubbleRegion?.bbox || region.bbox}
                        className={pendingRegionIds.includes(region.id) ? 'border-cyan/30 bg-cyan/5 font-mono' : undefined}
                        highlighted={hoveredRegionId === region.id}
                        onMouseEnter={() => setHoveredRegionId(region.id)}
                        onMouseLeave={() => setHoveredRegionId((current) => (current === region.id ? null : current))}
                        onClick={() => setHoveredRegionId(region.id)}
                        onToggleEdit={() => toggleBubbleEditor(region)}
                        onApply={() => applyBubbleEditorChanges(region.id)}
                        onDelete={() => removeRegion(selectedPanelIndex, region.id)}
                        typeValue={region.class_name}
                        typeOptions={OVERRIDE_BUBBLE_TYPE_OPTIONS}
                        onTypeChange={(value) => updateRegionType(selectedPanelIndex, region.id, value as BubbleType)}
                        helperText="Resize or move the region on the panel canvas, then click the tick to apply."
                      />
                    ))
                  ) : (
                    <div className="rounded-lg border border-border bg-surface-container p-3 text-xs text-muted-foreground">
                      No bubble overrides yet for this panel.
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="pt-4 border-t border-border space-y-3">
              <h3 className="text-xs font-medium text-foreground">Panel Metadata</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Request ID</span>
                  <span className="text-foreground font-mono">{requestId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Filename</span>
                  <span className="text-foreground truncate ml-3">{payload.panelResult.filename}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Panel Mode</span>
                  <span className="text-foreground">{payload.panelMode}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Bubble Mode</span>
                  <span className="text-foreground">{payload.bubbleMode}</span>
                </div>
                {selectedCaption ? (
                  <>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Shot Type</span>
                      <span className="text-foreground">{selectedCaption.shot_type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">OCR</span>
                      <span className="text-foreground text-right ml-3">
                        {selectedDialogue?.text || '[no dialogue detected]'}
                      </span>
                    </div>
                  </>
                ) : null}
              </div>
            </div>

            <div className="pt-4 border-t border-border space-y-3">
              <h3 className="text-xs font-medium text-foreground">Summary</h3>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('speech').summaryClassName)}>
                  <div className="font-medium text-foreground">
                    {perPanelSummaries.find((summary) => summary.panel.index === selectedPanelIndex)?.speechCount || 0}
                  </div>
                  <div className="text-muted-foreground">Speech</div>
                </div>
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('narration').summaryClassName)}>
                  <div className="font-medium text-foreground">
                    {perPanelSummaries.find((summary) => summary.panel.index === selectedPanelIndex)?.narrationCount || 0}
                  </div>
                  <div className="text-muted-foreground">Narration</div>
                </div>
                <div className={cn('rounded border p-2 text-center', getRegionAppearance('sfx').summaryClassName)}>
                  <div className="font-medium text-foreground">
                    {perPanelSummaries.find((summary) => summary.panel.index === selectedPanelIndex)?.sfxCount || 0}
                  </div>
                  <div className="text-muted-foreground">SFX</div>
                </div>
              </div>
              {tab === 'captions' ? (
                <div className="rounded-lg border border-border bg-surface-container p-3 text-xs text-muted-foreground">
                  Diff is read-only in this pass. The updated side changes only when the current panel or bubble overrides differ from the original.
                </div>
              ) : (
                <div className="rounded-lg border border-border bg-surface-container p-3 text-xs text-muted-foreground flex items-start gap-2">
                  <Tag className="w-4 h-4 mt-0.5" />
                  Draw one rectangle at a time. Enabling selection clears the previous draft and starts a fresh rectangular zone.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
