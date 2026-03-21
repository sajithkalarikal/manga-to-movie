export type DetectionMode = 'heuristic' | 'detector';
export type OverrideBubbleType = 'speech' | 'narration' | 'sfx';
export type AnnotationBubbleType = 'speech_bubble' | 'narration_box' | 'sfx';
export type RegionType = 'panel' | OverrideBubbleType | AnnotationBubbleType;

export interface EditorOption<T extends string = string> {
  value: T;
  label: string;
}

export interface RegionAppearance {
  label: string;
  badgeLabel: string;
  borderClassName: string;
  badgeClassName: string;
  handleClassName: string;
  summaryClassName: string;
}

export const DETECTION_MODE_OPTIONS: EditorOption<DetectionMode>[] = [
  { value: 'heuristic', label: 'Heuristic' },
  { value: 'detector', label: 'Object Model' },
];

export const OVERRIDE_BUBBLE_TYPE_OPTIONS: EditorOption<OverrideBubbleType>[] = [
  { value: 'speech', label: 'Speech' },
  { value: 'narration', label: 'Narration' },
  { value: 'sfx', label: 'SFX' },
];

export const ANNOTATION_BUBBLE_TYPE_OPTIONS: EditorOption<AnnotationBubbleType>[] = [
  { value: 'speech_bubble', label: 'Speech' },
  { value: 'narration_box', label: 'Narration' },
  { value: 'sfx', label: 'SFX' },
];

const REGION_APPEARANCE: Record<RegionType, RegionAppearance> = {
  panel: {
    label: 'Panel',
    badgeLabel: 'panel',
    borderClassName: 'border-crimson bg-crimson/10',
    badgeClassName: 'bg-crimson text-crimson-foreground',
    handleClassName: 'bg-crimson',
    summaryClassName: 'border-crimson/20 bg-crimson/10',
  },
  speech: {
    label: 'Speech',
    badgeLabel: 'speech',
    borderClassName: 'border-cyan bg-cyan/10',
    badgeClassName: 'bg-cyan text-cyan-foreground',
    handleClassName: 'bg-cyan',
    summaryClassName: 'border-cyan/20 bg-cyan/10',
  },
  narration: {
    label: 'Narration',
    badgeLabel: 'narration',
    borderClassName: 'border-emerald-500 bg-emerald-500/10',
    badgeClassName: 'bg-emerald-500 text-white',
    handleClassName: 'bg-emerald-500',
    summaryClassName: 'border-emerald-500/20 bg-emerald-500/10',
  },
  sfx: {
    label: 'SFX',
    badgeLabel: 'sfx',
    borderClassName: 'border-crimson bg-crimson/10',
    badgeClassName: 'bg-crimson text-crimson-foreground',
    handleClassName: 'bg-crimson',
    summaryClassName: 'border-crimson/20 bg-crimson/10',
  },
  speech_bubble: {
    label: 'Speech',
    badgeLabel: 'speech',
    borderClassName: 'border-cyan bg-cyan/10',
    badgeClassName: 'bg-cyan text-cyan-foreground',
    handleClassName: 'bg-cyan',
    summaryClassName: 'border-cyan/20 bg-cyan/10',
  },
  narration_box: {
    label: 'Narration',
    badgeLabel: 'narration',
    borderClassName: 'border-emerald-500 bg-emerald-500/10',
    badgeClassName: 'bg-emerald-500 text-white',
    handleClassName: 'bg-emerald-500',
    summaryClassName: 'border-emerald-500/20 bg-emerald-500/10',
  },
};

export function getRegionAppearance(type: RegionType): RegionAppearance {
  return REGION_APPEARANCE[type];
}

export function getRegionLabel(type: RegionType): string {
  return getRegionAppearance(type).label;
}

