export interface RectFrame {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function rectToStyle(rect: RectFrame | null, width: number, height: number) {
  if (!rect || width <= 0 || height <= 0) {
    return undefined;
  }
  return {
    left: `${(rect.x / width) * 100}%`,
    top: `${(rect.y / height) * 100}%`,
    width: `${(rect.width / width) * 100}%`,
    height: `${(rect.height / height) * 100}%`,
  };
}

export function formatBboxLabel(bbox: number[] | null | undefined) {
  if (!bbox || bbox.length !== 4) {
    return '[n/a]';
  }
  return `[${bbox.map((value) => Math.round(Number(value))).join(', ')}]`;
}

