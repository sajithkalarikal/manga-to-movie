import type { PointerEventHandler, ReactNode } from 'react';
import { getRegionAppearance, type RegionType } from '../lib/editor-config';
import { rectToStyle } from '../lib/region-utils';
import { cn } from './ui/utils';

interface RegionOverlayBoxProps {
  regionType: RegionType;
  label?: string;
  rect: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  canvasWidth: number;
  canvasHeight: number;
  isHovered?: boolean;
  isEditing?: boolean;
  title?: string;
  className?: string;
  badgeClassName?: string;
  handleClassName?: string;
  onClick?: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onPointerDown?: PointerEventHandler<HTMLDivElement>;
  children?: ReactNode;
}

export function RegionOverlayBox({
  regionType,
  label,
  rect,
  canvasWidth,
  canvasHeight,
  isHovered = false,
  isEditing = false,
  title,
  className,
  badgeClassName,
  handleClassName,
  onClick,
  onMouseEnter,
  onMouseLeave,
  onPointerDown,
  children,
}: RegionOverlayBoxProps) {
  const appearance = getRegionAppearance(regionType);

  return (
    <div
      className={cn(
        'absolute border-2 transition-all',
        appearance.borderClassName,
        isHovered && 'ring-4 ring-amber-400/60 shadow-[0_0_0_2px_rgba(251,191,36,0.25)]',
        className,
      )}
      style={rectToStyle(rect, canvasWidth, canvasHeight)}
      title={title}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onPointerDown={onPointerDown}
    >
      <div
        className={cn(
          'absolute -top-5 left-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase',
          appearance.badgeClassName,
          badgeClassName,
        )}
      >
        {label || appearance.badgeLabel}
      </div>
      {isEditing ? (
        <div
          className={cn(
            'absolute right-0 bottom-0 h-3.5 w-3.5 translate-x-1/2 translate-y-1/2 rounded-sm border border-background shadow-sm',
            appearance.handleClassName,
            handleClassName,
          )}
        />
      ) : null}
      {children}
    </div>
  );
}

