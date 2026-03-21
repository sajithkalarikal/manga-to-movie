import { Check, Pencil, Trash2 } from 'lucide-react';
import { getRegionLabel, type EditorOption, type RegionType } from '../lib/editor-config';
import { formatBboxLabel } from '../lib/region-utils';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { cn } from './ui/utils';

interface RegionPropertyCardProps<T extends string = string> {
  title: string;
  regionType: RegionType;
  badge?: string;
  isEditing: boolean;
  currentBbox: number[] | null | undefined;
  previousBbox?: number[] | null;
  updatedBbox?: number[] | null;
  onToggleEdit: () => void;
  onApply?: () => void;
  onDelete?: () => void;
  editTitle?: string;
  applyTitle?: string;
  deleteTitle?: string;
  typeValue?: T;
  typeOptions?: EditorOption<T>[];
  onTypeChange?: (value: T) => void;
  helperText?: string;
  highlighted?: boolean;
  className?: string;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onClick?: () => void;
}

export function RegionPropertyCard<T extends string = string>({
  title,
  regionType,
  badge,
  isEditing,
  currentBbox,
  previousBbox,
  updatedBbox,
  onToggleEdit,
  onApply,
  onDelete,
  editTitle,
  applyTitle,
  deleteTitle,
  typeValue,
  typeOptions,
  onTypeChange,
  helperText,
  highlighted = false,
  className,
  onMouseEnter,
  onMouseLeave,
  onClick,
}: RegionPropertyCardProps<T>) {
  const label = getRegionLabel(regionType);

  return (
    <div
      className={cn(
        'rounded-lg border bg-surface-container p-3 text-xs transition-all',
        highlighted ? 'border-amber-400/40 ring-2 ring-amber-400/50' : 'border-border',
        className,
      )}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
    >
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="font-medium text-foreground">{title}</div>
          </div>
          <div className="flex items-center gap-2">
            {badge ? (
              <span className="rounded-full border border-border bg-background px-2 py-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                {badge}
              </span>
            ) : null}
            <button
              onClick={(event) => {
                event.stopPropagation();
                onToggleEdit();
              }}
              className={cn(
                'inline-flex h-7 w-7 items-center justify-center rounded-md border transition-colors',
                isEditing
                  ? 'border-cyan bg-cyan/10 text-cyan'
                  : 'border-border text-muted-foreground hover:bg-surface-container-high hover:text-foreground',
              )}
              title={editTitle || `${isEditing ? 'Close' : 'Edit'} ${label.toLowerCase()}`}
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
            {isEditing && onApply ? (
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  onApply();
                }}
                className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-cyan bg-cyan/10 text-cyan transition-colors hover:bg-cyan/20"
                title={applyTitle || `Apply ${label.toLowerCase()} changes`}
              >
                <Check className="h-3.5 w-3.5" />
              </button>
            ) : null}
            {onDelete ? (
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  onDelete();
                }}
                className="inline-flex h-7 w-7 items-center justify-center rounded-md text-crimson transition-colors hover:bg-crimson/10"
                title={deleteTitle || `Remove ${label.toLowerCase()}`}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            ) : null}
          </div>
        </div>

        {isEditing ? (
          <div className="space-y-3 rounded-lg border border-border bg-background p-3">
            {typeValue && typeOptions && onTypeChange ? (
              <label className="block space-y-1">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Type</span>
                <Select
                  value={typeValue}
                  onValueChange={(value) => onTypeChange(value as T)}
                >
                  <SelectTrigger
                    className="w-full rounded-md border-border bg-surface text-xs"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent className="max-h-72">
                    {typeOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>
            ) : null}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Previous Coords</div>
                <div className="mt-1 rounded-md border border-border bg-surface px-2 py-2 font-mono text-foreground">
                  {formatBboxLabel(previousBbox || currentBbox)}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Updated Coords</div>
                <div className="mt-1 rounded-md border border-border bg-surface px-2 py-2 font-mono text-foreground">
                  {formatBboxLabel(updatedBbox || currentBbox)}
                </div>
              </div>
            </div>

            {helperText ? <div className="text-[11px] text-muted-foreground">{helperText}</div> : null}
          </div>
        ) : (
          <div className="rounded-md border border-border bg-background px-2 py-1.5 text-xs text-muted-foreground">
            {formatBboxLabel(currentBbox)}
          </div>
        )}
      </div>
    </div>
  );
}

