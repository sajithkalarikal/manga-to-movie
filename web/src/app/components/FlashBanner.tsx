import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { cn } from '../components/ui/utils';

type FlashBannerTone = 'success' | 'error';

interface FlashBannerState {
  id: number;
  message: string;
  tone: FlashBannerTone;
}

interface FlashBannerProps {
  banner: FlashBannerState | null;
}

export function useFlashBanner(durationMs = 3000) {
  const [banner, setBanner] = useState<FlashBannerState | null>(null);
  const timerRef = useRef<number | null>(null);

  const showBanner = useCallback(
    (message: string, tone: FlashBannerTone) => {
      if (!message.trim()) {
        return;
      }
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
      }
      setBanner({
        id: Date.now(),
        message,
        tone,
      });
      timerRef.current = window.setTimeout(() => {
        setBanner(null);
      }, durationMs);
    },
    [durationMs],
  );

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  return { banner, showBanner };
}

export function FlashBanner({ banner }: FlashBannerProps) {
  if (!banner) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute top-4 right-4 z-50">
      <div
        className={cn(
          'min-w-72 max-w-sm rounded-lg border px-3 py-2 shadow-lg backdrop-blur-sm',
          banner.tone === 'success'
            ? 'border-cyan/30 bg-cyan/10 text-foreground'
            : 'border-crimson/30 bg-crimson/10 text-foreground',
        )}
      >
        <div className="flex items-start gap-2">
          {banner.tone === 'success' ? (
            <CheckCircle2 className="mt-0.5 h-4 w-4 text-cyan" />
          ) : (
            <AlertCircle className="mt-0.5 h-4 w-4 text-crimson" />
          )}
          <p className="text-xs leading-5">{banner.message}</p>
        </div>
      </div>
    </div>
  );
}
