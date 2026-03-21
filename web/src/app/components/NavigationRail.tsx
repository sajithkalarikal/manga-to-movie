import { Link, useLocation } from 'react-router';
import { Activity, Home, Image, PenTool } from 'lucide-react';
import { useEffect, useState } from 'react';
import { cn } from '../components/ui/utils';

function readOverridePath() {
  try {
    const raw = sessionStorage.getItem('phase1_override_payload');
    if (!raw) {
      return null;
    }
    const payload = JSON.parse(raw);
    const requestId = payload?.panelResult?.request_id;
    return requestId ? `/ui_v2/${requestId}/override` : null;
  } catch {
    return null;
  }
}

export function NavigationRail() {
  const location = useLocation();
  const [overridePath, setOverridePath] = useState<string | null>(() => readOverridePath());

  useEffect(() => {
    const syncOverridePath = () => {
      setOverridePath(readOverridePath());
    };

    syncOverridePath();
    window.addEventListener('storage', syncOverridePath);
    window.addEventListener('phase1-override-updated', syncOverridePath);
    return () => {
      window.removeEventListener('storage', syncOverridePath);
      window.removeEventListener('phase1-override-updated', syncOverridePath);
    };
  }, []);

  const navItems = [
    { path: '/ui_v2/home', icon: Home, label: 'Home' },
    { path: overridePath || '/ui_v2/home', icon: Image, label: 'Override', available: Boolean(overridePath) },
    { path: '/ui_v2/annotate', icon: PenTool, label: 'Annotate' },
    { path: '/ui_v2/health', icon: Activity, label: 'Health' },
  ];

  return (
    <div className="w-20 h-full bg-surface-container-low border-r border-border flex flex-col items-center py-4 gap-2">
      <Link to="/ui_v2" aria-label="Go to Kokyu landing" className="mb-4 flex h-12 w-12 items-center justify-center">
        <div className="relative flex h-11 w-11 items-center justify-center rounded-full bg-[conic-gradient(from_180deg_at_50%_50%,rgba(35,176,201,0.98),rgba(239,71,111,0.98),rgba(35,176,201,0.98))] shadow-lg">
          <div className="absolute inset-[2px] rounded-full bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.16),_rgba(18,18,18,0.86))]" />
          <span className="relative text-sm font-black tracking-[0.08em] text-white">呼吸</span>
        </div>
      </Link>

      <div className="flex-1 flex flex-col gap-1 w-full px-2">
        {navItems.map((item) => {
          const isActive =
            (item.label === 'Home' && location.pathname === '/ui_v2/home') ||
            (item.label === 'Override' && /^\/ui_v2\/[^/]+\/override$/.test(location.pathname)) ||
            (item.label === 'Annotate' && location.pathname === '/ui_v2/annotate') ||
            (item.label === 'Health' && location.pathname === '/ui_v2/health');
          const Icon = item.icon;
          
          return (
            <Link
              key={item.label}
              to={item.path}
              title={item.label === 'Override' && !item.available ? 'Run analysis on Home to unlock Override' : item.label}
              className={cn(
                'flex flex-col items-center justify-center py-3 px-2 rounded-lg transition-all group relative',
                isActive
                  ? 'bg-surface-container-highest text-foreground'
                  : 'text-muted-foreground hover:bg-surface-container hover:text-foreground',
                item.label === 'Override' && !item.available && 'opacity-60'
              )}
            >
              <Icon className="w-6 h-6 mb-1" />
              <span className="text-[10px] text-center leading-tight">{item.label}</span>
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-crimson rounded-r-full" />
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
