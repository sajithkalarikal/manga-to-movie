import { useEffect } from 'react';
import { Outlet, useLocation } from 'react-router';
import { NavigationRail } from './components/NavigationRail';

export function Root() {
  const location = useLocation();
  const showRail = location.pathname !== '/ui_v2' && location.pathname !== '/ui_v2/';

  useEffect(() => {
    const navigationEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
    if (navigationEntry?.type !== 'reload') {
      return;
    }

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
  }, []);

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      {showRail ? <NavigationRail /> : null}
      <Outlet />
    </div>
  );
}
