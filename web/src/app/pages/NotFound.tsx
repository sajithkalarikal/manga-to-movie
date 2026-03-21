import { Link } from 'react-router';
import { Home } from 'lucide-react';

export function NotFound() {
  return (
    <div className="flex-1 flex items-center justify-center bg-surface">
      <div className="text-center">
        <div className="text-6xl font-bold text-crimson mb-4">404</div>
        <h1 className="text-xl font-medium text-foreground mb-2">Page Not Found</h1>
        <p className="text-sm text-muted-foreground mb-6">
          The page you're looking for doesn't exist.
        </p>
        <Link
          to="/ui_v2/home"
          className="inline-flex items-center gap-2 px-4 py-2 bg-crimson text-crimson-foreground hover:opacity-90 rounded-lg text-sm transition-opacity"
        >
          <Home className="w-4 h-4" />
          Back to Home
        </Link>
      </div>
    </div>
  );
}
