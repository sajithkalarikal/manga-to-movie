import { cn } from '../components/ui/utils';

interface MaterialCardProps {
  children: React.ReactNode;
  className?: string;
  elevation?: 'low' | 'medium' | 'high';
  hover?: boolean;
  onClick?: () => void;
}

export function MaterialCard({ 
  children, 
  className, 
  elevation = 'medium',
  hover = false,
  onClick 
}: MaterialCardProps) {
  const elevationClasses = {
    low: 'bg-surface-container-low',
    medium: 'bg-surface-container',
    high: 'bg-surface-container-high',
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        'rounded-lg border border-border transition-all',
        elevationClasses[elevation],
        hover && 'hover:bg-surface-container-high cursor-pointer',
        className
      )}
    >
      {children}
    </div>
  );
}
