import { Link } from 'react-router';
import { ArrowRight, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';

export function Landing() {
  const [breatheScale, setBreatheScale] = useState(1);

  useEffect(() => {
    let animationFrame: number;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const scale = 1 + Math.sin(elapsed / 2000) * 0.05;
      setBreatheScale(scale);
      animationFrame = requestAnimationFrame(animate);
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  return (
    <div className="flex-1 flex flex-col h-full overflow-auto bg-gradient-to-br from-background via-background to-surface/30">
      <div className="flex items-center justify-between px-16 py-6 border-b border-border/50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-crimson to-cyan flex items-center justify-center text-white font-bold text-lg shadow-lg">
            K
          </div>
          <div>
            <div className="text-sm font-semibold text-foreground">Kokyu</div>
            <div className="text-xs text-muted-foreground">Manga-to-Movie Pipeline</div>
          </div>
        </div>

        <Link
          to="/ui_v2/home"
          className="inline-flex items-center gap-3 px-6 py-3 bg-crimson text-crimson-foreground rounded-lg text-sm font-medium hover:opacity-90 transition-opacity shadow-lg group"
        >
          Enter Kokyu
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </Link>
      </div>

      <div className="flex-1 px-16 py-12">
        <div className="mx-auto flex h-full w-full max-w-7xl flex-col gap-12 xl:flex-row xl:items-center xl:justify-between">
          <div className="min-w-0 flex-1">
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-cyan/10 border border-cyan/20 rounded-full">
                <Sparkles className="w-3.5 h-3.5 text-cyan" />
                <span className="text-xs font-medium text-foreground uppercase tracking-wide">
                  Phase 1 Story Breathing Engine
                </span>
              </div>

              <p className="text-sm text-muted-foreground tracking-wide">
                Japanese storytelling workspace
              </p>

              <div className="relative">
                <div className="grid grid-cols-[auto,1fr] gap-8 items-center">
                  <div className="space-y-4">
                    <h1 className="text-8xl font-bold text-foreground leading-none tracking-tight">
                      Kokyu
                    </h1>
                    <p className="text-2xl text-muted-foreground italic">
                      breathing between panels
                    </p>
                  </div>

                  <div className="relative h-[140px] flex items-center justify-start">
                    <div
                      className="text-[160px] font-bold text-foreground/8 leading-none select-none"
                      style={{
                        transform: `scale(${breatheScale})`,
                        transformOrigin: 'left center',
                        willChange: 'transform',
                      }}
                    >
                      呼吸
                    </div>
                  </div>
                </div>
              </div>

              <div className="max-w-3xl space-y-5 text-base text-foreground/80 leading-relaxed pt-6">
                <p>
                  Like a child discovering worlds between inked frames, our AI takes its <em>first breath</em> through the sequential art of manga.
                </p>
                <p className="text-muted-foreground">
                  Comics are the first gateway-panels that teach rhythm, silence between motion,
                  and the patience to understand stories told in stillness. Phase 1 is where Kokyu learns
                  to <em>breathe</em>: to parse structure, trace intention, and feel the heartbeat of visual narrative.
                </p>
              </div>

              <div className="flex items-center gap-6 text-xs text-muted-foreground border-t border-border pt-6 mt-8">
                <span>Phase 1: Story Breathing</span>
                <span className="w-1 h-1 bg-muted-foreground rounded-full" />
                <span>Learning to read like a child</span>
              </div>
            </div>
          </div>

          <div className="flex w-full flex-col gap-5 xl:w-[360px] xl:shrink-0">
            <div className="p-5 bg-surface-container-low border border-border rounded-lg space-y-2.5 hover:bg-surface-container transition-colors">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Capture
              </div>
              <h3 className="text-sm font-semibold text-foreground">
                Read the page rhythm
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Phase 1 begins with panel boundaries, composition, and breathing room.
              </p>
            </div>

            <div className="p-5 bg-surface-container-low border border-border rounded-lg space-y-2.5 hover:bg-surface-container transition-colors">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Interpret
              </div>
              <h3 className="text-sm font-semibold text-foreground">
                Trace voices and motion
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Bubble structure, narration, and SFX are separated for editorial control.
              </p>
            </div>

            <div className="p-5 bg-surface-container-low border border-border rounded-lg space-y-2.5 hover:bg-surface-container transition-colors">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Refine
              </div>
              <h3 className="text-sm font-semibold text-foreground">
                Override with precision
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Correct structure, preserve intent, and move into annotation with confidence.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
