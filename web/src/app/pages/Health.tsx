import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Gauge,
  HardDrive,
  Info,
  RefreshCw,
  ServerCrash,
  Sparkles,
  Timer,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, XAxis, YAxis } from 'recharts';
import { MaterialCard } from '../components/MaterialCard';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '../components/ui/chart';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Tooltip, TooltipContent, TooltipTrigger } from '../components/ui/tooltip';
import { cn } from '../components/ui/utils';
import {
  buildHealthSuggestions,
  buildModelPerformanceSuggestions,
  fetchHealthReport,
  formatAge,
  formatBytes,
  formatDurationSeconds,
  formatMetricNumber,
  formatMetricPercent,
  getEvalRunDetails,
  getTrainingRunDetails,
  type HealthArtifactStatus,
  type SystemHealthReport,
} from '../lib/health';

interface MetricCardProps {
  title: string;
  explanation: string;
  value: string;
  status: 'ok' | 'degraded';
  detail?: string;
  detailTitle?: string;
  detailTruncate?: boolean;
  degradedWhy?: string;
  degradedRecommendation?: string;
}

interface MeaningCardProps {
  title: string;
  value: string;
  explanation: string;
  impact: string;
}

interface TinyStatProps {
  label: string;
  value: string;
  hint: string;
}

const trainingLossChartConfig = {
  train_loss: { label: 'Train loss', color: '#23b0c9' },
  valid_loss: { label: 'Valid loss', color: '#ef476f' },
};

const evalMetricChartConfig = {
  mAP_50_95: { label: 'mAP@0.50:0.95', color: '#0ea5a6' },
  mAP_50: { label: 'mAP@0.50', color: '#2f7cf6' },
  recall: { label: 'Recall', color: '#f59e0b' },
};

function StatusPill({
  status,
  degradedWhy,
  degradedRecommendation,
}: {
  status: 'ok' | 'degraded';
  degradedWhy?: string;
  degradedRecommendation?: string;
}) {
  const pill = (
    <span
      className={cn(
        'rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide',
        status === 'ok'
          ? 'border-cyan/20 bg-cyan/10 text-cyan'
          : 'border-crimson/20 bg-crimson/10 text-crimson',
      )}
    >
      {status}
    </span>
  );

  if (status !== 'degraded' || (!degradedWhy && !degradedRecommendation)) {
    return pill;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>{pill}</TooltipTrigger>
      <TooltipContent side="top" sideOffset={8} className="max-w-[240px] space-y-1.5">
        {degradedWhy ? (
          <p className="text-xs">
            <span className="font-semibold">Degrading:</span> {degradedWhy}
          </p>
        ) : null}
        {degradedRecommendation ? (
          <p className="text-xs">
            <span className="font-semibold">Fix:</span> {degradedRecommendation}
          </p>
        ) : null}
      </TooltipContent>
    </Tooltip>
  );
}

function MetricCard({
  title,
  explanation,
  value,
  status,
  detail,
  detailTitle,
  detailTruncate = false,
  degradedWhy,
  degradedRecommendation,
}: MetricCardProps) {
  return (
    <MaterialCard className="space-y-3 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          <p className="text-xs text-muted-foreground">{explanation}</p>
        </div>
        <StatusPill
          status={status}
          degradedWhy={degradedWhy}
          degradedRecommendation={degradedRecommendation}
        />
      </div>
      <div className="text-xl font-semibold text-foreground">{value}</div>
      {detail ? (
        <p
          className={cn('text-xs text-muted-foreground', detailTruncate && 'truncate')}
          title={detailTitle || detail}
        >
          {detail}
        </p>
      ) : null}
    </MaterialCard>
  );
}

function TinyStat({ label, value, hint }: TinyStatProps) {
  return (
    <div className="rounded-2xl border border-border bg-surface-container-low p-4">
      <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-lg font-semibold text-foreground">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

function MetricMeaningCard({ title, value, explanation, impact }: MeaningCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-surface-container-low p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <span className="rounded-full bg-surface-container-high px-2.5 py-1 text-xs font-medium text-foreground">
          {value}
        </span>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">{explanation}</p>
      <p className="mt-2 text-xs text-foreground/80">{impact}</p>
    </div>
  );
}

function EmptyChartState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex h-[240px] items-center justify-center rounded-2xl border border-dashed border-border bg-surface-container-low px-6 text-center">
      <div className="max-w-sm space-y-2">
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-xs text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}

function getPathLeaf(value: string | null | undefined) {
  if (!value) {
    return 'unknown';
  }
  const segments = value.split('/');
  return segments[segments.length - 1] || value;
}

function formatEvalRunLabel(artifact: HealthArtifactStatus) {
  const details = getEvalRunDetails(artifact);
  const modelName = details?.weights ? getPathLeaf(details.weights) : null;
  const runName = details?.run_name ? getPathLeaf(details.run_name) : null;
  return [modelName, details?.split, runName].filter(Boolean).join(' • ');
}

export function Health() {
  const [report, setReport] = useState<SystemHealthReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedEvalPath, setSelectedEvalPath] = useState('');

  const loadReport = async () => {
    setLoading(true);
    setError('');
    try {
      const payload = await fetchHealthReport();
      setReport(payload);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Failed to load system health.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadReport();
  }, []);

  const suggestions = useMemo(() => (report ? buildHealthSuggestions(report) : []), [report]);
  const topModels = report?.models.filter((model) => model.available) || [];
  const evaluationRuns = report?.evaluation_runs || [];
  const trainingDetails = useMemo(() => getTrainingRunDetails(report?.last_training_run), [report]);
  const selectedEvalArtifact = useMemo(() => {
    if (evaluationRuns.length === 0) {
      return report?.last_eval_run || null;
    }
    return evaluationRuns.find((artifact) => artifact.path === selectedEvalPath) || evaluationRuns[0];
  }, [evaluationRuns, report, selectedEvalPath]);
  const evalDetails = useMemo(() => getEvalRunDetails(selectedEvalArtifact), [selectedEvalArtifact]);
  const modelSuggestions = useMemo(
    () => buildModelPerformanceSuggestions(trainingDetails, evalDetails),
    [trainingDetails, evalDetails],
  );
  const overallStatusHint = useMemo(() => {
    if (!report || report.status === 'ok') {
      return null;
    }
    const degradedAreas = [
      !report.redis.ok && 'Redis',
      report.worker.live_workers === 0 && 'workers',
      !report.ocr.ok && 'OCR',
      report.disk.free_percent < 15 && 'disk',
      report.queue.backlog > 10 && 'queue',
      report.models.some((model) => !model.ok) && 'models',
    ].filter(Boolean) as string[];

    return {
      why: degradedAreas.length ? degradedAreas.slice(0, 3).join(', ') : 'Multiple services',
      recommendation: suggestions[0]?.detail || 'Resolve the highlighted system issues and refresh the report.',
    };
  }, [report, suggestions]);

  useEffect(() => {
    if (evaluationRuns.length === 0) {
      setSelectedEvalPath('');
      return;
    }
    setSelectedEvalPath((current) => {
      if (current && evaluationRuns.some((artifact) => artifact.path === current)) {
        return current;
      }
      return evaluationRuns[0]?.path || '';
    });
  }, [evaluationRuns]);

  const trainingHistoryData = useMemo(() => {
    const history = trainingDetails?.history || [];
    return history
      .map((item) => ({
        epoch: Number(item?.epoch),
        train_loss: Number(item?.train_loss),
        valid_loss: Number(item?.valid_loss),
      }))
      .filter((item) => Number.isFinite(item.epoch) && (Number.isFinite(item.train_loss) || Number.isFinite(item.valid_loss)));
  }, [trainingDetails]);

  const evalMetricData = useMemo(() => {
    const values = [
      {
        key: 'mAP_50_95',
        label: 'mAP@0.50:0.95',
        value: Number(evalDetails?.metrics?.mAP_50_95),
      },
      {
        key: 'mAP_50',
        label: 'mAP@0.50',
        value: Number(evalDetails?.metrics?.mAP_50),
      },
      {
        key: 'recall',
        label: 'Recall',
        value: Number(evalDetails?.metrics?.recall),
      },
    ];
    return values.filter((item) => Number.isFinite(item.value)).map((item) => ({ ...item, percent: item.value * 100 }));
  }, [evalDetails]);

  const metricMeaningCards = useMemo(
    () => [
      {
        title: 'Train loss',
        value: formatMetricNumber(trainingDetails?.train_loss),
        explanation: 'How easily the detector fits the training annotations during optimization.',
        impact: 'Lower is usually better. If it keeps dropping while validation loss rises, the model is memorizing instead of generalizing.',
      },
      {
        title: 'Valid loss',
        value: formatMetricNumber(trainingDetails?.valid_loss),
        explanation: 'How the same detector behaves on held-out pages it did not train on.',
        impact: 'Lower means the checkpoint should travel better to unseen pages. Rising values usually signal overfit or noisy labels.',
      },
      {
        title: 'Recall',
        value: formatMetricPercent(evalDetails?.metrics?.recall),
        explanation: 'The share of real target regions the detector successfully finds during evaluation.',
        impact: 'Higher recall means fewer missed bubbles or panels. Pushing it too high can add false positives if thresholds are loose.',
      },
      {
        title: 'mAP@0.50',
        value: formatMetricPercent(evalDetails?.metrics?.mAP_50),
        explanation: 'A forgiving detection score that rewards finding the right class with roughly correct boxes.',
        impact: 'Higher values mean the detector is usually catching the right region at all, even if box edges are still a bit loose.',
      },
      {
        title: 'mAP@0.50:0.95',
        value: formatMetricPercent(evalDetails?.metrics?.mAP_50_95),
        explanation: 'A stricter score that rewards tight boxes across a range of IoU thresholds.',
        impact: 'Higher values mean more trustworthy localization. If it lags far behind mAP@0.50, the detector sees the region but boxes need tightening.',
      },
    ],
    [evalDetails, trainingDetails],
  );

  return (
    <main className="flex-1 min-w-0 overflow-auto bg-[radial-gradient(circle_at_top,_rgba(35,176,201,0.10),_transparent_35%),linear-gradient(180deg,_rgba(255,255,255,0.98),_rgba(245,241,234,0.96))]">
      <div className="mx-auto max-w-7xl space-y-6 px-8 py-8">
        <section className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.24em] text-muted-foreground">System Health</p>
            <h1 className="text-3xl font-semibold text-foreground">Operational report for the detector stack</h1>
            <p className="max-w-3xl text-sm text-muted-foreground">
              Live infrastructure status, model readiness, and a compact readout of the latest training and evaluation signals.
            </p>
          </div>
          <button
            onClick={() => {
              void loadReport();
            }}
            disabled={loading}
            className={cn(
              'inline-flex items-center gap-2 self-start rounded-xl border px-4 py-2 text-sm font-medium transition-colors',
              loading
                ? 'cursor-not-allowed border-border bg-muted text-muted-foreground'
                : 'border-border bg-surface-container text-foreground hover:bg-surface-container-high',
            )}
          >
            <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
            Refresh report
          </button>
        </section>

        {error ? (
          <MaterialCard className="border-crimson/20 bg-crimson/10 p-5">
            <div className="flex items-start gap-3">
              <ServerCrash className="mt-0.5 h-5 w-5 text-crimson" />
              <div className="space-y-1">
                <h2 className="text-sm font-semibold text-foreground">Health fetch failed</h2>
                <p className="text-sm text-foreground">{error}</p>
                <p className="text-xs text-muted-foreground">
                  If you are running the React dev server, make sure the Vite proxy and FastAPI server are both up.
                </p>
              </div>
            </div>
          </MaterialCard>
        ) : null}

        <section className="grid gap-4 lg:grid-cols-[1.4fr,1fr]">
          <MaterialCard className="space-y-5 p-6">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Snapshot</p>
                <h2 className="text-xl font-semibold text-foreground">Current system state</h2>
              </div>
              {report?.status === 'ok' || report?.status === 'degraded' ? (
                <StatusPill
                  status={report.status}
                  degradedWhy={overallStatusHint?.why}
                  degradedRecommendation={overallStatusHint?.recommendation}
                />
              ) : (
                <span className="rounded-full border border-border px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  {report?.status || (loading ? 'loading' : 'unknown')}
                </span>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-2xl bg-surface-container-low p-4">
                <p className="text-xs text-muted-foreground">Environment</p>
                <p className="mt-2 text-lg font-semibold text-foreground">{report?.environment || 'n/a'}</p>
              </div>
              <div className="rounded-2xl bg-surface-container-low p-4">
                <p className="text-xs text-muted-foreground">Checked</p>
                <p className="mt-2 text-lg font-semibold text-foreground">
                  {report?.checked_at ? new Date(report.checked_at).toLocaleString() : 'n/a'}
                </p>
              </div>
              <div className="rounded-2xl bg-surface-container-low p-4">
                <p className="text-xs text-muted-foreground">Live workers</p>
                <p className="mt-2 text-lg font-semibold text-foreground">{report?.worker.live_workers ?? 0}</p>
              </div>
              <div className="rounded-2xl bg-surface-container-low p-4">
                <p className="text-xs text-muted-foreground">Queue backlog</p>
                <p className="mt-2 text-lg font-semibold text-foreground">{report?.queue.backlog ?? 0}</p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-cyan" />
                <h3 className="text-sm font-semibold text-foreground">Concise improvement suggestions</h3>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                {suggestions.length > 0
                  ? suggestions.map((suggestion) => (
                      <div key={suggestion.title} className="rounded-2xl border border-border bg-surface-container-low p-4">
                        <p className="text-sm font-medium text-foreground">{suggestion.title}</p>
                        <p className="mt-2 text-xs text-muted-foreground">{suggestion.detail}</p>
                      </div>
                    ))
                  : [0, 1, 2].map((index) => (
                      <div
                        key={index}
                        className="rounded-2xl border border-border bg-surface-container-low p-4 text-xs text-muted-foreground"
                      >
                        Loading suggestion...
                      </div>
                    ))}
              </div>
            </div>
          </MaterialCard>

          <MaterialCard className="space-y-4 p-6">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-crimson" />
              <h2 className="text-lg font-semibold text-foreground">Loaded models</h2>
            </div>
            <div className="space-y-3">
              {topModels.length > 0 ? (
                topModels.map((model) => (
                  <div key={model.name} className="rounded-2xl border border-border bg-surface-container-low p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-foreground">{model.name}</p>
                      <CheckCircle2 className={cn('h-4 w-4', model.ok ? 'text-cyan' : 'text-crimson')} />
                    </div>
                    <p className="mt-2 truncate text-xs text-muted-foreground" title={model.weights_path || 'not configured'}>
                      {model.weights_path || 'not configured'}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">Classes: {(model.classes || []).join(', ') || 'none'}</p>
                    {model.load_error ? <p className="mt-2 text-xs text-crimson">{model.load_error}</p> : null}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-border bg-surface-container-low p-4 text-xs text-muted-foreground">
                  No configured model weights were reported.
                </div>
              )}
            </div>
          </MaterialCard>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.2fr,0.8fr]">
          <MaterialCard className="space-y-5 p-6">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Model Performance</p>
                <h2 className="text-xl font-semibold text-foreground">Latest training and evaluation summary</h2>
              </div>
              <div className="rounded-full border border-border bg-surface-container-low px-3 py-1.5 text-xs text-muted-foreground">
                {evalDetails?.run_name || 'Awaiting eval metadata'}
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
              <div className="space-y-3 rounded-3xl border border-border bg-surface-container p-4">
                <div className="flex items-center gap-2">
                  <BrainCircuit className="h-4 w-4 text-cyan" />
                  <div>
                    <p className="text-sm font-semibold text-foreground">Training losses</p>
                    <p className="text-xs text-muted-foreground">
                      Epoch {trainingDetails?.epoch ?? 'n/a'} {trainingDetails?.detector_type ? `• ${trainingDetails.detector_type}` : ''}
                    </p>
                  </div>
                </div>
                {trainingHistoryData.length > 0 ? (
                  <ChartContainer config={trainingLossChartConfig} className="h-[240px] w-full aspect-auto">
                    <LineChart accessibilityLayer data={trainingHistoryData} margin={{ left: 6, right: 20, top: 8, bottom: 0 }}>
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="epoch"
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        allowDecimals={false}
                        tickLine={false}
                        axisLine={false}
                        tickMargin={12}
                      />
                      <YAxis tickLine={false} axisLine={false} tickMargin={12} tickFormatter={(value) => Number(value).toFixed(2)} />
                      <ChartTooltip
                        cursor={false}
                        content={
                          <ChartTooltipContent
                            labelFormatter={(value) => `Epoch ${value}`}
                            formatter={(value, _name, item) => (
                              <>
                                <span className="text-muted-foreground">{item.name === 'train_loss' ? 'Train loss' : 'Valid loss'}</span>
                                <span className="font-mono font-medium text-foreground">{formatMetricNumber(Number(value))}</span>
                              </>
                            )}
                          />
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="train_loss"
                        stroke="var(--color-train_loss)"
                        strokeWidth={2.5}
                        dot={{ r: 4, fill: 'var(--color-train_loss)' }}
                        activeDot={{ r: 5 }}
                        connectNulls
                      />
                      <Line
                        type="monotone"
                        dataKey="valid_loss"
                        stroke="var(--color-valid_loss)"
                        strokeWidth={2.5}
                        dot={{ r: 4, fill: 'var(--color-valid_loss)' }}
                        activeDot={{ r: 5 }}
                        connectNulls
                      />
                    </LineChart>
                  </ChartContainer>
                ) : (
                  <EmptyChartState
                    title="No training losses found"
                    detail="Once a checkpoint with train_loss and valid_loss is available, this chart will summarize overfit pressure at a glance."
                  />
                )}
                {trainingDetails?.history_source === 'current_epoch_only' ? (
                  <p className="text-xs text-amber-800">
                    Earlier epochs were not stored in this checkpoint. The next training run will populate the full trend automatically.
                  </p>
                ) : null}
                <p className="text-xs text-muted-foreground">
                  Dataset: {(trainingDetails?.dataset_roots || []).join(', ') || 'No dataset root stored in checkpoint.'}
                </p>
              </div>

              <div className="space-y-3 rounded-3xl border border-border bg-surface-container p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-amber-600" />
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-semibold text-foreground">Evaluation quality</p>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              className="rounded-full p-1 text-muted-foreground transition-colors hover:bg-surface-container-high hover:text-foreground"
                              aria-label="Explain evaluation metrics"
                            >
                              <Info className="h-3.5 w-3.5" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent side="top" sideOffset={6} className="max-w-xs space-y-2">
                            <div>
                              <p className="font-semibold">Recall</p>
                              <p>Higher means fewer real regions are missed.</p>
                            </div>
                            <div>
                              <p className="font-semibold">mAP@0.50</p>
                              <p>Higher means the detector usually finds the right class with roughly correct boxes.</p>
                            </div>
                            <div>
                              <p className="font-semibold">mAP@0.50:0.95</p>
                              <p>Higher means the boxes stay accurate even under stricter overlap checks.</p>
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {evalDetails?.split ? `${evalDetails.split} split` : 'Split unknown'}
                        {evalDetails?.dataset_root ? ' • selectable eval artifact' : ''}
                      </p>
                    </div>
                  </div>
                  <div className="min-w-[250px] flex-1 max-w-sm">
                    <Select value={selectedEvalPath} onValueChange={setSelectedEvalPath}>
                      <SelectTrigger className="w-full bg-background [&>span]:block [&>span]:truncate">
                        <SelectValue placeholder="Select an eval run" />
                      </SelectTrigger>
                      <SelectContent className="max-h-80 w-[var(--radix-select-trigger-width)]">
                        {evaluationRuns.map((artifact) => (
                          <SelectItem key={artifact.path || artifact.updated_at || 'eval-run'} value={artifact.path || ''}>
                            <span className="block truncate" title={formatEvalRunLabel(artifact)}>
                              {formatEvalRunLabel(artifact)}
                            </span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {evalMetricData.length > 0 ? (
                  <ChartContainer config={evalMetricChartConfig} className="h-[240px] w-full aspect-auto">
                    <BarChart accessibilityLayer data={evalMetricData} layout="vertical" margin={{ left: 12, right: 24 }}>
                      <CartesianGrid horizontal={false} />
                      <YAxis
                        dataKey="label"
                        type="category"
                        tickLine={false}
                        axisLine={false}
                        width={98}
                        tickMargin={12}
                      />
                      <XAxis
                        type="number"
                        tickLine={false}
                        axisLine={false}
                        domain={[0, 100]}
                        tickFormatter={(value) => `${value}%`}
                      />
                      <ChartTooltip
                        cursor={false}
                        content={
                          <ChartTooltipContent
                            hideLabel
                            formatter={(value, _name, item) => (
                              <>
                                <span className="text-muted-foreground">{item.payload.label}</span>
                                <span className="font-mono font-medium text-foreground">{`${Number(value).toFixed(1)}%`}</span>
                              </>
                            )}
                          />
                        }
                      />
                      <Bar dataKey="percent" radius={10}>
                        {evalMetricData.map((entry) => (
                          <Cell key={entry.key} fill={`var(--color-${entry.key})`} />
                        ))}
                        <LabelList
                          dataKey="percent"
                          position="right"
                          formatter={(value: number) => `${Number(value).toFixed(1)}%`}
                          className="fill-foreground text-xs"
                        />
                      </Bar>
                    </BarChart>
                  </ChartContainer>
                ) : (
                  <EmptyChartState
                    title="No evaluation metrics found"
                    detail="Run the latest detector eval to surface recall and mAP trends here."
                  />
                )}
                <p className="text-xs text-muted-foreground break-all">
                  Weights: {evalDetails?.weights || 'No weights path recorded in eval artifact.'}
                </p>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
              <TinyStat
                label="Eval images"
                value={evalDetails?.images != null ? String(evalDetails.images) : 'n/a'}
                hint="Pages included in the latest eval run."
              />
              <TinyStat
                label="Predictions"
                value={evalDetails?.predictions != null ? String(evalDetails.predictions) : 'n/a'}
                hint="Total detections emitted during that eval."
              />
              <TinyStat
                label="Eval time"
                value={formatDurationSeconds(evalDetails?.elapsed_seconds)}
                hint="How long the latest eval took end to end."
              />
              <TinyStat
                label="Score threshold"
                value={formatMetricNumber(evalDetails?.score_threshold, 2)}
                hint="Higher trims weak detections, lower raises recall."
              />
              <TinyStat
                label="Max detections"
                value={evalDetails?.max_detections != null ? String(evalDetails.max_detections) : 'n/a'}
                hint="Per-image cap before extra predictions are dropped."
              />
            </div>

            {evalDetails?.metric_notes?.precision ? (
              <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4 text-xs text-amber-900">
                <span className="font-semibold">Precision note:</span> {evalDetails.metric_notes.precision}
              </div>
            ) : null}
          </MaterialCard>

          <MaterialCard className="space-y-5 p-6">
            <div className="flex items-center gap-2">
              <Gauge className="h-4 w-4 text-cyan" />
              <h2 className="text-lg font-semibold text-foreground">Metric meaning and next moves</h2>
            </div>

            <div className="grid gap-3">
              {metricMeaningCards.map((card) => (
                <MetricMeaningCard
                  key={card.title}
                  title={card.title}
                  value={card.value}
                  explanation={card.explanation}
                  impact={card.impact}
                />
              ))}
            </div>

            <div className="space-y-3 rounded-3xl border border-border bg-surface-container-low p-4">
              <div className="flex items-center gap-2">
                <Timer className="h-4 w-4 text-amber-600" />
                <h3 className="text-sm font-semibold text-foreground">Performance improvement nudges</h3>
              </div>
              <div className="space-y-2">
                {modelSuggestions.map((suggestion) => (
                  <div key={suggestion.title} className="rounded-2xl border border-border bg-background/60 p-3">
                    <p className="text-sm font-medium text-foreground">{suggestion.title}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{suggestion.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          </MaterialCard>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <MetricCard
            title="Redis connectivity"
            explanation="Redis stores queue state, job status, and worker heartbeats."
            value={report?.redis.ok ? 'Connected' : 'Unavailable'}
            status={report?.redis.ok ? 'ok' : 'degraded'}
            detail={report?.redis.detail || 'No extra details reported.'}
            degradedWhy={report?.redis.ok ? undefined : 'Queue state and heartbeats cannot round-trip to Redis.'}
            degradedRecommendation={report?.redis.ok ? undefined : 'Check REDIS_URL and restart Redis.'}
          />
          <MetricCard
            title="Worker health"
            explanation="Workers process background jobs and publish heartbeats for liveliness."
            value={`${report?.worker.live_workers ?? 0} live / ${report?.worker.stale_workers ?? 0} stale`}
            status={report?.worker.ok ? 'ok' : 'degraded'}
            detail={
              report?.worker.workers?.length
                ? `Newest heartbeat ${formatAge(report.worker.workers[0]?.age_seconds)}.`
                : 'No worker heartbeat found yet.'
            }
            degradedWhy={report?.worker.ok ? undefined : 'Background jobs are not being picked up reliably.'}
            degradedRecommendation={report?.worker.ok ? undefined : 'Restart the ARQ worker and confirm heartbeats resume.'}
          />
          <MetricCard
            title="OCR availability"
            explanation="OCR readiness determines whether panel text can be extracted reliably."
            value={report?.ocr.tesseract_available ? 'Tesseract ready' : 'OCR unavailable'}
            status={report?.ocr.ok ? 'ok' : 'degraded'}
            detail={`manga-ocr: ${report?.ocr.manga_ocr_loaded ? 'loaded' : 'off'} | easyocr: ${
              report?.ocr.easyocr_loaded ? 'loaded' : 'off'
            }`}
            degradedWhy={report?.ocr.ok ? undefined : 'Text extraction services are not fully available.'}
            degradedRecommendation={report?.ocr.ok ? undefined : 'Restore Tesseract or the OCR runtime before extraction-heavy flows.'}
          />
          <MetricCard
            title="Disk space"
            explanation="Exports, checkpoints, and OCR temp files depend on free local storage."
            value={report ? `${report.disk.free_percent}% free` : 'n/a'}
            status={report?.disk.ok ? 'ok' : 'degraded'}
            detail={report ? `${formatBytes(report.disk.free_bytes)} free at ${report.disk.path}` : 'Waiting for report.'}
            degradedWhy={report?.disk.ok ? undefined : 'Low free storage can interrupt exports and temp files.'}
            degradedRecommendation={report?.disk.ok ? undefined : 'Archive or clear older outputs to restore headroom.'}
          />
          <MetricCard
            title="Queue backlog"
            explanation="Backlog shows how many jobs are waiting before a worker can pick them up."
            value={report ? `${report.queue.backlog} pending` : 'n/a'}
            status={report?.queue.ok ? 'ok' : 'degraded'}
            detail={report ? `Queue: ${report.queue.queue_name}` : 'Waiting for report.'}
            degradedWhy={report?.queue.ok ? undefined : 'Interactive runs are waiting behind queued jobs.'}
            degradedRecommendation={report?.queue.ok ? undefined : 'Add worker capacity or reduce heavy jobs.'}
          />
          <MetricCard
            title="Training freshness"
            explanation="This tracks how recently the latest checkpoint was produced."
            value={formatAge(report?.last_training_run.seconds_since_update)}
            status={report?.last_training_run.ok ? 'ok' : 'degraded'}
            detail={report?.last_training_run.path ? getPathLeaf(report.last_training_run.path) : 'No training artifact found.'}
            detailTitle={report?.last_training_run.path || 'No training artifact found.'}
            detailTruncate
            degradedWhy={report?.last_training_run.ok ? undefined : 'Checkpoint freshness is lagging behind recent annotation work.'}
            degradedRecommendation={report?.last_training_run.ok ? undefined : 'Run a fresh training pass after the latest review changes.'}
          />
          <MetricCard
            title="Eval freshness"
            explanation="Fresh evaluation tells us whether recent detector changes were measured."
            value={formatAge(report?.last_eval_run.seconds_since_update)}
            status={report?.last_eval_run.ok ? 'ok' : 'degraded'}
            detail={report?.last_eval_run.path ? getPathLeaf(report.last_eval_run.path) : 'No eval artifact found.'}
            detailTitle={report?.last_eval_run.path || 'No eval artifact found.'}
            detailTruncate
            degradedWhy={report?.last_eval_run.ok ? undefined : 'Recent model changes have not been measured on a fresh eval run.'}
            degradedRecommendation={report?.last_eval_run.ok ? undefined : 'Run detector eval again to refresh recall and mAP.'}
          />
          <MetricCard
            title="Model readiness"
            explanation="Detectors must load weights successfully before panel or bubble inference can run."
            value={report ? `${report.models.filter((model) => model.ok).length}/${report.models.length} ready` : 'n/a'}
            status={report?.models.every((model) => model.ok) ? 'ok' : 'degraded'}
            detail={
              report ? report.models.map((model) => `${model.name}: ${model.ok ? 'ok' : 'issue'}`).join(' | ') : 'Waiting for report.'
            }
            degradedWhy={report?.models.every((model) => model.ok) ? undefined : 'One or more detector weights failed to load cleanly.'}
            degradedRecommendation={report?.models.every((model) => model.ok) ? undefined : 'Repoint the failing weights and reload the model service.'}
          />
          <MaterialCard className="space-y-3 bg-gradient-to-br from-surface-container to-surface-container-high p-5">
            <div className="flex items-center gap-2">
              <HardDrive className="h-4 w-4 text-cyan" />
              <h2 className="text-sm font-semibold text-foreground">Worker heartbeat details</h2>
            </div>
            <p className="text-xs text-muted-foreground">
              Helps confirm whether ARQ is alive and checking Redis on schedule.
            </p>
            <div className="space-y-2">
              {report?.worker.workers?.length ? (
                report.worker.workers.map((worker) => (
                  <div key={worker.worker_key} className="rounded-xl border border-border bg-surface-container-low p-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs font-medium text-foreground">
                        {worker.hostname}:{worker.pid}
                      </p>
                      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{worker.status}</span>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">{worker.queue_name}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{worker.updated_at}</p>
                  </div>
                ))
              ) : (
                <div className="rounded-xl border border-border bg-surface-container-low p-3 text-xs text-muted-foreground">
                  No worker heartbeat has been seen yet. Restart the worker once to start reporting.
                </div>
              )}
            </div>
          </MaterialCard>
        </section>
      </div>
    </main>
  );
}
