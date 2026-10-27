export type RunStatus =
  | 'pending'
  | 'running'
  | 'stopping'
  | 'succeeded'
  | 'passed'
  | 'failed'
  | 'cancelled'
  | 'finished'
  | 'incomplete'
  | 'error'

export type RunKind = 'distributed' | 'single' | 'tensorboard'

export interface SplitMetrics {
  samples?: number
  loss?: number
  accuracy?: number
  mse?: number
  angle_mae?: number
  angle_rmse?: number
  throttle_mae?: number
}

export interface RunSummary {
  id: string
  source: string
  label: string
  kind: RunKind
  task: string | null
  mode: string | null
  transport: string | null
  workers: number
  status: RunStatus
  started_unix_s: number | null
  finished_unix_s: number | null
  elapsed_s: number | null
  device: string | null
  optimizer?: string | null
  global_batch?: number | null
  epochs?: number | null
  learning_rate?: number | null
  final: {
    val?: SplitMetrics
    test?: SplitMetrics
    train?: SplitMetrics
    samples_per_s?: number
    epoch_train_wall_s?: number
    last_mean_loss?: number
    best_loss?: number
    steps?: number
  }
  error?: string | null
  job_id?: string | null
  path: string
}

export interface RankStatus {
  rank: number
  pid: number | null
  steps_done: number
  epochs_done: number
  epoch?: number
  step?: number
  loss?: number
  step_wall_s?: number
  gradient_sync_s?: number
  sync_fraction?: number | null
  seconds_since_last?: number | null
}

export type Cursor = Record<string, number>
export type Columns = Record<string, (number | null)[]>

export interface Job {
  id: string
  type: 'distributed' | 'single'
  label: string
  task: string
  status: RunStatus
  created_unix_s: number
  started_unix_s: number | null
  finished_unix_s: number | null
  pid: number | null
  returncode: number | null
  error: string | null
  command: string[]
  run_dir: string
  log_path: string
  config: Record<string, unknown> | null
}

export interface RunDetail {
  summary: RunSummary
  kind: RunKind
  ranks: number[]
  cursor: Cursor
  config: Record<string, unknown>
  manifest: Record<string, unknown> | null
  results: Record<string, unknown> | null
  epochs: Record<string, number | null>[]
  rank_status?: RankStatus[]
  rank_pids?: Record<string, number>
  job?: Job | null
  scalar_tags?: string[]
}

export interface StepsResponse {
  origin: number | null
  ranks: Record<string, Columns>
  cursor: Cursor
}

export interface EpochRow {
  rank?: number
  epoch: number
  t_start?: number | null
  t_end?: number | null
  [key: string]: number | string | null | undefined
}

export interface EpochsResponse {
  ranks: Record<string, EpochRow[]>
  cursor: Cursor
}

export const RESOURCE_KEYS = ['cpu_utilization_pct', 'ram_used_bytes', 'gpu_utilization_pct', 'gpu_memory_mib',
  'gpu_temperature_c', 'gpu_power_w'] as const

export type ResourceSeries = { t: number[]; rank_rss_mib: Record<string, (number | null)[]> } &
  Partial<Record<(typeof RESOURCE_KEYS)[number], (number | null)[]>>

export interface ResourcesResponse {
  origin: number | null
  series: ResourceSeries
  sampling_errors: number
  cursor: Cursor
}

export interface GpuProcess {
  pid: number
  used_memory_mib: number | null
  owner?: { job_id: string; label: string; rank?: number } | null
}

export interface GpuInfo {
  index: number
  name: string
  uuid: string
  utilization_pct: number | null
  memory_used_mib: number | null
  memory_total_mib: number | null
  temperature_c: number | null
  power_w: number | null
  power_limit_w: number | null
  processes: GpuProcess[] | null
  missing: Record<string, string>
}

export interface GpuSample {
  unix_s: number
  cpu_utilization_pct: number
  ram_used_bytes: number
  ram_total_bytes: number
  gpus: GpuInfo[]
  sampling_error?: string
}

export interface GpuPayload {
  backend: string | null
  driver_version: string | null
  backend_errors: Record<string, string>
  interval_s: number
  latest: GpuSample | null
  history?: GpuSample[]
}

export interface ModeOption {
  mode: string
  transport: string
  label: string
}

export interface Preset {
  id: string
  type: 'distributed' | 'single'
  task: string
  label: string
  description: string
  config?: Record<string, unknown>
  args?: Record<string, unknown>
  data_ready: boolean
  data_hint: string | null
}

/** Conv/pool backend of the donkey_* tasks and apps.train; `default` marks the device's default. */
export interface ConvBackendOption {
  value: string
  device: 'cuda' | 'cpu'
  label: string
  description: string
  default: boolean
}

/** Native CUDA library on the console host; jobs build a missing or stale one before training. */
export interface NativeLibraryStatus {
  path: string
  built: boolean
  current: boolean
  reason: string | null
  cmake: string | null
  dependencies: boolean
}

export interface PresetsResponse {
  modes: ModeOption[]
  tasks: { task: string; label: string }[]
  max_workers: number
  worker_choices: number[]
  conv_backends: ConvBackendOption[]
  native_library: NativeLibraryStatus
  distributed: Preset[]
  single: Preset[]
}

export interface SourceInfo {
  key: string
  label: string
  kind: string
  root: string
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}

export const runPath = (id: string) => `/api/runs/${id.split('/').map(encodeURIComponent).join('/')}`

function query(params: Record<string, string | number | undefined | null>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  if (!entries.length) return ''
  return '?' + new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString()
}

export const api = {
  gpu: () => request<GpuPayload>('/api/gpu'),
  sources: () => request<SourceInfo[]>('/api/sources'),
  runs: (params: { source?: string; kind?: string; task?: string; mode?: string; status?: string; q?: string } = {}) =>
    request<RunSummary[]>('/api/runs' + query(params)),
  run: (id: string) => request<RunDetail>(runPath(id)),
  steps: (id: string, params: { ranks?: string; fields?: string; max_points?: number } = {}) =>
    request<StepsResponse>(runPath(id) + '/steps' + query(params)),
  epochs: (id: string) => request<EpochsResponse>(runPath(id) + '/epochs'),
  resources: (id: string, maxPoints = 3000) =>
    request<ResourcesResponse>(runPath(id) + '/resources' + query({ max_points: maxPoints })),
  log: (id: string, tailKb = 64) => request<{ path: string | null; text: string }>(runPath(id) + '/log' + query({ tail_kb: tailKb })),
  presets: () => request<PresetsResponse>('/api/presets'),
  jobs: () => request<Job[]>('/api/jobs'),
  submitJob: (body: { type: 'distributed'; config: Record<string, unknown>; label?: string } | { type: 'single'; args: Record<string, unknown>; label?: string }) =>
    request<Job>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  stopJob: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}/stop`, { method: 'POST' }),
}

export const streamUrl = (id: string, cursor: Cursor) => runPath(id) + '/stream' + query({ cursor: JSON.stringify(cursor) })

// ---- AutoPilot ------------------------------------------------------------------------------------------

export type CampaignStatus =
  | 'draft'
  | 'running'
  | 'pausing'
  | 'paused'
  | 'stopping'
  | 'finishing'
  | 'completed'
  | 'stopped'
  | 'failed'

export type CampaignPhase = 'baseline' | 'explore' | 'exploit' | 'confirm' | 'done'

export interface CampaignSpec {
  schema_version: number
  campaign_id: string | null
  title: string
  mode: 'dev' | 'formal'
  task: string
  objective: { metric: string; direction: 'minimize' | 'maximize' }
  baseline: { lr: number; batch: number; num_workers: number; seeds: number[] }
  fixed: Record<string, string | number | boolean | null>
  space: Record<string, (number | boolean)[]>
  strategy: { name: 'candidate_list' | 'tpe'; max_candidates: number; seed: number }
  budget: { max_trials: number; trial_timeout_s: number; wall_time_s: number; max_retries: number; max_gpu_memory_mb: number | null }
  online: {
    divergence_windows: number
    divergence_factor: number
    window_steps: number
    hang_timeout_s: number
    gpu_idle_pct: number
    explosion_factor: number
  }
  confirm: { top_k: number; seeds: number[] | null }
  specialists: string[]
  llm: { enabled: boolean; max_calls: number; max_total_tokens: number | null }
  rollback: { enabled: boolean; max_rollbacks: number; lr_factor: number; state_every: number; oom_half_batch: boolean }
  created_by: 'user' | 'llm'
}

export interface BudgetUsed {
  trials_started: number
  trials_committed: number
  attempts: number
  run_s: number
}

export interface CampaignSummary {
  campaign_id: string
  title: string
  task: string
  mode: 'dev' | 'formal'
  status: CampaignStatus
  phase: CampaignPhase
  created_by: 'user' | 'llm'
  created_unix_s: number | null
  started_unix_s: number | null
  finished_unix_s: number | null
  objective: CampaignSpec['objective']
  budget: CampaignSpec['budget']
  used: BudgetUsed
  trials: number
  valid: number
  best: { trial_id: string; value: number; params: Record<string, number> } | null
  pending: number
  llm: { enabled: boolean; calls: number; disabled_reason: string | null }
  stop_reason: string | null
}

export interface Precheck {
  passed: boolean
  reasons: string[]
  warnings?: string[]
}

export interface PendingItem {
  id: string
  kind: 'trials' | 'campaign_start'
  campaign_id: string
  source: string
  proposals: { delta: Record<string, number>; base_trial?: string | null; rationale: string; expected: string; precheck: Precheck }[]
  precheck: { passed: boolean; reasons: string[] }
  status: 'pending_confirmation' | 'confirmed' | 'rejected' | 'expired'
  created_unix_s: number
  resolved_unix_s?: number
}

export interface Finding {
  id: string
  agent: string
  kind: string
  message: string
  trial_id: string | null
  severity: 'info' | 'warning' | 'action'
  evidence: { trial_id: string; file: string; metric: string; value: number | null; threshold?: number }[]
  data: Record<string, unknown>
}

export interface ChainSide {
  trial_id: string
  objective: number | null
  throughput_sps: number | null
  data_share: number | null
  gpu_util_median: number | null
  mem_share: number | null
  params: Record<string, number>
}

export interface DiagnosisChain {
  trial_id: string
  base_trial: string
  finding_ids: string[]
  findings: (string | null)[]
  proposal: { delta: Record<string, number>; rationale: string; agent: string; kind: string }
  before: ChainSide
  after: ChainSide
  decision: 'adopted' | 'rejected' | 'failed' | 'better' | 'not_better' | 'recovered'
  reason: string
  throughput_gain?: number | null
}

export interface NoiseLevel {
  metric: string
  seeds: number[]
  values: number[]
  mean: number
  std: number
  throughput_cv: number | null
}

export interface CampaignDetail extends CampaignSummary {
  spec: CampaignSpec
  state: {
    status: CampaignStatus
    phase: CampaignPhase
    phase_log: { phase: string; note: string | null }[]
    adopted: Record<string, number>
    noise: NoiseLevel | null
    pending: PendingItem[]
    notes: { text: string; title: string | null; source: string; unix_s: number }[]
    provenance: {
      split?: Record<string, unknown>
      code?: { code_hash: string; repos: Record<string, { head: string | null; dirty_hash: string | null; dirty_files: string[] }> }
    }
    llm: { session_id: string | null; calls: number; tokens: number; failures: number; disabled_reason: string | null }
    unavailable: Record<string, string>
    stop_reason: string | null
    best: { trial_id: string; value: number } | null
  }
  report_available: boolean
}

export interface TrialAttempt {
  attempt: number
  job_id: string
  run_dir: string
  status: string
  failure: { kind: string; message: string; source?: string } | null
  run_s?: number
  lock_wait_s?: number | null
  online_stop?: { kind: string; message: string } | null
  cause?: 'first' | 'retry' | 'rollback'
}

export interface Trial {
  trial_id: string
  phase: CampaignPhase
  params: Record<string, number>
  resolved_args: Record<string, unknown>
  origin: {
    strategy: string
    agent?: string | null
    source: string
    base_trial?: string | null
    rationale?: string
    pending_id?: string | null
    finding_ids?: string[]
  }
  status: 'queued' | 'running' | 'done' | 'skipped' | 'cancelled'
  attempts: TrialAttempt[]
  verdict: { status: 'valid' | 'invalid' | 'failed' | 'cancelled'; kind: string | null; reasons: string[] } | null
  result: {
    metrics: Record<string, number | null>
    summary: {
      throughput_sps?: number | null
      data_share?: number | null
      gpu_util_median?: number | null
      mem_peak_mb?: number | null
      mem_share?: number | null
    }
  } | null
  skip_reason?: string
  rank: number | null
  objective: number | null
}

export interface CampaignEvent {
  seq: number
  unix_s: number
  kind: string
  [key: string]: unknown
}

export interface AutoPilotDefaults {
  spec: CampaignSpec
  formal: CampaignSpec
  tasks: { task: string; label: string; metrics: Record<string, string> }[]
  modes: Record<string, string>
  strategies: string[]
  specialists: Record<string, { label: string; available: boolean; reason: string | null }>
  llm_available: boolean
}

export interface DshStatus {
  available: boolean
  llm_ready: boolean
  checking?: boolean
  reason?: string | null
  context_campaign?: string | null
  web?: { state: string; url: string | null; pid?: number | null; error?: string | null; port?: number }
  key_configured?: boolean
  credentials?: { api_key: boolean; account: boolean }
  clone?: { path: string; commit: string | null; expected: string; built: boolean; node?: string | null }
  composition?: { ok: boolean; errors: string[] }
}

const ap = (cid: string) => `/api/autopilot/campaigns/${encodeURIComponent(cid)}`

export const autopilot = {
  defaults: () => request<AutoPilotDefaults>('/api/autopilot/defaults'),
  campaigns: () => request<CampaignSummary[]>('/api/autopilot/campaigns'),
  campaign: (cid: string) => request<CampaignDetail>(ap(cid)),
  create: (spec: Partial<CampaignSpec>) =>
    request<CampaignDetail>('/api/autopilot/campaigns', { method: 'POST', body: JSON.stringify(spec) }),
  update: (cid: string, patch: Partial<CampaignSpec>) =>
    request<CampaignDetail>(ap(cid), { method: 'PATCH', body: JSON.stringify(patch) }),
  action: (cid: string, action: 'start' | 'pause' | 'resume' | 'stop') =>
    request<CampaignDetail>(`${ap(cid)}/${action}`, { method: 'POST' }),
  trials: (cid: string) => request<Trial[]>(`${ap(cid)}/trials`),
  stopTrial: (cid: string, tid: string) =>
    request<Trial>(`${ap(cid)}/trials/${encodeURIComponent(tid)}/stop`, { method: 'POST' }),
  findings: (cid: string) =>
    request<{ findings: Finding[]; chains: DiagnosisChain[]; noise: NoiseLevel | null; unavailable: Record<string, string> }>(
      `${ap(cid)}/findings`,
    ),
  pending: (cid: string) => request<PendingItem[]>(`${ap(cid)}/pending`),
  resolve: (cid: string, pid: string, confirm: boolean) =>
    request<PendingItem>(`${ap(cid)}/pending/${encodeURIComponent(pid)}/${confirm ? 'confirm' : 'reject'}`, { method: 'POST' }),
  events: (cid: string, kinds?: string) => request<CampaignEvent[]>(`${ap(cid)}/events` + query({ kinds })),
  report: (cid: string) => request<{ markdown: string; path: string; figures: string[] }>(`${ap(cid)}/report`),
  regenerateReport: (cid: string) =>
    request<{ markdown: string; path: string; figures: string[] }>(`${ap(cid)}/report`, { method: 'POST' }),
  figureUrl: (cid: string, name: string) => `${ap(cid)}/report/figures/${encodeURIComponent(name)}`,
  streamUrl: (cid: string) => `${ap(cid)}/stream`,
  dshStatus: (campaign?: string | null) => request<DshStatus>('/api/autopilot/dsh/status' + query({ ensure_web: 1, campaign })),
  dshRestart: () => request<DshStatus>('/api/autopilot/dsh/restart', { method: 'POST' }),
  dshContext: (campaign: string) =>
    request<{ campaign_id: string }>('/api/autopilot/dsh/context', { method: 'POST', body: JSON.stringify({ campaign_id: campaign }) }),
}
