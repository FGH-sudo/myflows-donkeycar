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

export interface PresetsResponse {
  modes: ModeOption[]
  tasks: { task: string; label: string }[]
  max_workers: number
  worker_choices: number[]
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
