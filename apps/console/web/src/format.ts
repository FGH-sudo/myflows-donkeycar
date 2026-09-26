import dayjs from 'dayjs'
import type { RunStatus, RunSummary } from './api'

export const ACTIVE_STATUSES: RunStatus[] = ['pending', 'running', 'stopping']

export const isActive = (status?: string | null) => !!status && (ACTIVE_STATUSES as string[]).includes(status)

export const STATUS_META: Record<string, { color: string; label: string }> = {
  pending: { color: 'default', label: '排队中' },
  running: { color: 'processing', label: '运行中' },
  stopping: { color: 'warning', label: '停止中' },
  succeeded: { color: 'success', label: '成功' },
  passed: { color: 'success', label: '通过' },
  finished: { color: 'success', label: '已结束' },
  failed: { color: 'error', label: '失败' },
  cancelled: { color: 'default', label: '已取消' },
  incomplete: { color: 'warning', label: '不完整' },
  error: { color: 'error', label: '读取错误' },
}

export const MODE_TEXT: Record<string, string> = {
  single: '单进程',
  ps: 'PS',
  ring: 'Ring',
}

export const TRANSPORT_TEXT: Record<string, string> = {
  none: '—',
  socket_json: 'Socket JSON',
  grpc_proto: 'gRPC',
}

export const KIND_TEXT: Record<string, string> = {
  distributed: '分布式',
  single: '单进程',
  tensorboard: 'TensorBoard',
}

export function fmtTime(unix?: number | null): string {
  return unix ? dayjs(unix * 1000).format('MM-DD HH:mm:ss') : '—'
}

export function fmtDuration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m < 60) return `${m} 分 ${s} 秒`
  return `${Math.floor(m / 60)} 时 ${m % 60} 分`
}

export function fmtNum(value?: number | null, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(2)
  return Number(value.toFixed(digits)).toString()
}

export function fmtBytes(bytes?: number | null): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

export function fmtPct(value?: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${value.toFixed(0)}%`
}

export function modeLabel(run: Pick<RunSummary, 'mode' | 'transport'>): string {
  if (!run.mode) return '—'
  const mode = MODE_TEXT[run.mode] ?? run.mode
  if (!run.transport || run.transport === 'none') return mode
  return `${mode} · ${TRANSPORT_TEXT[run.transport] ?? run.transport}`
}

export function headlineMetric(run: RunSummary): { label: string; value: string } | null {
  const val = run.final?.val
  if (val?.accuracy !== undefined) return { label: 'val acc', value: `${(val.accuracy * 100).toFixed(2)}%` }
  if (val?.angle_mae !== undefined) return { label: 'val angle MAE', value: fmtNum(val.angle_mae) }
  if (val?.loss !== undefined) return { label: 'val loss', value: fmtNum(val.loss) }
  if (run.final?.best_loss !== undefined && run.final.best_loss !== null) return { label: 'best loss', value: fmtNum(run.final.best_loss) }
  if (run.final?.last_mean_loss !== undefined && run.final.last_mean_loss !== null) return { label: 'loss', value: fmtNum(run.final.last_mean_loss) }
  return null
}

/** Muted data palette shared by every chart and rank marker. */
export const RANK_COLORS = ['#3b6fd8', '#d4892a', '#2f9e7a', '#8866d6', '#cf5a80', '#3fa0b5', '#7a8394', '#b99a2a']

/** "NVIDIA GeForce RTX 4060 Laptop GPU" -> "RTX 4060 Laptop GPU" */
export const shortGpuName = (name: string) => name.replace(/^NVIDIA\s+(GeForce\s+)?/i, '')

export const rankColor = (rank: number | string) => RANK_COLORS[Number(rank) % RANK_COLORS.length]
