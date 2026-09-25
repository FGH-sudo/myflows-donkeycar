import type { Columns, EpochRow } from '../../api'

export type Point = [number, number | null]

export function ema(values: (number | null)[], alpha: number): (number | null)[] {
  if (alpha <= 0) return values
  let last: number | null = null
  let debias = 0
  let acc = 0
  return values.map((v) => {
    if (v === null || !Number.isFinite(v)) return last
    acc = acc * alpha + (1 - alpha) * v
    debias = debias * alpha + (1 - alpha)
    last = acc / debias
    return last
  })
}

export function zip(xs: (number | null)[] | undefined, ys: (number | null)[] | undefined): Point[] {
  if (!xs || !ys) return []
  const out: Point[] = []
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    const x = xs[i]
    if (x !== null && x !== undefined) out.push([x, ys[i] ?? null])
  }
  return out
}

/** Average consecutive rows into at most ``bins`` buckets (for readable stacked charts). */
export function binColumns(cols: Columns, fields: string[], xKey: string, bins = 300): Columns {
  const n = cols[xKey]?.length ?? 0
  if (n <= bins) return cols
  const size = Math.ceil(n / bins)
  const out: Columns = { [xKey]: [] }
  fields.forEach((f) => (out[f] = []))
  for (let start = 0; start < n; start += size) {
    const end = Math.min(n, start + size)
    out[xKey].push(cols[xKey][Math.floor((start + end - 1) / 2)])
    for (const f of fields) {
      let sum = 0
      let count = 0
      for (let i = start; i < end; i += 1) {
        const v = cols[f]?.[i]
        if (v !== null && v !== undefined && Number.isFinite(v)) {
          sum += v
          count += 1
        }
      }
      out[f].push(count ? sum / count : null)
    }
  }
  return out
}

/** Shaded epoch intervals; distributed epochs are 0-based, so ``offset`` = 1 shows them as 1-based. */
export function epochMarkAreas(rows: EpochRow[] | undefined, offset = 1) {
  return (rows ?? [])
    .filter((r) => r.t_start !== null && r.t_start !== undefined && r.t_end !== null && r.t_end !== undefined)
    .map((r, i) => [
      { xAxis: r.t_start as number, name: `E${Number(r.epoch) + offset}`,
        itemStyle: { color: i % 2 ? 'rgba(22,104,220,0.05)' : 'rgba(22,104,220,0.10)' } },
      { xAxis: r.t_end as number },
    ])
}

export const baseGrid = { left: 56, right: 56, top: 40, bottom: 56 }

export const zoom = [
  { type: 'inside' as const, xAxisIndex: 0 },
  { type: 'slider' as const, xAxisIndex: 0, height: 18, bottom: 8 },
]
