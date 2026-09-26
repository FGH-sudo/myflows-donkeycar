import { useId } from 'react'

interface Props {
  values: (number | null | undefined)[]
  color?: string
  width?: number
  height?: number
  /** Fixed value range; defaults to the data's own min/max. */
  min?: number
  max?: number
}

/** Tiny SVG trend line with a soft fill and an end dot. */
export default function Sparkline({ values, color = '#3b6fd8', width = 96, height = 32, min, max }: Props) {
  const gradientId = `spark${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const points = values
    .map((v, i) => (v === null || v === undefined || !Number.isFinite(v) ? null : ([i, v] as const)))
    .filter((p): p is readonly [number, number] => p !== null)
  if (points.length < 2) return <svg className="sparkline" width={width} height={height} aria-hidden />

  const lo = min ?? Math.min(...points.map((p) => p[1]))
  const hi = max ?? Math.max(...points.map((p) => p[1]))
  const span = hi - lo || 1
  const last = values.length - 1 || 1
  const pad = 3
  const xy = points.map(([i, v]) => [(i / last) * (width - pad * 2) + pad, height - pad - ((v - lo) / span) * (height - pad * 2)])
  const line = xy.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const [endX, endY] = xy[xy.length - 1]
  const area = `M${xy[0][0].toFixed(1)},${height} L${line.split(' ').join(' L')} L${endX.toFixed(1)},${height} Z`

  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.16} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradientId})`} />
      <polyline points={line} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={endX} cy={endY} r={2.5} fill={color} stroke="#fff" strokeWidth={1.5} />
    </svg>
  )
}
