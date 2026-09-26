import type { CSSProperties, ReactNode } from 'react'
import { useUiScale } from '../hooks/useUiScale'
import Sparkline from './Sparkline'

export interface StatItem {
  label: ReactNode
  value: ReactNode
  unit?: ReactNode
  hint?: ReactNode
  /** Colour of the label swatch and sparkline. */
  color?: string
  spark?: (number | null | undefined)[]
  sparkRange?: [number, number]
}

/** One bordered strip of metrics separated by hairlines; wraps on narrow screens. */
export default function StatStrip({ items }: { items: StatItem[] }) {
  const scale = useUiScale()
  return (
    <div className="stat-strip" style={{ '--cols': items.length } as CSSProperties}>
      {items.map((item, i) => (
        <div className="stat-cell" key={i}>
          <span className="stat-label">
            {item.color ? <span className="stat-swatch" style={{ background: item.color }} /> : null}
            {item.label}
          </span>
          <div className="stat-row">
            <span className="stat-value-wrap">
              <span className="stat-value">{item.value}</span>
              {item.unit ? <span className="stat-unit">{item.unit}</span> : null}
            </span>
            {item.spark ? (
              <Sparkline values={item.spark} color={item.color} width={Math.round(96 * scale)} height={Math.round(32 * scale)} min={item.sparkRange?.[0]} max={item.sparkRange?.[1]} />
            ) : null}
          </div>
          {item.hint ? <span className="stat-hint">{item.hint}</span> : null}
        </div>
      ))}
    </div>
  )
}
