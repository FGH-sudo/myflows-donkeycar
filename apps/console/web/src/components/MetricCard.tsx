import { Card } from 'antd'
import type { ReactNode } from 'react'

interface Props {
  label: string
  value: ReactNode
  unit?: string
  extra?: ReactNode
  footer?: ReactNode
}

export default function MetricCard({ label, value, unit, extra, footer }: Props) {
  return (
    <Card className="metric-card" size="small">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="metric-label">{label}</div>
          <div className="metric-value">
            {value}
            {unit ? <span className="metric-unit">{unit}</span> : null}
          </div>
        </div>
        {extra}
      </div>
      {footer ? <div style={{ marginTop: 8 }}>{footer}</div> : null}
    </Card>
  )
}
