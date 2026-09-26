import type { EChartsOption } from 'echarts'
import type { ReactNode } from 'react'
import EChart from './EChart'
import { Panel } from './Panel'

interface Props {
  option: EChartsOption
  height?: number
  title?: ReactNode
  subtitle?: ReactNode
  extra?: ReactNode
  className?: string
}

/**
 * A chart inside a panel. The title lives in the DOM (not in the canvas) so it
 * shares page typography; charts drawn here should put their legend top-left.
 */
export default function ChartPanel({ option, height = 320, title, subtitle, extra, className }: Props) {
  return (
    <Panel title={title} subtitle={subtitle} extra={extra} className={className}>
      <div className="panel-body">
        <EChart option={option} height={height} />
      </div>
    </Panel>
  )
}
