import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'
import { useEffect, useRef } from 'react'

interface Props {
  option: EChartsOption
  height?: number | string
  notMerge?: boolean
  onEvents?: Record<string, (params: unknown) => void>
}

export default function EChart({ option, height = 300, notMerge = true, onEvents }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const instance = echarts.init(ref.current, undefined, { renderer: 'canvas' })
    chart.current = instance
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(ref.current)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    chart.current?.setOption(option, { notMerge, lazyUpdate: true })
  }, [option, notMerge])

  useEffect(() => {
    const instance = chart.current
    if (!instance || !onEvents) return
    Object.entries(onEvents).forEach(([name, handler]) => instance.on(name, handler))
    return () => {
      Object.keys(onEvents).forEach((name) => instance.off(name))
    }
  }, [onEvents])

  return <div ref={ref} style={{ width: '100%', height }} />
}
