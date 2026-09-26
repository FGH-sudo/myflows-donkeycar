import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'
import { useEffect, useRef } from 'react'
import { useUiScale } from '../hooks/useUiScale'
import { chartThemeFor } from './chartTheme'

interface Props {
  option: EChartsOption
  /** Height in px at the base UI scale; scaled up on large monitors. */
  height?: number | string
  notMerge?: boolean
  onEvents?: Record<string, (params: unknown) => void>
}

export default function EChart({ option, height = 300, notMerge = true, onEvents }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const optionRef = useRef(option)
  const scale = useUiScale()
  const themeName = chartThemeFor(scale)

  useEffect(() => {
    optionRef.current = option
  })

  useEffect(() => {
    if (!ref.current) return
    const instance = echarts.init(ref.current, themeName, { renderer: 'canvas' })
    chart.current = instance
    instance.setOption(optionRef.current, { notMerge: true })
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(ref.current)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [themeName])

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
  }, [onEvents, themeName])

  const scaledHeight = typeof height === 'number' ? Math.round(height * scale) : height
  return <div ref={ref} style={{ width: '100%', height: scaledHeight }} />
}
