import * as echarts from 'echarts'
import { RANK_COLORS } from '../format'

/** Same stack as --font-sans so canvas text matches the DOM. */
export const CHART_FONT =
  '"Inter Variable", Inter, "Segoe UI Variable Text", "Segoe UI", -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif'

const TEXT = '#18181b'
const MUTED = '#71717a'
const SECONDARY = '#52525b'
const GRID = '#f0f0f3'
const AXIS = '#e4e4e7'

/** Theme for a UI scale factor (see hooks/useUiScale); text grows with the page. */
function buildTheme(f: number) {
  const px = (v: number) => Math.round(v * f)
  const axis = {
    axisLine: { show: false, lineStyle: { color: AXIS } },
    axisTick: { show: false },
    axisLabel: { color: MUTED, fontSize: px(11), hideOverlap: true },
    splitLine: { lineStyle: { color: GRID, type: 'solid' } },
    nameTextStyle: { color: MUTED, fontSize: px(11) },
  }
  return {
    color: RANK_COLORS,
    backgroundColor: 'transparent',
    textStyle: { fontFamily: CHART_FONT, color: SECONDARY, fontSize: px(12) },
    title: {
      textStyle: { color: TEXT, fontSize: px(14), fontWeight: 600 },
      subtextStyle: { color: MUTED, fontSize: px(12) },
    },
    legend: {
      icon: 'roundRect',
      itemWidth: px(10),
      itemHeight: 3,
      itemGap: px(16),
      textStyle: { color: SECONDARY, fontSize: px(12) },
      pageIconColor: TEXT,
      pageIconInactiveColor: '#d4d4d8',
      pageIconSize: 10,
      pageTextStyle: { color: MUTED, fontSize: px(11) },
    },
    tooltip: {
      backgroundColor: 'rgba(255, 255, 255, 0.98)',
      borderColor: '#e8e8ec',
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { color: TEXT, fontSize: px(12), fontFamily: CHART_FONT },
      extraCssText: 'box-shadow: 0 8px 24px rgba(24, 24, 27, 0.08); border-radius: 10px;',
      axisPointer: { lineStyle: { color: '#a1a1aa', type: 'dashed' }, shadowStyle: { color: 'rgba(24, 24, 27, 0.03)' } },
    },
    categoryAxis: {
      ...axis,
      axisLine: { show: true, lineStyle: { color: AXIS } },
      splitLine: { show: false, lineStyle: { color: GRID } },
    },
    valueAxis: axis,
    logAxis: axis,
    timeAxis: axis,
    line: { symbol: 'circle', symbolSize: 5, lineStyle: { width: 1.6 } },
    graph: { label: { fontFamily: CHART_FONT } },
  }
}

const registered = new Set<string>()

/** Registers (once) and returns the ECharts theme name for a UI scale factor. */
export function chartThemeFor(f: number) {
  const name = `myflows-${Math.round(f * 1000)}`
  if (!registered.has(name)) {
    echarts.registerTheme(name, buildTheme(f))
    registered.add(name)
  }
  return name
}
