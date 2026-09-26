import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp, ConfigProvider, theme } from 'antd'
import type { ThemeConfig } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { useMemo } from 'react'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import { CHART_FONT } from './components/chartTheme'
import { UiScaleContext, useUiScaleFactor } from './hooks/useUiScale'

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1, staleTime: 2000 } },
})

// Keep in sync with the tokens in index.css.
const INK = '#18181b'
const ACCENT = '#3b6fd8'

/** antd theme; sizes scale with the UI factor so large monitors get larger controls. */
function buildTheme(f: number): ThemeConfig {
  const px = (v: number) => Math.round(v * f)
  return {
    algorithm: theme.defaultAlgorithm,
    token: {
      colorPrimary: ACCENT,
      colorInfo: ACCENT,
      colorLink: ACCENT,
      colorSuccess: '#2f9e62',
      colorWarning: '#c47a12',
      colorError: '#d14343',
      colorBgLayout: '#f7f7f8',
      colorBorder: '#dcdce2',
      colorBorderSecondary: '#e8e8ec',
      colorSplit: '#ececf0',
      colorText: INK,
      colorTextSecondary: '#52525b',
      colorTextTertiary: '#71717a',
      colorTextQuaternary: '#a1a1aa',
      colorFillAlter: '#fafafb',
      colorFillSecondary: '#f1f1f4',
      borderRadius: 8,
      borderRadiusLG: 12,
      fontFamily: CHART_FONT,
      fontSize: px(14),
      controlHeight: px(34),
      boxShadowTertiary: 'none',
    },
    components: {
      Layout: {
        siderBg: 'transparent',
        bodyBg: '#f7f7f8',
        headerBg: 'transparent',
        headerHeight: px(52),
      },
      Menu: {
        itemBg: 'transparent',
        itemHeight: px(36),
        itemBorderRadius: 8,
        itemMarginInline: 14,
        itemMarginBlock: 3,
        itemColor: '#52525b',
        itemHoverColor: INK,
        itemHoverBg: 'rgba(24, 24, 27, 0.04)',
        itemSelectedBg: '#ffffff',
        itemSelectedColor: INK,
        iconSize: px(15),
        collapsedIconSize: px(16),
        activeBarBorderWidth: 0,
      },
      Button: {
        // Primary buttons are ink; blue stays reserved for links, selection and focus.
        colorPrimary: INK,
        colorPrimaryHover: '#3f3f46',
        colorPrimaryActive: '#000000',
        colorLink: ACCENT,
        colorLinkHover: '#5584e3',
        fontWeight: 500,
        primaryShadow: 'none',
        defaultShadow: 'none',
        dangerShadow: 'none',
      },
      Card: {
        headerFontSize: px(14),
        headerFontSizeSM: px(14),
      },
      Table: {
        headerBg: 'transparent',
        headerColor: '#71717a',
        headerSplitColor: 'transparent',
        rowHoverBg: '#fafafb',
        borderColor: '#efeff2',
        cellFontSize: px(14),
        cellFontSizeMD: px(14),
        cellFontSizeSM: px(13),
        cellPaddingBlock: px(14),
        cellPaddingBlockMD: px(14),
        cellPaddingBlockSM: px(11),
        cellPaddingInline: px(16),
        cellPaddingInlineMD: px(16),
        cellPaddingInlineSM: px(12),
        headerSortActiveBg: 'transparent',
        headerSortHoverBg: '#fafafb',
        bodySortBg: 'transparent',
        rowSelectedBg: 'rgba(59, 111, 216, 0.05)',
        rowSelectedHoverBg: 'rgba(59, 111, 216, 0.08)',
      },
      Tabs: {
        titleFontSize: px(14),
        itemColor: '#71717a',
        itemHoverColor: INK,
        itemSelectedColor: INK,
        inkBarColor: INK,
        horizontalItemGutter: px(28),
        horizontalMargin: `0 0 ${px(24)}px 0`,
      },
      Segmented: {
        trackBg: '#ececf0',
        itemColor: '#52525b',
        itemHoverColor: INK,
        itemSelectedBg: '#ffffff',
        itemSelectedColor: INK,
      },
      Tag: {
        defaultBg: '#f1f1f4',
        defaultColor: '#52525b',
      },
      Alert: {
        withDescriptionPadding: '14px 16px',
      },
      Descriptions: {
        labelBg: '#fafafb',
        labelColor: '#71717a',
      },
      Progress: {
        remainingColor: '#ececf0',
      },
      Slider: {
        railBg: '#e4e4e7',
        railHoverBg: '#d4d4d8',
      },
      Tooltip: {
        colorBgSpotlight: INK,
      },
    },
  }
}

export default function Root() {
  const factor = useUiScaleFactor()
  const themeConfig = useMemo(() => buildTheme(factor), [factor])
  return (
    <UiScaleContext.Provider value={factor}>
      <ConfigProvider locale={zhCN} tag={{ variant: 'filled' }} theme={themeConfig}>
        <AntApp>
          <QueryClientProvider client={queryClient}>
            <BrowserRouter>
              <App />
            </BrowserRouter>
          </QueryClientProvider>
        </AntApp>
      </ConfigProvider>
    </UiScaleContext.Provider>
  )
}

