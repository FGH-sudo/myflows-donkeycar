import {
  DashboardOutlined,
  LineChartOutlined,
  PlusCircleOutlined,
  PlusOutlined,
  RightOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Layout, Menu, Tooltip } from 'antd'
import { useEffect, useState } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { api } from './api'
import { isActive, shortGpuName } from './format'
import { useUiScale } from './hooks/useUiScale'
import Compare from './pages/Compare'
import NewJob from './pages/NewJob'
import Overview from './pages/Overview'
import RunDetailPage from './pages/RunDetail'
import Runs from './pages/Runs'

const { Sider, Content, Header } = Layout

const SECTIONS: Record<string, string> = {
  '/': '总览',
  '/runs': '运行记录',
  '/new': '新建训练',
  '/compare': '运行对比',
}

const NARROW_QUERY = '(max-width: 991.98px)'

function useScrolled(threshold = 4) {
  const [scrolled, setScrolled] = useState(() => window.scrollY > threshold)
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > threshold)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [threshold])
  return scrolled
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const scrolled = useScrolled()
  const scale = useUiScale()
  const [collapsed, setCollapsed] = useState(() => window.matchMedia(NARROW_QUERY).matches)
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 3000 })
  const health = useQuery({
    queryKey: ['health'],
    queryFn: () => fetch('/api/health').then((r) => r.json() as Promise<{ gpu_backend: string | null }>),
    refetchInterval: 10000,
  })
  const gpu = useQuery({ queryKey: ['gpu'], queryFn: api.gpu, staleTime: 60_000 })
  const active = (jobs.data ?? []).filter((j) => isActive(j.status)).length

  const isRunDetail = location.pathname.startsWith('/run/')
  const runId = isRunDetail ? decodeURIComponent(location.pathname.slice('/run/'.length)) : ''
  const run = useQuery({ queryKey: ['run', runId], queryFn: () => api.run(runId), enabled: isRunDetail })
  const section = isRunDetail ? '/runs' : '/' + (location.pathname.split('/')[1] ?? '')

  const items = [
    { key: '/', icon: <DashboardOutlined />, label: '总览' },
    {
      key: '/runs',
      icon: <UnorderedListOutlined />,
      label: (
        <span className="nav-label">
          <span>运行记录</span>
          {active ? <span className="nav-count">{active}</span> : null}
        </span>
      ),
    },
    { key: '/new', icon: <PlusCircleOutlined />, label: '新建训练' },
    { key: '/compare', icon: <LineChartOutlined />, label: '运行对比' },
  ]

  const gpuInfo = gpu.data?.latest?.gpus[0]
  const backend = health.data?.gpu_backend ?? gpu.data?.backend ?? null

  return (
    <Layout hasSider style={{ minHeight: '100vh' }}>
      <Sider
        className="app-sider"
        width={Math.round(232 * scale)}
        collapsedWidth={Math.round(72 * scale)}
        breakpoint="lg"
        collapsed={collapsed}
        onCollapse={setCollapsed}
        trigger={null}
      >
        <Link to="/" className="console-logo">
          <span className="logo-mark" aria-hidden>
            <svg viewBox="0 0 32 32" width="18" height="18">
              <path d="M7 21.5 L12.5 14.5 L17 18 L25 9.5" stroke="#fff" strokeWidth="2.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="25" cy="9.5" r="2.6" fill="#6f9bf0" />
            </svg>
          </span>
          {!collapsed ? (
            <span className="logo-text">
              <span className="logo-title">MyFlows</span>
              <span className="logo-subtitle">训练控制台</span>
            </span>
          ) : null}
        </Link>
        <Menu mode="inline" selectedKeys={[section]} items={items} onClick={(e) => navigate(e.key)} />
        {gpuInfo ? (
          <Tooltip title={`${gpuInfo.name}${gpu.data?.driver_version ? ` · 驱动 ${gpu.data.driver_version}` : ''}`} placement="right">
            <div className="sider-footer">
              <div className="sider-footer-title">
                <span className={`status-dot ${backend ? 'ok' : 'err'}`} />
                <span>{shortGpuName(gpuInfo.name)}</span>
              </div>
              <div className="sider-footer-desc">
                {gpuInfo.memory_total_mib ? `${(gpuInfo.memory_total_mib / 1024).toFixed(0)} GB 显存` : '显存未知'}
                {backend ? ` · ${backend.toUpperCase()}` : ''}
              </div>
            </div>
          </Tooltip>
        ) : null}
      </Sider>

      <Layout className="app-main">
        <Header className={`app-header${scrolled ? ' scrolled' : ''}`}>
          <nav className="app-breadcrumb" aria-label="breadcrumb">
            {isRunDetail ? (
              <>
                <Link to="/runs">运行记录</Link>
                <RightOutlined className="crumb-sep" />
                <span className="crumb-current">{run.data?.summary.label ?? runId}</span>
              </>
            ) : (
              <span className="crumb-current">{SECTIONS[section] ?? '控制台'}</span>
            )}
          </nav>

          <div className="header-actions">
            <Tooltip title="GPU 采样后端">
              <span className="status-chip">
                <span className={`status-dot ${health.isError ? 'err' : backend ? 'ok' : health.data ? 'err' : 'warn'}`} />
                {health.isError ? '后端未连接' : backend ? `GPU · ${backend.toUpperCase()}` : health.data ? 'GPU 不可用' : '检测中'}
              </span>
            </Tooltip>
            <span className="status-chip hide-sm">
              <span className={`status-dot ${active ? 'live' : ''}`} />
              {active ? `${active} 个任务进行中` : '队列空闲'}
            </span>
            {section !== '/new' ? (
              <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/new')}>
                <span className="hide-sm">新建训练</span>
              </Button>
            ) : null}
          </div>
        </Header>

        <Content className="app-content">
          <div className="page-container">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/runs" element={<Runs />} />
              <Route path="/run/*" element={<RunDetailPage />} />
              <Route path="/new" element={<NewJob />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}
