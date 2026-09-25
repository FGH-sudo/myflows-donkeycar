import {
  DashboardOutlined,
  LineChartOutlined,
  PlusCircleOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Badge, Layout, Menu, Tag, Tooltip } from 'antd'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { api } from './api'
import Compare from './pages/Compare'
import NewJob from './pages/NewJob'
import Overview from './pages/Overview'
import RunDetailPage from './pages/RunDetail'
import Runs from './pages/Runs'

const { Sider, Content, Header } = Layout

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 3000 })
  const health = useQuery({ queryKey: ['health'], queryFn: () => fetch('/api/health').then((r) => r.json()), refetchInterval: 10000 })
  const active = (jobs.data ?? []).filter((j) => ['pending', 'running', 'stopping'].includes(j.status)).length

  const selected = location.pathname.startsWith('/run') ? '/runs' : '/' + (location.pathname.split('/')[1] ?? '')
  const items = [
    { key: '/', icon: <DashboardOutlined />, label: '总览' },
    { key: '/runs', icon: <UnorderedListOutlined />, label: <span>运行记录 {active ? <Badge count={active} size="small" style={{ marginLeft: 6 }} /> : null}</span> },
    { key: '/new', icon: <PlusCircleOutlined />, label: '新建训练' },
    { key: '/compare', icon: <LineChartOutlined />, label: '运行对比' },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={208} breakpoint="lg" collapsedWidth={64}>
        <div className="console-logo">
          <img src="/favicon.svg" alt="" />
          <span>MyFlows 控制台</span>
        </div>
        <Menu theme="dark" mode="inline" selectedKeys={[selected === '/' ? '/' : selected]} items={items}
          style={{ background: 'transparent' }} onClick={(e) => navigate(e.key)} />
      </Sider>
      <Layout>
        <Header style={{ background: '#fff', padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 12, borderBottom: '1px solid #eef0f3' }}>
          <Tooltip title="GPU 采样后端">
            {health.data ? (
              <Tag color={health.data.gpu_backend ? 'blue' : 'red'}>GPU · {health.data.gpu_backend ?? '不可用'}</Tag>
            ) : (
              <Tag color={health.isError ? 'red' : 'default'}>{health.isError ? '后端未连接' : 'GPU · …'}</Tag>
            )}
          </Tooltip>
          <Tag color={active ? 'processing' : 'default'}>{active ? `${active} 个任务进行中` : '无运行任务'}</Tag>
        </Header>
        <Content style={{ padding: 24 }}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/run/*" element={<RunDetailPage />} />
            <Route path="/new" element={<NewJob />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  )
}
