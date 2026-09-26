import { CopyOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { App, Button, Segmented, Switch } from 'antd'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../api'

export default function LogTab({ runId, live }: { runId: string; live: boolean }) {
  const { message } = App.useApp()
  const [tailKb, setTailKb] = useState(64)
  const [follow, setFollow] = useState(true)
  const log = useQuery({
    queryKey: ['log', runId, tailKb],
    queryFn: () => api.log(runId, tailKb),
    refetchInterval: live ? 2000 : false,
  })
  const ref = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (follow && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }
  }, [log.data, follow])

  const copyLog = () => {
    if (log.data?.text) {
      navigator.clipboard.writeText(log.data.text)
      message.success('日志已复制')
    }
  }

  return (
    <div>
      <div className="toolbar tab-toolbar" style={{ justifyContent: 'space-between' }}>
        <span className="toolbar-item" style={{ minWidth: 0 }}>
          {log.data?.path ? (
            <code className="hide-sm" style={{ color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {log.data.path}
            </code>
          ) : (
            <span className="muted">运行日志</span>
          )}
        </span>
        <span className="toolbar" style={{ gap: 16 }}>
          <span className="toolbar-item">
            尾部
            <Segmented
              size="small"
              value={tailKb}
              onChange={(v) => setTailKb(Number(v))}
              options={[
                { value: 64, label: '64 KB' },
                { value: 256, label: '256 KB' },
                { value: 512, label: '512 KB' },
              ]}
            />
          </span>
          <span className="toolbar-item">
            自动滚动
            <Switch size="small" checked={follow} onChange={setFollow} />
          </span>
          <Button size="small" icon={<CopyOutlined />} onClick={copyLog} disabled={!log.data?.text}>
            复制
          </Button>
        </span>
      </div>
      <div className="terminal-window">
        <div className="terminal-header">
          <div className="terminal-dots">
            <span className="terminal-dot dot-red" />
            <span className="terminal-dot dot-yellow" />
            <span className="terminal-dot dot-green" />
          </div>
          <span className="terminal-title">stdout · stderr — 末尾 {tailKb} KB{live ? ' · 实时' : ''}</span>
        </div>
        {log.data?.text ? (
          <pre ref={ref} className="log-view">
            {log.data.text}
          </pre>
        ) : (
          <div style={{ padding: '56px 0', textAlign: 'center', color: '#a1a1aa', fontSize: 13 }}>
            {log.isLoading ? '正在获取日志…' : '没有日志文件或日志为空'}
          </div>
        )}
      </div>
    </div>
  )
}
