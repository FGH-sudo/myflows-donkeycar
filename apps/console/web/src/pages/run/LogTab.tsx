import { useQuery } from '@tanstack/react-query'
import { Card, Empty, Segmented, Switch } from 'antd'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../api'

export default function LogTab({ runId, live }: { runId: string; live: boolean }) {
  const [tailKb, setTailKb] = useState(64)
  const [follow, setFollow] = useState(true)
  const log = useQuery({ queryKey: ['log', runId, tailKb], queryFn: () => api.log(runId, tailKb), refetchInterval: live ? 2000 : false })
  const ref = useRef<HTMLPreElement>(null)
  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [log.data, follow])

  return (
    <Card size="small" title={log.data?.path ? <code style={{ fontSize: 12 }}>{log.data.path}</code> : '日志'}
      extra={<span style={{ display: 'inline-flex', gap: 16, alignItems: 'center' }}>
        <span>末尾 <Segmented size="small" value={tailKb} onChange={(v) => setTailKb(Number(v))} options={[{ value: 64, label: '64 KB' }, { value: 512, label: '512 KB' }]} /></span>
        <span>自动滚动 <Switch size="small" checked={follow} onChange={setFollow} /></span>
      </span>}>
      {log.data?.text ? <pre ref={ref} className="log-view">{log.data.text}</pre> : <Empty description={log.isLoading ? '加载中…' : '没有日志文件'} />}
    </Card>
  )
}
