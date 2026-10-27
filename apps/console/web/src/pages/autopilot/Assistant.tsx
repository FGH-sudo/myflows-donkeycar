import { ExportOutlined, ReloadOutlined, RobotOutlined } from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, App, Button, Descriptions, Space, Spin, Tag } from 'antd'
import { useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { autopilot } from '../../api'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'

const LOOPBACK = ['127.0.0.1', 'localhost']

/** The dsh web page on the console's own loopback name. Its login cookie is SameSite=Strict, so a console
 * opened as localhost cannot embed 127.0.0.1 (or the reverse): the frame would answer "authentication
 * required". dsh listens on IPv4 loopback and accepts either name. */
function onConsoleHost(url: string): string {
  try {
    const target = new URL(url)
    if (LOOPBACK.includes(target.hostname) && LOOPBACK.includes(window.location.hostname)) {
      target.hostname = window.location.hostname
    }
    return target.toString()
  } catch {
    return url
  }
}

/** The AutoPilot-only dsh web UI, embedded: chat, permission mode, approvals and API key settings. */
export default function Assistant() {
  const [search] = useSearchParams()
  const campaign = search.get('campaign')
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const status = useQuery({
    queryKey: ['autopilot', 'dsh', campaign],
    queryFn: () => autopilot.dshStatus(campaign),
    refetchInterval: (q) => (q.state.data?.checking || q.state.data?.web?.state === 'starting' ? 1500 : 10000),
  })

  useEffect(() => {
    if (campaign) autopilot.dshContext(campaign).catch(() => undefined)
  }, [campaign])

  const restart = async () => {
    try {
      await autopilot.dshRestart()
      queryClient.invalidateQueries({ queryKey: ['autopilot', 'dsh'] })
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  if (status.isLoading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  const s = status.data
  const url = s?.web?.url ? onConsoleHost(s.web.url) : undefined
  const ready = s?.web?.state === 'running' && !!url

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>AutoPilot 助手</h2>
          <div className="page-subtitle">
            AutoPilot 专用 dsh：只开放 AutoPilot 工具；配置类写操作按权限模式审批，启动训练一律由人确认
            {campaign ? (
              <>
                {' · 当前 campaign '}
                <Tag>{campaign}</Tag>
              </>
            ) : null}
          </div>
        </div>
        <Space>
          {url ? (
            <Button icon={<ExportOutlined />} href={url} target="_blank">
              新标签页打开
            </Button>
          ) : null}
          <Button icon={<ReloadOutlined />} onClick={restart} disabled={!s?.available}>
            重新启动
          </Button>
        </Space>
      </div>
      {!s?.available ? (
        <Panel>
          <EmptyState
            icon={s?.checking ? <Spin /> : <RobotOutlined />}
            title={s?.checking ? '正在检查 AutoPilot 专用 dsh…' : 'LLM 层不可用，campaign 按规则策略照常运行'}
            description={s?.reason ?? '无法读取 dsh 状态'}
          />
        </Panel>
      ) : (
        <>
          {s.key_configured === false ? (
            <Alert
              style={{ marginBottom: 12 }}
              type="warning"
              showIcon
              message="还没有 DeepSeek API Key 或账号登录"
              description="在下方 dsh 网页里添加 API Key 或登录 DeepSeek 账号；凭据只保存在 runs/autopilot/dsh-home。未配置时后台顾问跳过。"
            />
          ) : null}
          {s.web?.error ? <Alert style={{ marginBottom: 12 }} type="error" showIcon message={s.web.error} /> : null}
          {ready ? (
            <iframe className="assistant-frame" src={url} title="AutoPilot dsh" />
          ) : (
            <Panel>
              <EmptyState
                icon={<RobotOutlined />}
                title={s.web?.state === 'starting' ? 'dsh 网页正在启动…' : 'dsh 网页未运行'}
                action={
                  s.web?.state === 'starting' ? <Spin /> : (
                    <Button type="primary" onClick={restart}>
                      启动
                    </Button>
                  )
                }
              />
            </Panel>
          )}
          <Panel style={{ marginTop: 16 }} padded>
            <Descriptions
              size="small"
              column={{ xs: 1, md: 2 }}
              items={[
                { key: 'clone', label: '克隆', children: s.clone ? `${s.clone.path} @ ${(s.clone.commit ?? '').slice(0, 10)}${s.clone.node ? ` · Node ${s.clone.node}` : ''}` : '—' },
                { key: 'built', label: '构建产物', children: s.clone?.built ? '已构建' : '缺失' },
                { key: 'comp', label: '组合检查', children: s.composition?.ok ? '通过' : (s.composition?.errors ?? []).join('；') || '未检查' },
                { key: 'web', label: '网页进程', children: `${s.web?.state ?? '—'}${s.web?.pid ? ` · pid ${s.web.pid}` : ''}` },
              ]}
            />
          </Panel>
        </>
      )}
    </div>
  )
}
