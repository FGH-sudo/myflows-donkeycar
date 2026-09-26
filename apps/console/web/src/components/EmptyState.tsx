import { InboxOutlined } from '@ant-design/icons'
import type { ReactNode } from 'react'

interface Props {
  title?: ReactNode
  description?: ReactNode
  icon?: ReactNode
  action?: ReactNode
  /** Single-line variant for secondary areas. */
  compact?: boolean
}

export default function EmptyState({ title, description, icon = <InboxOutlined />, action, compact = false }: Props) {
  return (
    <div className={`empty-state${compact ? ' compact' : ''}`}>
      <span className="empty-state-icon">{icon}</span>
      <div>
        {title ? <div className="empty-state-title">{title}</div> : null}
        {description ? <div className="empty-state-desc">{description}</div> : null}
      </div>
      {action}
    </div>
  )
}
