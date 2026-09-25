import { Tag } from 'antd'
import { STATUS_META } from '../format'

export default function StatusTag({ status }: { status?: string | null }) {
  const meta = STATUS_META[status ?? ''] ?? { color: 'default', label: status ?? '—' }
  return <Tag color={meta.color}>{meta.label}</Tag>
}
