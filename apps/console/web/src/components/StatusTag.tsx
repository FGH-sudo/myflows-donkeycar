import { STATUS_META } from '../format'

interface Props {
  status?: string | null
  showDot?: boolean
}

export default function StatusTag({ status, showDot = true }: Props) {
  const key = (status ?? '').toLowerCase()
  const meta = STATUS_META[key] ?? { color: 'default', label: status ?? '—' }
  const pillClass = `status-pill status-pill-${key}`

  return (
    <span className={pillClass}>
      {showDot ? <span className="status-pill-dot" /> : null}
      <span>{meta.label}</span>
    </span>
  )
}
