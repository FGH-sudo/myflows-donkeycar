import type { CSSProperties, ReactNode } from 'react'

interface PanelProps {
  children?: ReactNode
  /** Optional in-panel heading; prefer a SectionHeader above the panel for page sections. */
  title?: ReactNode
  subtitle?: ReactNode
  extra?: ReactNode
  padded?: boolean
  className?: string
  style?: CSSProperties
  bodyStyle?: CSSProperties
}

/** Flat white surface with a hairline border. The base container of the console. */
export function Panel({ children, title, subtitle, extra, padded = false, className, style, bodyStyle }: PanelProps) {
  const head =
    title || extra ? (
      <div className="panel-head">
        <div style={{ minWidth: 0 }}>
          {title ? <div className="panel-title">{title}</div> : null}
          {subtitle ? <div className="panel-subtitle">{subtitle}</div> : null}
        </div>
        {extra}
      </div>
    ) : null
  const pad = padded ? { padding: head ? '12px 20px 20px' : 20 } : undefined
  return (
    <div className={`panel${className ? ` ${className}` : ''}`} style={style}>
      {head}
      <div style={{ ...pad, ...bodyStyle }}>{children}</div>
    </div>
  )
}

interface SectionHeaderProps {
  title: ReactNode
  description?: ReactNode
  extra?: ReactNode
  style?: CSSProperties
}

/** Heading that sits above a panel, so panels themselves need no header bar. */
export function SectionHeader({ title, description, extra, style }: SectionHeaderProps) {
  return (
    <div className="section-header" style={style}>
      <div style={{ minWidth: 0 }}>
        <h3>{title}</h3>
        {description ? <div className="section-header-desc">{description}</div> : null}
      </div>
      {extra}
    </div>
  )
}
