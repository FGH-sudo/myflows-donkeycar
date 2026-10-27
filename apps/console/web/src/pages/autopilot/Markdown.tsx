import type { ReactNode } from 'react'

/** Inline `code` and **bold**; everything else is plain text (no HTML is ever injected). */
function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = []
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g
  let last = 0
  let i = 0
  for (const match of text.matchAll(pattern)) {
    const at = match.index ?? 0
    if (at > last) out.push(text.slice(last, at))
    const token = match[0]
    if (token.startsWith('`')) out.push(<code key={`${key}-${i++}`}>{token.slice(1, -1)}</code>)
    else out.push(<strong key={`${key}-${i++}`}>{token.slice(2, -2)}</strong>)
    last = at + token.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

const cells = (line: string) =>
  line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((c) => c.trim())

/**
 * Renders the subset of Markdown the AutoPilot reporter writes: headings, paragraphs, lists (one nesting
 * level), tables, block quotes and images. ``resolveImage`` maps relative image paths to URLs.
 */
export default function Markdown({ text, resolveImage }: { text: string; resolveImage?: (src: string) => string }) {
  const lines = text.split(/\r?\n/)
  const blocks: ReactNode[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    const key = `b${i}`
    if (!line.trim()) {
      i += 1
      continue
    }
    const heading = /^(#{1,3})\s+(.*)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const content = inline(heading[2], key)
      blocks.push(level === 1 ? <h1 key={key}>{content}</h1> : level === 2 ? <h2 key={key}>{content}</h2> : <h3 key={key}>{content}</h3>)
      i += 1
      continue
    }
    if (line.trim().startsWith('|')) {
      const rows: string[] = []
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++])
      const [head, , ...body] = rows
      blocks.push(
        <table key={key}>
          <thead>
            <tr>
              {cells(head).map((c, j) => (
                <th key={j}>{inline(c, `${key}h${j}`)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row, r) => (
              <tr key={r}>
                {cells(row).map((c, j) => (
                  <td key={j}>{inline(c, `${key}r${r}c${j}`)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>,
      )
      continue
    }
    if (/^\s*- /.test(line)) {
      const items: { text: string; children: string[] }[] = []
      while (i < lines.length && /^\s*- /.test(lines[i])) {
        const nested = /^\s{2,}- /.test(lines[i])
        const content = lines[i].replace(/^\s*- /, '')
        if (nested && items.length) items[items.length - 1].children.push(content)
        else items.push({ text: content, children: [] })
        i += 1
      }
      blocks.push(
        <ul key={key}>
          {items.map((item, j) => (
            <li key={j}>
              {inline(item.text, `${key}l${j}`)}
              {item.children.length ? (
                <ul>
                  {item.children.map((child, k) => (
                    <li key={k} className="chain-evidence">
                      {inline(child, `${key}l${j}c${k}`)}
                    </li>
                  ))}
                </ul>
              ) : null}
            </li>
          ))}
        </ul>,
      )
      continue
    }
    if (line.startsWith('>')) {
      const quoted: string[] = []
      while (i < lines.length && lines[i].startsWith('>')) quoted.push(lines[i++].replace(/^>\s?/, ''))
      blocks.push(
        <blockquote key={key}>
          {quoted.map((q, j) => (q ? <div key={j}>{inline(q, `${key}q${j}`)}</div> : <br key={j} />))}
        </blockquote>,
      )
      continue
    }
    const image = /^!\[([^\]]*)\]\(([^)]+)\)$/.exec(line.trim())
    if (image) {
      blocks.push(<img key={key} alt={image[1]} src={resolveImage ? resolveImage(image[2]) : image[2]} />)
      i += 1
      continue
    }
    blocks.push(<p key={key}>{inline(line, key)}</p>)
    i += 1
  }
  return <div className="markdown-body">{blocks}</div>
}
