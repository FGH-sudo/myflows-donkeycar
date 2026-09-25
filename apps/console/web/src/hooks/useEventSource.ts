import { useEffect, useRef, useState } from 'react'

export type SseHandlers = Record<string, (data: unknown) => void>

/**
 * Subscribe to a Server-Sent Events endpoint while ``url`` is non-null.
 * Handlers are read through a ref so callers may pass inline closures.
 */
export function useEventSource(url: string | null, handlers: SseHandlers): { connected: boolean } {
  const handlersRef = useRef(handlers)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    handlersRef.current = handlers
  })

  useEffect(() => {
    if (!url) {
      setConnected(false)
      return
    }
    const source = new EventSource(url)
    const names = Object.keys(handlersRef.current)
    const listeners = names.map((name) => {
      const listener = (event: MessageEvent) => {
        try {
          handlersRef.current[name]?.(JSON.parse(event.data))
        } catch {
          /* malformed event: ignore */
        }
        if (name === 'end') source.close()
      }
      source.addEventListener(name, listener as EventListener)
      return [name, listener] as const
    })
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(source.readyState === EventSource.OPEN)
    return () => {
      listeners.forEach(([name, listener]) => source.removeEventListener(name, listener as EventListener))
      source.close()
      setConnected(false)
    }
  }, [url])

  return { connected }
}
