import { createContext, useContext, useEffect, useState } from 'react'

/**
 * Large monitors get a proportionally larger UI: type, controls, spacing and
 * chart heights all grow together so panels don't look tiny on a wide canvas.
 * Keep the breakpoints in sync with the @media blocks at the end of index.css.
 */
const LEVELS: { query: string; factor: number }[] = [
  { query: '(min-width: 2200px)', factor: 1.25 },
  { query: '(min-width: 1600px)', factor: 1.125 },
]

function currentFactor() {
  return LEVELS.find((l) => window.matchMedia(l.query).matches)?.factor ?? 1
}

export function useUiScaleFactor() {
  const [factor, setFactor] = useState(currentFactor)
  useEffect(() => {
    const lists = LEVELS.map((l) => window.matchMedia(l.query))
    const update = () => setFactor(currentFactor())
    lists.forEach((m) => m.addEventListener('change', update))
    return () => lists.forEach((m) => m.removeEventListener('change', update))
  }, [])
  return factor
}

export const UiScaleContext = createContext(1)

export const useUiScale = () => useContext(UiScaleContext)
