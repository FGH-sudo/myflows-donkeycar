import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useReducer } from 'react'
import { RESOURCE_KEYS, api, streamUrl } from '../../api'
import type {
  Columns, Cursor, EpochRow, EpochsResponse, RankStatus, ResourceSeries, ResourcesResponse, RunSummary, StepsResponse,
} from '../../api'
import { isActive } from '../../format'
import { useEventSource } from '../../hooks/useEventSource'

export interface RunState {
  loaded: boolean
  steps: Record<string, Columns>
  epochs: Record<string, EpochRow[]>
  resources: ResourceSeries | null
  cursor: Cursor
  summary: RunSummary | null
  rankStatus: RankStatus[]
  ranks: number[]
}

type Action =
  | { type: 'reset' }
  | { type: 'init'; steps: StepsResponse; epochs: EpochsResponse; resources: ResourcesResponse }
  | { type: 'steps'; data: StepsResponse }
  | { type: 'epochs'; data: EpochsResponse }
  | { type: 'resources'; data: ResourcesResponse }
  | { type: 'status'; summary: RunSummary; rankStatus: RankStatus[]; ranks: number[] }

const EMPTY: RunState = { loaded: false, steps: {}, epochs: {}, resources: null, cursor: {}, summary: null, rankStatus: [], ranks: [] }

function concatColumns(a: Columns | undefined, b: Columns): Columns {
  if (!a) return b
  const out: Columns = { ...a }
  for (const [k, v] of Object.entries(b)) out[k] = [...(a[k] ?? new Array(a.x?.length ?? 0).fill(null)), ...v]
  return out
}

function concatResources(a: ResourceSeries | null, b: ResourceSeries): ResourceSeries {
  if (!a) return b
  const out: ResourceSeries = { ...a, t: [...a.t, ...b.t] }
  for (const k of RESOURCE_KEYS) out[k] = [...(a[k] ?? new Array(a.t.length).fill(null)), ...(b[k] ?? new Array(b.t.length).fill(null))]
  const rss: Record<string, (number | null)[]> = { ...a.rank_rss_mib }
  for (const [rank, values] of Object.entries(b.rank_rss_mib ?? {})) {
    const prev = rss[rank] ?? new Array(a.t.length).fill(null)
    rss[rank] = [...prev, ...values]
  }
  out.rank_rss_mib = rss
  return out
}

function reducer(state: RunState, action: Action): RunState {
  switch (action.type) {
    case 'reset':
      return EMPTY
    case 'init':
      return {
        ...state,
        loaded: true,
        steps: action.steps.ranks,
        epochs: action.epochs.ranks,
        resources: action.resources.series,
        cursor: { ...action.steps.cursor, ...action.epochs.cursor, ...action.resources.cursor },
      }
    case 'steps': {
      const steps = { ...state.steps }
      for (const [rank, cols] of Object.entries(action.data.ranks)) steps[rank] = concatColumns(steps[rank], cols)
      return { ...state, steps, cursor: { ...state.cursor, ...action.data.cursor } }
    }
    case 'epochs': {
      const epochs = { ...state.epochs }
      for (const [rank, rows] of Object.entries(action.data.ranks)) epochs[rank] = [...(epochs[rank] ?? []), ...rows]
      return { ...state, epochs, cursor: { ...state.cursor, ...action.data.cursor } }
    }
    case 'resources':
      return { ...state, resources: concatResources(state.resources, action.data.series), cursor: { ...state.cursor, ...action.data.cursor } }
    case 'status':
      return { ...state, summary: action.summary, rankStatus: action.rankStatus, ranks: action.ranks }
  }
}

export function useRunData(runId: string) {
  const queryClient = useQueryClient()
  const detail = useQuery({ queryKey: ['run', runId], queryFn: () => api.run(runId) })
  const [state, dispatch] = useReducer(reducer, EMPTY)

  useEffect(() => {
    let cancelled = false
    dispatch({ type: 'reset' })
    Promise.all([api.steps(runId, { max_points: 4000 }), api.epochs(runId), api.resources(runId, 4000)])
      .then(([steps, epochs, resources]) => {
        if (!cancelled) dispatch({ type: 'init', steps, epochs, resources })
      })
      .catch(() => {
        if (!cancelled) dispatch({ type: 'init', steps: { origin: null, ranks: {}, cursor: {} }, epochs: { ranks: {}, cursor: {} },
          resources: { origin: null, series: { t: [], rank_rss_mib: {} }, sampling_errors: 0, cursor: {} } })
      })
    return () => {
      cancelled = true
    }
  }, [runId])

  const summary = state.summary ?? detail.data?.summary ?? null
  const live = state.loaded && isActive(summary?.status)
  const [streamCursor, setStreamCursor] = useReducer((_: Cursor | null, c: Cursor | null) => c, null)
  useEffect(() => {
    if (live && !streamCursor) setStreamCursor(state.cursor)
    if (!live && streamCursor) setStreamCursor(null)
  }, [live, streamCursor, state.cursor])

  const { connected } = useEventSource(streamCursor ? streamUrl(runId, streamCursor) : null, {
    steps: (data) => dispatch({ type: 'steps', data: data as StepsResponse }),
    epochs: (data) => dispatch({ type: 'epochs', data: data as EpochsResponse }),
    resources: (data) => dispatch({ type: 'resources', data: data as ResourcesResponse }),
    status: (data) => {
      const d = data as { summary: RunSummary; rank_status: RankStatus[]; ranks: number[] }
      const previous = summary?.status
      dispatch({ type: 'status', summary: d.summary, rankStatus: d.rank_status, ranks: d.ranks })
      if (previous && previous !== d.summary.status) queryClient.invalidateQueries({ queryKey: ['run', runId] })
    },
    end: () => {
      queryClient.invalidateQueries({ queryKey: ['run', runId] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const rankStatus = state.rankStatus.length ? state.rankStatus : detail.data?.rank_status ?? []
  const ranks = state.ranks.length ? state.ranks : detail.data?.ranks ?? []
  return { detail, state, summary, rankStatus, ranks, live, connected }
}
