// The permission table of the AutoPilot guard (spec §8.4), free of dsh imports so `node --test` can check it.
//
//    category            workspace-write   full access          auto                 headless profile
//    read/note/request   allow             allow                allow                allow (subset)
//    config              ask               allow                allow (Auto review)  not offered
//    start               ask               deny → use request_* deny → use request_* not offered
//
// Full access pins the `never` approval policy, which denies every ask, so the guard returns `allow` itself
// there instead of asking. Under Auto the review gate runs before this one and asks the user on its own when
// it judges a call risky.

export const PREFIX = 'mcp__autopilot__'

export const CATEGORIES = {
  read: ['list_campaigns', 'get_campaign', 'list_trials', 'get_trial_summary', 'get_findings', 'get_search_space',
    'list_runs'],
  config: ['create_campaign', 'update_campaign', 'pause_campaign', 'stop_trial'],
  start: ['start_campaign', 'submit_trials'],
  request: ['request_campaign_start', 'request_trials'],
  note: ['write_note'],
}

export const HEADLESS_TOOLS = [...CATEGORIES.read, 'request_trials', 'write_note']

/** Category of an AutoPilot tool from its raw (unprefixed) name, or undefined for an unknown tool. */
export function categoryOf(raw) {
  for (const [category, names] of Object.entries(CATEGORIES)) {
    if (names.includes(raw)) return category
  }
  return undefined
}

/** Permission column from the preset name and the effective approval policy: workspace (ask), full (never) or auto. */
export function modeOf(preset, approvalPolicy) {
  if (preset === 'auto') return 'auto'
  if (preset === 'danger-full-access' || approvalPolicy === 'never') return 'full'
  return 'workspace'
}

/** Whether an agent of this profile may see the tool at all (allowlist plus the per-profile subset). */
export function visible(toolName, profile = 'web') {
  if (!toolName.startsWith(PREFIX)) return false
  const raw = toolName.slice(PREFIX.length)
  if (categoryOf(raw) === undefined) return false
  return profile !== 'headless' || HEADLESS_TOOLS.includes(raw)
}

/**
 * Decision for one call.
 * @returns {{kind: 'allow'} | {kind: 'ask', reason: string} | {kind: 'deny', reason: string}}
 */
export function decide(toolName, { profile = 'web', mode = 'workspace' } = {}) {
  if (!toolName.startsWith(PREFIX)) {
    return { kind: 'deny', reason: `myflows-autopilot only allows ${PREFIX}* tools; ${toolName} is denied` }
  }
  const raw = toolName.slice(PREFIX.length)
  const category = categoryOf(raw)
  if (category === undefined) return { kind: 'deny', reason: `未知的 AutoPilot 工具：${raw}` }
  if (profile === 'headless') {
    return HEADLESS_TOOLS.includes(raw)
      ? { kind: 'allow' }
      : { kind: 'deny', reason: `后台一次性运行只能使用只读工具、request_trials 与 write_note；${raw} 不可用` }
  }
  if (category === 'read' || category === 'note' || category === 'request') return { kind: 'allow' }
  if (category === 'config') {
    return mode === 'workspace' ? { kind: 'ask', reason: `AutoPilot 配置操作 ${raw} 需要确认` } : { kind: 'allow' }
  }
  if (mode === 'workspace') return { kind: 'ask', reason: `${raw} 会启动训练，需要确认` }
  return {
    kind: 'deny',
    reason: `${raw} 会直接启动训练，当前权限模式不提供人工审批；请改用 ` +
      `${raw === 'start_campaign' ? 'request_campaign_start' : 'request_trials'}，在控制台的“待确认”列表里由人确认`,
  }
}
