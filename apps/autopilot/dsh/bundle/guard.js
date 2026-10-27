// myflows-autopilot-guard: the AutoPilot dsh only ever runs AutoPilot tools (spec §8.2, §8.4).
//
// 1. Allowlist: every agent sees only `mcp__autopilot__*` tools; calls to anything else are denied by a
//    monotonic guard, so no later listener can turn them back into permission.
// 2. Per-profile subset: the one-shot (headless) profile only gets read-only tools, request_trials and
//    write_note.
// 3. Permission decision in `tools/pre-execute` from the tool category and the session's permission
//    preset; the table lives in policy.js.

import { createScope } from '@deepseek-ai/dsh-scope'
import { PREFIX, decide, modeOf, visible } from './policy.js'

export const name = 'myflows-autopilot-guard'
export const inject = ['tools', 'permissionPresets', 'approval']

export function apply(ctx, config = {}) {
  const profile = config.profile === 'headless' ? 'headless' : 'web'
  const permissionPresets = ctx.permissionPresets
  const approval = ctx.approval

  ctx.tools.guard(execution => visible(execution.name, profile)
    ? undefined
    : `myflows-autopilot only allows ${PREFIX}* tools${profile === 'headless' ? ' (headless subset)' : ''}; ` +
      `${execution.name} is denied`)

  // tools.restrict() works only inside an agent scope and only names registered global tools, so the
  // deny list is computed when each agent is created.
  ctx.on('agent/created', ({ agent }) => {
    const denied = ctx.tools.schemas(agent).map(tool => tool.name).filter(n => !visible(n, profile))
    if (denied.length > 0) createScope(ctx, agent).ctx.tools.restrict({ deny: denied })
  })

  ctx.on('tools/pre-execute', async (execution, next) => {
    if (!execution.name.startsWith(PREFIX)) return next()
    const session = execution.agent?.session
    let mode = 'workspace'
    if (session !== undefined) {
      let preset
      try {
        preset = permissionPresets.current(session)
      } catch {
        preset = undefined
      }
      // The effective approval policy, as the approval service computes it: the session's own choice, else
      // the configured default (DSH_PERMISSION_MODE=danger-full-access configures `never`).
      mode = modeOf(preset, approval.overrideOf(session) ?? approval.config?.policy ?? 'ask')
    }
    const decision = decide(execution.name, { profile, mode })
    if (decision.kind === 'ask') {
      return { kind: 'ask', reason: decision.reason, displayReason: { en: decision.reason, zh: decision.reason } }
    }
    return decision
  })
}
