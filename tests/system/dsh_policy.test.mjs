// node --test: the AutoPilot guard's permission table (spec §8.4), run by tests/system/test_autopilot_dsh.py.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { CATEGORIES, HEADLESS_TOOLS, PREFIX, decide, modeOf, visible } from '../../apps/autopilot/dsh/bundle/policy.js'

const tool = raw => PREFIX + raw
const kinds = (raw, profile, mode) => decide(tool(raw), { profile, mode }).kind

test('permission column from the preset and approval override', () => {
  assert.equal(modeOf('workspace-write', 'ask'), 'workspace')
  assert.equal(modeOf('read-only', undefined), 'workspace')
  assert.equal(modeOf('custom', undefined), 'workspace')
  assert.equal(modeOf('danger-full-access', 'never'), 'full')
  assert.equal(modeOf('custom', 'never'), 'full')
  assert.equal(modeOf('auto', 'ask'), 'auto')
})

test('read, note and request tools are allowed in every Web mode', () => {
  for (const mode of ['workspace', 'full', 'auto']) {
    for (const raw of [...CATEGORIES.read, ...CATEGORIES.note, ...CATEGORIES.request]) {
      assert.equal(kinds(raw, 'web', mode), 'allow', `${raw} in ${mode}`)
    }
  }
})

test('config tools ask in workspace-write and are allowed otherwise', () => {
  for (const raw of CATEGORIES.config) {
    const asked = decide(tool(raw), { profile: 'web', mode: 'workspace' })
    assert.equal(asked.kind, 'ask')
    assert.match(asked.reason, /需要确认/)
    assert.equal(kinds(raw, 'web', 'full'), 'allow')
    assert.equal(kinds(raw, 'web', 'auto'), 'allow')
  }
})

test('start tools ask in workspace-write and are denied with a request_* hint otherwise', () => {
  for (const raw of CATEGORIES.start) {
    assert.equal(kinds(raw, 'web', 'workspace'), 'ask')
    for (const mode of ['full', 'auto']) {
      const denied = decide(tool(raw), { profile: 'web', mode })
      assert.equal(denied.kind, 'deny')
      assert.match(denied.reason, raw === 'start_campaign' ? /request_campaign_start/ : /request_trials/)
    }
  }
})

test('the headless profile gets only the read tools, request_trials and write_note', () => {
  assert.deepEqual([...HEADLESS_TOOLS].sort(), [...CATEGORIES.read, 'request_trials', 'write_note'].sort())
  for (const names of Object.values(CATEGORIES)) {
    for (const raw of names) {
      const expected = HEADLESS_TOOLS.includes(raw) ? 'allow' : 'deny'
      for (const mode of ['workspace', 'full', 'auto']) assert.equal(kinds(raw, 'headless', mode), expected, raw)
      assert.equal(visible(tool(raw), 'headless'), HEADLESS_TOOLS.includes(raw))
      assert.equal(visible(tool(raw), 'web'), true)
    }
  }
})

test('anything outside the AutoPilot tools is invisible and denied', () => {
  for (const name of ['bash', 'pwsh', 'read_file', 'mcp__other__x', 'mcp__autopilot__delete_campaign', 'subagent']) {
    assert.equal(visible(name, 'web'), false)
    assert.equal(decide(name, { profile: 'web', mode: 'full' }).kind, 'deny')
    assert.equal(decide(name, { profile: 'headless', mode: 'workspace' }).kind, 'deny')
  }
})
