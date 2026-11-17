// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Switchyard desktop half. Hermes Desktop copies this file into
 * `$HERMES_HOME/desktop-plugins/nvidia-switchyard/` when the Switchyard plugin is
 * installed, and the runtime loader rewrites the bare `@hermes/plugin-sdk` /
 * `react` imports.
 *
 * Adds a Switchyard page (live routing monitor, pin toggle, tier setup,
 * presets), a sidebar entry, palette commands, a card in empty chats until
 * a Switchyard preset is Hermes's model, and a line in the chat where a
 * turn moved to the cloud model or back. Every request goes
 * to this plugin's own API (`/api/plugins/nvidia-switchyard/*`). The monitor mirrors the web
 * dashboard's tab (`dashboard/dist/index.js`); the two SDKs cannot share code.
 */

import {
  Button,
  CHAT_EMPTY_AREA,
  cn,
  host,
  Input,
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  Textarea,
  ToggleRow,
  TRANSCRIPT_DIRECTIVE_AREA
} from '@hermes/plugin-sdk'
// Members an older Desktop may not export; a missing named import would stop the plugin loading.
import * as sdk from '@hermes/plugin-sdk'
import { createElement as h, useEffect, useRef, useState } from 'react'

const PATH = '/switchyard'
const PRODUCT_NAME = 'NVIDIA Switchyard'
const JOB_POLL_MS = 1000
const MONITOR_POLL_MS = 1000
const PROGRESS_POLL_MS = 1500
// The Python half writes route notes into replies only while this half reports in (sidecar.RENDERER_FRESH_S).
const RENDERER_PING_MS = 60000
const RECOMMENDED = 'vgr'

// Defaults of the tiers in plugins/switchyard/settings.py.
const DEFAULT_CLOUD_URL = 'https://api.anthropic.com'
const DEFAULT_LOCAL_URL = 'http://127.0.0.1:8080/v1'
// settings.AUTO_MODEL and settings.DEFAULT_MODEL, which both route with the default preset.
const AUTO_MODELS = ['switchyard/NVIDIA-Switchyard', 'switchyard/default']
// Hermes's own settings pages, where models built into Hermes and providers are set up.
const LOCAL_SETTINGS_PATH = '/settings?tab=providers&pview=local'
const PROVIDER_SETTINGS_PATH = '/settings?tab=providers'
// The Local Only and Cloud Only modes, by the pin value the plugin saves for each.
const MODES = { '': 'Auto', local: 'Local Only', cloud: 'Cloud Only' }

let rest = null
let os = null

const api = (path, opts) => (rest ? rest(path, opts) : Promise.reject(new Error('Switchyard plugin is not loaded')))
const errorText = error => (error && error.message) || String(error)
const sleep = ms => new Promise(resolve => window.setTimeout(resolve, ms))

// The sidecar's state; with pollMs it refreshes while the window is visible.
function useSwitchyardState(pollMs = 0) {
  const [state, setState] = useState(null)
  const [version, setVersion] = useState(0)

  useEffect(() => {
    let live = true
    let timer = null
    let first = true

    const tick = () => {
      const next = () => {
        if (live && pollMs) {
          timer = window.setTimeout(tick, pollMs)
        }
      }

      if (!first && document.hidden) {
        next()

        return
      }

      first = false
    api('/state')
        .then(value => live && setState(value))
        .catch(error => live && setState({ configured: false, error: errorText(error), algorithms: [] }))
        .finally(next)
    }

    tick()

    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [version, pollMs])

  return [state, () => setVersion(v => v + 1)]
}

async function setPin(pin) {
  const result = await api('/pin', { method: 'POST', body: { pin } })
  const label = `${PRODUCT_NAME}: ${MODES[pin]}`
  host.notify({ kind: 'success', message: result.live ? label : `${label} (applies when the sidecar starts)` })

  return result
}

async function openInBrowser() {
  host.notify({ kind: 'info', message: 'Opening the Switchyard monitor in your browser…' })

  try {
    const result = await api('/dashboard', { method: 'POST', timeoutMs: 90000 })

    if (!result.ok || !result.url) {
      throw new Error(result.error || 'The web dashboard did not start')
    }

    if (!(os && (await os.openExternal(result.url)))) {
      throw new Error(`Open ${result.url} in your browser`)
    }
  } catch (err) {
    host.notify({ kind: 'error', message: errorText(err) })
  }
}

function useLive() {
  const live = useRef(true)

  useEffect(() => {
    live.current = true

    return () => {
      live.current = false
    }
  }, [])

  return live
}

function setupStatus(job) {
  const step = job.step ? `Step ${job.step} of ${job.steps}: ` : ''

  return `${step}${job.detail || 'Configuring Switchyard'}… (${job.elapsed_s || 0} s)`
}

// The first setup enables the plugin, which can take minutes while Hermes
// prepares its environment, so it runs as a job instead of one long request.
async function runSetup(body, onJob, isLive) {
  const { job_id } = await api('/setup/jobs', { method: 'POST', body })

  for (;;) {
    await sleep(JOB_POLL_MS)

    if (!isLive()) {
      throw new Error('cancelled')
    }

    const job = await api(`/setup/jobs/${encodeURIComponent(job_id)}`)

    if (job.status === 'done') {
      return job.result
    }

    if (job.status === 'error') {
      throw new Error(job.error || `${PRODUCT_NAME} setup failed`)
    }

    onJob(job)
  }
}

function Field({ label, ...props }) {
  return h(Input, { 'aria-label': label, autoComplete: 'off', className: 'font-mono', placeholder: label, ...props })
}

function Choice({ disabled, hint, label, onSelect, selected, tag }) {
  return h(
    'button',
    {
      'aria-checked': selected,
      className: cn(
        'rounded-2xl border bg-background/60 p-3 text-left transition',
        disabled ? 'cursor-not-allowed opacity-50' : 'hover:bg-accent/50',
        selected ? 'border-primary ring-2 ring-primary/20' : 'border-transparent'
      ),
      disabled,
      onClick: onSelect,
      role: 'radio',
      type: 'button'
    },
    h(
      'span',
      { className: 'flex items-center gap-2 text-sm font-medium' },
      label,
      tag ? h('span', { className: 'rounded bg-primary/10 px-1.5 py-0.5 text-[0.65rem] text-primary' }, tag) : null
    ),
    h('p', { className: 'mt-1 text-xs text-muted-foreground' }, hint)
  )
}

function Section({ title, hint, children }) {
  return h(
    'section',
    { className: 'grid gap-3' },
    h('h2', { className: 'text-sm font-semibold' }, title),
    hint ? h('p', { className: '-mt-2 text-xs text-muted-foreground' }, hint) : null,
    children
  )
}

const isAvailable = algorithm => !algorithm || algorithm.available !== false

function AlgorithmPicker({ algorithms, onSelect, selected }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const offered = algorithms.filter(isAvailable)
  const guided = offered.filter(a => a.guided)
  const advanced = offered.filter(a => !a.guided)
  const shown = showAdvanced || advanced.some(a => a.name === selected) ? offered : guided

  return h(
    'div',
    { className: 'grid gap-2' },
    h(
      'div',
      { className: 'grid gap-2 sm:grid-cols-2', role: 'radiogroup' },
      shown.map(a =>
        h(Choice, {
          hint: a.summary,
          key: a.name,
          label: a.label,
          onSelect: () => onSelect(a.name),
          selected: selected === a.name,
          tag: a.recommended ? 'Recommended' : a.guided ? null : 'Advanced'
        })
      )
    ),
    advanced.length
      ? h(
          Button,
          {
            className: 'self-start',
            onClick: () => setShowAdvanced(v => !v),
            size: 'xs',
            type: 'button',
            variant: 'text'
          },
          showAdvanced ? 'Hide advanced algorithms' : `Show ${advanced.length} advanced algorithms`
        )
      : null
  )
}

// Each tier by where it usually runs, with the role Switchyard's algorithms and monitor call it by.
const TIERS = {
  local: ['Local model', 'efficient', 'The efficient model handles most requests. We recommend a local one.'],
  cloud: ['Cloud model', 'capable', 'The capable model takes what the efficient model can\u2019t. We recommend a frontier model.']
}
// A tier whose provider last refused for lack of credit; its turns go to the other tier until it answers again.
function CreditAlerts({ state }) {
  const alerts = (state.state && state.state.credit_alerts) || {}
  return ['local', 'cloud']
    .filter((tier) => alerts[tier])
    .map((tier) => {
      const role = state.config[tier].role
      const other = state.config[tier === 'local' ? 'cloud' : 'local'].role
      return h(
        'p',
        { className: 'rounded-md border border-destructive/60 px-3 py-2 text-xs text-destructive', key: tier },
        `${role[0].toUpperCase()}${role.slice(1)} model out of credits: ${state.config[tier].model} refused its ` +
          `last call, so its turns fall back to the ${other} model. Add credits or set another ${role} model ` +
          `with hermes switchyard setup. Its provider said: ${alerts[tier].message}`
      )
    })
}

// managed: a model built into Hermes (its llama.cpp runtime); hermes: another model Hermes has
// configured (a provider tier); url: a fixed endpoint.
const SOURCES = {
  managed: ['Local Model', 'Built into Hermes'],
  hermes: ['Existing Provider', 'Use a model configured in Hermes'],
  url: ['Custom URL', 'OpenAI- or Anthropic-compatible']
}
// The cloud tier offers no Local Model; a model built into Hermes stays pickable there under Existing Provider.
const TIER_SOURCES = { local: ['managed', 'hermes', 'url'], cloud: ['hermes', 'url'] }

const NO_HERMES_MODELS = { current: null, defaults: {}, providers: [] }

const hermesOption = (hermes, provider) => hermes.providers.find(p => p.provider === provider)
const builtInOptions = hermes => hermes.providers.filter(p => p.source === 'managed')
// The local tier's Existing Provider list leaves out the models it offers as Local Model.
const providerOptions = (name, hermes) => hermes.providers.filter(p => name === 'cloud' || p.source !== 'managed')

// A tier's starting model under `source`: its Hermes default if that source lists it, else the
// first signed-in option there, a local one first for the local tier.
function firstChoice(name, source, hermes) {
  const options = source === 'managed' ? builtInOptions(hermes) : providerOptions(name, hermes)
  const preferred = hermes.defaults[name]

  if (preferred && options.some(o => o.provider === preferred.provider)) {
    return { builtIn: preferred.source === 'managed', model: preferred.model, provider: preferred.provider }
  }

  const signedIn = options.filter(o => o.authenticated)
  const option = signedIn.find(o => o.local === (name === 'local')) || signedIn[0]

  return option
    ? { builtIn: option.source === 'managed', model: option.default_model, provider: option.provider }
    : { builtIn: false, model: '', provider: '' }
}

function sourceDraft(name, source, draft, hermes) {
  const kept = { api_key: draft.api_key, base_url: draft.base_url, source }

  return source === 'url'
    ? { ...kept, builtIn: false, model: '', provider: '' }
    : { ...kept, ...firstChoice(name, source, hermes) }
}

// The form's starting values for one tier: its saved settings, else the Hermes model it defaults to.
function tierDraft(name, tier, hermes) {
  const fallback = name === 'local' ? DEFAULT_LOCAL_URL : DEFAULT_CLOUD_URL
  const blank = { api_key: '', base_url: fallback, builtIn: false, model: '', provider: '' }

  if (!tier) {
    const choice = hermes.defaults[name]
    const source = name === 'local' && (!choice || choice.source === 'managed') ? 'managed' : 'hermes'

    return sourceDraft(name, source, blank, hermes)
  }

  const saved = { ...blank, base_url: tier.base_url || fallback, model: tier.model || '' }

  if (tier.source === 'managed') {
    const option = builtInOptions(hermes).find(p => p.models.includes(tier.model))

    return {
      ...saved,
      builtIn: true,
      provider: option ? option.provider : '',
      source: name === 'local' ? 'managed' : 'hermes'
    }
  }

  if (tier.source === 'provider') {
    return { ...saved, provider: tier.provider, source: 'hermes' }
  }

  return { ...saved, source: 'url' }
}

// A tier still waiting for a model takes one Hermes has now.
function fillDraft(name, draft, hermes) {
  return draft.source === 'url' || draft.model ? draft : { ...draft, ...firstChoice(name, draft.source, hermes) }
}

function tierReady(draft) {
  const model = draft.model.trim()

  if (draft.source === 'managed') {
    return Boolean(model)
  }

  return Boolean(model && (draft.source === 'hermes' ? draft.provider || draft.builtIn : draft.base_url.trim()))
}

function tierBody(draft, hermes) {
  const model = draft.model.trim()

  if (draft.source === 'managed' || (draft.source === 'hermes' && draft.builtIn)) {
    return { model, source: 'managed' }
  }

  if (draft.source === 'hermes') {
    return { model, provider: draft.provider, source: 'provider' }
  }

  const body = { base_url: draft.base_url.trim(), model, source: 'url' }

  if (draft.api_key.trim()) {
    body.api_key = draft.api_key.trim()
  }

  return body
}

function useHermesModels() {
  const [models, setModels] = useState(null)
  const [version, setVersion] = useState(0)

  useEffect(() => {
    let live = true
    api('/providers')
      .then(result => live && setModels({ ...NO_HERMES_MODELS, ...result }))
      .catch(() => live && setModels(current => current || NO_HERMES_MODELS))

    return () => {
      live = false
    }
  }, [version])

  return [models, () => setVersion(v => v + 1)]
}

// Tier choices made before opening Hermes's settings, for a page that is rebuilt on the way back.
let heldDrafts = null

// Calls `refresh` when the window regains focus or `ref`'s element comes back into view, as it
// does on returning from Hermes's settings.
function useRefreshOnReturn(ref, refresh) {
  useEffect(() => {
    let hidden = false
    const observer =
      typeof IntersectionObserver === 'function'
        ? new IntersectionObserver(entries => {
            const visible = entries.some(e => e.isIntersecting)

            if (visible && hidden) {
              refresh()
            }

            hidden = !visible
          })
        : null

    if (ref.current && observer) {
      observer.observe(ref.current)
    }

    window.addEventListener('focus', refresh)

    return () => {
      observer?.disconnect()
      window.removeEventListener('focus', refresh)
    }
  }, [])
}

function optionLabel(option, current) {
  const where = option.source === 'managed' ? ' · built into Hermes' : option.local ? ' · local' : ''
  const inUse = current && current.provider === option.provider ? ' · in use' : ''

  return `${option.name}${where}${inUse}${option.authenticated ? '' : ' (not signed in)'}`
}

const selectBox = (props, ...children) =>
  h('select', { className: 'h-9 rounded-md border bg-background px-2 text-sm', ...props }, ...children)

function settingsLink(key, label, path, openSettings) {
  return h(
    'div',
    { className: 'flex', key },
    h(Button, { onClick: () => openSettings(path), size: 'xs', type: 'button', variant: 'text' }, label)
  )
}

const mutedNote = (key, text) => h('p', { className: 'text-xs text-muted-foreground', key }, text)

function LocalModelFields({ draft, hermes, openSettings, set }) {
  const models = builtInOptions(hermes).flatMap(o => o.models.map(model => ({ model, provider: o.provider })))

  return [
    models.length || draft.model
      ? selectBox(
          {
            'aria-label': 'Local model',
            key: 'model',
            onChange: e => set(models.find(m => m.model === e.target.value) || { model: e.target.value }),
            value: draft.model
          },
          draft.model ? null : h('option', { value: '' }, 'Pick a local model…'),
          models.map(m => h('option', { key: `${m.provider}/${m.model}`, value: m.model }, m.model)),
          draft.model && !models.some(m => m.model === draft.model)
            ? h('option', { value: draft.model }, draft.model)
            : null
        )
      : null,
    models.length ? null : mutedNote('none', 'Set up a model in Settings \u2192 Providers \u2192 Local Models, then return here.'),
    settingsLink('settings', models.length ? 'Manage local models' : 'Set up local model', LOCAL_SETTINGS_PATH, openSettings)
  ]
}

function HermesModelFields({ name, draft, hermes, openSettings, set }) {
  const chosen = hermesOption(hermes, draft.provider)
  // Local models first for the local tier, remote ones first for the cloud tier.
  const wantLocal = name === 'local'
  const options = providerOptions(name, hermes).sort((a, b) => Number(b.local === wantLocal) - Number(a.local === wantLocal))
  const models = chosen ? chosen.models : []

  const pick = provider => {
    const option = hermesOption(hermes, provider)
    const current = hermes.current && hermes.current.provider === provider ? hermes.current.model : ''
    set({
      builtIn: Boolean(option && option.source === 'managed'),
      model: option ? current || option.default_model : '',
      provider
    })
  }

  return [
    options.length || draft.provider
      ? selectBox(
          { 'aria-label': 'Hermes provider', key: 'provider', onChange: e => pick(e.target.value), value: draft.provider },
          h('option', { value: '' }, 'Pick a provider…'),
          options.map(o => h('option', { key: o.provider, value: o.provider }, optionLabel(o, hermes.current))),
          draft.provider && !chosen
            ? h('option', { value: draft.provider }, `${draft.provider} (no longer configured in Hermes)`)
            : null
        )
      : null,
    options.length || draft.provider
      ? selectBox(
          {
            'aria-label': 'Model',
            disabled: !draft.provider,
            key: 'model',
            onChange: e => set({ model: e.target.value }),
            value: draft.model
          },
          models.map(m => h('option', { key: m, value: m }, m)),
          draft.model && !models.includes(draft.model) ? h('option', { value: draft.model }, draft.model) : null
        )
      : null,
    options.length || draft.provider ? null : mutedNote('none', 'Configure a provider in Settings \u2192 Providers, then return here.'),
    chosen && !chosen.authenticated
      ? mutedNote('auth', `Sign in first: hermes auth add ${chosen.provider}, or Settings \u2192 Providers.`)
      : null,
    settingsLink('settings', 'Configure provider', PROVIDER_SETTINGS_PATH, openSettings)
  ]
}

function TierFields({ name, draft, onChange, onKeyDown, hermes, openSettings }) {
  const set = patch => onChange({ ...draft, ...patch })
  const [legend, role, about] = TIERS[name]
  const sources = TIER_SOURCES[name]

  return h(
    'fieldset',
    { className: 'grid gap-2' },
    h('legend', { className: 'mb-2 text-sm font-semibold' }, `${legend} (${role})`),
    mutedNote('about', about),
    h(
      'div',
      { className: cn('grid gap-2', sources.length === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-2'), role: 'radiogroup' },
      sources.map(source =>
        h(Choice, {
          hint: SOURCES[source][1],
          key: source,
          label: SOURCES[source][0],
          onSelect: () => source !== draft.source && onChange(sourceDraft(name, source, draft, hermes)),
          selected: draft.source === source,
          tag: source === 'managed' ? 'Recommended' : null
        })
      )
    ),
    draft.source === 'managed' ? LocalModelFields({ draft, hermes, openSettings, set }) : null,
    draft.source === 'hermes' ? HermesModelFields({ draft, hermes, name, openSettings, set }) : null,
    draft.source === 'url'
      ? [
          h(Field, {
            key: 'url',
            label: 'Endpoint URL (e.g. http://127.0.0.1:8080/v1 or https://api.anthropic.com)',
            onChange: e => set({ base_url: e.target.value }),
            onKeyDown,
            value: draft.base_url
          }),
          h(Field, {
            key: 'model',
            label: 'Model name (for llama-server, its --alias)',
            onChange: e => set({ model: e.target.value }),
            onKeyDown,
            value: draft.model
          }),
          h(Field, {
            key: 'key',
            label: 'API key (not needed on this machine or network; empty keeps the saved key)',
            onChange: e => set({ api_key: e.target.value }),
            onKeyDown,
            type: 'password',
            value: draft.api_key
          })
        ]
      : null
  )
}

// The form starts from the models Hermes has, so it renders once they are loaded.
function SetupForm(props) {
  const [hermes, refresh] = useHermesModels()
  const ref = useRef(null)
  useRefreshOnReturn(ref, () => {
    heldDrafts = null
    refresh()
  })

  return h(
    'div',
    { ref },
    hermes
      ? h(SetupFields, { ...props, hermes })
      : h('p', { className: 'text-xs text-muted-foreground' }, 'Loading your Hermes models…')
  )
}

function Warnings({ warnings }) {
  return (warnings || []).map((w, i) =>
    h('p', { className: 'rounded-md border border-amber-500/50 px-3 py-2 text-xs', key: i }, w)
  )
}

function initialAlgorithm(algorithms, defaultPreset) {
  return defaultPreset && isAvailable(algorithms.find(a => a.name === defaultPreset.algorithm))
    ? defaultPreset.algorithm
    : RECOMMENDED
}

function setupReady({ chosen, cloud, keepsDefault, local }) {
  return tierReady(local) && tierReady(cloud) && isAvailable(chosen) && (!chosen || chosen.guided || keepsDefault)
}

// The notices of the subscription sign-ins the tiers use, and whether the user has yet to agree to one.
function subscriptionUse({ cloud, consent, hermes, local }) {
  const agreed = new Set(consent || [])
  const used = [local, cloud]
    .filter(d => d.source === 'hermes')
    .map(d => ({ agreed: agreed.has(d.provider), option: hermesOption(hermes, d.provider) }))
    .filter(u => u.option && u.option.subscription_notice)

  return {
    notices: [...new Set(used.map(u => u.option.subscription_notice))],
    pending: used.some(u => !u.agreed)
  }
}

function setupRequest({ allowSubscription, algorithm, bounded, chosen, cloud, defaultPreset, hermes, keepsDefault, local }) {
  return {
    body: {
      local: tierBody(local, hermes),
      cloud: tierBody(cloud, hermes),
      preset_name: keepsDefault ? defaultPreset.name : algorithm,
      preset: { algorithm, options: {} },
      bounded_local: bounded,
      ...(allowSubscription === undefined ? {} : { allow_subscription_use: allowSubscription })
    },
    label: chosen ? chosen.label : algorithm
  }
}

async function applySetup({ body }, onStatus, live) {
  onStatus('Configuring Switchyard…', null)

  const result = await runSetup(body, job => onStatus(setupStatus(job), null), () => live.current)
  await host.request('reload.env').catch(() => undefined)

  return result
}

// The setup form's save: its progress and error, and run(), which calls onSaved once setup has answered.
function useSetupRun(onSaved) {
  const [status, setStatus] = useState(null)
  const [percent, setPercent] = useState(null)
  const [error, setError] = useState(null)
  const live = useLive()
  const show = (text, value) => {
    setStatus(text)
    setPercent(value)
  }

  const run = async request => {
    setError(null)

    try {
      const result = await applySetup(request, show, live)

      if (!live.current) {
        return
      }

      show(null, null)

      if (result.running) {
        host.notify({ kind: 'success', message: `${PRODUCT_NAME} (${request.label}) is now Hermes\u2019s model` })
      } else {
        setError(result.error || 'Switchyard is configured but did not start')
      }

      onSaved()
    } catch (err) {
      if (live.current) {
        show(null, null)
        setError(errorText(err))
      }
    }
  }

  return { busy: status !== null, error, percent, run, status }
}

function SetupProgress({ percent, status }) {
  if (!status) {
    return null
  }

  return h(
    'div',
    { className: 'grid gap-2', role: 'status' },
    percent !== null
      ? h(
          'div',
          { className: 'h-1.5 overflow-hidden rounded-full bg-muted' },
          h('div', { className: 'h-full bg-primary transition-all', style: { width: `${percent}%` } })
        )
      : null,
    h('p', { className: 'text-xs text-muted-foreground' }, status)
  )
}

function SetupFields({ algorithms, config, hermes, onSaved }) {
  const current = config || {}
  const defaultPreset = (current.presets || []).find(p => p.name === current.default_preset)
  const [held] = useState(() => heldDrafts)
  const [local, setLocal] = useState(() => (held ? held.local : tierDraft('local', current.local, hermes)))
  const [cloud, setCloud] = useState(() => (held ? held.cloud : tierDraft('cloud', current.cloud, hermes)))
  // Setup keeps the default preset's algorithm; other algorithms are set up as presets.
  const algorithm = initialAlgorithm(algorithms, defaultPreset)
  const [bounded, setBounded] = useState(held ? held.bounded : Boolean(current.bounded_local))
  // The user's answer and the notices it answers; picking another subscription asks again.
  const [answer, setAnswer] = useState(null)
  const save = useSetupRun(() => {
    heldDrafts = null
    setLocal(d => ({ ...d, api_key: '' }))
    setCloud(d => ({ ...d, api_key: '' }))
    onSaved()
  })
  const busy = save.busy
  const chosen = algorithms.find(a => a.name === algorithm)
  // Keeping the default preset's algorithm keeps that preset (and its route settings).
  const keepsDefault = Boolean(defaultPreset && defaultPreset.algorithm === algorithm)
  const subscription = subscriptionUse({ cloud, consent: current.subscription_consent, hermes, local })
  const noticeText = subscription.notices.join(' ')
  const allowed = answer && answer.text === noticeText ? answer.value : !subscription.pending
  const canSave = setupReady({ chosen, cloud, keepsDefault, local }) && (!subscription.pending || allowed)
  const allowSubscription = subscription.notices.length ? allowed : undefined

  const submit = async () => {
    if (!canSave || busy) {
      return
    }

    await save.run(
      setupRequest({ allowSubscription, algorithm, bounded, chosen, cloud, defaultPreset, hermes, keepsDefault, local })
    )
  }

  const onKeyDown = e => e.key === 'Enter' && !e.nativeEvent.isComposing && void submit()

  const openSettings = path => {
    heldDrafts = { bounded, cloud, local }
    host.navigate(path)
  }

  useEffect(() => {
    heldDrafts = null
  }, [])

  // Models set up in Hermes meanwhile fill tiers still waiting for one.
  useEffect(() => {
    setLocal(d => fillDraft('local', d, hermes))
    setCloud(d => fillDraft('cloud', d, hermes))
  }, [hermes])

  const note = chosen && (bounded ? chosen.bounded_note : chosen.note)

  return h(
    'div',
    { className: 'grid gap-5' },
    h(
      'fieldset',
      { className: 'grid gap-5', disabled: busy },
      h(TierFields, { draft: local, hermes, name: 'local', onChange: setLocal, onKeyDown, openSettings }),
      h(TierFields, { draft: cloud, hermes, name: 'cloud', onChange: setCloud, onKeyDown, openSettings })
    ),
    h(Warnings, { warnings: current.warnings }),
    h(ToggleRow, {
      checked: bounded,
      description:
        'Escalate after 5 minutes per request or 10 minutes per message, including tools and approvals. ' +
        'Replies may be faster, with higher costs.',
      disabled: busy,
      label: 'Bound local thinking time',
      onChange: setBounded
    }),
    note ? h('p', { className: 'text-xs text-muted-foreground' }, note) : null,
    subscription.notices.length
      ? h(ToggleRow, {
          checked: allowed,
          description: subscription.pending
            ? noticeText
            : `${noticeText} Turn this off and save to withdraw your agreement; Switchyard then refuses ` +
              'every request until you agree again.',
          disabled: busy,
          label: 'Allow Switchyard to use this subscription',
          onChange: value => setAnswer({ text: noticeText, value })
        })
      : null,
    h(SetupProgress, { percent: save.percent, status: save.status }),
    save.error ? h('p', { className: 'text-xs text-destructive' }, save.error) : null,
    h(
      'div',
      { className: 'flex justify-end' },
      h(
        Button,
        { disabled: !canSave || busy, onClick: () => void submit() },
        busy ? 'Working…' : 'Save and start Switchyard'
      )
    )
  )
}

function PresetEditor({ algorithms, editing, onDone }) {
  const [name, setName] = useState(editing ? editing.name : '')
  const [algorithm, setAlgorithm] = useState(editing ? editing.algorithm : RECOMMENDED)
  const [options, setOptions] = useState(editing ? JSON.stringify(editing.options, null, 2) : '{}')
  const [makeDefault, setMakeDefault] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const chosen = algorithms.find(a => a.name === algorithm)

  const save = async () => {
    let parsed

    try {
      parsed = options.trim() ? JSON.parse(options) : {}
    } catch {
      setError('Route settings must be a JSON object')

      return
    }

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      setError('Route settings must be a JSON object')

      return
    }

    setBusy(true)
    setError(null)

    try {
      await api(`/presets/${encodeURIComponent(name.trim())}`, {
        method: 'PUT',
        body: { algorithm, options: parsed, make_default: makeDefault }
      })
      host.notify({ kind: 'success', message: `Saved preset ${name.trim()}` })
      onDone(true)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  return h(
    'div',
    { className: 'grid gap-3 rounded-2xl border p-4' },
    h(Field, {
      disabled: Boolean(editing) || busy,
      label: 'Preset name (lowercase, e.g. classifier)',
      onChange: e => setName(e.target.value),
      value: name
    }),
    h(AlgorithmPicker, { algorithms, onSelect: setAlgorithm, selected: algorithm }),
    h(
      'label',
      { className: 'grid gap-1 text-xs text-muted-foreground' },
      chosen && !chosen.guided
        ? 'Route settings (required for this algorithm; see Switchyard\u2019s routing docs). Tiers are "local" and "cloud".'
        : 'Route settings (optional overrides as JSON; null removes a default). Tiers are "local" and "cloud".',
      h(Textarea, {
        className: 'min-h-28 font-mono text-xs',
        disabled: busy,
        onChange: e => setOptions(e.target.value),
        value: options
      })
    ),
    chosen
      ? h(
          'p',
          { className: 'font-mono text-[0.7rem] text-muted-foreground' },
          `Defaults: ${JSON.stringify(chosen.defaults)}`
        )
      : null,
    h(
      'label',
      { className: 'flex items-center gap-2 text-xs' },
      h('input', { checked: makeDefault, onChange: e => setMakeDefault(e.target.checked), type: 'checkbox' }),
      'Make this the default preset and Hermes\u2019s model'
    ),
    error ? h('p', { className: 'text-xs text-destructive' }, error) : null,
    h(
      'div',
      { className: 'flex justify-end gap-2' },
      h(Button, { disabled: busy, onClick: () => onDone(false), type: 'button', variant: 'text' }, 'Cancel'),
      h(
        Button,
        { disabled: busy || !name.trim() || !isAvailable(chosen), onClick: () => void save() },
        busy ? 'Saving…' : 'Save preset'
      )
    )
  )
}

function PresetList({ state, onChanged }) {
  const [editing, setEditing] = useState(null)
  const [error, setError] = useState(null)
  const { config, algorithms } = state

  const act = async (path, opts, message) => {
    setError(null)

    try {
      await api(path, opts)
      host.notify({ kind: 'success', message })
      onChanged()
    } catch (err) {
      setError(errorText(err))
    }
  }

  return h(
    'div',
    { className: 'grid gap-2' },
    config.presets.map(p =>
      h(
        'div',
        { className: 'flex flex-wrap items-center gap-3 rounded-xl border px-3 py-2', key: p.name },
        h(
          'div',
          { className: 'min-w-0 flex-1' },
          h(
            'div',
            { className: 'flex items-center gap-2 text-sm font-medium' },
            p.name,
            p.name === config.default_preset
              ? h('span', { className: 'rounded bg-primary/10 px-1.5 py-0.5 text-[0.65rem] text-primary' }, 'Default')
              : null,
            state.selected_model === p.model ||
              (p.name === config.default_preset && AUTO_MODELS.includes(state.selected_model))
              ? h('span', { className: 'rounded bg-muted px-1.5 py-0.5 text-[0.65rem]' }, 'In use')
              : null
          ),
          h('p', { className: 'font-mono text-xs text-muted-foreground' }, `${p.model} · ${p.label}`)
        ),
        p.name === config.default_preset || !isAvailable(algorithms.find(a => a.name === p.algorithm))
          ? null
          : h(
              Button,
              {
                onClick: () =>
                  void act(
                    `/presets/${encodeURIComponent(p.name)}`,
                    { method: 'PUT', body: { algorithm: p.algorithm, options: p.options, make_default: true } },
                    `${p.name} is now the default`
                  ),
                size: 'xs',
                variant: 'text'
              },
              'Make default'
            ),
        h(Button, { onClick: () => setEditing(p), size: 'xs', variant: 'text' }, 'Edit'),
        config.presets.length > 1
          ? h(
              Button,
              {
                onClick: () =>
                  void act(`/presets/${encodeURIComponent(p.name)}`, { method: 'DELETE' }, `Deleted preset ${p.name}`),
                size: 'xs',
                variant: 'text'
              },
              'Delete'
            )
          : null
      )
    ),
    error ? h('p', { className: 'text-xs text-destructive' }, error) : null,
    editing
      ? h(PresetEditor, {
          algorithms,
          editing: editing === 'new' ? null : editing,
          key: editing === 'new' ? 'new' : editing.name,
          onDone: saved => {
            setEditing(null)

            if (saved) {
              onChanged()
            }
          }
        })
      : h(
          Button,
          { className: 'self-start', onClick: () => setEditing('new'), size: 'xs', variant: 'text' },
          'Add preset'
    )
  )
}

function StatusLine({ state }) {
  if (!state) {
    return h('p', { className: 'text-sm text-muted-foreground' }, 'Loading…')
  }

  if (!state.configured) {
    return h('p', { className: 'text-sm text-destructive' }, state.error || 'Switchyard is not configured')
  }

  return state.active && state.running ? null : h('p', { className: 'text-sm' }, `${routeStatus(state)}.`)
}

function routeStatus(state) {
  const sidecar = state.running ? 'running' : 'not running'
  const route = state.active ? `Hermes uses ${state.selected_model}` : 'Switchyard is not Hermes\u2019s model yet'

  return `${route}; sidecar ${sidecar}`
}

function SidecarCard({ state }) {
  const { config } = state
  const runtime = state.state && state.state.runtime
  const line = (key, text, title) =>
    h('p', { className: 'font-mono text-xs text-muted-foreground', key, title }, text)

  return h(
    Card,
    { title: 'Sidecar' },
    h('p', { className: 'text-xs' }, `${routeStatus(state)}.`),
    ['local', 'cloud'].map(name =>
      line(name, `${config[name].role} (${name}): ${config[name].model || '(none)'} \u00b7 ${config[name].endpoint}`)
    ),
    state.running && runtime && runtime.version ? line('version', `Switchyard ${runtime.version}`, runtime.location) : null
  )
}

// Monitor: the web dashboard's Switchyard cards, on the same /state poll.
const OUTCOMES = {
  local_committed: ['Committed local', 'local'],
  local_tool_call: ['Tool call on local', 'local'],
  local_after_cloud_failure: ['Local after cloud failure', 'local'],
  escalated_after_verification: ['Escalated after verification', 'cloud'],
  escalated_by_trajectory_judge: ['Escalated by trajectory judge', 'cloud'],
  escalated_on_time_limit: ['Escalated on time limit', 'cloud'],
  escalated_without_attempt: ['Escalated without attempt', 'cloud'],
  shadow_tool_call_allowed: ['Shadow: tool call allowed', 'local'],
  shadow_checked: ['Shadow: checked, cloud served', 'muted'],
  shadow_unchecked: ['Shadow: not checked, cloud served', 'muted'],
  served_local: ['Served locally', 'local'],
  served_cloud: ['Served from cloud', 'cloud'],
  cancelled: ['Cancelled by client', 'muted'],
  failed: ['Failed', 'error']
}

// Desktop's Tailwind build only ships the classes the app itself uses, so palette
// classes like bg-sky-400 may not exist there; tier colors are set inline instead.
const TIER_COLOR = { local: '#10b981', cloud: '#38bdf8' }
const TIER_TEXT = { muted: 'text-muted-foreground', error: 'text-destructive' }
const tierText = tier => (TIER_COLOR[tier] ? { color: TIER_COLOR[tier] } : undefined)
const tierBg = tier => (TIER_COLOR[tier] ? { backgroundColor: TIER_COLOR[tier] } : undefined)
const WORK_STAGES = { local_draft: true, cloud_serve: true }

function fmt(n) {
  if (n == null) {
    return '-'
  }

  if (n >= 1e6) {
    return `${(n / 1e6).toFixed(1)}M`
  }

  return n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n)
}

function ms(n) {
  if (n == null) {
    return ''
  }

  return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${n}ms`
}

function ago(ts, now) {
  const s = Math.max(0, Math.round(now - ts))

  if (s < 60) {
    return `${s}s ago`
  }

  return s < 3600 ? `${Math.round(s / 60)}m ago` : `${Math.round(s / 3600)}h ago`
}

const tokens = t => (t ? t.prompt + t.completion : 0)

function Card({ title, actions, children }) {
  return h(
    'section',
    { className: 'grid gap-3 rounded-2xl border bg-background/60 p-4' },
    h(
      'div',
      { className: 'flex items-center justify-between gap-2' },
      h('h2', { className: 'text-sm font-semibold' }, title),
      actions || null
    ),
    children
  )
}

function TierTag({ tier }) {
  return h(
    'span',
    {
      className: cn('font-mono text-[0.7rem] font-semibold uppercase', !TIER_COLOR[tier] && TIER_TEXT.muted),
      style: tierText(tier)
    },
    tier || 'idle'
  )
}

function Segments({ options, value, onSelect, disabled }) {
  return h(
    'div',
    { className: 'inline-flex overflow-hidden rounded-lg border', role: 'radiogroup' },
    options.map(([id, label, hint]) =>
      h(
        'button',
        {
          'aria-checked': value === id,
          className: cn(
            'px-3 py-1 text-xs transition',
            value === id ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent/50'
          ),
          disabled,
          key: id,
          onClick: () => value !== id && onSelect(id),
          role: 'radio',
          title: hint,
          type: 'button'
        },
        label
      )
    )
  )
}

function PinToggle({ config, onChanged }) {
  const [busy, setBusy] = useState(false)

  const choose = async pin => {
    setBusy(true)

    try {
      await setPin(pin)
      onChanged()
    } catch (err) {
      host.notify({ kind: 'error', message: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  return h(
    'div',
    { className: 'grid gap-2' },
    h(Segments, {
      disabled: busy,
      onSelect: pin => void choose(pin),
      options: [
        ['', MODES[''], 'Switchyard picks the model for each request with the selected preset'],
        ['local', MODES.local, `Every request goes to the local (${config.local.role}) model, ${config.local.model}`],
        ['cloud', MODES.cloud, `Every request goes to the cloud (${config.cloud.role}) model, ${config.cloud.model}`]
      ],
      value: config.pin || ''
    }),
    config.pin
      ? h(
          'p',
          { className: 'text-xs text-muted-foreground' },
          `${MODES[config.pin]}: every request goes to ${config[config.pin].model}; the monitor reflects the mode, not the preset.`
        )
      : null
  )
}

function NowCard({ current, last }) {
  if (!current) {
    return h(
      Card,
      { title: 'Now' },
      h(
        'div',
        { className: 'flex items-center gap-3' },
        h(TierTag, { tier: last ? last.served_tier : null }),
        h(
          'p',
          { className: 'text-xs text-muted-foreground' },
          last
            ? `Idle. Last turn served by ${last.served_tier || '-'}${last.why ? `: ${last.why.headline}` : ''}`
            : 'Idle. Waiting for the first routed turn.'
        )
      )
    )
  }

  const turn = current.turn

  return h(
    Card,
    { title: `Now: turn #${turn.id} \u00b7 ${turn.preset}` },
    h(
    'div',
      { className: 'flex items-center gap-3' },
      h(TierTag, { tier: current.tier }),
      h(
        'div',
        { className: 'min-w-0' },
        h('p', { className: 'animate-pulse text-sm font-medium' }, current.label),
        h(
          'p',
          { className: 'text-xs text-muted-foreground' },
          `${ms(current.elapsed_ms)}${turn.task_type ? ` \u00b7 typed ${turn.task_type}` : ''} \u00b7 ` +
            `local ${fmt(tokens(turn.tokens.local))} / cloud ${fmt(tokens(turn.tokens.cloud))} tokens`
        )
      )
    ),
    h(StageChips, { stages: turn.stages })
  )
}

function StageChips({ stages }) {
  return h(
    'div',
    { className: 'flex flex-wrap gap-1.5' },
    stages.map((s, i) =>
      h(
        'span',
        {
          className: cn(
            'rounded-md border px-1.5 py-0.5 text-[0.7rem]',
            s.status === 'running' && 'animate-pulse'
          ),
          style: tierText(s.tier),
          key: i,
          title: `${s.label} on ${s.model} (${s.prompt_tokens} in / ${s.completion_tokens} out)`
        },
        s.label,
        s.ms != null ? h('span', { className: 'ml-1 text-muted-foreground' }, ms(s.ms)) : null
      )
    )
  )
}

function ShareBar({ label, local, cloud, show }) {
  const total = local + cloud
  const share = total ? Math.round((100 * local) / total) : 0

  return h(
    'div',
    { className: 'grid gap-1' },
    h(
      'div',
      { className: 'flex justify-between text-xs' },
      h('span', null, label),
      h(
        'span',
        { className: 'text-muted-foreground' },
        total ? `local ${show(local)} \u00b7 cloud ${show(cloud)}` : '-'
      )
    ),
    h(
      'div',
      { className: cn('flex h-2 overflow-hidden rounded-full', !total && 'bg-muted'), style: total ? tierBg('cloud') : undefined },
      total ? h('div', { className: 'h-full transition-all', style: { ...tierBg('local'), width: `${share}%` } }) : null
    )
  )
}

const minus = (a, b) => Object.fromEntries(Object.keys(a).map(k => [k, a[k] - ((b && b[k]) || 0)]))

// The sidecar's totals without background tasks (titles, compression), which the timeline leaves out too.
function foregroundTotals(totals) {
  const bg = totals.background
  if (!bg) return totals
  const byStage = {}
  Object.keys(totals.by_stage).forEach(k => {
    const b = minus(totals.by_stage[k], bg.by_stage[k])
    if (b.calls) byStage[k] = b
  })
  return {
    ...totals,
    local: minus(totals.local, bg.local),
    cloud: minus(totals.cloud, bg.cloud),
    served: minus(totals.served, bg.served),
    by_stage: byStage
  }
}

function SplitCard({ state }) {
  const totals = foregroundTotals(state.totals)
  let routing = 0
  let work = 0

  Object.keys(totals.by_stage).forEach(k => {
    if (WORK_STAGES[k]) {
      work += totals.by_stage[k].ms || 0
    } else {
      routing += totals.by_stage[k].ms || 0
    }
  })

  const served = totals.served.local + totals.served.cloud

  return h(
    Card,
    { title: 'Local vs cloud this sidecar run' },
    h(
      'div',
      { className: 'grid gap-3 sm:grid-cols-2' },
      ['local', 'cloud'].map(tier =>
        h(
          'div',
          { className: 'grid gap-0.5', key: tier },
          h(
            'div',
            { className: 'flex items-center gap-2' },
            h(TierTag, { tier }),
            h('span', { className: 'truncate text-xs text-muted-foreground' }, state[`${tier}_model`])
          ),
          h('p', { className: 'text-xl font-semibold' }, fmt(tokens(totals[tier]))),
          h(
            'p',
            { className: 'text-xs text-muted-foreground' },
            `${fmt(totals[tier].prompt)} in \u00b7 ${fmt(totals[tier].completion)} out \u00b7 ${totals[tier].calls} calls`
          )
        )
      )
    ),
    h(ShareBar, { label: 'Tokens', local: tokens(totals.local), cloud: tokens(totals.cloud), show: fmt }),
    h(ShareBar, { label: 'Time in model calls', local: totals.local.ms || 0, cloud: totals.cloud.ms || 0, show: ms }),
    h(ShareBar, { label: 'Turns served', local: totals.served.local, cloud: totals.served.cloud, show: String }),
    h(
      'p',
      { className: 'text-xs text-muted-foreground' },
      `${served} turns \u00b7 routing calls ${ms(routing)} vs task work ${ms(work)} (classifiers, judges and verifiers count as routing; background tasks left out)`
    )
  )
}

function TimelineCard({ timeline }) {
  const segments = (timeline || []).filter(s => s.kind !== 'auxiliary')
  const runs = []

  segments.forEach(s => {
    let run = runs[runs.length - 1]

    if (!run || run.tier !== s.tier || run.turn !== s.turn_id) {
      run = {
        tier: s.tier,
        turn: s.turn_id,
        newTurn: Boolean(run) && run.turn !== s.turn_id,
        ms: 0,
        labels: [],
        running: false
      }
      runs.push(run)
    }

    run.ms += s.ms || 0
    run.running = run.running || s.running
    run.labels.push(`${s.label} ${ms(s.ms)}`)
  })

  const total = runs.reduce((sum, run) => sum + run.ms, 0)

  return h(
    Card,
    { title: 'Routing timeline' },
    !total
      ? h('p', { className: 'text-xs text-muted-foreground' }, 'No model calls yet.')
      : h(
          'div',
          { className: 'flex h-6 gap-px overflow-hidden rounded-md' },
          runs.map((run, i) =>
            h('div', {
              className: cn(!TIER_COLOR[run.tier] && 'bg-muted', run.newTurn && 'ml-1', run.running && 'animate-pulse'),
              key: i,
              style: { ...tierBg(run.tier), flexGrow: Math.max(run.ms, total * 0.004) },
              title: `${run.tier}: ${ms(run.ms)}${run.turn != null ? ` \u00b7 turn #${run.turn}` : ''}\n${run.labels.join('\n')}`
            })
          )
        ),
    total
      ? h(
          'p',
          { className: 'text-xs text-muted-foreground' },
          'Oldest on the left; gaps separate turns. Hover a block for its calls.'
        )
      : null
  )
}

function Why({ why }) {
  return h(
    'div',
    { className: 'grid gap-1 rounded-md bg-muted/40 px-2 py-1.5 text-xs' },
    h(
      'p',
      null,
      h(
        'span',
        { className: 'font-semibold', style: tierText(why.served) },
        why.served === 'cloud' ? 'Why cloud: ' : 'Why local: '
      ),
      why.headline
    ),
    h('p', { className: 'text-muted-foreground' }, why.summary),
    why.checks.length
      ? h(
          'ul',
          { className: 'grid gap-0.5' },
          why.checks.map((c, i) =>
            h('li', { key: i }, `${c.ok === true ? '\u2713' : c.ok === false ? '\u2717' : '?'} ${c.label}: ${c.result}`)
          )
        )
      : null
  )
}

function EscalationsCard({ turns }) {
  const groups = {}
  let escalated = 0

  turns.forEach(t => {
    if (t.why && t.why.served === 'cloud') {
      escalated += 1
      const g = groups[t.why.headline] || (groups[t.why.headline] = { count: 0, summary: t.why.summary })
      g.count += 1
    }
  })

  const vgrTurns = turns.filter(t => t.why).length

  return h(
    Card,
    { title: 'Why turns went to the cloud' },
    !escalated
      ? h(
          'p',
          { className: 'text-xs text-muted-foreground' },
          vgrTurns ? 'No recent VGR turn was escalated.' : 'No VGR turns yet.'
        )
      : h(
          'div',
          { className: 'grid gap-1.5' },
          h(
            'p',
            { className: 'text-xs text-muted-foreground' },
            `${escalated} of the last ${vgrTurns} VGR turns went to the cloud.`
          ),
          Object.keys(groups)
            .sort((a, b) => groups[b].count - groups[a].count)
            .map(name =>
              h(
                'div',
                { className: 'flex items-center gap-2 text-xs', key: name, title: groups[name].summary },
                h(
                  'div',
                  { className: 'h-1.5 w-20 overflow-hidden rounded-full bg-muted' },
                  h('div', {
                    className: 'h-full',
                    style: { ...tierBg('cloud'), width: `${Math.round((100 * groups[name].count) / escalated)}%` }
                  })
                ),
                h('span', { className: 'font-mono' }, groups[name].count),
                h('span', null, name)
              )
            )
        )
  )
}

function TurnRow({ turn, now }) {
  const outcome = OUTCOMES[turn.outcome] || [turn.outcome || '-', 'error']
  const duration = turn.ended ? Math.round((turn.ended - turn.started) * 1000) : null

  return h(
    'div',
    { className: 'grid gap-1.5 border-t pt-2 first:border-t-0 first:pt-0' },
    h(
      'div',
      { className: 'flex flex-wrap items-center gap-2 text-xs' },
      h('span', { className: 'font-mono' }, `#${turn.id}`),
      h('span', { className: 'rounded bg-muted px-1.5 py-0.5' }, turn.preset),
      h('span', { className: cn('font-medium', TIER_TEXT[outcome[1]]), style: tierText(outcome[1]) }, outcome[0]),
      h('span', { className: 'text-muted-foreground' }, `${ms(duration)} \u00b7 ${ago(turn.started, now)}`),
      h(
        'span',
        { className: 'ml-auto font-mono text-muted-foreground' },
        `L ${fmt(tokens(turn.tokens.local))} \u00b7 C ${fmt(tokens(turn.tokens.cloud))}`
      )
    ),
    turn.preview ? h('p', { className: 'truncate text-xs text-muted-foreground' }, turn.preview) : null,
    turn.why ? h(Why, { why: turn.why }) : null,
    h(StageChips, { stages: turn.stages })
  )
}

function MonitorView({ state, onChanged }) {
  const [starting, setStarting] = useState(false)
  const monitor = state.state

  const start = async () => {
    setStarting(true)

    try {
      const result = await api('/start', { method: 'POST', timeoutMs: 60000 })

      if (!result.ok) {
        host.notify({ kind: 'error', message: result.error || 'Switchyard sidecar failed to start' })
      }

      onChanged()
    } catch (err) {
      host.notify({ kind: 'error', message: errorText(err) })
    } finally {
      setStarting(false)
    }
  }

  if (!monitor) {
    return h(
      'div',
      { className: 'grid gap-4' },
      h(
        Card,
        { title: 'Sidecar stopped' },
        h(
          'p',
          { className: 'text-xs text-muted-foreground' },
          'The Switchyard sidecar is not running, so there is nothing to monitor.'
        ),
        h(
          Button,
          { className: 'self-start', disabled: starting, onClick: () => void start() },
          starting ? 'Starting…' : 'Start sidecar'
        )
      ),
      h(SidecarCard, { state })
    )
  }

  const turns = monitor.turns || []

  return h(
    'div',
    { className: 'grid gap-4' },
    h(NowCard, { current: monitor.current, last: turns.find(t => t.served_tier) }),
    h(SplitCard, { state: monitor }),
    h(TimelineCard, { timeline: monitor.timeline }),
    h(EscalationsCard, { turns }),
    h(
      Card,
      { title: 'Recent decisions' },
      turns.length
        ? h(
            'div',
            { className: 'grid gap-2' },
            turns.map(t => h(TurnRow, { key: t.id, now: monitor.now, turn: t }))
          )
        : h(
            'p',
            { className: 'text-xs text-muted-foreground' },
            'No routed turns yet. Chat with Hermes to see decisions here.'
          )
    ),
    h(SidecarCard, { state })
  )
}

// The page outlives a visit, keeping its tab, so a request for the Monitor reaches it here.
const monitorRequests = new Set()

function openMonitor() {
  monitorRequests.forEach(show => show())
  host.navigate(PATH)
}

function PageHeader({ onChanged, state }) {
  const ready = Boolean(state && state.configured)

  return h(
    'header',
    { className: 'grid gap-2' },
    h('h1', { className: 'text-lg font-semibold' }, `${PRODUCT_NAME} (Beta)`),
    h(
      'p',
      { className: 'text-xs text-muted-foreground' },
      'Automatic inference routing between local and cloud models, escalating tasks only when needed.'
    ),
    h(StatusLine, { state }),
    ready ? h(PinToggle, { config: state.config, onChanged }) : null,
    ready ? h(Warnings, { warnings: state.config.warnings }) : null,
    ready ? h(CreditAlerts, { state }) : null
  )
}

function PageTabs({ onSelect, shown }) {
  return h(
    'div',
    { className: 'flex items-center justify-between gap-2' },
    h(Segments, {
      onSelect,
      options: [
        ['monitor', 'Monitor', 'Live routing decisions'],
        ['setup', 'Setup', 'Tiers and presets']
      ],
      value: shown
    }),
    h(Button, { onClick: () => void openInBrowser(), size: 'xs', variant: 'text' }, 'Open in browser')
  )
}

function SetupTab({ algorithms, configured, onChanged, state }) {
  const ready = Boolean(state && state.configured)

  return [
    h(
      Section,
      { key: 'tiers', title: 'Select your models' },
      state
        ? h(SetupForm, {
            algorithms,
            config: state.config,
            configured,
            key: ready ? 'configured' : 'new',
            onSaved: onChanged
          })
        : null
    ),
    ready
      ? h(
          Section,
          {
            hint: 'Pick a preset with /model <name> --provider switchyard, or from the model picker.',
            key: 'presets',
            title: 'Presets'
          },
          h(PresetList, { onChanged, state })
        )
      : null
  ]
}

function SwitchyardPage() {
  const [tab, setTab] = useState(null)
  const [polling, setPolling] = useState(false)
  const [state, refresh] = useSwitchyardState(polling ? MONITOR_POLL_MS : 0)
  const algorithms = (state && state.algorithms) || []
  const ready = Boolean(state && state.configured)
  const configured = ready && Boolean(state.active)
  const shown = tab || (configured ? 'monitor' : 'setup')

  useEffect(() => setPolling(shown === 'monitor'), [shown])
  useEffect(() => {
    const show = () => setTab('monitor')
    monitorRequests.add(show)

    return () => monitorRequests.delete(show)
  }, [])

  return h(
    'div',
    { className: 'mx-auto grid w-full max-w-3xl gap-6 overflow-y-auto p-6' },
    h(PageHeader, { onChanged: refresh, state }),
    ready ? h(PageTabs, { onSelect: setTab, shown }) : null,
    shown === 'monitor' && ready
      ? h(MonitorView, { onChanged: refresh, state })
      : h(SetupTab, { algorithms, configured, onChanged: refresh, state })
  )
}

function EmptyChatCard() {
  const [state] = useSwitchyardState()

  if (!state || state.active) {
    return null
  }

  return h(
    'button',
    {
      className:
        'mx-auto mt-6 flex w-full max-w-md items-center justify-between gap-4 rounded-[8px] bg-primary/[0.06] px-3 py-2.5 text-left transition-colors hover:bg-primary/10',
      onClick: () => host.navigate(PATH),
      type: 'button'
    },
    h(
      'div',
      { className: 'min-w-0' },
      h('span', { className: 'text-sm font-semibold' }, 'Set up local and cloud routing (Beta)'),
      h(
        'p',
        { className: 'mt-1 text-xs leading-5 text-muted-foreground' },
        `${PRODUCT_NAME} runs most turns on a local model and escalates to a cloud model only when needed.`
      )
    ),
    h('span', { 'aria-hidden': true, className: 'text-primary' }, '\u203a')
  )
}

const STEP = /^\d{1,4}$/

// The `::switchyard-route{...}` line the plugin opens a reply with when Switchyard changed tier in that turn.
// Its attributes are text from the backend's transcript, so each one is checked before it is shown.
function RouteMarker({ attrs }) {
  const change = attrs.change

  if (change !== 'escalated' && change !== 'deescalated') {
    return null
  }

  const tier = change === 'escalated' ? 'cloud' : 'local'
  const verb = change === 'deescalated' ? 'Back to' : attrs.algorithm === 'vgr' ? 'Escalated to' : 'Routed to'
  const model = (attrs.model || '').split('/').pop().slice(0, 80) || `the ${tier} model`
  const where = STEP.test(attrs.step || '') && STEP.test(attrs.steps || '') ? `at step ${attrs.step} of ${attrs.steps}` : ''
  const reason = (attrs.reason || '').slice(0, 160)

  return h(
    'div',
    {
      className: 'not-prose my-1 flex min-w-0 items-center gap-2 border-y border-border/60 px-1 py-1 text-xs',
      title: attrs.summary ? attrs.summary.slice(0, 400) : undefined
    },
    h('span', { className: 'size-1.5 shrink-0 rounded-full', style: tierBg(tier) }),
    h('span', { className: 'shrink-0 font-semibold', style: tierText(tier) }, `${verb} ${model}`),
    where ? h('span', { className: 'shrink-0 text-muted-foreground' }, where) : null,
    reason ? h('span', { className: 'min-w-0 truncate text-muted-foreground' }, reason) : null,
    h(
      Button,
      { className: 'ml-auto shrink-0', onClick: openMonitor, size: 'xs', type: 'button', variant: 'text' },
      'Why?'
    )
  )
}

function useHostValue(atom) {
  const [value, setValue] = useState(() => (atom ? atom.get() : null))
  useEffect(() => (atom ? atom.subscribe(setValue) : undefined), [atom])
  return value
}

// What Switchyard is doing for the focused chat's turn, above the composer while the turn runs.
function TurnProgress() {
  const state = host.state || {}
  const busy = useHostValue(state.busy)
  const storedId = useHostValue(state.focusedStoredSessionId)
  const runtimeId = useHostValue(state.focusedSessionId)
  const [current, setCurrent] = useState(null)

  useEffect(() => {
    if (!busy) {
      setCurrent(null)
      return undefined
    }
    let live = true
    let timer = null
    const poll = () =>
      api('/state')
        .then(value => live && setCurrent(value?.state?.current || null))
        .catch(() => live && setCurrent(null))
        .finally(() => {
          if (live) timer = window.setTimeout(poll, PROGRESS_POLL_MS)
        })
    poll()
    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [busy])

  const session = current?.turn?.session_id
  if (!busy || !session || (session !== storedId && session !== runtimeId)) {
    return null
  }
  const tokens = current.tokens || {}
  return h(
    'div',
    { className: 'flex min-w-0 items-center gap-2 px-2 py-0.5 text-xs text-muted-foreground' },
    h('span', { className: 'size-1.5 shrink-0 animate-pulse rounded-full', style: tierBg(current.tier) }),
    h('span', { className: 'min-w-0 truncate' }, `${PRODUCT_NAME}: ${current.label}`),
    h('span', { className: 'shrink-0 tabular-nums' }, `${Math.round((current.elapsed_ms || 0) / 1000)} s`),
    tokens.prompt || tokens.completion
      ? h(
          'span',
          {
            className: 'shrink-0 tabular-nums',
            title: 'Tokens this turn sent to the models and got back, routing checks included'
          },
          `↑ ${fmt(tokens.prompt || 0)} ↓ ${fmt(tokens.completion || 0)}`
        )
      : null
  )
}

export default {
  id: 'nvidia-switchyard',
  name: `${PRODUCT_NAME} (Beta)`,
  register(ctx) {
    rest = ctx.rest
    os = ctx.os

    const reportLoaded = loaded => {
      api('/renderer', { method: 'POST', body: { loaded } }).catch(() => {})
    }
    reportLoaded(true)
    const heartbeat = window.setInterval(() => reportLoaded(true), RENDERER_PING_MS)
    ctx.onDispose?.(() => {
      window.clearInterval(heartbeat)
      reportLoaded(false)
    })

    const composerTop = sdk.COMPOSER_AREAS?.top
    if (composerTop) {
      try {
        ctx.registerMany([{ id: 'turn-progress', area: composerTop, render: () => h(TurnProgress) }])
      } catch (error) {
        console.warn('Switchyard: no turn progress above the composer', error)
      }
    }

    const command = (id, label, run) => ({
      id,
      area: PALETTE_AREA,
      data: { id: `switchyard.${id}`, label, keywords: ['switchyard', 'routing', 'pin', 'mode', 'auto', 'only', 'local', 'cloud'], run }
    })

    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: PATH }, render: () => h(SwitchyardPage) },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 60,
        data: { codicon: 'server-process', label: PRODUCT_NAME, path: PATH }
      },
      {
        id: 'setup',
        area: PALETTE_AREA,
        data: {
          id: 'switchyard.setup',
          label: `Set up ${PRODUCT_NAME} (Beta) with local and cloud models`,
          keywords: ['switchyard', 'vgr', 'local', 'cloud', 'llama', 'router', 'routing'],
          run: () => host.navigate(PATH)
        }
      },
      command(
        'pin-local',
        'Switchyard: Local Only (every request to the local, efficient model)',
        () => void setPin('local').catch(err => host.notify({ kind: 'error', message: errorText(err) }))
      ),
      command(
        'pin-cloud',
        'Switchyard: Cloud Only (every request to the cloud, capable model)',
        () => void setPin('cloud').catch(err => host.notify({ kind: 'error', message: errorText(err) }))
      ),
      command(
        'pin-off',
        'Switchyard: Auto (pick the model for each request)',
        () => void setPin('').catch(err => host.notify({ kind: 'error', message: errorText(err) }))
      ),
      command('open-browser', 'Switchyard: open the routing monitor in the browser', () => void openInBrowser()),
      { id: 'empty', area: CHAT_EMPTY_AREA, data: { render: () => h(EmptyChatCard) } },
      {
        id: 'route-marker',
        area: TRANSCRIPT_DIRECTIVE_AREA,
        data: { name: 'switchyard-route', render: ({ attrs }) => h(RouteMarker, { attrs }) }
      }
    ])
  }
}
