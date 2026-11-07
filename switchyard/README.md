# Switchyard Local-first Routing plugin for Hermes (Beta)

> **Beta.** Routing, settings and commands may change between releases. Not
> yet supported for production use.

This plugin routes [Hermes](https://github.com/NousResearch/hermes-agent)
between two models, usually a local one and a cloud one, with VGR
(Verification-Gated Routing): each request goes to the local model first, its
answer is checked, and the request escalates to the cloud model only when the
check fails.

[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) runs in a loopback
sidecar next to Hermes. Each routing preset appears as a model under the
Switchyard provider (`switchyard/<preset>`). The plugin adds `hermes switchyard`
commands and a Switchyard page in the dashboard and in Hermes Desktop.

## Requirements

- **Hermes** 0.21.7 or later, CLI or Desktop.
- **Windows** (x64 or ARM64), **macOS** (Apple silicon on macOS 11 or later,
  or Intel on 10.12 or later) or **Linux** (x86_64 or ARM64), with access to
  pypi.org while the plugin installs.
- **A local model** on an OpenAI-compatible Chat Completions server that
  returns token probabilities, such as
  [`llama-server`](https://github.com/ggml-org/llama.cpp). VGR scores the local
  model's answers from them. Hermes's one-click llama.cpp setup works too.
- **A cloud model**: any OpenAI- or Anthropic-compatible server, or a provider
  you are signed in to in Hermes (`hermes auth`). Users can configure
  Switchyard with third-party model providers. Each provider's terms and
  conditions apply. A Claude Pro or Max login can't be routed.

## Install

**In Hermes Desktop:** open **Capabilities ▸ Plugins**, find **Switchyard** and
click **Install**.

**From a terminal:**

```sh
hermes plugins install nvidia-switchyard
```

Hermes installs the plugin and `nemo-switchyard`, the Switchyard runtime, from
PyPI into its own Python environment. That can take a few minutes the first
time.

To install from a clone of this repository instead, point Hermes at the
`switchyard/` folder in it:

```sh
hermes plugins install file:///path/to/this/repository#switchyard
```

## Set up

**In Hermes Desktop:** under **Capabilities ▸ Plugins**, check that both of
**Switchyard**'s switches are on: the one under the profile picker (for example
**Hermes (default)**) turns on Switchyard Local-first Routing for that profile, and
**Desktop** adds its page and chat notes. Then open **Switchyard** in the sidebar. Its
setup form picks each model from:

- **Your Hermes models**: a model or signed-in provider Hermes already has.
  Switchyard uses Hermes's login on each call and doesn't copy the token.
- **Set up llama.cpp for me**: Hermes downloads a model sized for this machine
  and runs it.
- **Server URL**: any compatible server. An API key you paste is saved in
  Hermes's `.env` as `SWITCHYARD_LOCAL_API_KEY` or `SWITCHYARD_CLOUD_API_KEY`.

Leave **Routing algorithm** on **VGR** and click **Save and start Switchyard**.
The VGR preset (`switchyard/vgr`) becomes the default model.

**From a terminal:** put the cloud key in Hermes's `.env` as
`SWITCHYARD_CLOUD_API_KEY`, then:

```sh
hermes switchyard setup \
  --local-url http://127.0.0.1:8080/v1 --local-model <local model id> \
  --cloud-url <cloud server URL> --cloud-model <cloud model id> \
  --cloud-key-env SWITCHYARD_CLOUD_API_KEY
```

Use `--<tier>-provider <name>` instead of the URL options for a provider you
signed in to with `hermes auth`. `--<tier>-key-env` names an environment
variable; never pass a key on the command line. With no tier options, `setup`
asks you to pick each model. `hermes switchyard setup --help` lists every option.

A tier on a provider's subscription sign-in needs your agreement first: setup
shows what Switchyard does with it and asks, the Desktop form adds a switch for
it, and `--allow-subscription-use` agrees from a script. See [Subscription sign-ins](#subscription-sign-ins).

Setup also sends Hermes's session titles and smart-approval checks straight to
the local model (`switchyard/only-efficient`), since a VGR check can outlast
Hermes's 30-second limit for both. Models you set yourself for those are kept.
In any chat on a Switchyard model, the sidecar also sends Hermes's side tasks
(titles, approval checks, context compression and the like) to the local model
without VGR checks, unless they include images or you chose
`switchyard/only-capable`.

## What to expect

- **Replies don't stream.** VGR checks the local answer before showing any of
  it, so a reply appears whole. Don't press Stop or resend while you wait; the
  Monitor tab shows what VGR is doing. In a terminal, run
  `hermes switchyard watch` in a second window to follow each step.
- **No per-message time limit on the local model by default.** Each request
  gets up to 30 minutes before the cloud model takes it. See
  [Bound local thinking time](#bound-local-thinking-time).
- **Requests with images go to the cloud model.** VGR treats the local model as
  text-only.
- **The cloud model sees your requests.** "Local first" means the local model
  answers first, not that requests stay on your machine: to check an answer,
  the cloud model receives the task and the local answer. VGR masks text that
  matches its secret and personal-data patterns first; treat that as a backstop,
  not a guarantee.
- **Escalating doesn't undo tool calls.** The cloud model continues from the
  commands and edits the local model already ran.
- **If the cloud model fails**, the local model answers that step unchecked,
  and the Monitor shows **Local after cloud failure**.

### Subscription sign-ins

When a tier uses a subscription you signed in to in Hermes rather than an API
key, Switchyard calls it on its own as it routes, including VGR's checks of the
local model's answers, without asking you each time. Those calls count against
the subscription's usage. Each provider's terms and conditions apply, and the
provider can restrict your account, or Hermes's sign-in with it, for use its
terms do not allow; check them before you agree. Setup records your agreement
for each such provider, and the sidecar won't start on one you haven't agreed
to. `hermes switchyard setup --no-allow-subscription-use` withdraws it, and
`hermes switchyard pin local` keeps every request on the local model.

## What the plugin does on your machine

- **Runs background processes.** Switchyard runs in a sidecar process
  listening on `127.0.0.1` (port 4110 by default). Hermes starts it, and it
  keeps running after Hermes quits until you run `hermes switchyard stop`.
  **Open in browser** also starts Hermes's web dashboard in the background
  when none is running.
- **Makes network calls.** Hermes installs `nemo-switchyard` from pypi.org
  when it installs the plugin. After that, the sidecar connects only to the
  local and cloud model servers you set up. The plugin sends no telemetry.
- **Sees every model request.** Its request hooks and middleware read each
  model request and response Hermes sends, to route it and to show it in the
  Monitor.
- **Stores credentials.** The plugin saves these in Hermes's `.env`, through
  Hermes's own config API:
  - `SWITCHYARD_API_KEY`: a random token it generates, which Hermes uses to
    call the sidecar.
  - `SWITCHYARD_LOCAL_API_KEY` and `SWITCHYARD_CLOUD_API_KEY`: API keys you
    paste during setup.
  - For a provider tier, it gets that provider's credential from Hermes on
    each call, sign-ins included, and sends it only to that provider. It
    doesn't store it. The exception is a custom endpoint from Hermes's
    `model` settings: setup saves that endpoint's key as the tier's key.
- **Changes Hermes settings.** Setup makes `switchyard/<preset>` the default
  model and sends session titles and smart-approval checks to the local model.
  It also sets the Switchyard provider's timeouts and records the models it
  replaced in its plugin settings.
- **Writes outside its plugin folder.**
  - It keeps its logs, routing records and generated Switchyard config in
    `<HERMES_HOME>/switchyard/`.
  - Each time the plugin loads, it copies its provider profile, from
    `provider/` in this folder, to
    `<HERMES_HOME>/plugins/model-providers/switchyard/`. That's the only place
    Hermes loads provider profiles from, and the copy is what lists the
    `switchyard/*` models in every Hermes process.
- **Cleans up after removal.** Once the plugin is disabled or removed, the
  provider profile switches Hermes back to the models setup replaced and stops
  the sidecar. Once the plugin is removed, the profile takes the plugin's old
  id, `switchyard`, out of Hermes's plugin lists, then deletes its own copy. It
  doesn't change saved conversations: replies that Desktop showed with a route
  note keep it, as raw `::switchyard-route{...}` text.

## Using Switchyard

### Smart routing or always one model

The toggle at the top of the Switchyard page switches between **Smart routing**
(the preset decides), **Always <local model>** and **Always <cloud model>**, for
every chat and across restarts. From a terminal:
`hermes switchyard pin local|cloud|off`.

To pin one chat instead, pick **Only Efficient** (local) or **Only Capable**
(cloud) in its model picker, or `/model switchyard/only-capable --provider
switchyard` in a terminal chat.

Once VGR escalates a turn, it stays on the cloud model until your next message.

### Bound local thinking time

**Bound local thinking time**, on the Setup tab or with
`hermes switchyard setup --bounded-local` (`--no-bounded-local` to undo), gives
VGR a 300-second decision deadline per request and a 600-second local budget
per message. When the local model runs over, the cloud model takes the work:
replies come back sooner, but cloud usage and cost are much higher. The 600
seconds count from the message's first step, including time spent running tools
and waiting for approvals. `hermes switchyard status` shows which is in effect.

### Presets

Each preset is one VGR route, served as `switchyard/<name>`. Add, delete or
make one the default on the Switchyard page, or with `hermes switchyard
list`, `configure <name> --set KEY=VALUE` and `delete <name>`. Switch presets in
chat with `/model <name> --provider switchyard`.

### Watching the routing

The **Monitor** tab shows the current turn, tokens and time per model, the
routing timeline and why turns went to the cloud. **Open in browser** opens the
same view in the web dashboard (<http://127.0.0.1:9119/switchyard>). A line at
the top of each escalated reply says when and why, with **Why?** linking to the
Monitor; Switchyard strips it from what models are sent.

Switchyard keeps its records in `<HERMES_HOME>/switchyard/`: `events.jsonl` has
one line per turn (rotated to `events.jsonl.1` at 5 MB), and `sidecar.log` has
errors.

Other commands: `start`, `stop`, `status`, `serve` and `config` manage the
sidecar.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The install fails after a `cryptography`, `maturin` or `cargo` error | Open Hermes once and let it finish setting up, then install the plugin again. |
| No **Switchyard** in the sidebar | Switch on Switchyard's **Desktop** switch under **Capabilities ▸ Plugins**. It reinstalls nothing, unlike the profile's switch next to it. |
| An earlier reply opens with raw `::switchyard-route{...}` text | The reply was saved with a route note, which only Switchyard's **Desktop** half draws. Switch **Desktop** back on to see it as a note. While it's off or the plugin is removed, older notes show as text, and new replies get none. Once the plugin is disabled or removed, nothing takes the notes out of the conversation, so the model sees them too when you continue that chat. |
| The sidecar won't start and says the installed Switchyard engine does not have VGR routing | Hermes rebuilt its Python environment with a Switchyard build that has no VGR. Reinstall the plugin the way you installed it: `hermes plugins install nvidia-switchyard --force`, or the install script of the clone you installed from. |
| After an install from an administrator terminal, the **Switchyard** page shows `404: {"detail":"Not Found"}` over a blank setup form, or chats keep retrying with "Provider temporarily unavailable" | The plugin folder belongs to Administrators, so Hermes can't load the plugin. Don't press **Save and start** on that form: it would replace your setup with its defaults. From an administrator terminal, run `icacls "<HERMES_HOME>\plugins" /reset /t`, then restart Hermes. Your settings are kept. |
| No `switchyard/*` models in the model picker | Restart Hermes. If they're still missing, check that `<HERMES_HOME>/plugins/model-providers/switchyard/` exists; the plugin recreates it each time it loads. |
| The sidecar won't start and `sidecar.log` says `nemo-switchyard is not installed` | Run `hermes plugins install nvidia-switchyard --force` on a machine that can reach pypi.org. |
| `SWITCHYARD_CLOUD_API_KEY is not set` | Paste the key on the Setup tab and click **Save and start Switchyard**, or add it to Hermes's `.env`. |
| An error says "Before this error, N tool calls for this message had already run" | The cloud model failed partway through. The named calls ran and their changes stay; check them before you resend. |

## Switchyard runtime

Switchyard itself is the `nemo-switchyard` package from
[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard). VGR ships in it from 0.4.0,
starting with the 0.4.0 alpha releases. The plugin declares it as a Python dependency
pinned to one release (`nemo-switchyard==0.4.0a1` in `plugin.yaml`). Hermes installs
it from PyPI, together with the plugin, into Hermes's own Python environment. The
plugin downloads nothing itself. Before the sidecar loads Switchyard, it checks every
installed Switchyard file against the SHA-256 of the published 0.4.0a1 wheels, and
does not start if any file differs, is missing or was added. The `runtime:` line of
`hermes switchyard status` shows the package and version serving.

## Update or remove

To update, click **Update** on the plugin in **Capabilities ▸ Plugins**, or run
`hermes plugins update nvidia-switchyard`. Your settings live in Hermes's config and
are kept.

To remove, remove it in **Capabilities ▸ Plugins**, or run
`hermes plugins remove nvidia-switchyard`. On its next start, Hermes goes back to the
models setup replaced, except a main model you have chosen since, stops the
sidecar, and Switchyard leaves the model picker. Turning the plugin off does the
same. If setup recorded no earlier main model, Hermes asks you to pick one.

## Develop

The tests import the plugin as `plugins.switchyard`, so run them from a Hermes
checkout, in the test environment Hermes's `CONTRIBUTING.md` describes:

```sh
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
cp -r ../<this repository>/switchyard plugins/switchyard
cp -r ../<this repository>/switchyard/provider plugins/model-providers/switchyard
python -m pm.build_env --source . --out .venv --group dev --group test
uv pip install --python .venv/bin/python "nemo-switchyard==0.4.0a1"
.venv/bin/python -m pytest plugins/switchyard/tests
```

On Windows, clone with `git -c core.longpaths=true` into a short path, and run
the tests with `.venv\Scripts\python.exe`.

Before submitting a change to the plugin catalog, run the check its CI runs:

```sh
hermes plugins validate /path/to/this/repository/switchyard --install-deps
```

## License

Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. Licensed under the Apache
License, Version 2.0; see [LICENSE](LICENSE). The provider profile in
`provider/` carries its own copy. `nemo-switchyard` carries its own license
files.

OpenAI, Anthropic and Claude are trademarks of their owners, named here only
to say what the plugin works with. This plugin is
not affiliated with, sponsored by or endorsed by them.
