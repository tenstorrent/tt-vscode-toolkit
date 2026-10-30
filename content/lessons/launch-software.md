---
id: launch-software
title: Launch Software
description: >-
  tt launch connects a chat window or a coding agent to the model you are
  serving. See what can connect, preview, launch Open WebUI or a terminal
  agent, and meet TT-Studio.
category: connect
tags:
  - launch
  - open-webui
  - coding
  - agent
  - studio
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: validated
validatedOn:
  - p300c
estimatedMinutes: 15
order: 1
---

# Launch Software

An endpoint is the foundation. `tt launch` connects a ready-made application to the
model you are already serving, so you never copy a URL into a settings page. You do not
tell the app where the model is: `tt` asks every server it started what it serves
(`GET /v1/models`) and configures the client from the answer.

Keep the model from the last group running, and see what can connect:

```bash
tt launch list
```

[📋 What can connect](command:tenstorrent.launch.list)

On a TT-QuietBox 2 (2026-10-29):

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ client      ┃ kind        ┃ model          ┃ status         ┃ configures                             ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ opencode    │ terminal    │ tool calling   │ needs opencode │ ~/.config/opencode/opencode.json       │
│ pi          │ terminal    │ tool calling   │ needs pi       │ ~/.pi/agent/models.json                │
│ aider       │ terminal    │ tool calling   │ needs aider    │ environment (this run only)            │
│ qwencode    │ terminal    │ tool calling   │ ready          │ command-line arguments (this run only) │
│ hermes      │ terminal    │ tool calling   │ ready          │ environment (this run only)            │
│ openwebui   │ web service │ any chat model │ stopped        │ container tt-open-webui                │
│ anythingllm │ web service │ any chat model │ ready          │ container tt-anythingllm               │
└─────────────┴─────────────┴────────────────┴────────────────┴────────────────────────────────────────┘
```

There are two kinds of client, and the difference is who installs them:

* **Web UIs** (Open WebUI, AnythingLLM). `tt` pulls and runs these in a container for
  you, after asking once. They need Docker, so they are for your own machine; the cloud
  pod has none.
* **Terminal coding agents** (OpenCode, pi, Aider, Qwen Code, Hermes Agent). You install
  the agent. `tt launch` writes the small config it needs to reach your model and hands
  your terminal to it. These need a model that supports **tool calling**; `tt` checks,
  and stops with a suggestion if the model cannot.

The `status` column says what is missing: `needs opencode` means the agent is not
installed, `ready` means it can launch now, and `stopped` is a web UI container that
exists and is not running.

On the cloud pod the same listing reads `needs docker or podman` for both web UIs and
`needs qwen` / `needs hermes` for the agents that need Node.js. Aider installs there
with `uv`; it reads `ready` once `uv tool install aider-chat` has run (below).

## Where the Model Comes From

`tt launch` looks at the containers `tt` started and asks each one. With no Docker, or
nothing running there, it asks `tt`'s default port, 20000. That is why the
[Serve Without Containers](command:tenstorrent.lesson.show?["api-server"]) route on the
cloud pod uses `--port 20000`: a server you started by hand on that port is found the
same way. For any other address, say so: `tt launch aider --port 8000`, or `--url
http://host:port/v1`.

## A Chat Window in Your Browser

> **Your own machine only.** Open WebUI runs in a container, so this section needs Docker.
> On the cloud pod the buttons tell you so and run nothing; skip to the terminal agents.

```bash
tt launch openwebui --dry-run     # show what would happen, change nothing
tt launch openwebui
```

[👀 Dry run: Open WebUI](command:tenstorrent.launch.openwebuiDryRun)
[🌐 Launch Open WebUI](command:tenstorrent.launch.openwebui)

The dry run prints what `tt` found and what it will do:

```
┌───────────────────────┬───────────────────────────────────────────────────────────┐
│ model                 │ meta-llama/Llama-3.1-8B-Instruct  (Llama-3.1-8B-Instruct) │
│ endpoint              │ http://127.0.0.1:20000/v1                                 │
│ tool-call parser      │ llama3_json                                               │
│ container             │ tt-open-webui (stopped)                                   │
│ image                 │ ghcr.io/open-webui/open-webui:main                        │
│ endpoint in container │ http://host.docker.internal:20000/v1                      │
│ open                  │ http://localhost:3000                                     │
└───────────────────────┴───────────────────────────────────────────────────────────┘
```

The first time, `tt` asks before it pulls the image (a few GB) and runs it as
`tt-open-webui`. It waits until the page answers and prints `tt-open-webui is ready at
http://localhost:3000`. Open that address in a browser. The container is configured with
your endpoint, so its model list comes from your server.

If two models are being served, say which one: `tt launch openwebui --model Qwen3-32B`.
Web UIs start on port 3000, or the next free port above it; `--web-port 3080` asks for
another. The container remembers the server it was created for. After you serve on a
different port, `tt launch openwebui` stops with `tt-open-webui is pointed at a different
model server` and tells you to `docker rm -f tt-open-webui` first.

When you are done:

```bash
tt launch stop openwebui      # stops the container, keeps your chats and settings
```

[⏹️ Stop Open WebUI](command:tenstorrent.launch.stopOpenwebui)

> **⚠️ Open WebUI starts with sign-in turned off** (`WEBUI_AUTH=False`), and anyone who
> can reach port 3000 can read your chats and use the model. To chat from a laptop, tunnel
> the port and keep it private: `ssh -L 3000:localhost:3000 <you>@<your-box>`, then browse
> to `http://localhost:3000`. A tunnel does not close the port on the box itself, so use
> this only on a network you trust or behind a firewall.

## A Coding Agent in Your Terminal

Terminal agents work on your files, so they need a model that can call tools.
`Llama-3.1-8B-Instruct` can. `Qwen3-32B` can, with one caveat:

> **⚠️ Qwen3-32B thinks first, and that can drop tool calls.** With thinking on, a
> reasoning model sometimes ends a turn without making the tool call the agent asked for.
> In a test on the cloud Qwen3-32B deployment, 8 of 20 runs did this (tt-inference-server
> issue [#4386](https://github.com/tenstorrent/tt-inference-server/issues/4386), still
> open when we checked). If an agent stalls or answers without acting, send
> `chat_template_kwargs: {"enable_thinking": false}` with the request if your client lets
> you add request fields, or serve a model without a reasoning mode, such as
> `Llama-3.1-8B-Instruct`.

### Aider, on any machine

Aider is a Python package, so `uv` installs it anywhere, including the cloud pod, which
has `uv` and no Node.js:

```bash
uv tool install aider-chat
tt launch aider --dry-run
tt launch aider
```

[📦 Install Aider](command:tenstorrent.launch.installAider)
[👀 Dry run: Aider](command:tenstorrent.launch.aiderDryRun)
[🤖 Launch Aider](command:tenstorrent.launch.aider)

The dry run shows the exact command `tt launch` runs: your agent, pointed at your
endpoint, with no cloud account. Run it without `--dry-run` to start the agent.
[Coding Assistant with Aider](command:tenstorrent.lesson.show?["coding-assistant"]) goes
on from here: what Aider can do, and how to shape it with a system prompt.

### Qwen Code and Hermes, where Node.js is installed

A TT-QuietBox 2 ships with Node.js, Qwen Code and Hermes Agent, so both read `ready`
above:

```bash
tt launch qwencode --dry-run
tt launch qwencode
```

[👀 Dry run: Qwen Code](command:tenstorrent.launch.qwencodeDryRun)
[🤖 Launch Qwen Code](command:tenstorrent.launch.qwencode)

```
┌──────────────────┬───────────────────────────────────────────────────────────┐
│ model            │ meta-llama/Llama-3.1-8B-Instruct  (Llama-3.1-8B-Instruct) │
│ endpoint         │ http://127.0.0.1:20000/v1                                 │
│ tool-call parser │ llama3_json                                               │
└──────────────────┴───────────────────────────────────────────────────────────┘

commands:
qwen --auth-type openai --openai-api-key tt-local --openai-base-url http://127.0.0.1:20000/v1 --model meta-llama/Llama-3.1-8B-Instruct
```

For a community bundle, `tt` adds `warning: <model> is not in the model catalog — tt
cannot confirm it supports the tool calling qwencode needs`, and launches anyway. If the
agent then never acts, that is why.

| Client | Install it first | What `tt launch` writes |
|---|---|---|
| `aider` | `uv tool install aider-chat` | Nothing on disk (environment variables, this run only) |
| `qwencode` | Node.js 22+, then `npm install -g @qwen-code/qwen-code` | Nothing on disk (command-line arguments) |
| `hermes` | [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/) | Nothing on disk (environment). `--web` opens its browser dashboard |
| `opencode` | [opencode.ai/docs](https://opencode.ai/docs/) | A `tenstorrent` provider in `~/.config/opencode/opencode.json` |
| `pi` | [github.com/earendil-works/pi](https://github.com/earendil-works/pi) | A `tenstorrent` provider in `~/.pi/agent/models.json` |

`tt` shows what it will change and asks before writing a config file.
`tt launch disconnect opencode` removes what it wrote.

The developer blog walks through the same flow with a community model:
[Deploying Community Models with the Tenstorrent CLI](https://tenstorrent.com/developers/blog/tt-cli-and-community-model-bundles).

### Claude Code with Tenstorrent skills

The development version of `tt` (after release 1.2.1) adds `tt agent`, which sets up
Claude Code with the plugins from the
[tenstorrent/skills](https://github.com/tenstorrent/skills) marketplace for what you are
doing today (`deploy`, `bringup` or `develop`) and starts it. Preview with
`tt agent deploy --dry-run`. Check `tt --help` for an `agent` entry to see whether your
release has it.

## Prefer Point and Click? TT-Studio

[TT-Studio](https://github.com/tenstorrent/tt-studio) serves the model and provides its
own web interface. It is started by `tt serve --studio`; `tt launch` has no part in it.
It needs Docker, so it is for your own machine:

```bash
tt model stop                              # TT-Studio takes the chips and port 3000 for itself
tt serve Llama-3.1-8B-Instruct --studio    # one chip
tt serve Qwen3-32B --studio                # TT-QuietBox 2
```

[🎛️ TT-Studio: Llama-3.1-8B-Instruct](command:tenstorrent.launch.studioN150)
[🎛️ TT-Studio: Qwen3-32B](command:tenstorrent.launch.studioQb2)

On first use `tt` fetches the latest TT-Studio release, and TT-Studio asks you to accept
its terms. It prints a `TT Studio is ready` box with its address, `http://localhost:3000`.
Stopping the model with `tt model stop` also takes TT-Studio's services down and keeps
your settings and chat history. Start TT-Studio *before* any `tt launch` web UI, since
both want port 3000 and only the web UIs move when it is taken.

## Check Your Work

```bash
tt model ps               # your model is healthy
tt launch list            # your chosen client is ready or connected
tt launch stop openwebui  # clean up the web UI when you are done
tt model stop             # and the model
```

[📊 What is running](command:tenstorrent.model.ps)

## Go Deeper

Every client above speaks the OpenAI API, and so does anything else you write: an SDK
call, a `curl`, a notebook. The next three lessons build on that, with agent frameworks
and an assistant that answers questions about your own hardware.

**Next:** [Coding Assistant with Aider →](command:tenstorrent.lesson.show?["coding-assistant"])
