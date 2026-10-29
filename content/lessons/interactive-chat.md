---
id: interactive-chat
title: Serve a Model
description: >-
  tt serve starts a model in a container, waits until it answers, and gives you
  an OpenAI-compatible endpoint. Preview, serve, ask it a question, watch logs,
  stop it.
category: serve
tags:
  - inference
  - model
  - api
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
  - n150
  - p300c
estimatedMinutes: 15
order: 2
---

# Serve a Model

One command. `tt serve` downloads or finds the weights, starts the model in a container,
waits until it answers, and tells you the address. What you get is an OpenAI-compatible
HTTP API on your own hardware, which every chat app, coding agent and SDK already knows
how to talk to.

> **This lesson needs Docker.** `tt serve` runs the model in a container. The cloud pod
> has no Docker, so on the pod go to
> [Serve Without Containers](command:tenstorrent.lesson.show?["api-server"]) instead; it
> ends at the same endpoint, and every later lesson works the same way.

Two models, one per machine. Press the buttons for yours; `tt device status` told you
which you have in [Hardware Detection](command:tenstorrent.lesson.show?["hardware-detection"]).

| You have | Serve | Why |
|---|---|---|
| one chip (n150, p150) | `Llama-3.1-8B-Instruct` | COMPLETE on n150, calls tools, 64K context. Accept Meta's license on Hugging Face first. |
| a TT-QuietBox 2 (p300x2) | `Qwen3-32B` | Ships on the box. Weights and compiled cache are local, so the first serve downloads nothing. |

## Preview First

`--dry-run` starts nothing. It shows what `tt serve` found and what it would run:

```bash
tt serve Llama-3.1-8B-Instruct --dry-run    # one chip
tt serve Qwen3-32B --dry-run                 # TT-QuietBox 2
```

[👀 Dry run: Llama-3.1-8B-Instruct](command:tenstorrent.model.serveDryRunN150)
[👀 Dry run: Qwen3-32B](command:tenstorrent.model.serveDryRunQb2)

On a TT-QuietBox 2:

```
✓ Detecting device configuration  p300x2 (override with --device)
                    tt serve Qwen3-32B — dry run
┌──────────────────┬──────────────────────────────────────────────────────────┐
│ backend          │ tt-inference-server                                      │
│ workflow         │ server                                                   │
│ device           │ p300x2                                                   │
│ engines          │ vLLM                                                     │
│ status           │ FUNCTIONAL                                               │
│ docker image     │ ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-…  │
│ tool-call parser │ hermes                                                   │
│ reasoning parser │ qwen3                                                    │
│ weights          │ /home/ttuser/data/tt-cache                               │
│                  │ pre-seeded volume — weights and tt_metal_cache reused;   │
│                  │ the HF cache is not consulted                            │
│ HF token         │ from the Hugging Face login store                        │
│ port             │ 20000  (server default)                                  │
└──────────────────┴──────────────────────────────────────────────────────────┘
```

Read the `weights` row. **`pre-seeded volume`** means `tt` found the shipped weights and
the compiled kernel cache, so the first start takes minutes. `~/.cache/huggingface` means
it will use (or download into) the Hugging Face cache. `HF token` says where your login
came from without printing it.

## Serve It

```bash
tt serve Llama-3.1-8B-Instruct    # one chip
tt serve Qwen3-32B                 # TT-QuietBox 2
```

[🚀 Serve Llama-3.1-8B-Instruct](command:tenstorrent.model.serveN150)
[🚀 Serve Qwen3-32B](command:tenstorrent.model.serveQb2)

`tt serve` shows the boot as a live checklist: host ready, weights ready, image ready,
container started, engine started, device opened, KV cache configured, model warmed up,
endpoint answering. It returns once the endpoint answers, so whatever you run next will
work. When it finishes you get a box with the endpoint (`http://127.0.0.1:20000/v1`) and
the commands to look at logs and stop the model.

To watch the raw server log while it boots, run this in a second terminal:

```bash
tt model logs Llama-3.1-8B-Instruct --follow
```

Pressing Ctrl-C in the `tt serve` terminal stops *watching*. The server is a separate
container and keeps booting; `tt model stop <model>` ends it.

> **A first serve of a model you have not run before takes longer.** The image (about
> 18 GB) and the weights are downloaded, then the model is compiled for your chips and
> warmed up. A first serve of a 27B model on a TT-QuietBox 2 took about 15 minutes, 7 of
> them downloading weights. The shipped Qwen3-32B skips the download and the compile.

> **⚠️ The endpoint has no password.** `tt serve` starts the server with authentication
> off, and the server's default bind address is `0.0.0.0` (every network interface). Anyone
> who can reach port 20000 can use your model. Serve only on a network you trust or behind
> a firewall, and do not forward the port to the internet. To bind to loopback only, or to
> turn authentication on, run `tt-inference-server` yourself as shown in
> [tt-inference-server by Hand](command:tenstorrent.lesson.show?["tt-inference-server"]).

## Ask It a Question

```bash
tt model curl "Explain what an API is in one sentence." --max-tokens 256
```

[💬 Ask the model](command:tenstorrent.model.curl)

`tt model curl` asks the server which model it is serving, sends your prompt as a chat
completion, and prints the JSON reply. The text is in `choices[0].message.content`.
Unknown options become fields of the request, so `--max-tokens 256` sets `max_tokens`.
Add `--print` to see the equivalent `curl` command.

Qwen3-32B is a **reasoning model** (the dry run shows a `qwen3` reasoning parser): it
thinks first, in a separate `reasoning` field, and then answers in `content`. A small
token budget can be used up by the thinking, so give it room (`--max-tokens 2048`) when
`content` comes back empty.

> **Pick prompts that do not depend on the model knowing about Tenstorrent.** We asked
> Llama-3.1-8B "In one sentence, what is Tenstorrent?" and it replied with a confident,
> invented company history. Small models do not know recent or niche facts. That is a
> property of the model and says nothing about your hardware.

The same endpoint speaks the standard OpenAI API, so any client works. Note the model
name: the server reports the **full Hugging Face id**, which differs from the short name
you gave `tt serve`.

```bash
curl -s http://localhost:20000/v1/models | python3 -m json.tool | grep '"id"'
```

```bash
curl -s http://localhost:20000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain what an API is in one sentence."}],
    "max_tokens": 256
  }' | python3 -m json.tool
```

## Manage What Is Running

```bash
tt model ps                          # what is served, on which port, and whether it is healthy
tt model logs Llama-3.1-8B-Instruct  # the newest server log
```

[📊 What is running](command:tenstorrent.model.ps)
[📜 Server log](command:tenstorrent.model.logs)

```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ name                  ┃ backend          ┃ container                    ┃ port  ┃ health  ┃ uptime ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ Llama-3.1-8B-Instruct │ inference-server │ tt-inference-server-162a825f │ 20000 │ healthy │ 5m     │
└───────────────────────┴──────────────────┴──────────────────────────────┴───────┴─────────┴────────┘
```

`healthy` means the server answers `GET /v1/models`. `starting` means the container is
running and not answering yet. Keep the model running for the Connect Software lessons.
When you are finished with it:

```bash
tt model stop Llama-3.1-8B-Instruct
```

[⏹ Stop the model](command:tenstorrent.model.stop)

`tt model stop` stops only the named model's container.

## Serving on Half the Machine

A TT-QuietBox 2 is two p300 boards. A model with a `p300` profile uses one board (two
chips). `tt serve Llama-3.1-8B-Instruct --device p300` runs it that way. `tt serve` does
not let you choose *which* board: the container is given every Tenstorrent device node on
the machine, so run a half-machine model only when nothing else is using the chips. How to
pin a server to specific chips is in
[tt-inference-server by Hand](command:tenstorrent.lesson.show?["tt-inference-server"]).

## What Just Happened

`tt serve` ran `tt-inference-server`, which started a container holding vLLM and the
Tenstorrent plugin. The container opened a mesh across your chips, loaded the weights into
device memory, compiled and warmed up the kernels, and published the API. The detail of
each stage, and how to run `tt-inference-server` yourself, is under Go Deeper.

**Next:** [Launch Software →](command:tenstorrent.lesson.show?["launch-software"])
