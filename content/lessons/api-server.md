---
id: api-server
title: Serve Without Containers
description: >-
  The cloud pod has no Docker. Serve Llama-3.1-8B-Instruct from the vLLM
  environment the pod already has, and get the same OpenAI-compatible endpoint
  tt serve gives everywhere else.
category: serve
tags:
  - inference
  - model
  - api
  - cloud
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
estimatedMinutes: 20
order: 3
---

# Serve Without Containers

`tt serve` runs a model in a Docker container. The Tenstorrent Cloud pod has no Docker, so
this lesson serves the same model from what the pod already has: a built tt-metal and a
vLLM environment with the Tenstorrent plugin installed, at
`~/tt-metal/build/python_env_vllm`. It ends at the same `/v1` endpoint, so the Connect
Software lessons work the same way afterwards.

This is also the route on any machine without Docker, provided it has a built tt-metal
and the plugin; [Build TT-Metalium from Source](command:tenstorrent.lesson.show?["build-tt-metal"])
and the plugin's own README cover that.

## Log In to Hugging Face

Llama-3.1-8B-Instruct is gated. Accept the license on
[its Hugging Face page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then:

```bash
tt model login
```

[🔑 Log in to Hugging Face](command:tenstorrent.model.login)

On the pod, `HF_HOME` is `~/models`, so the token and the weights both land there, where
`hf`, the plugin and `tt` all look.

## Serve

```bash
source ~/tt-metal/build/python_env_vllm/bin/activate
export PYTHONPATH=$HOME/tt-metal/tools:$HOME/tt-metal/ttnn:$PYTHONPATH
MESH_DEVICE=N150 vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 --port 20000
```

[🚀 Serve Llama-3.1-8B-Instruct (vLLM, no container)](command:tenstorrent.vllm.serveCloudN150)

The `PYTHONPATH` line is there because the pod's vLLM environment finds `ttnn` through a
path file but not tt-metal's `tracy` package beside it; without it `import ttnn` fails
with `No module named 'tracy'` and the server never starts.

> **If `vllm` stops with `operator torchvision::nms does not exist`,** the image's
> `torchvision` does not match its `torch` (found in tt-code-server v0.63.0: torch 2.14.1
> with torchvision 0.26.0). Until the image is rebuilt with the fix, install the matching
> build into the venv once:
> ```bash
> uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torchvision==0.29.*"
> ```

`MESH_DEVICE` names the board the plugin should open: `N150` on the pod, `N300`, `T3K`,
`P100`, `P150` and so on elsewhere. `--max-model-len 32768` is the plugin's documented
setting for Llama 3.1 8B on an n150. `--port 20000` matches `tt serve`, so the Connect
Software lessons find it in the same place.

The first run downloads the weights (16 GB) and compiles the kernels for your chip; both
are cached, so the next start is minutes. The server is ready when the log says
`Application startup complete` and `Uvicorn running on http://0.0.0.0:20000`.

> **⚠️ The endpoint has no password**, and `vllm serve` binds every interface by default.
> Add `--host 127.0.0.1` to keep it on the pod, or set `--api-key` to require a key. On
> the pod the only route in is the pod's own network, which is yours; treat any other host
> with the same care as in [Serve a Model](command:tenstorrent.lesson.show?["interactive-chat"]).

## Ask It a Question

In a second terminal:

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

[💬 Ask the model](command:tenstorrent.vllm.curlCloud)

The text is in `choices[0].message.content`. The endpoint speaks the standard OpenAI API,
so any client works. `tt launch` in the next group finds servers `tt` started; for this
one you give the client the URL yourself (`http://127.0.0.1:20000/v1`), which each Connect
Software lesson shows.

## Stop

Ctrl-C in the terminal running `vllm serve`. There is no container to clean up.

## Go Deeper

`tt-inference-server` can also run without Docker (`run.py --local-server`), pointed at a
built tt-metal and this same venv; that is how `tt serve` would run on a host without
Docker, with its model specs and warm-up. It is covered in
[tt-inference-server by Hand](command:tenstorrent.lesson.show?["tt-inference-server"]).
The plugin's own options (`MESH_DEVICE`, `--additional-config '{"tt": {...}}'`, vision
models, speculative decoding) are in the
[vllm-tt-plugin README](https://github.com/tenstorrent/vllm-tt-plugin#readme).

**Next:** [Launch Software →](command:tenstorrent.lesson.show?["launch-software"])
