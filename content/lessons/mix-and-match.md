---
id: mix-and-match
title: Mix and Match Versions
description: >-
  Where tt keeps its pinned tools, how they relate to a tt-metal you built,
  the cloud pod's venvs and a container, when each applies, and how to find
  the exact commits a release was tested with.
category: deeper
tags:
  - versions
  - environments
  - tt-cli
  - tt-inference-server
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
estimatedMinutes: 10
order: 9
---

# Mix and Match Versions

The golden path uses one set of versions: the ones `tt` pins. This lesson is for when you
need a different one. A newer tt-metal for a model that just landed, an older
tt-inference-server release that still serves a model the newest dropped, or a tt-metal you
built yourself beside the stack `tt` manages. Four things can hold a copy of the software,
and knowing which one a command uses is most of the work.

## What `tt` Manages

`tt` keeps the tools it runs under `~/.local/share/tenstorrent/`:

```bash
tt update --dry-run                              # every component, installed vs golden
cat ~/.local/share/tenstorrent/installed.toml    # what tt installed, where, when
ls ~/.local/share/tenstorrent/tools/             # one directory per tool and version
```

[📋 Update plan](command:tenstorrent.setup.updateDryRun)
[📂 What tt installed](command:tenstorrent.setup.listTools)

On a TT-QuietBox 2 (2026-10-29):

```
[tools.tt-inference-server]
version = "v0.25.0"
path = "/home/ttuser/.local/share/tenstorrent/tools/tt-inference-server-v0.25.0/src/run.py"
installed_at = "2026-10-23T23:10:30+00:00"
```

```
tools/
  tt-flash/
  tt-inference-server-v0.18.0/
  tt-inference-server-v0.22.0/
  tt-inference-server-v0.25.0/      # src/ (the checkout) + venv/ (packaging, pyyaml)
  tt-model/
  tt-smi/
```

Three tt-inference-server versions sit side by side because each `tt` release pins one and
earlier pins stay. `tt serve` runs the pinned one; `installed.toml` says which. The
`golden` column of `tt update --dry-run` is the set of versions `tt` was released against:
tt-smi, tt-flash, the kernel driver, firmware and a `metal-version`. `tt update` moves the
machine to that set and nothing else.

> `tt update --dry-run` may warn about `~/.local/lib/tt-inference-server` and
> `~/.local/lib/tt-studio`. Those are copies an earlier `tt-installer` left; `tt` does not
> use or update them. They are safe to delete once nothing of yours points at them.

## The Four Places Software Lives

| Where | What it is | Who chose the version |
|---|---|---|
| `~/.local/share/tenstorrent/tools/` | `tt`'s pinned copies | the `tt` release you have |
| a container (`tt serve`, `tt-metalium`) | an image with tt-metal and vLLM built in | the model spec, by image tag |
| `~/tt-metal` and its venvs | a built source tree | you, or the cloud image |
| a venv you made | one tool's PyPI wheel | you |

Only one of these answers at a time. `which -a tt`, `which python`, and the `docker image`
line of a `tt serve --dry-run` tell you which.

**The cloud pod** is the third row. `~/tt-metal` is a built tree (`cd ~/tt-metal && git
describe --tags` prints its version), `python_env` is active in every shell, and the
other stacks are separate venvs: `~/tt-metal/build/python_env_vllm`, `~/tt-forge-venv`,
`~/tt-lang-venv`. There is no container and no `tt update`; what the image shipped is what
you have, and `/mnt/data` is the only place that outlives the pod.

**A TT-QuietBox 2** is the first two rows. The host has `tt` and its tools; tt-metal lives
in containers (`tt serve`'s image, or the `tt-metalium` wrapper). There is no `~/tt-metal`
unless you build one.

## How a Release Pins Its Commits

`tt serve` does not build anything. The model spec names an image, and the image tag
carries the commits it was built from:

```
ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.2.0-9b67e09-a91b644
                                                                                      ^^^^^ ^^^^^^^ ^^^^^^^
                                                                                      spec  tt-metal vLLM
```

The same three fields are in the spec file itself:

```bash
cd ~/.local/share/tenstorrent/tools/tt-inference-server-v0.25.0/src
python3 -c "
import json; d = json.load(open('release_model_spec.json'))
def walk(o):
    if isinstance(o, dict):
        if o.get('model_name') == 'Llama-3.1-8B-Instruct' and o.get('device_type') == 'N150': return o
        for v in o.values():
            if (r := walk(v)): return r
    if isinstance(o, list):
        for v in o:
            if (r := walk(v)): return r
m = walk(d['model_specs'])
print({k: m[k] for k in ('tt_metal_commit', 'vllm_commit', 'version', 'docker_image')})
"
```

```
{'tt_metal_commit': '9b67e09', 'vllm_commit': 'a91b644', 'version': '0.2.0',
 'docker_image': 'ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.2.0-9b67e09-a91b644'}
```

`tt serve <model> --dry-run` prints the `docker image` line without the script. Those
commits are the pair that model was validated on. A tt-metal you build at any other
commit is a different pair, and the model may or may not run on it.

## When to Leave the Golden Path

| You want | Do |
|---|---|
| a model the current release serves | nothing; `tt serve` |
| a model only an older release served | `cd` into that `tools/tt-inference-server-<ver>/src` and run `run.py` yourself ([tt-inference-server by Hand](command:tenstorrent.lesson.show?["tt-inference-server"])) |
| a newer tt-metal than any image has | build it ([Build TT-Metalium from Source](command:tenstorrent.lesson.show?["build-tt-metal"])), then `run.py --local-server --tt-metal-home ~/tt-metal` |
| a community bundle built for a specific chip | `tt serve ns/name`; the bundle carries its own engine ([Find and Pull a Model](command:tenstorrent.lesson.show?["download-model"])) |
| a change inside TT-NN without a fork | [Monkeypatching TT-NN](command:tenstorrent.lesson.show?["monkeypatch-ttnn"]) |
| no hardware at all | [ttsim](command:tenstorrent.lesson.show?["ttsim-twenty-and-ten"]) |

One rule covers all of them: a `tt` command uses `tt`'s copy, a `python` command uses
whichever environment is active. `tt config set tools.override.<tool> <path>` points `tt`
at a copy of yours when you want both to agree.

**Next:** [tt-inference-server by Hand →](command:tenstorrent.lesson.show?["tt-inference-server"])
