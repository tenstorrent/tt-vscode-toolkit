---
id: arrive
title: Where Am I?
description: >-
  Find out which Tenstorrent environment you landed in and which setup steps
  apply to it: the cloud pod, a TT-QuietBox 2, or a machine of your own.
category: setup
tags:
  - setup
  - cloud
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
  - simulator
status: validated
validatedOn:
  - n150
  - p300c
estimatedMinutes: 3
order: 1
---

# Where Am I?

This toolkit teaches by making things go. You read a step, then either type the command
into the terminal below or press the button that types and runs it for you. Before the
first button, it helps to know where you are, because three kinds of machine run these
lessons and not every step applies to each.

| | Tenstorrent Cloud pod | TT-QuietBox 2 | Your own machine |
|---|---|---|---|
| Chips | one n150 | four Blackhole (two p300 boards) | whatever you installed |
| `tt` CLI | install it (next lesson) | installed | install it |
| Docker | no | yes | your call |
| Firmware and driver updates | **not allowed** | `tt update` | `tt update` |
| tt-metal source | built at `~/tt-metal`, its Python env active in every shell | no (lives in a container) | if you built it |
| Where files persist | `/mnt/data` only | disk | disk |

## Two Rules for the Cloud Pod

The pod's welcome note states them, and they bear repeating before any button:

1. **Do not change drivers or firmware.** The pod is a Kubernetes pod with one n150
   attached. Its driver and firmware are managed for you. Buttons that would change them
   refuse to run here and tell you so.
2. **Keep work under `/mnt/data`.** Everything outside it is lost when the pod stops.

## Run the Check

One block answers the questions the next lessons depend on:

```bash
echo "uv:       $(uv --version 2>/dev/null || echo none)"
echo "tt:       $(which -a tt 2>/dev/null | tr '\n' ' ' || true)"
echo "chips:    $(tt-smi -s 2>/dev/null | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["device_info"]))' 2>/dev/null || echo 0)"
echo "tt-metal: $([ -d ~/tt-metal ] && echo ~/tt-metal || echo none)"
echo "docker:   $(docker --version 2>/dev/null || echo none)"
```

[🧭 Where am I?](command:tenstorrent.setup.whereAmI)

## Read the Answer

- **`tt:` is empty** → go to [Install and Upgrade with tt](command:tenstorrent.lesson.show?["tt-installer"])
  and install it with `uv`. On the cloud pod this is the expected state, and that lesson
  has one extra pod step: pointing `tt` at the `tt-smi` the pod already has.
- **`chips: 1`, `tt-metal: /home/user/tt-metal`, `docker: none`** → you are on the cloud
  pod. Set Up and Build on the Stack are yours in full. In Serve Models, use the lesson
  for serving without containers.
- **`chips: 4`, `docker:` prints a version** → a TT-QuietBox 2. Every lesson applies.
- **`chips: 0`** → no card is visible. On a machine that should have one, go to
  [Hardware Detection](command:tenstorrent.lesson.show?["hardware-detection"]) for the
  diagnosis steps. With no card at all, the simulator lessons
  ([ttsim](command:tenstorrent.lesson.show?["ttsim-twenty-and-ten"])) run the same
  kernels in software.

## What Comes Next

The sidebar is in the order to read it. **Set Up** gets `tt` installed and your system
current. **Serve Models** gets a model answering on an HTTP endpoint. **Connect Software**
puts a chat window or a coding agent on that endpoint. **Build on the Stack** is where you
write your own code against the hardware. **Go Deeper** is the same ground by hand, for
when you need to choose versions, pin chips or run without `tt`.

**Next:** [Install and Upgrade with tt →](command:tenstorrent.lesson.show?["tt-installer"])
