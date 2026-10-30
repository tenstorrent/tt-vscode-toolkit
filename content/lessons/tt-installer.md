---
id: tt-installer
title: Install and Upgrade with tt
description: >-
  Install the tt command line with uv, see what your machine has, preview and
  apply the tested golden set of driver, firmware and tools, and keep tt itself
  current.
category: setup
tags:
  - installation
  - setup
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

# Install and Upgrade with `tt`

`tt` is the Tenstorrent command line. It is the one tool you use to install the software
stack, keep it current, look at your hardware, serve models and connect apps to them.
Under the hood it runs [`tt-installer`](https://github.com/tenstorrent/tt-installer),
`tt-smi`, `tt-flash`, `tt-inference-server` and `tt-studio` for you, each at a version
Tenstorrent has tested together.

A TT-QuietBox 2 ships with `tt` installed, so there this lesson is about **upgrading**.
The cloud pod does not have `tt`, so there it starts with installing it, and stops before
the upgrade step.

## Install `tt`

`tt` is the PyPI package named **`tenstorrent`**, installed as a tool with
[uv](https://docs.astral.sh/uv/). (The unrelated `tt-cli` package on PyPI is a time
tracker.) The cloud pod and the QuietBox both have `uv` already.

```bash
uv tool install --python 3.12 tenstorrent
```

[📦 Install tt](command:tenstorrent.setup.installTt)

`--python 3.12` matters. On the cloud pod the default interpreter is Python 3.10 (the
tt-metal environment), and with it `uv` resolved an old `tt` (1.0.1) instead of the
current release. With 3.12 it installed 1.2.1. `uv` downloads 3.12 if the machine does
not have it.

If you get `tt: command not found` afterwards, `~/.local/bin` is not on your `PATH` in
this shell. Run `uv tool update-shell`, or add `export PATH="$HOME/.local/bin:$PATH"` to
`~/.bashrc`, and open a new terminal.

If `tt --help` does not list `device`, `model`, `serve` and `update`, another program
named `tt` is earlier on your `PATH`. `which -a tt` lists every match in the order the
shell tries them. Put `~/.local/bin` first.

### Cloud pod: point `tt` at the tools it already has

`tt` normally installs its own pinned copies of `tt-smi` and `tt-flash` during
`tt update`. The pod already has both (tt-installer put them in `~/.tt-installer-venv`),
and `tt update` is not for the pod. Tell `tt` where they are once:

```bash
tt config set tools.override.tt-smi ~/.tt-installer-venv/bin/tt-smi
tt config set tools.override.tt-flash ~/.tt-installer-venv/bin/tt-flash
```

[🔗 Use the pod's tt-smi and tt-flash](command:tenstorrent.setup.useInstalledTools)

After this, `tt device status` and `tt model list` work on the pod, and `tt update
--dry-run` shows those two rows as `external`. Skip this on a QuietBox or your own
machine: there, `tt update` installs the pinned copies.

## See What You Have

```bash
tt --version
tt device status
```

[🔍 Device status](command:tenstorrent.setup.deviceStatus)

`tt device status` lists one row per chip. A QB2 has four, on two p300 boards:

```
                 Tenstorrent devices (driver: TT-KMD 2.11.0)
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ # ┃ Board ┃ Bus ID       ┃ Temp    ┃ Power  ┃ AIClk      ┃ Voltage ┃ DRAM ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ 0 │ p300c │ 0000:01:00.0 │ 55.0 °C │ 74.0 W │ 1337.0 MHz │ 0.81 V  │ True │
│ 1 │ p300c │ 0000:02:00.0 │ 59.9 °C │ 49.0 W │ 1343.0 MHz │ 0.82 V  │ True │
│ 2 │ p300c │ 0000:03:00.0 │ 45.9 °C │ 13.0 W │ 800.0 MHz  │ 0.73 V  │ True │
│ 3 │ p300c │ 0000:04:00.0 │ 47.9 °C │ 18.0 W │ 800.0 MHz  │ 0.73 V  │ True │
└───┴───────┴──────────────┴─────────┴────────┴────────────┴─────────┴──────┘
```

In that capture, chips 0 and 1 were running a model (high power, 1.3 GHz clock) and
chips 2 and 3 were idle (800 MHz). The clock column is the quickest way to see which
chips are in use. On the cloud pod you get one row, an n150.

## Preview an Upgrade

Tenstorrent publishes a **golden set**: the versions of the kernel driver, firmware,
HugePages setup, `tt-smi`, `tt-flash`, `tt-inference-server` and `tt-studio` that have
been tested together. `tt update` moves your machine onto it. Preview first. This changes
nothing, and it is safe everywhere, including the cloud pod:

```bash
tt update --dry-run
```

[👀 Preview upgrade](command:tenstorrent.setup.updateDryRun)

```
  Update plan (goldens: bundled supplement.toml + golden.json v3.0.0 (cached))
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ component       ┃ kind     ┃ installed       ┃ golden           ┃ action     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ tt-flash        │ uv-tool  │ 4.0.0           │ 4.0.0            │ up-to-date │
│ tt-inference-s… │ git-venv │ v0.25.0         │ v0.25.0          │ up-to-date │
│ tt-model        │ uv-tool  │ a7aa9cefdd3086… │ a7aa9cefdd3086b… │ up-to-date │
│ tt-smi          │ uv-tool  │ 6.6.0           │ 6.6.0            │ up-to-date │
│ tt-studio       │ git-venv │ —               │ v2.12.0          │ optional   │
│ system-stack    │ system   │ —               │ hugepages 1.4.1, │ converge   │
│                 │          │                 │ kmd 2.11.0, ...  │            │
└─────────────────┴──────────┴─────────────────┴──────────────────┴────────────┘
```

Read the `action` column:

| Action | Meaning |
|---|---|
| `up-to-date` | Nothing to do. |
| `install` / `update` | `tt update` will install or upgrade it. |
| `optional` | Fetched the first time a command needs it (`tt-studio` here). Add `--include-lazy` to fetch it now. |
| `converge` | The system layer (driver, firmware, HugePages) is checked and brought to the golden versions by `tt-installer`. |

**On the cloud pod, stop here.** This is the pod's plan after the override step above
(captured in the tt-code-server image, `tt` 1.2.1):

```
  Update plan (goldens: bundled supplement.toml + golden.json v4.0.0 (cached))
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ component           ┃ kind     ┃ installed ┃ golden               ┃ action   ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ tt-flash            │ uv-tool  │ —         │ 4.1.0                │ external │
│ tt-inference-server │ git-venv │ —         │ v0.25.0              │ optional │
│ tt-model            │ uv-tool  │ —         │ 713eeb2f33354881bec… │ install  │
│ tt-smi              │ uv-tool  │ —         │ 6.7.0                │ external │
│ tt-studio           │ git-venv │ —         │ v2.12.0              │ optional │
│ system-stack        │ system   │ —         │ hugepages 1.4.1, kmd │ converge │
│                     │          │           │ 2.11.0, ...          │          │
└─────────────────────┴──────────┴───────────┴──────────────────────┴──────────┘
```

`external` is the override from the step above. `install` and `optional` tools are put
under your home directory by `tt` the first time a command needs them. The
`system-stack` row is the driver and firmware, which the pod manages for you; that is
why the upgrade button below refuses to run there.

## Upgrade

```bash
tt update
```

[⬆️ Upgrade to the golden set](command:tenstorrent.setup.update)

`tt update` runs in three phases: it checks the golden manifest, installs or upgrades the
tools into isolated environments under `~/.local/share/tenstorrent/`, and then downloads
the matching `tt-installer` release and runs it. The last phase uses `sudo`, so it asks
for your password. It also asks before flashing firmware, because flashing resets the
devices. Stop any model you are serving first (`tt model ps` shows what is running).

> **Firmware only moves forward.** A plain `tt update`, with no `--force` and no version
> argument, flashes a device only when its firmware is older than the golden bundle. A box
> that is already ahead is left alone. `--force`, or naming an older installer version, is
> what allows a downgrade. Run `tt update` on the machine itself, not inside a `docker run`
> shell: it reloads the kernel driver and can install Docker through systemd.

> **⚠️ Reboot after a first install, or if the driver changed.** `tt update` never reboots
> for you. If it installed the kernel driver or changed the HugePages configuration, run
> `sudo reboot` before you serve anything. On a first install it also adds you to the
> `docker` group, which takes effect at your next login.

> `tt update` ignores older copies of tools that a previous `tt-installer` run left in
> `~/.local/lib/` (for example `tt-inference-server` and `tt-studio`) and warns you about
> them. It manages its own pinned copies under `~/.local/share/tenstorrent/tools/`. The
> warning prints the `rm -rf` commands that reclaim the space.

## Upgrade `tt` Itself

`tt update` brings the system and its tools onto the golden set. `tt self update`
upgrades the `tt` command. Run both when you want to be current.

```bash
tt self update --check    # is there a newer release?
tt self update
```

[🔄 Upgrade tt](command:tenstorrent.setup.selfUpdate)

## Installing on a Fresh Machine

You need Ubuntu 22.04 or 24.04, `sudo`, and a network connection. Install `uv`, then
`tt`, then let `tt update` install everything else:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # uv and tt install here; this makes them work in this terminal
uv tool install tenstorrent
uv tool update-shell        # puts ~/.local/bin on PATH for future terminals
tt update
```

`tt update` installs the kernel driver, firmware tooling and HugePages configuration as
root, so reboot afterwards.

## Using `tt-installer` Directly

`tt update` runs `tt-installer` for you, and for almost everyone that is the right way to
use it. `tt update <VERSION>` runs a specific installer release instead (and implies
`--force`, which allows downgrades). The installer's own options, container mode, the
package repository it configures and the `tt-metalium` containers are covered in
[tt-installer by Hand](command:tenstorrent.lesson.show?["tt-installer-by-hand"]) under
Go Deeper.

## Check Your Work

```bash
tt --version
tt update --dry-run      # every row should say up-to-date (cloud pod: tools rows may say install; that is fine)
tt device status         # one row per chip, all DRAM True
tt model list            # models for your hardware
```

If `tt device status` shows fewer chips than you expect, go to
[Hardware Detection](command:tenstorrent.lesson.show?["hardware-detection"]) for the
diagnosis steps.

**Next:** [Hardware Detection →](command:tenstorrent.lesson.show?["hardware-detection"])
