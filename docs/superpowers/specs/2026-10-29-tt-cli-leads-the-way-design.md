# tt CLI Leads the Way — design

Branch: `updates/tt-cli-leads-the-way`. Written 2026-10-29 from a brainstorm with Taylor
Singletary, after the same rework landed in tt-quietbox2-guide (PR #27) and
tenstorrent.github.io.

## Intent

The toolkit's lessons were written when installing, serving and connecting software to a
model each meant a different tool run by hand: `tt-installer`'s curl-pipe-bash,
`tt-inference-server`'s `run.py`, a vLLM checkout, `tt-studio`'s own scripts. The `tt` CLI
(PyPI package `tenstorrent`, 1.2.1 at time of writing) now does all of that with one
vocabulary: `tt update`, `tt device status`, `tt model list|info|pull|serve`, `tt launch`.
The lessons, the buttons that run them, and the sidebar that orders them should teach that
vocabulary first and keep the by-hand routes for people who need to go deeper.

The software landscape has also moved since the project started: tt-awesome curates
models and projects, the developer portal has a blog, and the cloud pod ships a built
tt-metal with the vLLM plugin already installed. The corpus should point at those rather
than describe a world that no longer exists.

People learn in this toolkit by making things go. They read a lesson in the webview and
either type the command into the terminal below or press the button that types and runs
it for them. Both must stay in step, and every command must work where the reader is.

## Where readers are

Three environments, and a lesson must say which steps apply to each.

| | Cloud pod (`tt-code-server`) | TT-QuietBox 2 | Own machine |
|---|---|---|---|
| chips | one n150 | four Blackhole (two p300) | varies |
| `tt` installed | **no** | yes (1.2.0 on the test box) | maybe |
| `uv` | yes | yes | maybe |
| Docker | **no** | yes | maybe |
| firmware / driver changes | **not allowed** (WELCOME.md) | `tt update` | `tt update` |
| built tt-metal | `~/tt-metal` (v0.69.0), `python_env` active in every shell | no (container only) | maybe |
| vLLM + TT plugin | `~/tt-metal/build/python_env_vllm`, ready | inside `tt serve`'s container | via `tt serve` |
| Forge / XLA / tt-lang | `~/tt-forge-venv`, `~/tt-lang-venv` | install | install |
| model cache | `HF_HOME=~/models` | `~/.cache/huggingface` + shipped `~/data/tt-cache` | default |
| persistence | `/mnt/data` only | disk | disk |
| sudo | passwordless | password | varies |

Source for the cloud column: `~/code/tt-code-server/tt-code-server/Dockerfile` and
`WELCOME.md` on the QuietBox checkout (image `ghcr.io/tenstorrent/tt-code-server/code-server:v0.63.0`).
WELCOME.md still says tt-metal v0.65.1; the Dockerfile builds v0.69.0. Lessons should read
the version from the box (`git -C ~/tt-metal describe --tags`) rather than state one.

Most cloud users are there to explore the hardware (path 4). Serving without containers is
a branch, not the main event, but it must exist so nobody is left with a button that cannot
work.

## Goals

1. Four golden paths, in this order, each a lesson group at the top of the sidebar:
   **Set Up**, **Serve Models**, **Connect Software**, **Build on the Stack**.
2. A **Go Deeper** group below them for the by-hand routes: `tt-inference-server` and
   `run.py` directly, vLLM in a venv, building tt-metal, monkeypatching, ttsim, bounties,
   mixing versions and environments.
3. The four series (Cookbook, CS Fundamentals, LLM from Scratch, Custom Training) keep
   their own groups beside Go Deeper, unchanged in content except where a setup step they
   depend on has moved to `tt`.
4. Every install, model, serve and vLLM button runs a `tt` command where `tt` has one, and
   the lesson text shows the same command.
5. Every command id follows `tenstorrent.<feature>.<action>`. Hard cut: old ids are removed,
   not aliased.
6. Every lesson that changes is run, button by button, on the QuietBox and in the cloud
   image before it is called done.
7. tt-awesome (model browser, curated projects) and the developer portal (blog) are linked
   where a reader would next want them.

## Non-goals

- No new lesson-system machinery. The tree is ordered by `content/lesson-registry.json`;
  buttons are `[text](command:id)` links rendered by `CommandButtonRenderer`; command text
  lives in `TERMINAL_COMMANDS`. All three stay as they are.
- No rewrite of the four series beyond re-pointing their setup steps.
- No attempt to make `tt update` flash firmware in the cloud, or to make `tt serve` run
  without Docker. The cloud serving branch uses what the pod already has.
- No new content for `tt agent`: it is not in a release yet. One sentence pointing at it,
  as the QB2 guide does.
- Not renaming product names in prose to satisfy Vale's Tenstorrent style (QB2, TTNN,
  lowercase `tt-inference-server` stay). Decided on the QB2 guide; same call here.

## Anti-goals

- A button that drops a command into a cloud terminal and fails because it needs Docker,
  sudo-flashing, or a `tt` that is not installed. Every button either works in every
  environment or its lesson says in plain words which environment it is for, before the
  button.
- A lesson that reads as finished while its proof was never run. Each lesson's CLAUDE.md
  log entry says where it was run and what was not.
- Prose that frames `tt` against the old tools ("X, not Y"). State what `tt` does. The old
  route is described on its own terms in Go Deeper.

## Constraints

- **Interactive.** Work goes one lesson group at a time, with Taylor reviewing each before
  the next. Nothing is batched across groups.
- **Test rig.** The cloud image is pulled on the QuietBox and run with one chip mapped in
  (`--device /dev/tenstorrent/N` for a leased chip, hugepages mounted), and the lesson's
  commands are executed inside it. QB2-native lessons run on the host. Both behind a
  `gozer` lease; two chips at most, when free.
- **Version.** `package.json` version bumps with every change that touches the extension
  (global CLAUDE.md rule), and after any bug Taylor finds in testing.
- **Writing.** Global CLAUDE.md "plain English" rules and the sweep list. Vale runs with the
  repo's `.vale.ini`; style-only hits are ignored as above.
- **Docs site.** `scripts/lib/command-map-parser.js` reads `terminalCommands.ts` to print
  the command under each button on the published site. The rename must keep it parsing;
  its test stays green.
- **Hardware.** Nothing opens `/dev/tenstorrent/*` without a `gozer` lease.

## Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Happy-path model, n150 | `Llama-3.1-8B-Instruct` | `tt model info` reports `on n150: vLLM, COMPLETE, 65536 context`. Tool calling works for path 3. |
| Happy-path model, QB2 | `Qwen3-32B` | Ships on the box with weights and compiled cache. Same choice as the QB2 guide. |
| Cloud serving route | `source ~/tt-metal/build/python_env_vllm/bin/activate`, then vLLM's OpenAI server with the TT plugin, one chip | The venv is already built in the image. `run.py --local-server` also exists (needs `--tt-metal-home` and the venv) and is the Go Deeper variant. |
| Command namespacing | `tenstorrent.<feature>.<action>`, hard cut | Convention already used by ttsim and monkeypatch work. Aliases rejected by Taylor. |
| Namespaces | `setup`, `model`, `launch`, `inferenceServer`, `vllm`, `metal`, `forge`, `xla`, `ttlang`, `ttsim`, `monkeypatch`, `cookbook`, `lfs`, `training`, `apps`, `lesson`, `viz` | One per feature or stack; `lesson` and `viz` for the extension's own UI. |
| Sidebar | Set Up · Serve Models · Connect Software · Build on the Stack · Go Deeper · the four series | Matches the four golden paths; series stay beside, not under, Build on the Stack. |
| Old lessons | Rewritten in place where they are a golden-path step; moved to Go Deeper where they are the by-hand route; deleted only if nothing of them survives | Keeps lesson ids and site URLs where the lesson still exists. |
| `tt agent` | one sentence, with the "check `tt --help`" test | Not released. |

## The lessons, group by group

Each group is one review cycle with Taylor. Lesson ids in `content/lesson-registry.json`;
new ids marked *new*.

### Set Up

1. **`arrive`** *(new)* — Where am I? Detects the environment and says which of the next
   lessons apply. `uv --version`, `which -a tt`, `tt-smi -s` chip count, `ls ~/tt-metal`,
   `docker --version`. Explains the three environments in the table above in a few lines
   each. The cloud pod's WELCOME.md rules (no firmware changes, `/mnt/data`) are repeated
   here so a reader meets them before any button.
2. **`tt-installer`** → rewritten as **Install and Upgrade with `tt`**. `uv tool install
   tenstorrent`, `tt --version`, `tt update --dry-run`, `tt update`, `tt self update`.
   Cloud readers stop after the dry run; the lesson says so above the `tt update` button,
   and the button's template refuses on a cloud marker (see "Environment guard" below).
   The curl-pipe-bash installer moves to Go Deeper as `tt-installer-by-hand`.
3. **`hardware-detection`** — `tt device status`, `tt device info`, `tt smi`, then the
   `tt-smi -s` JSON it already teaches. Reorder so `tt` is first.
4. **`verify-installation`** — one `tt`-first check per environment: `tt update --dry-run`
   all up to date; `python -c "import ttnn"` in whichever env holds it (cloud:
   `python_env`; QB2: `tt-metalium` container). Keeps its existing container notes.

### Serve Models

5. **`download-model`** → **Find and Pull a Model**. `tt model list`, `tt model list
   --catalog`, `tt model info`, `tt model search` with the "I want to…" table from the QB2
   guide, `tt model login`, `tt model pull`. Community bundles carry the same "not reviewed
   by Tenstorrent" warning and the tt-awesome model browser link.
6. **`interactive-chat`** → **Serve a Model**. `tt serve <model> --dry-run`, `tt serve`,
   `tt model ps|logs|curl|stop`. QB2 shows Qwen3-32B; n150 shows Llama-3.1-8B-Instruct; the
   lesson picks by `tt device status` output. Carries the unauthenticated-endpoint warning
   from the QB2 guide.
7. **`api-server`** → **Serve Without Containers** (cloud branch). Activate
   `python_env_vllm`, `vllm serve meta-llama/Llama-3.1-8B-Instruct` with the plugin's
   flags, same `/v1` endpoint, same `curl` test. Says plainly that this is the route for
   the cloud pod and any host without Docker.
8. **`image-generation`**, **`video-generation-ttmetal`** — re-pointed at `tt serve
   FLUX.1-schnell` / `tt serve Wan2.2-…` where `tt model list` offers them for the reader's
   hardware; the by-hand route they describe today moves into their own "Go deeper"
   footer.

### Connect Software

9. **`launch-software`** *(new)* — `tt launch list`, Open WebUI (with the
   `WEBUI_AUTH=False` warning), terminal agents (`qwencode` and `claude` are already in the
   cloud pod), `tt serve --studio`. The Qwen3-32B thinking-and-tool-calls warning carries
   over for QB2 readers.
10. **`coding-assistant`**, **`qb2-local-agents`**, **`qb2-openclaw-assistant`** — the
    server-start steps become `tt serve` and `tt launch`; the agent-specific content
    stays.

### Build on the Stack

11. **`explore-metalium`**, **`tt-lang-intro`**, **`forge-image-classification`**,
    **`tt-xla-jax`**, **`cookbook-overview`** — setup steps re-pointed at what the
    environment has (cloud venvs by name; QB2 container or `tt`-installed tool), and a
    "where to go next" footer with tt-awesome projects in the same area and the relevant
    developer-portal posts. Content otherwise unchanged.

### Go Deeper

12. **`tt-inference-server`**, **`vllm-production`**, **`build-tt-metal`**,
    **`monkeypatch-ttnn`**, **`ttsim-*`**, **`bounty-program`** — move here. Their
    setup steps say which `tt`-managed thing they replace (`tt serve` runs this `run.py`
    for you; this is how to run it yourself, pin chips, choose a version).
13. **`mix-and-match`** *(new, short)* — how `tt`'s pinned tools under
    `~/.local/share/tenstorrent/tools/` relate to a tt-metal you built, the cloud pod's
    venvs, and a container; when to use which; how to find the version a release pins.

### Applications and the series

`animatediff-video-generation` and `qb2-video-generation` stay in an **Applications**
group below Go Deeper; `coding-assistant`, `qb2-local-agents` and `qb2-openclaw-assistant`
move to Connect Software. The four series keep their groups. All of these get setup steps
re-pointed only.

## Calls made after the builder check

These were open in the first draft. Each is my call unless marked otherwise.

- **`tt` version.** The QuietBox's `tt` is an editable install of `~/code/tt-cli`. It is
  pinned to the `v1.2.1` tag (Taylor's call, 2026-10-29) so the box matches what
  `uv tool install tenstorrent` gives a reader. Every flag in a lesson is verified against
  that tag. `tt agent` is on `main` only and stays a one-sentence pointer.
- **Model names** are read from `tt model list` on the box at build time, never typed from
  memory. Where a model is not offered for the reader's hardware, the lesson says so above
  the button and points at the Go Deeper route.
- **Cloud readers install `tt`** with `uv tool install tenstorrent` in the Set Up group.
  Read-only commands (`tt device status`, `tt model list|info|search`, `tt update
  --dry-run`) work there. The environment guard covers only `tt update` proper and
  anything that pulls a container. `tt device status` reads telemetry and does not take
  the chip.
- **The guard lives in the TypeScript handler**, never in the template. Templates stay the
  bare command, so the docs-site parser prints what the reader types.
- **Choosing a model in a static lesson:** two buttons, one per happy-path model, with the
  text saying which to press from the `tt device status` output. No shell conditionals in
  templates.
- **"Button by button" in the cloud image** means: commands the pod can run are run there;
  commands it cannot run are verified to refuse with the guard's message there and are run
  for real on the QuietBox host. Both results go in the CLAUDE.md log entry.
- **Lesson ids** stay where a lesson is rewritten in place (`api-server` keeps its id with
  the new title). Moved lessons keep their ids. New category ids: `setup`, `serve`,
  `connect`, `build`, `deeper`, `applications` (existing); `first-inference`, `serving`,
  `advanced` are dropped from the registry. `welcome` is left alone.
- **The new command-link test** checks two things: every `command:tenstorrent.*` link in
  `content/` resolves to an id in `contributes.commands`; and every `TERMINAL_COMMANDS`
  key is used by some handler. It does not require every registered id to appear in a
  lesson (UI commands are not lesson buttons). The ~24 ids registered in `extension.ts`
  but absent from `package.json` are inventoried in the first commit and either added to
  `contributes.commands` or deleted.
- **UI commands** (`showLesson`, `refreshLessons`, `filterLessons`,
  `toggleShowAllLessons`, `showCommandMenu`, …) become `tenstorrent.lesson.*`. The hard
  cut applies to them too; a user with one of the old ids in `keybindings.json` has to
  re-bind, and CHANGELOG says so.
- **Zero chips** (simulator, no card): the arrival lesson names it as a fourth case and
  points at the ttsim lessons. `supportedHardware` and `validatedOn` front matter follow
  the registry's existing vocabulary, filled per lesson by the builder from what was
  actually run.
- **Cloud vLLM flags** come from the image's Dockerfile and `vllm-tt-plugin`'s docs, then
  a live run in the rig. No command from the cloud team is assumed to exist.

## Commands

All 123 ids in `package.json` are renamed in one commit per group, touching `package.json`,
`extension.ts` registrations, `TERMINAL_COMMANDS` keys and every `command:` link in
`content/`. The install, model, serve and vLLM set (48 today) also gets new templates.

New test: every `command:tenstorrent.*` link under `content/` resolves to an id in
`package.json`'s `contributes.commands`, and every registered id is referenced from at
least one lesson or the extension UI. Written before the first rename and run in
`npm test`.

**Environment guard.** Some commands must not run in the cloud pod (`tt update` proper,
anything that pulls a container). The handler checks a cloud marker and prints one line
explaining why instead of running. Marker choice is left to the builder; candidates are
`TT_CODE_SERVER_*` env vars from the image, `/home/user/WELCOME.md`, or absence of
`docker`. Prefer a positive marker the image sets over absence of a tool.

## tt-awesome and the developer portal

- Model browser: `https://docs.tenstorrent.com/tt-awesome/models/`, linked from Find and
  Pull a Model and from the community-bundle warning.
- Projects: linked per stack from the Build on the Stack footers, drawn from the local
  `~/code/tt-awesome` checkout's categories at build time, not hand-copied (so the list is
  the curated one, and a lesson is not a second copy to keep in sync). The builder decides
  whether this is a build-time include or a plain link to the category page.
- Developer blog: `https://tenstorrent.com/developers/blog`, linked from Connect Software
  (the Tenstorrent CLI post) and Build on the Stack where a post covers the topic.

## Left to the builder

- Exact wording of every template and lesson; the arrival lesson's detection script.
- The cloud marker for the environment guard.
- Whether tt-awesome project lists are included at build time or linked.
- Test design for the command-link resolver and the parser after the rename.
- Order of commits inside a group.

## What done looks like

Per group: lessons rewritten, buttons renamed and retemplated, the group's commands run
on the QuietBox and in the cloud image with the output pasted into the lesson where the
lesson shows output, `npm test` green, version bumped, CLAUDE.md log entry stating what
was run where and what was not, Taylor's review. The branch is one PR at the end, or one
per group if Taylor wants to merge as we go.
