# tt CLI Leads the Way — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the toolkit's lesson tree, lessons and command buttons around the `tt` CLI, with every command namespaced `tenstorrent.<feature>.<action>` and every changed lesson proven on the QuietBox and in the cloud image.

**Architecture:** Lessons are markdown under `content/lessons/`, ordered by `content/lesson-registry.json`. A `[text](command:tenstorrent.x)` link becomes a button; `src/extension.ts` registers each id to a handler that runs a template from `TERMINAL_COMMANDS` in `src/commands/terminalCommands.ts`. The docs site (`scripts/lib/command-map-parser.js`) prints those templates under the published buttons. This plan renames every id in one task (Tasks 1–2), adds one environment guard (Task 3), re-cuts the registry (Task 4), then rewrites lessons one sidebar group per task (Tasks 5–10), each proven on hardware before review.

**Tech Stack:** TypeScript VS Code extension, `marked`, mocha + chai (`npm test`, `tsconfig.test.json`), Node for scripts, Docker on the QuietBox for the cloud rig.

**Spec:** `docs/superpowers/specs/2026-10-29-tt-cli-leads-the-way-design.md`

## Global Constraints

- Every command id is `tenstorrent.<feature>.<action>`; namespaces: `setup`, `model`, `launch`, `inferenceServer`, `vllm`, `metal`, `forge`, `xla`, `ttlang`, `ttsim`, `monkeypatch`, `cookbook`, `lfs`, `training`, `apps`, `lesson`, `viz`. Hard cut: no aliases.
- `tt` is verified against the `v1.2.1` tag only. `tt agent` gets one sentence.
- Templates stay the bare command; any guard lives in the TypeScript handler.
- Happy-path models: n150 `Llama-3.1-8B-Instruct`; QB2 `Qwen3-32B`. Model names are read from `tt model list` on the box, never typed from memory.
- One sidebar group per task; Taylor reviews each before the next starts.
- Every lesson change is run on the QuietBox host (QB2 steps) and inside the cloud rig (cloud steps) under a `gozer` lease. Commands the rig cannot run are verified to refuse with the guard's message.
- `package.json` version bumps in every commit that touches `src/` or `package.json`, and after any bug Taylor finds in review.
- Writing follows the global "plain English" rules; Vale product-name hits are ignored.
- Nothing opens `/dev/tenstorrent/*` without `gozer run`/`gozer acquire`.

## Review Focus

1. **A lesson link to a command that no longer exists** renders a dead button. Task 1's resolver test pins it; it must stay in `npm test`.
2. **`tt update` pressed in the cloud pod** must print the guard's one line and run nothing. Task 3's test pins the marker; Task 5's rig step verifies the refusal.
3. **`tt serve` pressed on a machine without Docker** fails mid-boot with a Docker error. Task 6's Serve lesson states the requirement above the button and names the no-container lesson; Task 3 guards container pulls.
4. **The docs-site parser after the rename** must still print each button's command. Task 2 runs `command-map-parser.test.ts` and a site build.
5. **A registry `completionEvents` entry naming an old id** silently stops marking lessons complete. Task 2 renames them with the links and Task 1's test covers `onCommand:` events.

---

## The cloud rig

Image `tt-code-server:local`, built on the QuietBox from `~/code/tt-code-server/tt-code-server` (log `~/logs/tt-code-server-build.log`). To run it with one leased chip:

```bash
gozer acquire --chips 1 --who "claude:vscode-toolkit" --reason "cloud rig for lesson verification"
# TT_VISIBLE_DEVICES gives the chip index N; map that node only
docker run -d --name tt-cloud-rig --device /dev/tenstorrent/$N -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME/rig-data:/mnt/data tt-code-server:local sleep infinity
docker exec -it -u user -w /home/user tt-cloud-rig bash -l
```

Inside, the shell matches the pod: `python_env` active, no `tt`, no Docker. Release the lease (`gozer release`) and `docker rm -f tt-cloud-rig` when a group is done.

---

### Task 1: Command inventory and the link-resolver test

**Files:**
- Create: `test/lesson-tests/command-links.test.ts`
- Create: `scripts/lib/command-inventory.js`
- Modify: `package.json` (`contributes.commands`), `src/extension.ts` (registrations)

**Interfaces:**
- Produces: `buildCommandInventory(repoRoot: string): { packageIds: string[]; registeredIds: string[]; contentLinks: {id: string; file: string}[]; completionEvents: {id: string; lessonId: string}[]; templateKeys: string[]; templateUses: string[] }` in `scripts/lib/command-inventory.js` (CommonJS, pure file reads; `registeredIds` from `registerCommand('tenstorrent.…')` literals and from template-literal loops, which must be rewritten as literals in this task).

- [ ] **Step 1: Write the failing tests** in `test/lesson-tests/command-links.test.ts`: `every command: link in content/ resolves to contributes.commands`; `every onCommand: completionEvent resolves to contributes.commands`; `every registered id is in contributes.commands and vice versa`; `every TERMINAL_COMMANDS key is used by a handler`. Assert with `expect(missing).to.deep.equal([])` so the failure lists the offenders.
- [ ] **Step 2: Run `npx mocha test/lesson-tests/command-links.test.ts`.** Expected: FAIL listing the 28 registered-not-in-package and 13 package-not-registered ids found 2026-10-29.
- [ ] **Step 3: Implement `buildCommandInventory`.** Rewrite any `registerCommand(\`tenstorrent.${…}\`)` loop in `extension.ts` as explicit literals so the inventory sees them.
- [ ] **Step 4: Reconcile.** For each of the 28 ids registered but absent from `package.json`: add to `contributes.commands` if a lesson links it, else delete the registration and handler. For each of the 13 in `package.json` but never registered: register if a lesson links it, else remove from `package.json`. Record both lists in the commit message.
- [ ] **Step 5: Run `npm test`.** Expected: all green, including the existing suites.
- [ ] **Step 6: Commit** `test: command inventory and link resolver; reconcile registrations`.

### Task 2: The rename

**Files:**
- Create: `scripts/rename-commands.js` (one-shot, kept for the record), `scripts/command-rename-map.json`
- Modify: `package.json`, `src/extension.ts`, `src/commands/terminalCommands.ts`, `src/commands/vizCommands.ts`, `content/lesson-registry.json`, every file under `content/` with a `command:` link, `CHANGELOG.md`

**Interfaces:**
- Consumes: `buildCommandInventory` (Task 1).
- Produces: `scripts/command-rename-map.json` as `{ "tenstorrent.runQuickInstall": "tenstorrent.setup.install", … }` covering every id; later tasks use the new ids only.

- [ ] **Step 1: Write the map.** One entry per id from the inventory, new name from the namespace list. UI ids go to `lesson.*`; `showTensixViz` and the viz commands to `viz.*`; existing `ttsim.*` and `monkeypatch.*` keep their names. Ids the `tt` rewrite will delete in Tasks 5–7 (the 48 install/model/serve/vLLM ones) are still renamed here so the cut is one event.
- [ ] **Step 2: Write `scripts/rename-commands.js`:** reads the map, rewrites every occurrence of each old id (as a whole token) in the files listed above, prints a count per file, exits non-zero on any old id left behind in `src/`, `content/` or `package.json`.
- [ ] **Step 3: Run it.** Expected: counts printed, exit 0, `grep -rn "tenstorrent\.[a-z][A-Za-z]*'" src content package.json` returns only namespaced ids.
- [ ] **Step 4: Run `npm test` and `node scripts/build-site.js` (or the repo's docs-site build command).** Expected: green; the site's command map prints a command under every button it did before.
- [ ] **Step 5: CHANGELOG entry** under a new version: "All command ids are now `tenstorrent.<feature>.<action>`. Old ids are removed; re-bind any you use in `keybindings.json`." Bump `package.json` version.
- [ ] **Step 6: Commit** `refactor: namespace every command id (hard cut)`.

### Task 3: Environment guard

**Files:**
- Create: `src/utils/environment.ts`, `test/lesson-tests/environment.test.ts`
- Modify: `src/extension.ts` (the handlers for `setup.update` and every `launch.*` web-UI command)

**Interfaces:**
- Produces: `detectEnvironment(env: NodeJS.ProcessEnv = process.env): 'cloud-pod' | 'host'` and `guardCloud(action: string, env?: NodeJS.ProcessEnv): boolean` (returns `false` and shows one information message `"${action} is not available in the Tenstorrent Cloud pod: it changes firmware or needs Docker. See the Where Am I? lesson."` when the environment is `cloud-pod`).

- [ ] **Step 1: Write failing tests:** `detectEnvironment` returns `'cloud-pod'` when `TT_VLLM_VENV` and `TT_INSTALLER_VENV` are both set (the image's own `ENV` lines), `'host'` when either is missing; `guardCloud` returns `true` on host without showing a message, `false` on cloud-pod with the message text above.
- [ ] **Step 2: Run them.** Expected: FAIL, module not found.
- [ ] **Step 3: Implement.** Marker is the two env vars; nothing is inferred from a missing tool.
- [ ] **Step 4: Run `npm test`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: cloud-pod environment guard`.

### Task 4: Registry re-cut

**Files:**
- Modify: `content/lesson-registry.json`, `src/types/LessonMetadata.ts` (`LessonCategory`), `test/lesson-tests/markdown-validation.test.ts` if it checks categories

- [ ] **Step 1: Write a failing test** in `test/lesson-tests/registry.test.ts`: category ids are exactly `welcome, setup, serve, connect, build, deeper, applications, cookbook, cs-fundamentals, llm-from-scratch, custom-training` in that `order`; every lesson's `category` is one of them; no lesson has category `first-inference`, `serving`, `advanced` or `compilers`.
- [ ] **Step 2: Run it.** Expected: FAIL on the old categories.
- [ ] **Step 3: Edit the registry.** New categories with titles: `Set Up`, `Serve Models`, `Connect Software`, `Build on the Stack`, `Go Deeper`. Move lessons per the spec's groups (Set Up: tt-installer, hardware-detection, verify-installation; Serve: download-model, interactive-chat, api-server, image-generation, video-generation-ttmetal; Connect: coding-assistant, qb2-local-agents, qb2-openclaw-assistant; Build: explore-metalium, tt-lang-intro, forge-image-classification, tt-xla-jax, cookbook-overview; Deeper: tt-inference-server, vllm-production, build-tt-metal, monkeypatch-ttnn, ttsim-twenty-and-ten, ttsim-qemu-bridge, bounty-program; Applications: animatediff-video-generation, qb2-video-generation). Set `order` within each group to the spec's numbering. Add `'setup' | 'serve' | 'connect' | 'build' | 'deeper'` to `LessonCategory` and drop the stale members.
- [ ] **Step 4: Run `npm test`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: sidebar re-cut into the four golden paths`.

### Task 5: Set Up group

**Files:**
- Create: `content/lessons/arrive.md`, `content/lessons/tt-installer-by-hand.md` (moved content, category `deeper`)
- Modify: `content/lessons/tt-installer.md`, `hardware-detection.md`, `verify-installation.md`, `content/lesson-registry.json`, `src/commands/terminalCommands.ts`, `src/extension.ts`, `package.json`

**Interfaces:**
- Consumes: `guardCloud` (Task 3); new ids from Task 2's map.
- Produces: templates `SETUP_INSTALL_TT` (`uv tool install tenstorrent`), `SETUP_UPDATE_DRY_RUN` (`tt update --dry-run`), `SETUP_UPDATE` (`tt update`, guarded), `SETUP_SELF_UPDATE` (`tt self update`), `SETUP_DEVICE_STATUS` (`tt device status`), `SETUP_DEVICE_INFO` (`tt device info`), `SETUP_WHERE_AM_I` (the detection one-liner from Step 2), `SETUP_VERIFY_TTNN_CLOUD` (`python -c "import ttnn; print(ttnn.__file__)"`), `SETUP_VERIFY_TTNN_QB2` (the existing `tt-metalium -c …` template renamed). Ids `tenstorrent.setup.{installTt,updateDryRun,update,selfUpdate,deviceStatus,deviceInfo,whereAmI,verifyTtnnCloud,verifyTtnnQb2}`.

- [ ] **Step 1: Delete** the old `QUICK_INSTALL`, `DOWNLOAD_INSTALLER`, `RUN_INTERACTIVE_INSTALL`, `RUN_NON_INTERACTIVE_INSTALL` templates and handlers; move their lesson text into `tt-installer-by-hand.md` under Go Deeper with the installer commands as plain code blocks (no buttons).
- [ ] **Step 2: Write `arrive.md`** ("Where Am I?"): the three-environment table from the spec in prose, the cloud pod's two rules (no firmware changes, `/mnt/data` persists), one detection block: `uv --version; which -a tt; tt-smi -s | python3 -c "import sys,json;print(len(json.load(sys.stdin)['device_info']),'chips')"; ls -d ~/tt-metal 2>/dev/null; docker --version 2>/dev/null || echo "no docker"`, and a line per outcome saying which lesson is next. Zero chips → the ttsim lessons.
- [ ] **Step 3: Rewrite `tt-installer.md`** as "Install and Upgrade with `tt`" from the QB2 guide's chapter 4, with the cloud paragraph: install `tt`, run the dry run, stop there. Button order: install, `--version`, dry run, update (guarded), self update.
- [ ] **Step 4: Edit `hardware-detection.md`** so `tt device status` and `tt device info` come first, then the existing `tt-smi -s` material. **Edit `verify-installation.md`** to open with `tt update --dry-run` and the two `verifyTtnn*` buttons, each labelled with its environment.
- [ ] **Step 5: Add templates, register handlers (`setup.update` wrapped in `guardCloud('tt update')`), add to `package.json`, update registry entries (`arrive` first in `setup`, `completionEvents` to the new ids).**
- [ ] **Step 6: Run `npm test`.** Expected: PASS.
- [ ] **Step 7: Prove on the QuietBox host:** run every template in order under `gozer run --chips 1 …` where it touches the device; paste real output into the lessons where output is shown.
- [ ] **Step 8: Prove in the cloud rig:** `uv tool install tenstorrent`, `tt --version` (expect 1.2.1), `tt device status` (one n150), `tt update --dry-run`; confirm the extension's `setup.update` refuses (open the toolkit in the rig's code-server, or call the handler through the test harness with `TT_VLLM_VENV` set). Record both in the lesson's CLAUDE.md log entry.
- [ ] **Step 9: Bump version, commit** `feat(setup): tt-first Set Up lessons`, hand to Taylor for review. Wait.

### Task 6: Serve Models group

**Files:**
- Modify: `content/lessons/download-model.md`, `interactive-chat.md`, `api-server.md`, `image-generation.md`, `video-generation-ttmetal.md`, registry, `terminalCommands.ts`, `extension.ts`, `package.json`

**Interfaces:**
- Produces: templates and ids `tenstorrent.model.{list,listCatalog,info,search,login,pull,serveDryRunN150,serveN150,serveDryRunQb2,serveQb2,ps,logs,curl,stop}` and `tenstorrent.vllm.{activateCloudVenv,serveCloudN150,curl}` (cloud route). `serve*N150` use `Llama-3.1-8B-Instruct`; `serve*Qb2` use `Qwen3-32B`. The cloud serve template is `source ~/tt-metal/build/python_env_vllm/bin/activate && vllm serve meta-llama/Llama-3.1-8B-Instruct <plugin flags>`, flags taken from `~/vllm-tt-plugin/docs/` in the rig and confirmed by a live run.

- [ ] **Step 1: Delete** the old download/inference/api-server/vLLM-start templates and handlers that `tt` replaces (`DOWNLOAD_MODEL`, `RUN_INFERENCE`, `INSTALL_INFERENCE_DEPS`, `CREATE_API_SERVER*`, `START_API_SERVER*`, `INSTALL_FLASK`, `START_VLLM_SERVER_*`, `CLONE_VLLM`, `INSTALL_VLLM`, `RUN_VLLM_OFFLINE`). Keep `TEST_VLLM_*` renamed under `vllm.*` for the cloud route.
- [ ] **Step 2: Rewrite `download-model.md`** ("Find and Pull a Model") from the QB2 guide's model-zoo chapter: list, catalog, info, search with the "I want to…" table, login, pull; the community-bundle warning with the tt-awesome model browser link.
- [ ] **Step 3: Rewrite `interactive-chat.md`** ("Serve a Model") from the QB2 guide's chapter 5: two serve buttons with the sentence telling the reader which to press from `tt device status`; the unauthenticated-endpoint warning; `ps`, `logs`, `curl`, `stop`; a paragraph above the serve buttons: needs Docker, and the next lesson is the route without it.
- [ ] **Step 4: Rewrite `api-server.md`** ("Serve Without Containers"): activate the venv, serve, same `curl` test, the `/v1` endpoint; the Go Deeper pointer to `run.py --local-server`.
- [ ] **Step 5: Edit the image and video lessons:** opening section becomes `tt model list --type image|video` and `tt serve <model>` where the box lists one; the existing by-hand content moves under a "Go deeper" heading at the end.
- [ ] **Step 6: Templates, handlers, package.json, registry. `npm test` PASS.**
- [ ] **Step 7: Prove on the QuietBox host** (two-chip lease; `tt serve Llama-3.1-8B-Instruct --device p300` is the half-machine proof; Qwen3-32B dry run only unless all four chips are free). Paste output.
- [ ] **Step 8: Prove in the cloud rig:** `tt model list/info/search` (no device needed beyond status); the full `vllm serve` route against the leased n150 through to a `curl` reply. Paste output; record flags used in the lesson.
- [ ] **Step 9: Bump version, commit** `feat(serve): tt model and tt serve lessons; cloud vLLM route`, Taylor review. Wait.

### Task 7: Connect Software group

**Files:**
- Create: `content/lessons/launch-software.md`
- Modify: `coding-assistant.md`, `qb2-local-agents.md`, `qb2-openclaw-assistant.md`, registry, `terminalCommands.ts`, `extension.ts`, `package.json`

**Interfaces:**
- Produces: `tenstorrent.launch.{list,openwebuiDryRun,openwebui,stopOpenwebui,qwencodeDryRun,qwencode,claude,studio}`; `launch.openwebui` wrapped in `guardCloud('tt launch openwebui')`; `launch.studio` template `tt serve <model> --studio` likewise guarded.

- [ ] **Step 1: Write `launch-software.md`** from the QB2 guide's chapter 6, with the `WEBUI_AUTH=False` warning, the Qwen3-32B thinking warning, the `tt agent` sentence, and the developer-blog CLI post link. `qwencode` and `claude` are noted as already installed in the cloud pod.
- [ ] **Step 2: Edit the three assistant lessons** so their server-start steps are `tt serve` + `tt launch` buttons (or the cloud vLLM route from Task 6 where the lesson is cloud-relevant); agent-specific content unchanged. Delete the replaced `START_QB2_AGENTS_SERVER_*` and `VERIFY_CODING_MODEL` templates.
- [ ] **Step 3: Templates, handlers, package.json, registry. `npm test` PASS.**
- [ ] **Step 4: Prove on the QuietBox host:** `tt launch list`, `tt launch qwencode --dry-run` against the Task 6 server; Open WebUI launch and stop. **Prove in the rig:** `tt launch list` against the cloud vLLM server (expect it listed), `qwencode --dry-run`; `launch.openwebui` refused by the guard.
- [ ] **Step 5: Bump version, commit** `feat(connect): tt launch lessons`, Taylor review. Wait.

### Task 8: Build on the Stack group

**Files:**
- Modify: `explore-metalium.md`, `tt-lang-intro.md`, `forge-image-classification.md`, `tt-xla-jax.md`, `cookbook-overview.md`, `terminalCommands.ts`, `extension.ts`, `package.json`
- Create: `scripts/lib/awesome-links.js` only if Step 1 chooses build-time inclusion

**Interfaces:**
- Produces: `tenstorrent.metal.*`, `forge.*`, `xla.*`, `ttlang.*` ids (renamed in Task 2); templates for environment activation per lesson: cloud `source ~/tt-forge-venv/bin/activate` / `~/tt-lang-venv` / `python_env`; QB2 the existing install commands.

- [ ] **Step 1: Decide tt-awesome inclusion** by reading `~/code/tt-awesome`'s category index format: if a category page URL exists per stack (`https://docs.tenstorrent.com/tt-awesome/<category>/`), link it; otherwise write `awesome-links.js` to render the category's entries into the lesson at site build. Record the choice in the commit message.
- [ ] **Step 2: Edit each lesson's setup section** to name the environment per the spec's table (cloud venv by path; QB2 route), and add a "Where to go next" footer with the tt-awesome link and any developer-blog post on the topic.
- [ ] **Step 3: `npm test` PASS. Prove in the rig:** each lesson's first runnable step in its named venv (`import ttnn` in `python_env`; `import tt_forge` in `tt-forge-venv`; `import ttlang` in `tt-lang-venv`). Prove on the QuietBox the QB2 routes the lessons already had.
- [ ] **Step 4: Bump version, commit** `feat(build): environment-aware setup and next steps`, Taylor review. Wait.

### Task 9: Go Deeper group

**Files:**
- Create: `content/lessons/mix-and-match.md`
- Modify: `tt-inference-server.md`, `vllm-production.md`, `build-tt-metal.md`, `monkeypatch-ttnn.md`, `bounty-program.md`, registry, `terminalCommands.ts`, `extension.ts`, `package.json`

**Interfaces:**
- Produces: `tenstorrent.inferenceServer.{clone,runServer,runServerLocal,…}` and `tenstorrent.vllm.*` for the by-hand routes; `runServerLocal` template `python run.py --model <m> --workflow server --local-server --tt-metal-home ~/tt-metal --tt-metal-python-venv-dir ~/tt-metal/build/python_env_vllm --device n150`.

- [ ] **Step 1: Edit `tt-inference-server.md` and `vllm-production.md`** to open with one paragraph: `tt serve` runs this for you; here is how to run it yourself, pin chips (`--device-id`), bind to loopback (`--bind-host 127.0.0.1`), choose a version. Keep the rest.
- [ ] **Step 2: Write `mix-and-match.md`:** `~/.local/share/tenstorrent/tools/` and what `tt update --dry-run` lists; the cloud venvs; a built tt-metal; a container; when each applies; how to find a release's pinned tt-metal commit (tt-inference-server `model_spec`). Short.
- [ ] **Step 3: `npm test` PASS. Prove in the rig:** `runServerLocal` boots against the n150 and answers `curl` (this is the rig's second serving route; if it fails, the lesson says what was seen and the vLLM route stays primary). Prove on the QuietBox: `run.py` with `--device-id` on the leased pair.
- [ ] **Step 4: Bump version, commit** `feat(deeper): by-hand serving and mix-and-match`, Taylor review. Wait.

### Task 10: Applications, series, docs site, PR

**Files:**
- Modify: `animatediff-video-generation.md`, `qb2-video-generation.md`, the series lessons whose setup steps named a replaced command, `CLAUDE.md`, `README.md`, `llms.txt`

- [ ] **Step 1: Re-point setup steps** in the Applications lessons and the four series to the Set Up / Serve lessons' buttons or the cloud venvs; no other content change. Run `grep -rn "tt-installer\|install.sh\|run.py" content/lessons` and justify each remaining hit in the commit message.
- [ ] **Step 2: Build the docs site and open three pages** (arrive, serve, launch) to confirm the command map prints the `tt` commands.
- [ ] **Step 3: CLAUDE.md log entry:** prompt, decisions, per-group "run on / not run" lines from Tasks 5–9. README and `llms.txt` describe the four paths.
- [ ] **Step 4: Full `npm test`, Vale over `content/lessons/` (ignore product-name hits), bump version.**
- [ ] **Step 5: Commit, push `updates/tt-cli-leads-the-way`, open the PR** with the group-by-group summary and the unverified list.
