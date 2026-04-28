---
id: qb2-local-agents
title: Local AI Agents on QuietBox 2
description: Build real AI agents in pure Python — web research, codebase navigation, multi-agent pipelines, and stateful text adventures — all running locally on your QB2's 32B/70B models
category: applications
tags:
  - qb2
  - p300x2
  - agents
  - smolagents
  - crewai
  - openai-agents-sdk
  - tool-calling
  - 32b
  - 70b
  - local-inference
supportedHardware:
  - p300x2
status: validated
validatedOn:
  - p300x2
estimatedMinutes: 60
---

# Local AI Agents on QuietBox 2

> **QB2-only lesson.** Everything here requires a 32B or 70B model. The agent reliability gap between 7B and 32B+ is large enough that 7B is not a supported path — the demos will fail or loop indefinitely at smaller scales.

You have a QuietBox 2. You're probably using it to chat with a 70B model and thinking "this is fast." That's true. But that's not the interesting part.

The interesting part is that you now have the hardware to run **AI agents that actually work** — systems that search the web, read files, hand tasks between specialized roles, and maintain state across a full session. These patterns fall apart at 7B. They come together at 32B. They shine at 70B.

This lesson is the raw-Python antidote to the OpenClaw lesson. No frameworks to configure, no services to restart, no JSON files to edit. Seven scripts. `pip install`. `python3 run.py`. You see every tool call, every step, every decision the model makes.

---

## What You'll Build

Seven scripts that grow from single tool calls to multi-model pipelines:

```
00_verify_tools.py         ── Confirm inference + tool calling work (run this first)
01_research_agent.py       ── Web search + synthesis via smolagents (60 lines)
02_code_explorer.py        ── Codebase Q&A via OpenAI Agents SDK (80 lines)
03_writing_pipeline.py     ── Researcher → Writer → Editor via CrewAI (100 lines)
04_dungeon_master.py       ── Stateful interactive agent via smolagents (120 lines)
05_storyboard_to_pixelart.py ── Two-stage storyboard pipeline: CrewAI + smolagents (280 lines)
06_landscape_svg.py        ── Parameterized generative landscape SVG, no framework (120 lines)
```

Each script uses a different pattern — from single-call structured generation to multi-model lifecycle management. By the end you'll know which approach to reach for and why.

---

## Why 32B Changes Everything for Agents

A language model doing a single task — answer this question, translate this text — is forgiving. A model operating as an **agent** is not. Agents must:

1. Decide *which tool* to call from a list of options
2. Format *valid JSON* arguments for that tool
3. Interpret the tool's return value
4. Decide *whether to call another tool* or respond
5. Maintain coherent intent across multiple steps

Each decision is a place to fail. The math compounds:

| Task | 7B success rate | 32B success rate | 70B success rate |
|------|----------------|-----------------|-----------------|
| Single tool call | ~78% | ~93% | ~96% |
| 3-step ReAct loop | ~52% | ~78% | ~88% |
| Multi-agent pipeline | ~30% | ~70% | ~82% |

A 3-step loop with 7B succeeds barely half the time. The same loop with 32B works 3 out of 4 times. Your QB2 runs Qwen3-32B at ~8 seconds per response and Llama-3.3-70B-Instruct at ~14 seconds. Both are genuinely usable. Neither requires a cloud subscription or an NVIDIA GPU.

---

## Performance at a Glance

| What | Time | Model |
|------|------|-------|
| Single tool call | ~8–14 s | Qwen3-32B / Llama-3.3-70B |
| Research agent (5 steps) | ~3–5 min | Qwen3-32B |
| Code explorer (3 hops) | ~1–2 min | Qwen3-32B or Llama-3.3-70B |
| Writing pipeline (3 agents) | ~5–15 min | Qwen3-32B |
| DM response (per turn) | ~10–16 s | Llama-3.3-70B (recommended) |
| Storyboard pipeline (--single-model) | ~8–20 min | Qwen3-32B or Llama-3.3-70B |
| Storyboard pipeline (full switch) | ~25–45 min | 70B → 32B (includes ~10–15 min switch) |
| Landscape SVG (single call) | ~30–60 s | Qwen3-32B or Llama-3.3-70B |

---

## Prerequisites

- QB2 hardware healthy: `tt-smi -s` shows 4 chips
- `~/code/tt-inference-server` cloned and working
- Docker running
- Internet access (for the web-search demo and model downloads)

**Verify your QB2:**

```bash
tt-smi -s | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"device_info\"])} chips detected')"
# Should print: 4 chips detected
```

[Run Hardware Detection](command:tenstorrent.runHardwareDetection)

---

## Step 1: Start the Inference Server

The demos need a vLLM endpoint running on port 8000. Use **Qwen3-32B** for the fastest iteration, or **Llama-3.3-70B-Instruct** for maximum output quality — especially for the Dungeon Master demo.

### Option A — Qwen3-32B (~8 s/response, recommended for most demos)

```bash
cd ~/code/tt-inference-server

python3 run.py \
    --model Qwen3-32B \
    --tt-device p300x2 \
    --workflow server \
    --docker-server \
    --no-auth \
    --vllm-override-args '{"enable_auto_tool_choice": true, "tool_call_parser": "hermes"}'
```

[Start Qwen3-32B Server](command:tenstorrent.startQb2AgentsServerQwen)

### Option B — Llama-3.3-70B-Instruct (~14 s/response, best for Dungeon Master)

```bash
cd ~/code/tt-inference-server

python3 run.py \
    --model Llama-3.3-70B-Instruct \
    --tt-device p300x2 \
    --workflow server \
    --docker-server \
    --no-auth \
    --vllm-override-args '{"enable_auto_tool_choice": true, "tool_call_parser": "llama3_json"}'
```

[Start Llama-3.3-70B Server](command:tenstorrent.startQb2AgentsServerLlama)

> **`--enable_auto_tool_choice` is not optional.** Without it, the model will respond in prose instead of making tool calls. The `tool_call_parser` must match the model — `hermes` for Qwen, `llama3_json` for Llama.

**Wait for warmup** (5 minutes on a warm cache, up to 20 minutes if downloading for the first time). The server is ready when:

```bash
curl http://localhost:8000/v1/models
# → {"data":[{"id":"Qwen/Qwen3-32B",...}]}
```

[Check Server Health](command:tenstorrent.checkAgentServerHealth)

---

## Step 2: Get the Agent Scripts

Clone the demo repository and install Python dependencies:

```bash
# Clone the demos (if not already present)
git clone https://github.com/tenstorrent/tt-agents.git ~/code/tt-agents
cd ~/code/tt-agents

# Upgrade pip first — some crewai sub-dependencies require it
pip install --upgrade pip setuptools wheel

# Install all framework dependencies
pip install -r requirements.txt
```

[Clone tt-agents](command:tenstorrent.cloneTtAgents)

**What gets installed:**

```
smolagents     — HuggingFace's lightweight agent framework (CodeAgent + ToolCallingAgent)
openai-agents  — OpenAI Agents SDK (@function_tool decorator, async Runner)
crewai         — Role-based multi-agent orchestration
ddgs           — DuckDuckGo search (smolagents 1.13+ requires ddgs, not duckduckgo-search)
beautifulsoup4 — Web page parsing
markdownify    — HTML-to-markdown converter (required by smolagents' visit_webpage tool)
openai         — OpenAI-compatible client (works against local vLLM)
```

**Copy to tt-scratchpad** so you can modify freely without touching the originals:

```bash
mkdir -p ~/tt-scratchpad/agents
cp ~/code/tt-agents/*.py ~/tt-scratchpad/agents/
cp ~/code/tt-agents/requirements.txt ~/tt-scratchpad/agents/
cp ~/code/tt-agents/world.json ~/tt-scratchpad/agents/
```

[Copy Scripts to Scratchpad](command:tenstorrent.copyAgentsToScratchpad)

> Scripts in `~/tt-scratchpad/agents/` are yours to hack. The originals in `~/code/tt-agents/` stay clean.

---

## Step 3: Verify Everything Works

Before running any demos, confirm inference and tool calling are both functioning:

```bash
python3 ~/code/tt-agents/00_verify_tools.py
```

[Run Tool Verification](command:tenstorrent.runAgentsVerify)

**Expected output:**

```
============================================================
tt-agents: Tool Calling Verification
============================================================

Endpoint: http://localhost:8000/v1
  Found model: Qwen/Qwen3-32B

[1/3] Basic inference...
  ✓ Response: INFERENCE_OK

[2/3] Tool call (function calling)...
  ✓ Tool called: get_weather({'city': 'San Francisco'})

[3/3] Structured output (JSON mode)...
  ✓ Parsed JSON: {'status': 'ok', 'count': 42}

============================================================
  ✓ inference
  ✓ tool_call
  ✓ structured

✅ All checks passed. Ready to run agent demos!
```

**If tool_call fails:** The most common cause is a mismatch between the model and the parser flag. Check that you used `--tool-call-parser hermes` with Qwen3-32B, or `--tool-call-parser llama3_json` with Llama-3.3-70B-Instruct.

---

## Demo 1: Research Agent

**Framework:** smolagents `CodeAgent` | **~60 lines** | **Model:** Qwen3-32B

An agent that searches the web, reads pages, and synthesizes a cited report. This is the demo that makes people say "oh, *this* is what a local LLM is actually for."

```bash
python3 ~/code/tt-agents/01_research_agent.py
```

[Run Research Agent](command:tenstorrent.runResearchAgent)

The script opens a **topic picker** — eight research prompts across tech, travel, philosophy, gaming, and more. Pick a number, paste your own query, or just press Enter to accept today's highlighted suggestion. It auto-continues after 10 seconds if you walk away.

```
Research topic — pick a number, paste your own query, or press Enter:

    [1] Python AI agent libraries
    [2] East Coast music festivals 2026
    [3] Walter Benjamin on AI video and the prompt artist
    [4] Attracting beneficial insects to any garden
    [5] Roguelike vs roguelite — genre lineage 1980–2026
    [6] Best blog platforms for GitHub Pages in 2026
    [7] Invent an original word puzzle mechanic
  → [8] Underrated US national parks and public lands

  Auto-continuing with [8] in 10s...
>
```

The highlighted suggestion (`→`) rotates daily, so every time you run it you see a different default.

**Run without any prompt (useful in scripts or scheduled jobs):**

```bash
python3 ~/code/tt-agents/01_research_agent.py --headless
```

**Pass your own query directly:**

```bash
python3 ~/code/tt-agents/01_research_agent.py \
    --query "What are the most-cited safety concerns with agentic AI systems in 2025-2026? Summarize 3 key papers or posts with links."
```

**Sample output:**

```
======================================================================
tt-agents Proof 1: Research Agent (smolagents)
======================================================================
Model:    Qwen/Qwen3-32B
Endpoint: http://localhost:8000/v1

Query:
  Research the current state of open-source AI inference hardware...
----------------------------------------------------------------------

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing this code:
    results = web_search(query="open source AI inference hardware 2025 alternatives NVIDIA")
    print(results)

 ─ Execution logs:
    [search results...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing this code:
    page = visit_webpage(url="https://tenstorrent.com/hardware/quietbox")
    print(page[:3000])

 ─ Execution logs:
    [page content...]

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing this code:
    final_answer("Open-source AI inference hardware has matured significantly...")

======================================================================
FINAL RESULT:
======================================================================
Open-source AI inference hardware has matured significantly heading into 2026.
Three companies have moved beyond vaporware into shipping products:

**Tenstorrent (QuietBox 2 / Blackhole):** The QB2 runs Qwen3-32B at ~8 s/response
and Llama-3.3-70B-Instruct at ~14 s/response on 4× Blackhole ASICs...

✓ Completed in 5 steps
```

> **Qwen3 extended thinking:** Qwen3-32B uses chain-of-thought reasoning internally. You may see `<think>...</think>` blocks appear in step output before the actual tool call — this is the model working through its reasoning and is normal. The final answer does not include these tags.


### How It Works

The `CodeAgent` is smolagents' most powerful mode. Instead of emitting JSON tool call objects, it **writes Python code** to call tools:

```python
# What the model generates internally:
results = web_search("open source AI inference hardware 2025")
page = visit_webpage(results[0]['url'])
# ... then synthesizes all gathered info into a final answer
```

This is why it's robust even with imperfect tool calling config — the model is writing code, not formatting JSON. It handles multi-hop research loops naturally.

### Real-World Uses

**Before committing to a library:**
```bash
python3 ~/code/tt-agents/01_research_agent.py \
    --query "Research known security vulnerabilities and CVEs in the 'requests' Python library versions 2.28-2.32. What versions are safe?"
```

**Before a technical meeting:**
```bash
python3 ~/code/tt-agents/01_research_agent.py \
    --query "Summarize the last 3 months of activity on the vLLM GitHub repo. What major features shipped? What are the open blockers?"
```

**Competitive intelligence:**
```bash
python3 ~/code/tt-agents/01_research_agent.py \
    --query "Compare Anthropic Claude API, OpenAI API, and local Llama pricing for a workload of 10M tokens/day. Include current pricing from their websites."
```

**Understanding a new topic before diving in:**
```bash
python3 ~/code/tt-agents/01_research_agent.py \
    --query "I'm new to Kubernetes. What are the top 5 concepts I need to understand first, and what's a common mistake beginners make with each?"
```

---

## Demo 2: Codebase Explorer

**Framework:** OpenAI Agents SDK | **~80 lines** | **Model:** Qwen3-32B or Llama-3.3-70B

Point this at any directory and ask questions. This is what a local coding assistant looks like at the architectural level — it actually reads your files.

Run it from any project directory and it explores that directory by default:

```bash
cd ~/code/tt-agents
python3 02_code_explorer.py
```

[Run Code Explorer](command:tenstorrent.runCodeExplorer)

**Point it at a specific directory:**

```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/YOUR_PROJECT \
    --query "How does authentication work in this codebase? Walk me through the request flow."
```

**Compare 32B vs 70B output quality side by side:**

```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/tt-agents \
    --compare
```

**Sample output** (run from `~/code/tt-agents`):

```
======================================================================
tt-agents Proof 2: Codebase Explorer (OpenAI Agents SDK)
======================================================================
Endpoint: http://localhost:8000/v1
Directory: /home/ttuser/code/tt-agents

Query:
  Summarize this codebase, how it is organized, and which files
  are most important to understand first.

======================================================================
Model: Qwen/Qwen3-32B
======================================================================
Based on my analysis:

**What it does:** Five standalone agent demos spanning three frameworks
(smolagents, OpenAI Agents SDK, CrewAI), each demonstrating a different
agentic pattern against a local vLLM endpoint.

**Most important files:**

1. **00_verify_tools.py** — Run this first. Confirms the vLLM server is
   up, tool calling is working, and structured output parses correctly.

2. **01_research_agent.py** — smolagents CodeAgent. Web search + page
   reading + synthesis. The simplest demonstration of a real work loop.

3. **04_dungeon_master.py** — The most complex. Shows persistent state
   via world.json, generative tools (cast_spell, examine_item, manage_lore),
   and how tool-grounded agents avoid hallucinating world state.

4. **world.json** — The DM's ground truth. Edit this to extend the world.

5. **requirements.txt** — Install order matters; upgrade pip first.

✓ Code exploration complete
```

### How It Works

The agent uses three tools with explicit JSON schemas via the `@function_tool` decorator:

```python
@function_tool
def read_file(path: str) -> str:
    """Read a text file from disk (truncated at 32KB)."""
    ...

@function_tool
def list_files(directory: str, pattern: str = "*") -> str:
    """List files matching a glob pattern."""
    ...

@function_tool
def grep_code(pattern: str, directory: str, file_extension: str = ".py") -> str:
    """Search for a regex pattern, returning file:line matches."""
    ...
```

Every tool call is a real JSON function call object — you can see exactly what the model decided to read and why. This is the right pattern for production tools where auditability matters.

> **Qwen3 extended thinking:** With Qwen3-32B you may see `<think>...</think>` blocks in the raw output between tool calls — the model working through its reasoning before committing to a file read or search. The final answer is clean. If you prefer to suppress it, add `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` when constructing the OpenAI client.

### Real-World Uses

**Day 1 at a new job:**
```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/company-app/src \
    --query "Give me a tour of this codebase. What does it do, how is it organized, and what are the most important files to understand first?"
```

**Security audit:**
```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/my-api \
    --query "Find all places where user-provided input is passed to SQL queries, shell commands, or eval(). List file:line for each."
```

**Documentation generation:**
```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/my-library \
    --query "List every public function and class with its parameters and a one-sentence description of what it does."
```

**Understanding a PR before reviewing:**
```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/project \
    --query "I'm reviewing a PR that changes authentication. Explain the current auth flow in detail so I can spot regressions."
```

**Exploring your QB2's inference server config:**
```bash
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir ~/code/tt-inference-server \
    --query "How does run.py decide which Docker image to use? Trace the logic from command-line flags to the final image selection."
```

---

## Demo 3: Multi-Agent Writing Pipeline

**Framework:** CrewAI | **~100 lines** | **Model:** Qwen3-32B

Three specialized agents collaborate sequentially: a **Researcher** builds a factual outline, a **Writer** drafts from it, an **Editor** polishes the result. The output is measurably better than any single agent produces.

**This demo chains with Demo 1.** If you ran the Research Agent first, its output is loaded automatically and the Researcher agent is skipped — the pipeline goes straight to Writer → Editor with real research already in hand. Run them back to back and watch the full chain:

```bash
python3 ~/code/tt-agents/01_research_agent.py   # saves last_research.txt
python3 ~/code/tt-agents/03_writing_pipeline.py  # picks it up automatically
```

[Run Writing Pipeline](command:tenstorrent.runWritingPipeline)

The pipeline opens a **format picker** — choose how the piece is written, not just what it covers:

```
Writing format — pick a number, paste your own instruction, or press Enter:

    [1] Developer blog post
    [2] Tweet thread (≤140 chars each)
  → [3] Explain it like I'm curious but new
    [4] Devil's advocate — argue the opposite
    [5] Lyrical / prose poetry
    [6] Executive one-pager
    [7] Reddit thread — multiple POVs

  Auto-continuing with [3] in 10s...
>
```

The same research can become a tweet thread, an executive briefing, a Reddit discussion, or prose poetry — just run it again and pick a different format.

**Run without prompts:**

```bash
python3 ~/code/tt-agents/03_writing_pipeline.py --headless
```

**Force a fresh research run (ignore saved research):**

```bash
python3 ~/code/tt-agents/03_writing_pipeline.py --no-research \
    --topic "How Tenstorrent's Blackhole architecture differs from GPU inference"
```

**Expected timeline:**

```
With pre-loaded research (Writer → Editor only):
  [Writer]  Drafting in chosen format...  (~2–5 min)
  [Editor]  Polishing...                  (~1–3 min)
  Total:    ~3–8 minutes

Without pre-loaded research (Researcher → Writer → Editor):
  [Researcher] Building factual outline... (~1–3 min)
  [Writer]     Drafting...                 (~2–5 min)
  [Editor]     Polishing...                (~1–3 min)
  Total:       ~5–15 minutes
```

> Each agent makes multiple LLM calls at ~8 s each — grab a coffee either way.

**Sample output (excerpt):**

```
======================================================================
FINAL ARTICLE:
======================================================================

# Why 32B Models Finally Make Local AI Agents Viable

Seven billion parameters sounds like a lot until you ask your model to search
the web, read a file, and synthesize an answer — in that order. At 7B, that
loop fails about half the time. At 32B, it works three times out of four.
That's not a marginal improvement. That's the difference between a toy and a tool.

The bottleneck isn't intelligence — it's reliability. Agents must format valid
JSON for every tool call, interpret the response, and decide what to do next.
Each step is an opportunity to hallucinate an argument, call the wrong tool, or
lose track of the goal. Smaller models accumulate these errors; larger ones
mostly don't.

Tenstorrent's QuietBox 2 runs Qwen3-32B at roughly eight seconds per response
on four Blackhole ASICs...

✓ Pipeline complete
```

### How It Works

Each agent has a narrowly scoped **role**, **goal**, and **backstory**. When research is pre-loaded from Demo 1, the Researcher is skipped entirely and its output is fed directly into the Writer's task. Role separation is what makes this better than one big prompt — and the format instruction travels with the research so every agent knows the target form.

```python
# With pre-loaded research — 2 agents
writing_task = Task(
    description=(
        "FORMAT INSTRUCTION — follow this exactly:\n"
        "Write a thread of 10-14 tweets. EVERY tweet must be ≤140 characters...\n\n"
        "RESEARCH:\n<output from 01_research_agent.py>"
    ),
    agent=writer,
)

# Without — 3 agents, Researcher runs first
research_task = Task(description="Research the topic...", agent=researcher)
writing_task = Task(context=[research_task], ...)
editing_task = Task(context=[writing_task], ...)
```

### Real-World Uses

The format picker changes everything here. The same facts become different artifacts:

**Research once, publish everywhere:**
```bash
python3 ~/code/tt-agents/01_research_agent.py --query "What's new in Rust 2025?"
python3 ~/code/tt-agents/03_writing_pipeline.py        # → tweet thread
python3 ~/code/tt-agents/03_writing_pipeline.py        # → exec one-pager
python3 ~/code/tt-agents/03_writing_pipeline.py        # → ELI5 for onboarding docs
```

**Postmortem in every format your team needs:**
```bash
python3 ~/code/tt-agents/03_writing_pipeline.py --no-research \
    --topic "Database outage: 45 minutes, connection pool misconfiguration, 14:23–15:08, no data loss"
# Pick [6] Executive one-pager for your VP
# Pick [7] Reddit thread to share the learning publicly
# Pick [1] Blog post for the engineering blog
```

**Turn research agent output into a tweet thread:**
```bash
python3 ~/code/tt-agents/01_research_agent.py  # research any topic
python3 ~/code/tt-agents/03_writing_pipeline.py # pick [2] Tweet thread
```

**Make any topic approachable:**
```bash
python3 ~/code/tt-agents/01_research_agent.py --query "How does RDMA networking work?"
python3 ~/code/tt-agents/03_writing_pipeline.py  # pick [3] ELI5
```

---

## Demo 4: Dungeon Master Agent

**Framework:** smolagents `ToolCallingAgent` | **~120 lines** | **Model:** Llama-3.3-70B (recommended)

An interactive text adventure where the DM is your local LLM — with tools to enforce world state consistency, large context to remember your full session, and enough reasoning depth to make narrative decisions feel genuine. This is the demo that shows what stateful agents look like.

```bash
# Recommended: 70B for narrative quality
python3 ~/code/tt-agents/04_dungeon_master.py

# Faster (still fun, ~8 s/turn) — use this if you started Qwen3-32B in Step 1
python3 ~/code/tt-agents/04_dungeon_master.py --model "Qwen/Qwen3-32B"

# Start fresh (clear session state)
python3 ~/code/tt-agents/04_dungeon_master.py --reset
```

> If you started Qwen3-32B (Option A) in Step 1, pass `--model "Qwen/Qwen3-32B"` to all DM commands. The default assumes Llama-3.3-70B-Instruct is loaded. Both work well — 70B has richer narrative prose, 32B responds in ~8 s.

[Start Dungeon Master](command:tenstorrent.runDungeonMaster)

> **State persists between sessions** in `world_session.json`. Quit and come back — the DM remembers where you were.

**Full sample session:**

```
======================================================================
tt-agents Proof 4: Dungeon Master Agent (smolagents)
======================================================================
Model:    meta-llama/Llama-3.3-70B-Instruct
Endpoint: http://localhost:8000/v1
State:    world.json (fresh)

Tip: 70B gives the best narrative. Qwen3-32B is faster (~8s vs ~14s/turn).
Commands: 'status' | 'lore' | 'quit'
----------------------------------------------------------------------

The DM is setting the scene...

  [get_player_status]

DM: You stand in the Town Square of Millhaven, where the fountain gurgles
    with clear water cold from the mountain springs. Three paths open before
    you: north toward the Dark Forest where the treeline begins to thicken
    and the shadows pool early; east to the Rusty Axe Tavern, its sign
    swinging in the afternoon breeze; and south toward the dungeon entrance,
    which the locals avoid discussing over dinner. Your leather armor sits
    comfortably across your shoulders. Your short sword is at your hip.
    You have 10 gold coins and a torch.

You: Go to the rusty axe

  [get_player_status]
  [move_player]
  [manage_lore]

DM: You push through the heavy oak door of the Rusty Axe. Woodsmoke and
    roasted meat. A barkeep polishes mugs with professional disinterest.
    Two farmers argue quietly over a corner table. In the dim light behind
    the bar, half-hidden by a coil of rope, you notice something glinting
    on the shelf — a healing potion, apparently forgotten or deliberately
    hidden. The barkeep hasn't looked up yet.

You: Pick up the healing potion

  [get_player_status]
  [pick_up_item]

DM: You reach over the bar with practiced nonchalance. The potion slides
    into your pack while the barkeep's attention stays on his mug. It's
    warm from sitting near the hearthfire.
    Inventory: torch, short sword, leather armor, healing_potion.

You: status

[Status] HP: 100/100 | Gold: 10 g | Location: The Rusty Axe Tavern
[Inventory] torch, short sword, leather armor, healing_potion
[Spells] magic_missile, healing_word, shield  |  Slots: L1:3  L2:2
[Exits] town_square

You: Ask the barkeep about the dungeon

  [manage_lore]
  [manage_lore]

DM: The barkeep sets the mug down deliberately and meets your eyes for the
    first time. "People go in," he says. "Most of them even come back out,
    the first time." He goes back to polishing. A long pause. "The key to
    level one — rusty thing, opens a chest nobody's found yet — it's yours
    if you bring me proof someone cleared the skeleton on level two." He
    names a price that doesn't involve gold.

You: examine the mysterious coin

  [get_player_status]
  [examine_item]

DM: You turn the coin over in your fingers. The metal is wrong — too light,
    too warm, with a slight give to it that copper shouldn't have. The face
    shows a figure you don't recognize. You flip it to the date side and
    read it twice before it registers: forty years from now.

You: lore

[World Lore — recorded by the DM this session]

  barkeep: Former adventurer named Oswin. Lost two fingers to a pressure
    plate on Level 3. Retired the day his partner didn't come back. Knows
    more about the dungeon than he admits.

  mysterious_coin: Origin unknown. The date suggests either a forgery, a
    time anomaly, or something the archivist would recognize immediately.

  crypt runes: Pre-date the town by at least three centuries. The archivist
    has a partial translation she won't share.

You: quit

Farewell, adventurer!
[World state saved to world_session.json]
[Lore entries created this session: 3]
[Turns played: 5]
```

> **Reading the tool lines:** The bracketed names (`[move_player]`, `[manage_lore]`, ...) appear while the DM is thinking — each one is a real JSON tool call made against the world state. They tell you what the agent is doing during the wait without flooding the screen with argument JSON. A turn with `[manage_lore]` twice means the DM invented something new about that NPC and recorded it for future consistency.

### How It Works

The script ships with **ten tools** in two categories. The first five enforce world state — the DM physically cannot invent an exit, teleport the player, or ignore HP. The second five are *generative* — the model calls them to create new material and have it persist.

**State enforcement tools** (the DM cannot cheat):
```python
@tool
def move_player(destination: str) -> str:
    """Move the player — only succeeds if destination is a valid exit."""

@tool
def pick_up_item(item_name: str) -> str:
    """Pick up an item — only if it's actually in the current location."""

@tool
def update_player_hp(change: int, reason: str) -> str:
    """Modify HP — negative = damage, positive = healing."""
```

**Generative tools** (the DM creates real, persistent material):
```python
@tool
def cast_spell(spell_name: str, target: str) -> str:
    """Cast a spell from the player's spellbook. Consumes a slot. Returns the mechanical
    effect — the DM narrates it. Slots are tracked and saved."""

@tool
def examine_item(item_name: str) -> str:
    """Examine an item closely. Returns its surface description AND hidden properties —
    the mysterious coin's date is forty years in the future; the short sword's maker's mark
    matches one in the torn journal. None of this is in the narrative until the player looks."""

@tool
def manage_lore(subject: str, description: str = "") -> str:
    """Record or retrieve lore about any person, place, or object.
    When the DM invents the barkeep's backstory or the crypt's history, it calls this
    to save the canon. On later turns it retrieves it — so the barkeep stays consistent
    across every session, not just within one conversation."""

@tool
def check_rules(action: str) -> str:
    """Validate whether an action is possible given current state — spell slots, weapons
    present, available exits. Call before resolving anything unusual."""
```

**Adding a new tool takes about 10 lines.** Name it, write a docstring the model will read, touch world state, add it to the tools list. The model figures out the rest:

```python
# Want a crafting tool? This is all you need:
@tool
def craft_item(ingredient_a: str, ingredient_b: str) -> str:
    """Combine two items from inventory to attempt crafting something new.
    Only works if both items are in the player's inventory.

    Args:
        ingredient_a: First item name.
        ingredient_b: Second item name.
    """
    inv = _world_state["player"]["inventory"]
    if ingredient_a not in inv or ingredient_b not in inv:
        return f"Need both items in inventory. Have: {inv}"
    # Remove ingredients, add result, save — the DM narrates the outcome
    inv.remove(ingredient_a)
    inv.remove(ingredient_b)
    result = f"{ingredient_a}+{ingredient_b} creation"
    inv.append(result)
    save_world(_world_state)
    return f"Combined {ingredient_a} and {ingredient_b} → {result}"
```

The world itself reflects this depth. The crypt has a spirit NPC and items with dark history. Every item in `item_details` has a surface description and a `hidden` property that only surfaces when `examine_item` is called — a coin with a future date, a sword whose maker's mark connects to the dungeon's history, a journal whose last line reads: *"The crypt is not a dead end. It is a door."* The model builds on all of this through tool calls, not hallucination.

**The `lore` command** (type it at the prompt) shows you everything the DM has recorded during the session — every NPC's backstory, every invented fact, every secret it decided to commit to. By the end of a real session, it's a living document the model wrote itself.

### The Pattern Behind the Game

The Dungeon Master is a toy, but the architecture is real. Here's what it actually demonstrates:

**Pattern: Tool-grounded stateful agent.** The LLM is a reasoning and language layer. Actual state lives in structured storage (JSON, a database, an API). The model reads state via tools before acting, and writes state via tools after acting. It cannot hallucinate state it doesn't have access to.

This is exactly how you'd build:

**A personal task assistant:**
```
Tools: list_tasks(), add_task(), mark_complete(), get_priorities()
State: tasks.json / SQLite
Prompt: "What should I focus on this afternoon?"
```

**A customer service agent:**
```
Tools: get_order_status(), lookup_account(), create_ticket(), update_ticket()
State: your CRM / order management system
Prompt: customer message → agent handles → state updated
```

**A PR review assistant with memory:**
```
Tools: read_file(), get_diff(), add_comment(), list_open_issues()
State: GitHub API (via tools)
Prompt: "Review this PR for security issues, remembering we're migrating to OAuth"
```

The Dungeon Master is the smallest possible version of all of these. Once you understand it, the others are just different tools and different state.

---

## Demo 5: Storyboard to Pixel Art Pipeline

**Frameworks:** CrewAI (Stage 1) + smolagents ToolCallingAgent (Stage 2) | **~280 lines** | **Models:** Llama-3.3-70B → Qwen3-32B

[Run Storyboard Pipeline](command:tenstorrent.runStoryboardPipeline)

> The VSCode button runs with `--simulate` so it works even when your QB2 is busy. Remove that flag when you want to see the full server lifecycle.

This demo shows what none of the others do: **the multi-model lifecycle**. Different tasks benefit from different model sizes. The storyboard stage needs the 70B's narrative creativity — it's inventing visual language, palette, and scene-by-scene descriptions from scratch. The prompt engineering stage needs precise, structured output (JSON tool calls against a known schema) — the 32B handles that reliably and twice as fast.

What the pipeline teaches:

1. **Model selection by task** — choose the right size for the right job, not the same model for everything
2. **Server lifecycle from Python** — `docker stop` → `tt-smi -r` → `run.py` — all as managed subprocesses
3. **`tt-smi -r` as a required step** — loading a second model without resetting chips causes hardware faults; the script explains why inline
4. **JSON artifact handoff** — `storyboard.json` is the bridge between two entirely separate model invocations
5. **Chip targeting** — `--device-id 0,1` and `--device-id 2,3` shown in the dual-server launch commands

### Pipeline Architecture

```
[User picks theme]
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Storyboard         │  CrewAI, 3 agents, Llama-3.3-70B
│  ConceptDirector             │  → visual concept + named palette
│  StoryboardWriter            │  → 4-6 scene descriptions
│  ArtDirector                 │  → pixel art constraints per scene
└────────────┬────────────────┘
             │ saves storyboard.json
             ▼
┌─────────────────────────────┐
│  Model Switch               │  orchestrator (stdlib only)
│  docker stop <container>    │
│  tt-smi -r                  │  ← required between model loads
│  run.py --model Qwen3-32B   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Prompt Engineering │  smolagents ToolCallingAgent, Qwen3-32B
│  refine_scene_prompt         │  → structured pixel art prompt per scene
│  check_palette_consistency   │  → palette coherence check across all scenes
└────────────┬────────────────┘
             │ saves pixelart_prompts.json
             ▼
        [pipeline_summary.txt — timing + all prompts]
```

### Running It

```bash
# --simulate: shows all server commands but doesn't run them
# Works with whatever model is currently loaded — good for trying the pipeline
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --simulate

# Full lifecycle: Stage 1 on 70B, switch to 32B, Stage 2
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py

# Pick your own theme
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --theme "haunted lighthouse"

# One model for both stages (faster iteration)
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --single-model

# Dual-server: run both models simultaneously on separate chip pairs
# Stage 1 → port 8000 (chips 0,1)  |  Stage 2 → port 8001 (chips 2,3)
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --dual-server
```

**Auto-detect single-model:** If you already have a Qwen model loaded when you start the pipeline, the script detects it and automatically skips the model switch — no flags needed. You'll see a line like:

```
[auto: Qwen/Qwen3-32B detected — using for both stages, no model switch needed]
```

This covers the common case where you're iterating: start your Qwen server once, run the pipeline as many times as you like.

### What Gets Produced

```bash
storyboard.json         # Scene descriptions, palette, constraints (Stage 1 output)
pixelart_prompts.json   # Final prompts per scene (Stage 2 output)
pipeline_summary.txt    # Human-readable run log with timing per stage
```

**Running the prompts:** The prompts in `pixelart_prompts.json` are ready for any image generation API. If you have Flux running via tt-local-generator:

```bash
# Start Flux on TT hardware
cd ~/code/tt-local-generator && ./bin/start_flux.sh

# POST each scene prompt to the image generation endpoint
# http://localhost:8000/v1/images/generations
```

Or pipe them into any external image generation service that accepts text prompts — the format is standard and provider-agnostic.

Sample `storyboard.json` (from theme "a Game Boy with a cracked screen still running Tetris, battery low"):

```json
{
  "theme": "a Game Boy with a cracked screen still running Tetris, battery low",
  "hardware_era": "Game Boy",
  "palette": "Game Boy dark #0F380F, medium dark #306230, medium light #8BAC0F, lightest #9BBC0F",
  "lighting": "cold CRT phosphor glow, no warm tones",
  "technique": "dithered gradients, 2bpp tile constraints",
  "scenes": [
    {
      "id": 1,
      "title": "The Cracked Screen",
      "description": "A close-up of scratched plastic casing, a hairline crack bisecting the screen diagonally. Tetris blocks still fall in the bottom half, unaware.",
      "mood": "melancholy",
      "pixel_constraints": "16x16 tiles, 4-shade green palette, no off-palette colors, crack rendered as 1-pixel dark line"
    },
    {
      "id": 2,
      "title": "Battery Warning",
      "description": "The battery indicator blinks in the corner — one bar left. The play field dims slightly, the dithering pattern loosening at the edges.",
      "mood": "anxious",
      "pixel_constraints": "8x8 battery sprite, 2-color blink cycle, scanline fade on outer tiles"
    }
  ]
}
```

### The Server Lifecycle in Code

The model switch is the most instructive part — three operations, stdlib only:

```python
def switch_models(prompt_model: str, simulate: bool) -> None:
    # 1. Stop the running container (finds it by name filter)
    _stop_server(simulate=simulate)

    # 2. Reset chip state — required between model loads
    #    Without this: the second model may fail to initialize or corrupt chip state
    _reset_chips(simulate=simulate)

    # 3. Launch the new model as a background subprocess, then poll until ready
    _start_server(prompt_model, port=8000, simulate=simulate)
    if not simulate:
        _wait_for_server(port=8000)   # polls /v1/models every 10s, up to 15 min
```

The `--simulate` flag prints all three commands without executing them — so you can see the full lifecycle without touching the hardware.

### Dual-Server Mode

With four chips (QB2 = two P300 cards), you can run both models simultaneously:

```python
# chips 0,1 → Llama-3.3-70B on port 8000
# chips 2,3 → Qwen3-32B on port 8001
_start_server(creative_model, port=8000, device_ids=[0, 1])
_start_server(prompt_model, port=8001, device_ids=[2, 3])
```

Stage 1 hits port 8000, Stage 2 hits port 8001. No chip reset between stages — both models are loaded from the start and torn down together at the end. The tradeoff: startup takes longer (two models loading), but the inter-stage pause disappears entirely.

### Advanced: CPU Orchestrator Mode

There's a third option — and it's the most efficient when you only have one chip pair: run Stage 2 on a small Qwen3 model on the **host CPU** while Stage 1 occupies the TT hardware.

```bash
python3 05_storyboard_to_pixelart.py --cpu-orchestrator
python3 05_storyboard_to_pixelart.py --cpu-orchestrator --cpu-model Qwen/Qwen3-1.7B
```

The `--cpu-orchestrator` flag:
1. Launches `prompt_server.py` (from [tt-local-generator](https://github.com/tenstorrent/tt-local-generator)) as a background subprocess on port 8002
2. Stage 1 runs on TT hardware (70B) as normal
3. **No `docker stop`, no `tt-smi -r`, no restart** — Stage 2 hits the CPU server directly
4. At the end, the CPU server subprocess is terminated

The CPU model runs via HuggingFace `transformers` with `device_map="cpu"`:

```python
# From prompt_server.py — the pattern that makes this work
_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16,  # half-precision — fits in ~1.2 GB RAM
    device_map="cpu",
    trust_remote_code=True,
)
```

**Why Qwen3-0.6B works for orchestration**: The prompt-engineering task is structured and repetitive — take a scene description, append style tags, check consistency. A 0.6B model at ~19 tok/s on a modern CPU handles this well. Stage 1 (open-ended narrative creativity across a whole storyboard) genuinely needs the 70B.

**The trade-off**:
- `--cpu-orchestrator`: zero chip downtime, but Stage 2 is slower (~4-8 min for 5 scenes vs. ~1-2 min on 32B)
- Default model switch: 70B → 32B swap takes ~10-15 min (stop + reset + reload), then Stage 2 is fast
- `--dual-server`: fastest inter-stage, but requires four chips

| Mode | Chip downtime | Stage 2 speed | Hardware req |
|------|---------------|---------------|--------------|
| Default (switch) | ~10-15 min | Fast (32B on TT) | 2 chips |
| `--cpu-orchestrator` | 0 | Slow (0.6B on CPU) | 2 chips |
| `--dual-server` | 0 | Fast (32B on TT) | 4 chips |

The CPU orchestrator is particularly useful when the hardware is already committed to another job, or when you want to keep a long-running creative model loaded and fire off prompt-engineering passes on demand.

### Stage 3: Render the Storyboard

After Stage 2 produces `pixelart_prompts.json`, Stage 3 takes two independent paths depending on the flag you pass:

```bash
# ANSI mode: pixel art grid printed to terminal
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --single-model --ansi

# SVG mode: illustrated storyboard panels + assembled HTML
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --single-model --svg

# Both at once (separate LLM calls per scene)
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --single-model --ansi --svg
```

**ANSI mode** asks the LLM for a compact palette-indexed grid (16×12, letters as color keys) and renders it as Unicode block characters (`█`) with ANSI truecolor escape codes. Printed directly to the terminal; also saved as `scene_N_title.ans`. Requires truecolor terminal support.

**SVG mode** asks the LLM to draw a proper storyboard panel illustration per scene — 320×180 SVG with a gradient sky, silhouetted subjects, ground plane, and a mood accent. Each scene is saved as `scene_N_title.svg` and then all panels are assembled into a single `storyboard.html` with a 2-column grid layout, scene numbers, titles, moods, and descriptions under each panel.

Sample output panels from a "monolith awakening" theme run:

![Scene 1: Monolith Awaken](/assets/img/scene_1_monolith_awaken.svg) ![Scene 2: Temple Floor](/assets/img/scene_2_temple_floor.svg) ![Scene 3: Wind Eroded](/assets/img/scene_3_wind_eroded.svg)

![Scene 4: Celestial Shift](/assets/img/scene_4_celestial_shift.svg) ![Scene 5: Last Glyph](/assets/img/scene_5_last_glyph.svg) ![Scene 6: Buried Truth](/assets/img/scene_6_buried_truth.svg)

If the LLM produces invalid SVG for a scene, the script falls back to a simple placeholder panel so the HTML assembles cleanly regardless.

This is Stage 3 as a **third API pattern**: while Stage 1 uses CrewAI's `LLM` wrapper and Stage 2 uses smolagents' `OpenAIServerModel`, Stage 3 calls the OpenAI client directly — `client.chat.completions.create()` with no framework overhead.

---

## Demo 6: Generative Landscape SVG

**File:** `~/tt-scratchpad/agents/06_landscape_svg.py`
**Framework:** none — direct OpenAI client
**Demonstrates:** parameterized LLM prompts that produce SVG with gradients, layered terrain, and atmospheric effects

The simplest possible demo: one LLM call, one SVG file. Pass flags describing the scene you want; the script builds a structured prompt and asks the model to generate a complete 800×450 landscape with proper `<defs>` gradients, mountain `<polygon>` shapes, cloud `<ellipse>` groups, and a sun or moon.

```bash
# Sunset with mountains (default)
python3 ~/code/tt-agents/06_landscape_svg.py

# Night sky with stars
python3 ~/code/tt-agents/06_landscape_svg.py --palette blue --no-mountains --stars

# Full scene
python3 ~/code/tt-agents/06_landscape_svg.py --palette purple --mountains --clouds --stars

# See the prompt without calling the model
python3 ~/code/tt-agents/06_landscape_svg.py --palette red --simulate
```

**Palette choices:** `sunset` · `blue` · `purple` · `red` · `orange`

Each palette defines a complete color system — sky gradient (3 stops), mountain depth layers, cloud color, atmospheric glow, sun/moon, ground — so all colors in the generated SVG are harmonically related without the LLM having to invent them.

Sample output (default `sunset` palette):

![Generative landscape SVG sample output](/assets/img/landscape.svg)

**Why no framework?** The task doesn't need agents, tools, or multi-step reasoning. It's a single structured generation: prompt in, SVG out. Reaching for CrewAI or smolagents here would add complexity with no benefit. This demo exists to show the contrast — knowing when to use a framework is as important as knowing how.

**The key prompt technique:** Instead of asking the model to "draw a landscape," the prompt specifies every layer explicitly — which gradient IDs to define in `<defs>`, which colors to use for each mountain ridge, that polygon points must span `0,450` to `800,450` to close at the bottom edge. The LLM's job is to fill in the actual shape coordinates, not to make architectural decisions.

---

## Modify the Scripts

The scripts in `~/tt-scratchpad/agents/` are your starting point for building real things.

**Change the research agent's default query:**

```bash
# Edit ~/tt-scratchpad/agents/01_research_agent.py
# Line 33: DEFAULT_QUERY = "..."
# Set it to something you actually want to research every morning
```

**Add a new tool to the code explorer:**

```python
# ~/tt-scratchpad/agents/02_code_explorer.py

@function_tool
def summarize_function(file_path: str, function_name: str) -> str:
    """Read a specific function from a file and return just that function's code."""
    # Find the function, extract it, return it
    ...
```

**Give the writing pipeline a fourth agent:**

```python
# ~/tt-scratchpad/agents/03_writing_pipeline.py
fact_checker = Agent(
    role="Fact Checker",
    goal="Verify every specific claim against what you know is accurate",
    backstory="You flag anything that sounds made-up or imprecise...",
    llm=llm,
)
```

**Extend the dungeon master's world:**

```bash
# ~/tt-scratchpad/agents/world.json
# Add new locations to "locations" dict
# Add new enemies to "enemies" dict
# Add items to location "items" lists
# The DM automatically uses whatever's in the JSON
```

---

## Build Your Own Agent

Every agent in this lesson follows the same three-step pattern:

```python
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
import openai

# 1. Define tools
@tool
def my_tool(arg: str) -> str:
    """What this tool does — the model reads this docstring."""
    return do_real_work(arg)

# 2. Connect to your QB2's inference endpoint
model = OpenAIServerModel(
    model_id="Qwen/Qwen3-32B",
    api_base="http://localhost:8000/v1",
    api_key="none",
)

# 3. Build and run the agent
agent = ToolCallingAgent(tools=[my_tool], model=model, max_steps=10)
result = agent.run("Do something useful with my_tool")
print(result)
```

**Minimal research agent (15 lines):**

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel

model = OpenAIServerModel(model_id="Qwen/Qwen3-32B", api_base="http://localhost:8000/v1", api_key="none")
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, max_steps=5)
print(agent.run("What is the current state of the art in protein folding prediction?"))
```

**Minimal file agent (20 lines):**

```python
import asyncio, os, openai
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel

@function_tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return open(os.path.expanduser(path)).read()

# OpenAI Agents SDK parses the model string for a provider prefix — use
# OpenAIChatCompletionsModel to route directly to your local vLLM endpoint.
client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
model = OpenAIChatCompletionsModel(model="Qwen/Qwen3-32B", openai_client=client)

agent = Agent(name="FileReader", instructions="Answer questions about files.", tools=[read_file], model=model)
print(asyncio.run(Runner.run(agent, "Summarize ~/code/tt-agents/README.md")).final_output)
```

---

## Framework Cheat Sheet

| Framework | When to use | Why |
|-----------|-------------|-----|
| **smolagents CodeAgent** | Research, multi-hop tasks | Writes Python code to call tools — robust even with imperfect tool calling config |
| **smolagents ToolCallingAgent** | Stateful/interactive agents | JSON tool calls — more auditable, better for interactive sessions |
| **OpenAI Agents SDK** | Production tools, async | Explicit JSON schemas via `@function_tool`, async-first, close to the metal |
| **CrewAI** | Multi-role pipelines | Role-based orchestration — the cleanest pattern for Researcher → Writer → Editor flows |
| **Direct OpenAI client** | Single structured generation | No framework needed — prompt in, structured output out; use when there's nothing to orchestrate |

**Which model?**

- Qwen3-32B: 8 s/response, reliable tool calling, `hermes` parser — use for everything except narrative tasks
- Llama-3.3-70B: 14 s/response, better reasoning depth, `llama3_json` parser — use when output quality matters more than speed (writing pipeline, dungeon master, storyboard Stage 1)

---

## Troubleshooting

### Tool calls always fail / model responds in prose

**Cause:** Missing or wrong vLLM flags.

```bash
# Verify the flags are active
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# If the server started correctly you'll see the model ID

# Re-run verify script to confirm tool calling works
python3 ~/code/tt-agents/00_verify_tools.py
# If [2/3] fails: restart with correct --tool-call-parser for your model
```

### Research agent loops indefinitely or gives up after 2 steps

**Cause:** 7B-class reasoning behavior at tool call boundaries (but you're on QB2 with 32B — check model actually loaded).

```bash
# Confirm which model is actually running
curl http://localhost:8000/v1/models
# Should show Qwen/Qwen3-32B or meta-llama/Llama-3.3-70B-Instruct
# If something smaller loaded, restart the server
```

### Code explorer returns "Directory not found"

**Cause:** Path expansion issue with `~`.

```bash
# Always expand ~ before passing to the script
python3 ~/code/tt-agents/02_code_explorer.py \
    --dir /home/ttuser/code/my-project \
    --query "..."
```

### Writing pipeline hangs between agents

**Cause:** CrewAI's inter-agent context can sometimes exceed the model's working window.

```bash
# Reduce topic complexity or explicitly scope the output
python3 ~/code/tt-agents/03_writing_pipeline.py \
    --topic "Write 400 words about X. Be concise."
```

### Dungeon master ignores what I did last session

**Cause:** Session state not loading correctly. Check file exists:

```bash
ls -la ~/code/tt-agents/world_session.json
# Should exist after any previous run
# If corrupted, reset to fresh state:
python3 ~/code/tt-agents/04_dungeon_master.py --reset
```

### `pip install -r requirements.txt` fails on crewai

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Research agent fails with `ImportError: No module named 'ddgs'`

smolagents 1.13+ requires the `ddgs` package, not the older `duckduckgo-search`:

```bash
pip uninstall duckduckgo-search -y
pip install ddgs
```

If you cloned an older version of tt-agents or built your own requirements.txt, make sure it lists `ddgs>=9.0.0` (not `duckduckgo-search`).

### Storyboard pipeline Stage 1 produces invalid JSON

The art director agent sometimes wraps its output in markdown code fences. The script strips these automatically, but if the model outputs narrative prose instead of JSON, a fallback storyboard is used with a single scene.

```bash
# Check what Stage 1 actually produced
cat ~/code/tt-agents/storyboard.json

# If it looks wrong, run with a more constrained theme
python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --simulate \
    --theme "knight vs dragon" --single-model
```

A narrower, more concrete theme gives the art director less room to wander.

### `docker stop` fails in switch_models / `tt-smi -r` not found

Both commands require Docker and tt-smi to be installed and accessible. In `--simulate` mode, neither is called — the commands are only printed. If you see errors running without `--simulate`, verify:

```bash
which docker && docker ps   # Docker must be running
which tt-smi && tt-smi -s   # tt-smi must be on PATH
```

---

## What You've Built

You've run seven different patterns against your QB2's local inference endpoint:

- ✅ **Research synthesis** — multi-hop web search with cited output (smolagents CodeAgent)
- ✅ **Codebase Q&A** — file navigation and code comprehension (OpenAI Agents SDK)
- ✅ **Multi-role pipeline** — specialized agents handing off to each other (CrewAI)
- ✅ **Stateful interactivity** — tool-grounded persistent world state (smolagents ToolCallingAgent)
- ✅ **Multi-model lifecycle** — model selection by task, server lifecycle, JSON artifact handoff (CrewAI + smolagents)
- ✅ **Structured creative generation** — SVG storyboard panels assembled into a browsable HTML document
- ✅ **Parameterized generation** — CLI flags shape a single structured prompt; the model fills in the geometry (no framework)

More importantly, you've seen the three things that make these patterns work:

1. **32B+ models** that reliably format tool calls and follow multi-step instructions
2. **Large context** — vLLM defaults to 32K tokens in this config, which is enough for multi-turn sessions and full research loops; bump `max_model_len` in the server config for longer sessions
3. **Local inference** that lets you iterate without API bills, rate limits, or data leaving your machine

The scripts are short on purpose. Each one is a skeleton you can grow. The code explorer becomes your codebase assistant. The writing pipeline becomes your documentation tool. The dungeon master becomes your customer service bot or personal task tracker. The research agent becomes the thing that briefs you before every meeting. The storyboard pipeline becomes any multi-model workflow where different tasks genuinely need different model sizes. The landscape generator becomes any tool where structured prompting replaces a framework — data visualizations, diagrams, reports, config files.

---

## What's Next

**See how OpenClaw packages these patterns into a production service:**

[OpenClaw AI Assistant on QuietBox 2](command:tenstorrent.showLesson?["qb2-openclaw-assistant"])

OpenClaw is what you'd build if you took the dungeon master pattern seriously: persistent memory, multi-agent routing, a WebSocket API, and a terminal UI. Now that you've seen the raw version, the architecture makes more sense.

**Generate video on your QB2:**

[Generating Video on QuietBox 2](command:tenstorrent.showLesson?["qb2-video-generation"])

**Explore the framework comparison** (included in the tt-agents repo):

```bash
cat ~/code/tt-agents/framework_comparison.md
```

**Resources:**

- [smolagents docs](https://huggingface.co/docs/smolagents)
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)
- [CrewAI docs](https://docs.crewai.com)
- [tt-inference-server](https://github.com/tenstorrent/tt-inference-server)
- [Tenstorrent Discord](https://discord.gg/tenstorrent)
