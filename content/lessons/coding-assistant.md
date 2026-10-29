---
id: coding-assistant
title: Coding Assistant with Aider
description: >-
  Pair-program with Aider against the model tt is serving: one uv install,
  one tt launch. Then shape how it codes with a system prompt.
category: connect
tags:
  - coding
  - assistant
  - model
  - aider
  - vllm
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
order: 2
---

# Coding Assistant with Aider

## Overview

[Aider](https://aider.chat/) is an open source AI pair programming tool that runs in
your terminal. It reads your codebase, edits files, and auto-commits changes. It speaks
the OpenAI API, so `tt launch` can point it at the model you are serving.

**Works on all hardware**, and on the cloud pod. No `~/tt-metal` required.

> **Prerequisite:** a model is being served. On your own machine that is
> [Serve a Model](command:tenstorrent.lesson.show?["interactive-chat"]) (`tt serve`); on
> the cloud pod it is
> [Serve Without Containers](command:tenstorrent.lesson.show?["api-server"]). Aider needs
> a model that calls tools; `Llama-3.1-8B-Instruct` and `Qwen3-32B` both do.

---

## Step 1: Install Aider

```bash
uv tool install aider-chat
```

[📦 Install Aider](command:tenstorrent.launch.installAider)

That is the only install. Aider is a self-contained Python package with no TT-Metalium<sup>™</sup>
dependencies, and `uv tool` keeps it in its own environment, away from the tt-metal venv.
`pipx install aider-chat` does the same where you prefer pipx.

---

## Step 2: Connect to Your Local LLM

```bash
tt launch aider --dry-run
tt launch aider
```

[👀 Dry run: Aider](command:tenstorrent.launch.aiderDryRun)
[🤖 Launch Aider](command:tenstorrent.launch.aider)

The dry run prints the model `tt` found, its endpoint, and the exact `aider` command it
would run. `tt launch` sets `OPENAI_API_BASE` and `OPENAI_API_KEY` for that one run and
passes `--model openai/<served model>`, so nothing lands on disk. If `tt` finds no server,
it says so and names the port it tried; `--port 8000` or `--url http://host:port/v1`
points it elsewhere.

By hand, the same thing is two environment variables and a flag:

```bash
export OPENAI_API_BASE=http://localhost:20000/v1
export OPENAI_API_KEY=tt-local       # vLLM does not check the key; any value works
aider --model openai/meta-llama/Llama-3.1-8B-Instruct
```

> The `openai/` prefix tells Aider to use your custom API base instead of
> calling OpenAI's servers. The model name must match what the server reports at
> `/v1/models`, which is the Hugging Face name for a catalog model.

---

## Step 3: Make It Permanent (Optional but Recommended)

Drop a `.aider.conf.yml` in your project root to skip the env vars and flags on
every launch:

```yaml
# .aider.conf.yml  (commit this to your repo!)
openai-api-base: http://localhost:20000/v1
openai-api-key: tt-local
model: openai/meta-llama/Llama-3.1-8B-Instruct
```

Now in that project directory, just run:

```bash
aider
```

No flags, no env vars. Aider reads the config file automatically.

---

## What Aider Can Do

Once running, Aider operates in a REPL. Drop a question or instruction:

```
> Write a Python function to find all primes up to N using the Sieve of Eratosthenes

> Add type hints and a docstring to the function in sieve.py

> Refactor this: [paste code or just name the file]

> Debug this error: IndexError: list index out of range

> Write pytest tests for the sieve function
```

Aider will:
1. Read relevant files from your repo automatically
2. Generate code changes
3. Show you a diff
4. Ask to apply and auto-commit with a sensible message

**Exit:** Type `/exit` or press Ctrl+C.

---

## Customizing Aider's Behavior via Prompt Engineering

Aider uses a built-in system prompt optimized for code editing. You can augment it
with your own instructions in `.aider.conf.yml`:

```yaml
# .aider.conf.yml
openai-api-base: http://localhost:20000/v1
openai-api-key: tt-local
model: openai/meta-llama/Llama-3.1-8B-Instruct
system-prompt: |
  You are a Python expert. Always include:
  - Type hints on all functions
  - Docstrings in Google style
  - Time and space complexity as inline comments
  Prefer readability over cleverness.
```

Or via a separate file:

```bash
# Write your system prompt to a file
cat > ~/coding-assistant-prompt.txt << 'EOF'
You are an expert Python engineer at a systems programming company.
- Always add type hints
- Include complexity analysis
- Prefer stdlib over third-party where possible
- Flag security issues when you see them
EOF

# Use it with Aider
aider --system-prompt ~/coding-assistant-prompt.txt
```

### Why Prompt Engineering Matters

The system prompt shapes everything about how the model responds:

| Technique | Effect | Example |
|-----------|--------|---------|
| **Role Assignment** | Sets behavioral context | "You are an expert in memory management" |
| **Task Enumeration** | Defines scope | "specializing in: Rust, C++, systems code" |
| **Quality Markers** | Influences style | "Always add tests" |
| **Constraint Setting** | Focuses behavior | "Never use eval(); flag unsafe patterns" |
| **Response Format** | Shapes output structure | "Include complexity analysis in comments" |

These same techniques work for all LLMs — what you learn here applies to GPT, Claude,
Gemini, or any hosted model.

---

## One-Shot Mode (No REPL)

For quick questions without entering the REPL:

```bash
# Ask a question about a file
aider --message "Explain what this does and flag any bugs" mymodule.py

# Generate something from scratch
aider --message "Create a Flask app with a /health endpoint" --no-git

# Pipe code for a quick review
cat suspicious_code.py | aider --message "Review this for security issues" -
```

---

## Comparing Approaches

| Approach | When to Use |
|----------|-------------|
| **Aider (this lesson)** | Day-to-day coding; editing actual project files |
| **Open WebUI** ([Launch Software](command:tenstorrent.lesson.show?["launch-software"])) | Quick Q&A in a browser; no file editing needed |
| **Your own code against the endpoint** | Agents and pipelines, as in the next two lessons |

**Key insight:** Aider's system prompt + your local Tenstorrent LLM gives you ~80–90% of
a hosted coding assistant at zero API cost. The model runs on your hardware, your code
stays private, and you can swap models in seconds.

---

## Troubleshooting

**"Model not found" or API errors:**
- Check the server: `tt model ps`, or `curl http://localhost:20000/v1/models`
- `tt launch aider` reads the model name from the server, so a mismatch only happens by
  hand: `openai/` plus the exact `id` the server reports

**"No response" or very slow responses:**
- Check the server log: `tt model logs`
- An 8B model answers faster than a 32B one; larger models take longer per token
- Reduce context by starting Aider in a subdirectory rather than the repo root

**Aider edits files unexpectedly:**
- Aider only edits files you've mentioned or it has inferred are relevant
- Use `--dry-run` to preview changes without applying them
- All edits are git commits — `git log` and `git diff HEAD~1` let you review everything

**Which model:**
- One chip: `Llama-3.1-8B-Instruct`. TT-QuietBox<sup>®</sup> 2: `Qwen3-32B` (see the
  thinking caveat in [Launch Software](command:tenstorrent.lesson.show?["launch-software"]))
- The prompting techniques here work identically regardless of model size

---

## Key Takeaways

✅ **Aider is zero-conf** — one `uv tool install`, then `tt launch aider`

✅ **Your hardware, your data** — the LLM runs on-device, nothing leaves your machine

✅ **Prompt engineering is universal** — the system prompt techniques here work with
   every LLM you'll encounter

✅ **Persistent config** — `.aider.conf.yml` in your repo makes it permanent

✅ **All hardware supported** — any model `tt serve` or the pod's vLLM can serve, on n150 and above
