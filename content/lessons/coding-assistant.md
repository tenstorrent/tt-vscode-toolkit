---
id: coding-assistant
title: Coding Assistant with Aider
description: >-
  Run a real AI coding assistant (Aider) against your local Tenstorrent
  vLLM server. One install, one command — then pair-program with your own
  on-device LLM. Also covers prompt engineering to customize behavior.
category: applications
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
status: draft
estimatedMinutes: 15
---

# Coding Assistant with Aider

## Overview

[Aider](https://aider.chat/) is an open-source AI pair programming tool that runs in
your terminal. It reads your codebase, edits files, and auto-commits changes — and it
speaks the OpenAI API, so you can point it directly at your Tenstorrent vLLM server.

**Works on all hardware** — N150/N300/T3K/P100/P300c/Galaxy. No `~/tt-metal` required.

> **Prerequisites:**
> Your vLLM server must be running before launching Aider.
> See [vLLM Production](command:tenstorrent.showLesson?["vllm-production"]) to start it.

---

## Step 1: Install Aider

```bash
pip install aider-chat
```

That's the only install. Aider is a self-contained Python package with no tt-metal
dependencies. Install it in any environment — your system Python, a separate venv,
or even `pipx`:

```bash
# Isolated install (keeps dependencies separate from tt-metal venv)
pipx install aider-chat
```

---

## Step 2: Connect to Your Local LLM

Set two environment variables to point Aider at your running vLLM server:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=fake       # vLLM doesn't check the key; any value works
```

Then launch Aider with the model you started in vLLM:

```bash
# If you started vLLM with Qwen3-0.6B (works on all hardware):
aider --model openai/Qwen3-0.6B

# If you started vLLM with Qwen3-8B or Llama 3.1 8B:
aider --model openai/Qwen3-8B
aider --model openai/Llama-3.1-8B-Instruct
```

> The `openai/` prefix tells Aider to use your custom API base instead of
> calling OpenAI's servers.

---

## Step 3: Make It Permanent (Optional but Recommended)

Drop a `.aider.conf.yml` in your project root to skip the env vars and flags on
every launch:

```yaml
# .aider.conf.yml  (commit this to your repo!)
openai-api-base: http://localhost:8000/v1
openai-api-key: fake
model: openai/Qwen3-0.6B
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
openai-api-base: http://localhost:8000/v1
openai-api-key: fake
model: openai/Qwen3-0.6B
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
| **Aider + vLLM (this lesson)** | Day-to-day coding; editing actual project files |
| **Direct API (Lessons 4–5)** | Learning tt-metal internals; building custom tools |
| **vLLM chat interface** | Quick Q&A; no file editing needed |

**Key insight:** Aider's system prompt + your local Tenstorrent LLM gives you ~80–90% of
a hosted coding assistant at zero API cost. The model runs on your hardware, your code
stays private, and you can swap models in seconds.

---

## Troubleshooting

**"Model not found" or API errors:**
- Verify vLLM is running: `curl http://localhost:8000/v1/models`
- Check the model name in the response matches what you passed to `--model`
- The model name must match exactly: `openai/Qwen3-0.6B` where `Qwen3-0.6B` matches
  the model slug in vLLM's response

**"No response" or very slow responses:**
- Check vLLM server logs for errors
- Qwen3-0.6B is fastest; larger models take longer per token
- Reduce context by starting Aider in a subdirectory rather than the repo root

**Aider edits files unexpectedly:**
- Aider only edits files you've mentioned or it has inferred are relevant
- Use `--dry-run` to preview changes without applying them
- All edits are git commits — `git log` and `git diff HEAD~1` let you review everything

**N150 / QB2 model recommendation:**
- Use `Qwen3-0.6B` — it's fast, reasoning-capable, and fits comfortably in N150 DRAM
- The prompting techniques here work identically regardless of model size

---

## Key Takeaways

✅ **Aider is zero-conf** — one install, two env vars, then `aider`

✅ **Your hardware, your data** — the LLM runs on-device, nothing leaves your machine

✅ **Prompt engineering is universal** — the system prompt techniques here work with
   every LLM you'll encounter

✅ **Persistent config** — `.aider.conf.yml` in your repo makes it permanent

✅ **All hardware supported** — Qwen3-0.6B via vLLM works on N150, N300, P300c, and above
