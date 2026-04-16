---
id: coding-assistant
title: Coding Assistant with Prompt Engineering
description: >-
  Build an AI coding assistant using Llama 3.1 8B and prompt engineering. Learn
  how to shape model behavior through system prompts - a critical real-world
  skill!
category: applications
tags:
  - coding
  - assistant
  - model
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: draft
estimatedMinutes: 10
---

# Coding Assistant with Prompt Engineering

## Overview

Build a **coding assistant** powered by Llama 3.1 8B running on your Tenstorrent
hardware using tt-metal's Direct API. This lesson focuses on **prompt engineering** —
the art of shaping model behavior through system prompts and conversation structure.

> **Hardware requirement: N300 / T3K / P100 / P300c or larger.**
> Llama 3.1 8B exhausts DRAM on a single N150 chip.
>
> **No `~/tt-metal` built yet?** See
> [Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"]) first.
>
> **N150 or QB2 users:** Skip to
> [vLLM Production](command:tenstorrent.showLesson?["vllm-production"]) and use
> `--model Qwen3-8B` with a coding system prompt — same result, no source build needed.

---

### Model Options for Coding Assistants (April 2026)

Several coding-capable models now run on Tenstorrent hardware:

| Model | Hardware | API | Notes |
|-------|----------|-----|-------|
| **Llama 3.1 8B** | N300/T3K/P100/P300c | Direct API or vLLM | General-purpose; best via vLLM |
| **Qwen3-0.6B** | N150+ (all hardware) | vLLM | Tiny, fast, reasoning-capable |
| **Qwen3-8B** | N300/T3K/P100/P300c | vLLM | Strong coding & math |
| **Qwen3-32B+** | T3K/Galaxy | vLLM | SOTA coding performance |

**This lesson uses Llama 3.1 8B via the Direct API** — the same pattern as Lessons 4–5,
now specialized for coding tasks through prompt engineering.

For a vLLM-based coding assistant (no `~/tt-metal` required, easier startup),
see [vLLM Production](command:tenstorrent.showLesson?["vllm-production"]) and use
`--model Qwen3-8B` with a coding system prompt.

---

**Why Prompt Engineering?**
- ✅ **Works today** — Uses proven tt-metal compatible model
- ✅ **Already downloaded** — No additional model required (from Lesson 3)
- ✅ **Fast** — Direct API keeps model in memory (1-3 sec per query)
- ✅ **Educational** — Learn how prompting shapes model behavior
- ✅ **Transferable skills** — Prompt engineering works across all models
- ✅ **Real-world technique** — Production systems use prompt engineering heavily

**What You'll Build:**
- Interactive CLI coding assistant
- System prompt optimized for coding tasks
- Context management for multi-turn debugging
- Foundation for custom developer tools

**Performance:**
- Model loads once (2-5 min), then fast queries (1-3 sec)
- Same hardware acceleration as Lesson 4
- Native tt-metal performance on N150+

---

## Step 1: Verify Model Availability

Llama 3.1 8B should already be downloaded from Lesson 3. Let's verify:

**Check command:**
```bash
ls -lh ~/models/Llama-3.1-8B-Instruct/original/
```

**Expected files:**
- `consolidated.00.pth` - Model weights
- `params.json` - Model configuration
- `tokenizer.model` - Tokenizer

If missing, re-run the download from Lesson 3:

[📥 Download Model (Lesson 3)](command:tenstorrent.downloadModel)

---

## Step 2: Install Dependencies (If Not Already Done)

The tt-metal Python venv from `create_venv.sh` includes most dependencies. If you
need the Tenstorrent llama-models package (for the Generator API tokenizer):

```bash
source ~/tt-metal/python_env/bin/activate
pip install git+https://github.com/tenstorrent/llama-models.git@tt_metal_tag
```

[🔧 Install Dependencies](command:tenstorrent.installInferenceDeps)

---

## Step 3: Create Coding Assistant Script

This script uses the same Direct API pattern as Lesson 4, but with a **coding-optimized system prompt**.

**What makes it a coding assistant:**
1. **System prompt** - Instructs model to focus on code, algorithms, debugging
2. **Temperature** - Lower temperature (0.7) for more focused, deterministic outputs
3. **Context length** - Adequate tokens for code snippets
4. **Example prompts** - Guides users toward coding tasks

**Script location:** `~/tt-scratchpad/tt-coding-assistant.py`

[📝 Create Coding Assistant Script](command:tenstorrent.createCodingAssistantScript)

**Key features:**
- Coding-focused system prompt
- Multi-turn conversation support
- Clean REPL interface
- Educational code comments

---

## Step 4: Start Coding Assistant

Launch your coding assistant! Model loads once (2-5 min), then enjoy fast responses.

**Environment setup:**
```bash
source ~/tt-metal/python_env/bin/activate
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original

# For P100 / P300c (Blackhole):
# export TT_METAL_ARCH_NAME=blackhole

python3 ~/tt-scratchpad/tt-coding-assistant.py
```

[💬 Start Coding Assistant](command:tenstorrent.startCodingAssistant)

**Usage tips:**
- First message takes 2-5 minutes (model loading + compilation)
- Subsequent messages: 1-3 seconds
- Type coding questions, algorithms, debugging requests
- Exit with Ctrl+C

**Example prompts:**
```text
> Write a Python function to find the longest palindrome substring

> Explain how quicksort works with pseudocode

> Debug this code: [paste your code]

> Suggest a data structure for implementing a task scheduler

> What's the time complexity of this algorithm? [paste code]

> Refactor this function to be more efficient: [paste code]
```

---

## The Power of Prompt Engineering

**System Prompt Architecture:**

Our coding assistant uses a carefully crafted system prompt that guides Llama 3.1 8B to behave like a coding expert:

```python
SYSTEM_PROMPT = """You are an expert coding assistant specializing in:
- Algorithm design and analysis
- Data structures (trees, graphs, heaps, hash tables)
- Code debugging and optimization
- Explaining complex programming concepts
- Writing clean, well-documented code
- Time and space complexity analysis

When answering:
- Provide clear, concise explanations
- Include code examples when relevant
- Explain your reasoning
- Consider edge cases
- Suggest optimizations where applicable
- Use appropriate programming language syntax

Focus on being helpful, accurate, and educational."""
```

**Why This Works:**

1. **Explicit role definition** - "You are an expert coding assistant"
2. **Enumerated capabilities** - Lists exactly what it should do
3. **Response structure guidance** - "When answering..." section
4. **Quality guidelines** - "Clear, concise, educational"

**Prompt Engineering Techniques:**

| Technique | Effect | Example |
|-----------|--------|---------|
| **Role Assignment** | Sets behavioral context | "You are an expert coding assistant" |
| **Task Enumeration** | Defines scope | "specializing in: algorithm design..." |
| **Response Format** | Shapes output structure | "Provide clear explanations" |
| **Constraint Setting** | Focuses behavior | "Consider edge cases" |
| **Quality Markers** | Influences style | "Be helpful, accurate, educational" |

---

## Comparing Approaches: Prompt Engineering vs Model Specialization

| Aspect | Prompt Engineering (This lesson) | Dedicated Coder Model |
|--------|----------------------------------|-----------------------|
| **Model** | Llama 3.1 8B (general) | Qwen3-8B (coding-optimized) |
| **Setup** | System prompt only | vLLM with `--model Qwen3-8B` |
| **Hardware** | N300/T3K/P100/P300c | N300/T3K/P100/P300c |
| **Performance** | 1-3 sec/query | 1-3 sec/query |
| **Quality** | Excellent (Llama code quality is strong) | SOTA coding benchmark performance |
| **Flexibility** | Easy to modify prompts | vLLM OpenAI-compatible API |
| **Learning Value** | High — prompt skills transfer to all LLMs | Good — production API patterns |
| **Production Use** | ✅ Common approach | ✅ Preferred when hardware permits |

**Key Insight:** Prompt engineering often delivers 80%+ of specialized model quality with zero compatibility issues!

---

## Advanced Prompt Engineering Techniques

**1. Few-Shot Learning:**
Include examples in your prompt to guide behavior:

```python
> Explain bubble sort

# Add context in your prompt:
> Like you explained quicksort with pseudocode, explain bubble sort with pseudocode and complexity analysis
```

**2. Chain-of-Thought Prompting:**
Ask the model to show its reasoning:

```python
> Step by step, explain how to solve the two-sum problem
```

**3. Constrained Generation:**
Set specific output requirements:

```python
> Write a Python function to reverse a linked list. Include:
  - Function signature with type hints
  - Docstring explaining parameters and return value
  - Time and space complexity in comments
  - Example usage
```

**4. Iterative Refinement:**
Build on previous responses:

```python
> Write a binary search function
# ... model responds ...
> Now add error handling for empty lists
# ... model refines ...
> Now add type hints and a docstring
```

---

## Architecture: Direct API Pattern

**The Generator API Pattern (same as Lesson 4):**

```python
# Load model once (slow - 2-5 minutes)
model_args, model, tt_kv_cache, _ = create_tt_model(
    mesh_device,
    instruct=True,
    max_batch_size=1,
    max_seq_len=2048,
)
generator = Generator([model], [model_args], mesh_device, ...)

# Chat loop (fast - 1-3 seconds per query!)
while True:
    prompt = get_user_input()

    # Prepend system prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

    # Prefill: process prompt
    logits = generator.prefill_forward_text(tokens, ...)

    # Decode: generate response
    for i in range(max_tokens):
        next_token = generator.decode_forward_text(...)
        if is_end_token(next_token):
            break

    print(response)
```

**Why This Works:**
- Model loads once, stays in memory
- System prompt prepended to every query
- Same fast performance as Lesson 4
- Full control over generation parameters

---

## Customization Ideas

**1. Domain-Specific Assistants:**

Modify the system prompt for different domains:

**Web Development:**
```python
SYSTEM_PROMPT = """You are a web development expert specializing in:
- React, TypeScript, Node.js
- REST APIs and GraphQL
- Database design (SQL, MongoDB)
- Authentication and security
- Performance optimization
..."""
```

**Systems Programming:**
```python
SYSTEM_PROMPT = """You are a systems programming expert specializing in:
- C, C++, Rust
- Memory management and pointers
- Concurrency and parallelism
- Low-level optimization
- Debugging segfaults and race conditions
..."""
```

**2. Response Format Control:**

Add format requirements to system prompt:

```python
SYSTEM_PROMPT += """

Always format code in markdown code blocks with language tags:
```
def example():
    pass
```python

For algorithms, include:
1. High-level explanation
2. Code implementation
3. Time/space complexity
4. Example walkthrough
"""
```

**3. Context Management:**

Implement conversation history for multi-turn debugging:

```python
conversation_history = []

while True:
    user_input = input("> ")
    conversation_history.append({"role": "user", "content": user_input})

    # Build full context
    full_prompt = SYSTEM_PROMPT
    for msg in conversation_history[-5:]:  # Last 5 turns
        full_prompt += f"\n{msg['role']}: {msg['content']}"

    response = generate(full_prompt)
    conversation_history.append({"role": "assistant", "content": response})
```

**4. Code Execution Integration:**

Extend the assistant to run code:

```python
import subprocess

def execute_python(code):
    result = subprocess.run(['python3', '-c', code],
                          capture_output=True, text=True, timeout=5)
    return result.stdout or result.stderr

# In chat loop:
if "```python" in response:
    code = extract_code_block(response)
    print("\n🔧 Execute this code? (y/n)")
    if input().lower() == 'y':
        output = execute_python(code)
        print(f"Output:\n{output}")
```

---

## Real-World Applications

**1. Interactive Code Review:**
```python
> Review this function for bugs and suggest improvements:
  [paste code]
```

**2. Algorithm Learning:**
```python
> Teach me dynamic programming by explaining the coin change problem
```

**3. Debugging Assistant:**
```python
> I'm getting "IndexError: list index out of range" in this code:
  [paste code]
  Help me find and fix the bug
```

**4. Code Translation:**
```python
> Translate this Python function to Rust:
  [paste code]
```

**5. Test Generation:**
```python
> Write pytest unit tests for this function:
  [paste code]
```

**6. Documentation Generation:**
```python
> Write a comprehensive docstring for this function:
  [paste code]
```

---

## Performance

**Performance metrics (Llama 3.1 8B Direct API):**
- **First run:** 2-5 minutes (model load + kernel compilation)
- **Subsequent queries:** 1-3 seconds per response
- **Context length:** 128K tokens (Llama 3.1 native support)
- **Memory:** ~16GB for model weights + KV cache

**Optimization tips:**
- Keep system prompt concise (reduces prefill time)
- Use shorter user prompts for faster responses
- Adjust `max_tokens` parameter (default: 256 for coding tasks)
- Lower temperature (0.7) gives more consistent outputs

---

## What's Next?

**Extend Your Coding Assistant:**

1. **Add File I/O:**
   - Read source files from disk
   - Analyze entire codebases
   - Generate reports

2. **Implement RAG (Retrieval-Augmented Generation):**
   - Index your codebase with embeddings
   - Retrieve relevant context for queries
   - Answer questions about large projects

3. **Build IDE Integration:**
   - VSCode extension (you're building one now!)
   - Vim/Emacs plugins
   - LSP (Language Server Protocol) integration

4. **Add Specialized Tools:**
   - Static analysis integration (pylint, mypy)
   - Git integration (commit message generation)
   - Documentation generation (Sphinx, JSDoc)

5. **Multi-Model Orchestration:**
   - Use Llama for code generation
   - Use smaller models for classification
   - Ensemble different approaches

**Compare with Other Lessons:**
- **Interactive Chat** — Same Direct API, general chat → Learn the pattern
- **API Server** — Same approach + HTTP API → Add network access
- **Coding Assistant (this lesson)** — Same pattern + specialized prompting → Domain expertise

**Learning Path:**
1. Master Direct API (Interactive Chat, Coding Assistant)
2. Add HTTP layer (API Server)
3. Scale to production (vLLM Production)
4. Integrate into tools (VSCode Chat Integration)

---

## Troubleshooting

**Model load fails:**
- Ensure Llama 3.1 8B downloaded: `ls ~/models/Llama-3.1-8B-Instruct/original/`
- Verify LLAMA_DIR points to `original/` subdirectory
- Check PYTHONPATH includes tt-metal directory

**Slow responses:**
- First run is always slow (compilation)
- Check device with `tt-smi` - should show N150 healthy
- Ensure no other processes using tt-metal
- Reduce `max_tokens` if generating too much text

**Responses not coding-focused:**
- Check system prompt is being prepended
- Try more specific prompts ("Write Python code to...")
- Lower temperature for more deterministic output
- Add examples to prompt (few-shot learning)

**Out of memory:**
- Close other tt-metal processes
- Use device management commands to reset/clear state
- Reduce `max_seq_len` in model initialization

**Dependencies missing:**
- Run Step 2 again to install llama-models
- Ensure using Tenstorrent's llama-models fork (not the Meta upstream)
- Verify tt-metal venv is active: `which python3` should point to `python_env/`

---

## Key Takeaways

✅ **Prompt engineering is powerful** - 80%+ of specialized model quality

✅ **Works with available hardware** - No compatibility issues

✅ **Fast after initial load** - Model stays in memory between queries

✅ **Transferable skill** - Applies to all LLMs (GPT, Claude, Gemini, etc.)

✅ **Production-ready technique** - Real systems use prompt engineering heavily

✅ **Foundation for custom tools** - Extend with file I/O, RAG, integrations

**You now have:**
- A working AI coding assistant running on tt-metal
- Understanding of prompt engineering techniques
- Skills to customize behavior through prompting
- Foundation for building custom developer tools
- Experience with the Direct API pattern

**Next steps:**
- Experiment with different system prompts
- Try domain-specific customizations
- Add context management for multi-turn conversations
- Integrate with your development workflow
- Build custom tools using the same pattern

**Swap in a stronger coding model:**
Qwen3-8B (available via vLLM on N300/T3K/P100/P300c) delivers SOTA coding benchmark
performance. Apply the same prompt engineering techniques from this lesson with
`--model Qwen3-8B` in the
[vLLM Production lesson](command:tenstorrent.showLesson?["vllm-production"]).

**N150 and QB2 users:**
Run Qwen3-0.6B via vLLM with a coding system prompt — it's remarkably capable for
its size and is the easiest path to an on-device coding assistant on single-chip hardware.

