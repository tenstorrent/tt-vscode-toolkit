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
  - galaxy
status: draft
estimatedMinutes: 10
---

# Lesson 9: Coding Assistant with Prompt Engineering

## Future Model Options (Coming Soon as of December 2025)

As tt-metal model support expands, you'll have access to specialized coding models:

**🔮 On the Horizon:**
- **Llama 3.2 6B AlgoCode** - Specialized for algorithms, data structures, debugging
  - *Blocker:* Weight conversion needed for tt-metal tile alignment (32x32)
  - *Status:* Community fine-tune, requires model adaptation layer

- **Qwen 2.5 Coder 7B** - Code generation, explanation, documentation
  - *Blocker:* Requires N300 (TP=2, dual-chip) - N150 single-chip not supported
  - *Status:* Waiting for single-chip optimization or N300 availability

- **CodeLlama variants** - Multi-language code completion
  - *Blocker:* Model architecture compatibility with Generator API
  - *Status:* Research phase

- **StarCoder2** - Open-source code generation
  - *Blocker:* Architecture differs from Llama family
  - *Status:* Requires custom tt-metal implementation

**🎯 Today's Approach:**
While we wait for these specialized models, we'll use **Llama 3.1 8B** with coding-focused prompt engineering. This teaches a critical skill: **getting maximum value from available models through smart prompting**.

Prompt engineering often delivers 80% of the value of model specialization, with zero compatibility issues!

---

## Overview

Build a **coding assistant** powered by Llama 3.1 8B running on your N150 hardware using tt-metal's Direct API. This lesson focuses on **prompt engineering** - the art of shaping model behavior through system prompts and conversation structure.

**Why Prompt Engineering?**
- ✅ **Works today** - Uses proven tt-metal compatible model
- ✅ **Already downloaded** - No additional model required (from Lesson 3)
- ✅ **Fast** - Direct API keeps model in memory (1-3 sec per query)
- ✅ **Educational** - Learn how prompting shapes model behavior
- ✅ **Transferable skills** - Prompt engineering works across all models
- ✅ **Real-world technique** - Production systems use prompt engineering heavily

**What You'll Build:**
- Interactive CLI coding assistant
- System prompt optimized for coding tasks
- Context management for multi-turn debugging
- Foundation for custom developer tools

**Performance:**
- Model loads once (2-5 min), then fast queries (1-3 sec)
- Same hardware acceleration as Lesson 4
- Native tt-metal performance on N150

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

These were installed in Lesson 4, but run again if needed:

**Required packages:**
```bash
pip install pi && pip install git+https://github.com/tenstorrent/llama-models.git@tt_metal_tag
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
cd ~/tt-metal && \
  export LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original && \
  export PYTHONPATH=$(pwd) && \
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

| Aspect | Prompt Engineering (Today) | Specialized Model (Future) |
|--------|----------------------------|----------------------------|
| **Model** | Llama 3.1 8B (general) | AlgoCode/Qwen Coder (specialized) |
| **Setup** | Add system prompt | Download + convert weights |
| **Compatibility** | ✅ Works today | ❌ Requires model adaptation |
| **Performance** | 1-3 sec/query | 1-3 sec/query (similar) |
| **Quality** | 80-85% of specialized | 100% (trained on code) |
| **Flexibility** | Easy to modify prompts | Fixed model behavior |
| **Learning Value** | High - transferable skill | Medium - model-specific |
| **Production Use** | ✅ Common approach | ✅ When compatibility exists |

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

## Performance on N150

**Hardware:** Tenstorrent N150 (single chip, 72 Tensix cores)

**Performance metrics:**
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
- **Lesson 4:** Same Direct API, general chat → Learn the pattern
- **Lesson 5:** Same approach + HTTP API → Add network access
- **Lesson 9:** Same pattern + specialized prompting → Domain expertise

**Learning Path:**
1. Master Direct API (Lessons 4, 9)
2. Add HTTP layer (Lesson 5)
3. Scale to production (Lesson 6 - vLLM)
4. Integrate into tools (Lesson 7 - VSCode)

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
- Run Step 2 again to install pi and llama-models
- Ensure using Tenstorrent's llama-models fork

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

**Looking ahead:**
As tt-metal model support expands and weight conversion tools mature, you'll be able to swap in specialized coding models (AlgoCode, Qwen Coder, etc.) using the same architecture. The prompt engineering skills you learned here will remain valuable across all models!

