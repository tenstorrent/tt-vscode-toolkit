#!/usr/bin/env python3
"""
Proof 2: Codebase Explorer
Framework: OpenAI Agents SDK
Model: Qwen3-32B (default) or Llama-3.3-70B-Instruct (--model flag)

Demonstrates:
  - Explicit JSON function calling (read_file, list_files, grep_code)
  - File system navigation via tools
  - Quality difference: 32B vs 70B visible on code comprehension tasks

Usage:
  python3 02_code_explorer.py                           (explores current directory)
  python3 02_code_explorer.py --dir ~/code/myproject
  python3 02_code_explorer.py --model "meta-llama/Llama-3.3-70B-Instruct" --compare
  python3 02_code_explorer.py --query "What API endpoints does this expose?"

Install deps:
  pip install openai-agents openai
"""
import argparse
import asyncio
import glob as _glob
import os
import re as _re
import sys

try:
    from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_trace_processors
except ImportError:
    print("ERROR: openai-agents not installed. Run: pip install openai-agents")
    sys.exit(1)

# Disable SDK tracing — we're local-only, there's no OpenAI API key to authenticate with
set_trace_processors([])

import openai

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-32B")
MAX_FILE_BYTES = 32_000  # cap reads to avoid context overflow


# --- Tools ---

@function_tool
def read_file(path: str) -> str:
    """Read a text file from disk. Returns file contents (truncated at 32KB)."""
    try:
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            return f"ERROR: File not found: {path}"
        with open(path, "r", errors="replace") as f:
            content = f.read(MAX_FILE_BYTES)
        if len(content) == MAX_FILE_BYTES:
            content += "\n... [truncated at 32KB]"
        return content
    except Exception as e:
        return f"ERROR reading {path}: {e}"


@function_tool
def list_files(directory: str, pattern: str = "*") -> str:
    """List files in a directory matching an optional glob pattern (e.g. '*.py', '*.md')."""
    try:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            return f"ERROR: Directory not found: {directory}"
        matches = _glob.glob(os.path.join(directory, pattern))
        matches.sort()
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        lines = [os.path.relpath(m, directory) for m in matches[:50]]
        result = "\n".join(lines)
        if len(matches) > 50:
            result += f"\n... and {len(matches) - 50} more"
        return result
    except Exception as e:
        return f"ERROR listing {directory}: {e}"


@function_tool
def grep_code(pattern: str, directory: str, file_extension: str = ".py") -> str:
    """Search files in a directory for a regex pattern. Returns matching lines with file:line context."""
    try:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            return f"ERROR: Directory not found: {directory}"
        results = []
        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                if not fname.endswith(file_extension):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if _re.search(pattern, line):
                                rel = os.path.relpath(fpath, directory)
                                results.append(f"{rel}:{i}: {line.rstrip()}")
                                if len(results) >= 40:
                                    results.append("... [limited to 40 matches]")
                                    return "\n".join(results)
                except OSError:
                    continue
        if not results:
            return f"No matches for '{pattern}' in {directory} ({file_extension} files)"
        return "\n".join(results)
    except Exception as e:
        return f"ERROR grepping {directory}: {e}"


SYSTEM_PROMPT = """You are an expert code navigator with tools to read files, list directories, and search for patterns.

When asked a question about a codebase:
1. Use list_files and grep_code to locate relevant files
2. Use read_file to examine the code in detail
3. Answer precisely with file paths and line references for every claim

Be concise and cite specific file:line locations."""


def make_agent(model_id: str) -> Agent:
    # Use OpenAIChatCompletionsModel so the SDK routes to our local vLLM endpoint
    # instead of trying to resolve the model string as a provider prefix.
    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key="none")
    model = OpenAIChatCompletionsModel(model=model_id, openai_client=client)
    return Agent(
        name="CodeExplorer",
        instructions=SYSTEM_PROMPT,
        tools=[read_file, list_files, grep_code],
        model=model,
    )


async def run_query(agent: Agent, query: str) -> str:
    try:
        result = await Runner.run(agent, query)
        return result.final_output
    except Exception as e:
        if "maximum context length" in str(e) or "context_length_exceeded" in str(e):
            return (
                "[Context limit reached before the agent could finish.\n"
                "Tips:\n"
                "  • Use --query with a narrower question (one file, one function, one concept)\n"
                "  • Use --dir pointing at a subdirectory rather than the whole project\n"
                "  • Ask about specific files directly: --query 'explain src/auth/login.py'\n]"
            )
        raise


def main():
    parser = argparse.ArgumentParser(description="Codebase explorer agent demo (OpenAI Agents SDK)")
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory to explore (default: current working directory)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID on vLLM")
    parser.add_argument("--query", default=None, help="Custom question about the codebase")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the same query with both Qwen3-32B and Llama-3.3-70B and compare output",
    )
    args = parser.parse_args()

    target_dir = os.path.abspath(os.path.expanduser(args.dir))
    default_query = (
        f"Explore the directory {target_dir}. "
        "Summarize what this codebase does, how it is organized, "
        "and which files are most important to understand first. "
        "List the top 5 files with a one-sentence description of each."
    )
    query = args.query or default_query

    print("=" * 70)
    print("tt-agents Proof 2: Codebase Explorer (OpenAI Agents SDK)")
    print("=" * 70)
    print(f"Endpoint: {BASE_URL}")
    print(f"Directory: {target_dir}")
    print(f"\nQuery:\n  {query}\n")

    models = ["Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B-Instruct"] if args.compare else [args.model]

    for model_id in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print("=" * 70)
        agent = make_agent(model_id)
        result = asyncio.run(run_query(agent, query))
        print(result)

    print("\n✓ Code exploration complete")


if __name__ == "__main__":
    main()
