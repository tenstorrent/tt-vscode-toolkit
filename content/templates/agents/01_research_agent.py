#!/usr/bin/env python3
"""
Proof 1: Research Agent
Framework: smolagents (HuggingFace)
Model: Qwen3-32B (hermes tool parser) — or swap MODEL_ID for Llama-3.3-70B

Demonstrates:
  - Web search via DuckDuckGo
  - Web page fetching and parsing
  - Multi-step research synthesis
  - ~60 lines of agent code

Usage:
  python3 01_research_agent.py                          (interactive topic picker)
  python3 01_research_agent.py --headless               (auto-pick today's topic, no prompt)
  python3 01_research_agent.py --query "your question"  (skip picker entirely)
  python3 01_research_agent.py --model "meta-llama/Llama-3.3-70B-Instruct"

Install deps:
  pip install smolagents ddgs beautifulsoup4 markdownify requests
"""
import argparse
import datetime
import os
import select
import sys
from pathlib import Path

try:
    from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, VisitWebpageTool
except ImportError:
    print("ERROR: smolagents not installed. Run: pip install smolagents ddgs")
    sys.exit(1)

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-32B")
RESEARCH_FILE = Path(__file__).parent / "last_research.txt"

SUGGESTIONS = [
    # Tech
    (
        "Python AI agent libraries",
        "What Python libraries are most popular for building AI agents right now? "
        "Compare smolagents, LangChain, AutoGen, and any strong newcomers — "
        "with a real use case that shows each one's strengths."
    ),
    # Travel
    (
        "East Coast music festivals 2026",
        "Which music festivals on the US East Coast are worth attending in spring, "
        "summer, or fall 2026? Cover genre, vibe, rough ticket cost, and what "
        "makes each one distinctly worth the trip."
    ),
    # Philosophy / AI
    (
        "Walter Benjamin on AI video and the prompt artist",
        "Apply Walter Benjamin's theory of 'the aura' and mechanical reproduction "
        "to AI video generation, deepfakes, and tools like Wan 2.2. What does it "
        "mean to be an author when your primary creative act is writing a prompt?"
    ),
    # Home & Garden
    (
        "Attracting beneficial insects to any garden",
        "What are the most effective ways to attract and keep beneficial insects — "
        "ladybugs, lacewings, ground beetles — in a garden? Focus on strategies "
        "that work across climates and don't require buying insects online."
    ),
    # Gaming / Classification
    (
        "Roguelike vs roguelite — genre lineage 1980–2026",
        "What precisely distinguishes a 'roguelike' from a 'roguelite'? Trace the "
        "genre from Rogue (1980) through the defining games of each decade, and "
        "name the best example from each era with a one-sentence reason why."
    ),
    # Software discovery
    (
        "Best blog platforms for GitHub Pages in 2026",
        "Which static site generators and blog platforms work best with GitHub Pages "
        "hosting in 2026? Compare the top options on setup friction, theme quality, "
        "writing experience, and long-term maintenance burden."
    ),
    # Creative / Game design
    (
        "Invent an original word puzzle mechanic",
        "Design a word or language puzzle game mechanic that genuinely doesn't exist "
        "yet. Describe the core rules, walk through a sample round, explain why it's "
        "fun, and argue why no one has made it before."
    ),
    # My pick — outdoors / travel
    (
        "Underrated US national parks and public lands",
        "What are the most underrated US national parks and public lands — places "
        "that deliver a Yellowstone- or Yosemite-quality experience without the "
        "crowds? For each, give the best season to visit and one thing most visitors miss."
    ),
]

PROMPT_TIMEOUT = 10  # seconds to wait for input before auto-continuing


def pick_topic(headless: bool) -> str:
    # Rotate daily so each day highlights a different suggestion.
    # Same suggestion for everyone on the same calendar day.
    daily_idx = datetime.date.today().toordinal() % len(SUGGESTIONS)
    label, query = SUGGESTIONS[daily_idx]

    if headless:
        print(f"[headless] Auto-selected: {label}")
        return query

    print("\nResearch topic — pick a number, paste your own query, or press Enter:\n")
    for i, (lbl, _) in enumerate(SUGGESTIONS):
        marker = "→" if i == daily_idx else " "
        print(f"  {marker} [{i + 1}] {lbl}")

    print(f"\n  Auto-continuing with [{daily_idx + 1}] in {PROMPT_TIMEOUT}s...")
    print("> ", end="", flush=True)

    ready = select.select([sys.stdin], [], [], PROMPT_TIMEOUT)[0]
    user_input = sys.stdin.readline().strip() if ready else ""
    print()

    if not user_input:
        return query

    try:
        n = int(user_input)
        if 1 <= n <= len(SUGGESTIONS):
            return SUGGESTIONS[n - 1][1]
    except ValueError:
        pass

    return user_input  # treat any other text as a custom query


def build_agent(model_id: str, verbose: bool = True) -> CodeAgent:
    model = OpenAIServerModel(
        model_id=model_id,
        api_base=BASE_URL,
        api_key="none",
    )
    # CodeAgent writes Python code to call tools — more robust than JSON tool calling.
    # It handles multi-hop research naturally: search → read → search again → synthesize.
    # Works even with imperfect tool calling configs because it generates executable code.
    return CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        max_steps=10,
        verbosity_level=1 if verbose else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Research agent demo (smolagents)")
    parser.add_argument("--query", default=None, help="Research question (skips topic picker)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID on vLLM")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip interactive prompt — auto-pick today's topic and run immediately",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("tt-agents Proof 1: Research Agent (smolagents)")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Endpoint: {BASE_URL}")

    query = args.query if args.query else pick_topic(args.headless)

    print(f"\nQuery:\n  {query}\n")
    print("-" * 70)

    agent = build_agent(args.model, verbose=not args.quiet)
    result = agent.run(query)

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)
    print(f"\n✓ Completed in {agent.step_number} steps")

    # Save for the writing pipeline (03_writing_pipeline.py picks this up automatically)
    RESEARCH_FILE.write_text(f"Topic: {query}\n\n{result}")
    print(f"\n[Research saved → {RESEARCH_FILE.name}]")
    print("[Tip: run 03_writing_pipeline.py next to turn this into any format you choose]")


if __name__ == "__main__":
    main()
