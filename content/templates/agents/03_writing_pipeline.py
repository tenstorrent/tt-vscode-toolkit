#!/usr/bin/env python3
"""
Proof 3: Multi-Agent Writing Pipeline
Framework: CrewAI
Model: Qwen3-32B

If 01_research_agent.py has been run first, its output is loaded automatically
and the Researcher agent is skipped — the pipeline goes straight to Writer → Editor
with real research already in hand.

If no prior research exists, the full three-agent pipeline runs:
  Researcher → Writer → Editor

Either way, a format picker lets you choose HOW the piece is written:
  blog post, tweet thread, ELI5, devil's advocate, lyrical prose,
  exec one-pager, Reddit thread, or your own instruction.

Usage:
  python3 03_writing_pipeline.py                          (picks up last_research.txt if present)
  python3 03_writing_pipeline.py --topic "your topic"     (ignore saved research, research fresh)
  python3 03_writing_pipeline.py --no-research            (force fresh research even if file exists)
  python3 03_writing_pipeline.py --headless               (no prompts, auto-pick format)
  python3 03_writing_pipeline.py --model "meta-llama/Llama-3.3-70B-Instruct"

Install deps:
  pip install crewai openai
"""
import argparse
import datetime
import importlib.util
import os
import select
import sys
from pathlib import Path

try:
    from crewai import Agent, Crew, LLM, Process, Task
except ImportError:
    print("ERROR: crewai not installed. Run: pip install crewai")
    sys.exit(1)

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-32B")
RESEARCH_FILE = Path(__file__).parent / "last_research.txt"
PROMPT_TIMEOUT = 10

DEFAULT_TOPIC = (
    "Running AI agents locally on open hardware: "
    "why 32B+ parameter models change everything for agent reliability"
)

FORMATS = [
    (
        "Developer blog post",
        "Write a structured 600-900 word blog post for a technical developer audience. "
        "Use specific facts and numbers. Short paragraphs (3-4 sentences max). "
        "Include at least one concrete code example or command if it fits naturally. "
        "End with a clear call to action.",
    ),
    (
        "Tweet thread (≤140 chars each)",
        "Write a thread of 10-14 tweets. EVERY tweet must be ≤140 characters. "
        "Number them 1/N through N/N. Each must stand alone if quote-tweeted. "
        "Open with a hook tweet that earns the follow. Close with a punchy summary. "
        "No filler, no 'great thread!' tweets.",
    ),
    (
        "Explain it like I'm curious but new",
        "Write for someone who is genuinely curious and intelligent but has zero "
        "prior knowledge of this topic. No jargon without definition. Use analogies "
        "liberally. Make it feel like a conversation with a knowledgeable friend, "
        "not a lecture. Aim for 400-600 warm, readable words.",
    ),
    (
        "Devil's advocate — argue the opposite",
        "Write a skeptical, contrarian take. Steelman the subject first (one paragraph), "
        "then systematically challenge it. Find the weaknesses, the hype, the things "
        "that could go wrong or that the enthusiasts are glossing over. 400-600 words. "
        "End with a clear statement of what evidence would change your mind.",
    ),
    (
        "Lyrical / prose poetry",
        "Write this as literary nonfiction or prose poetry. Prioritize beauty of "
        "language, rhythm, and surprise over comprehensiveness. Let the form reflect "
        "the content. Aim for 300-500 words that someone would want to read aloud. "
        "Every sentence should earn its place.",
    ),
    (
        "Executive one-pager",
        "Write a single-page briefing for a decision-maker with 90 seconds. "
        "Lead with a one-sentence summary of the situation. Three bullet points "
        "of essential context. One paragraph recommendation that names the most "
        "important tradeoff explicitly. No jargon, no filler, fits on one page.",
    ),
    (
        "Reddit thread — multiple POVs",
        "Write it as a Reddit-style discussion. Start with a top-level post (2-3 paragraphs). "
        "Follow with 5 reply comments from distinct, authentic-feeling personas: "
        "an enthusiast, a skeptic, a practitioner with real experience, a complete "
        "newcomer with a naive but honest question, and someone with an unexpected "
        "angle nobody else mentioned. Each voice should feel like a real person.",
    ),
]

PROMPT_TIMEOUT = 10


def pick_format(headless: bool) -> tuple[str, str]:
    daily_idx = datetime.date.today().toordinal() % len(FORMATS)
    label, instruction = FORMATS[daily_idx]

    if headless:
        print(f"[headless] Auto-selected format: {label}")
        return label, instruction

    print("\nWriting format — pick a number, paste your own instruction, or press Enter:\n")
    for i, (lbl, _) in enumerate(FORMATS):
        marker = "→" if i == daily_idx else " "
        print(f"  {marker} [{i + 1}] {lbl}")

    print(f"\n  Auto-continuing with [{daily_idx + 1}] in {PROMPT_TIMEOUT}s...")
    print("> ", end="", flush=True)

    ready = select.select([sys.stdin], [], [], PROMPT_TIMEOUT)[0]
    user_input = sys.stdin.readline().strip() if ready else ""
    print()

    if not user_input:
        return label, instruction

    try:
        n = int(user_input)
        if 1 <= n <= len(FORMATS):
            return FORMATS[n - 1]
    except ValueError:
        pass

    return "Custom instruction", user_input


def load_research() -> str | None:
    if RESEARCH_FILE.exists():
        return RESEARCH_FILE.read_text().strip()
    return None


def run_research_agent(model_id: str) -> str:
    """Run script 01's research pipeline to generate last_research.txt, then return the result."""
    script_path = Path(__file__).parent / "01_research_agent.py"
    spec = importlib.util.spec_from_file_location("research_agent", script_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    print("\n[No saved research found — starting from the research agent]")
    print("[Running 01_research_agent.py first. The result will be available next time too.]\n")
    print("-" * 70)

    query = m.pick_topic(headless=False)

    print(f"\nResearch query:\n  {query}\n")
    print("-" * 70)

    agent = m.build_agent(model_id, verbose=True)
    result = agent.run(query)

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE:")
    print("=" * 70)
    print(result)
    print(f"\n✓ Research done in {agent.step_number} steps")
    print("-" * 70)

    full_text = f"Topic: {query}\n\n{result}"
    m.RESEARCH_FILE.write_text(full_text)
    print(f"[Research saved → {m.RESEARCH_FILE.name}]\n")
    return full_text


def build_crew(model_id: str, topic: str, format_label: str, format_instruction: str,
               existing_research: str | None) -> Crew:
    llm = LLM(
        model=f"openai/{model_id}",
        base_url=BASE_URL,
        api_key="none",
        temperature=0.7,
        max_tokens=2048,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Transform research into a compelling piece in the requested format",
        backstory=(
            "You are a skilled writer who adapts style, tone, and format to match "
            "any brief. You write with clarity, use concrete examples from the research, "
            "and never invent facts not provided to you."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    editor = Agent(
        role="Senior Editor",
        goal="Polish the piece so every sentence earns its place",
        backstory=(
            "You are a demanding editor who cuts redundant sentences, strengthens the "
            "opening, verifies the format instruction was followed exactly, and ensures "
            "the ending lands. You return only the final polished piece."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    if existing_research:
        # Research already done — go straight to writing
        writing_task = Task(
            description=(
                f"Write a piece about the following topic using the research provided below.\n\n"
                f"FORMAT INSTRUCTION — follow this exactly:\n{format_instruction}\n\n"
                f"RESEARCH (do not invent facts outside this):\n{existing_research}"
            ),
            agent=writer,
            expected_output=f"A complete piece in this format: {format_label}",
        )
    else:
        # No prior research — include a Researcher
        researcher = Agent(
            role="Research Specialist",
            goal="Gather accurate, specific facts and create a detailed outline",
            backstory=(
                "You are a meticulous researcher who finds concrete facts, statistics, "
                "and examples. You never make up numbers. You create structured outlines "
                "that writers can follow precisely."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        research_task = Task(
            description=(
                f"Research the topic: '{topic}'\n\n"
                "Produce:\n"
                "1. A hook opening sentence that stops a skeptical technical reader mid-scroll\n"
                "2. 5-7 bullet points of specific, factual supporting points\n"
                "   (include numbers/benchmarks where you know them)\n"
                "3. A suggested 5-section outline with one sentence per section\n"
                "4. A strong concluding argument\n\n"
                "Be specific. Cite the 7B vs 32B+ agent reliability difference if relevant."
            ),
            agent=researcher,
            expected_output="A structured research document with hook, key facts, outline, and conclusion",
        )

        writing_task = Task(
            description=(
                f"Using the research document above, write a piece on this topic.\n\n"
                f"FORMAT INSTRUCTION — follow this exactly:\n{format_instruction}\n\n"
                "Do NOT invent facts not in the research document."
            ),
            agent=writer,
            expected_output=f"A complete piece in this format: {format_label}",
            context=[research_task],
        )

        editing_task = Task(
            description=(
                f"Edit the piece. Verify the format instruction was followed exactly:\n"
                f"{format_instruction}\n\n"
                "Cut redundant sentences (aim for 10% reduction). "
                "Strengthen the opening. Ensure the ending lands. "
                "Return only the final polished piece, ready to post."
            ),
            agent=editor,
            expected_output="Final polished piece, ready to publish",
            context=[writing_task],
        )

        return Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, writing_task, editing_task],
            process=Process.sequential,
            verbose=True,
        )

    # 2-agent path when research was pre-loaded
    editing_task = Task(
        description=(
            f"Edit the piece. Verify the format instruction was followed exactly:\n"
            f"{format_instruction}\n\n"
            "Cut redundant sentences (aim for 10% reduction). "
            "Strengthen the opening. Ensure the ending lands. "
            "Return only the final polished piece, ready to post."
        ),
        agent=editor,
        expected_output="Final polished piece, ready to publish",
        context=[writing_task],
    )

    return Crew(
        agents=[writer, editor],
        tasks=[writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-agent writing pipeline demo (CrewAI)")
    parser.add_argument("--topic", default=None,
                        help="Topic to research and write about (skips saved research)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID on vLLM")
    parser.add_argument("--no-research", action="store_true",
                        help="Ignore last_research.txt and run the full Researcher → Writer → Editor pipeline")
    parser.add_argument("--headless", action="store_true",
                        help="Skip format picker — auto-select today's format and run immediately")
    args = parser.parse_args()

    print("=" * 70)
    print("tt-agents Proof 3: Multi-Agent Writing Pipeline (CrewAI)")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Endpoint: {BASE_URL}")

    # Determine research source
    existing_research = None
    topic = args.topic or DEFAULT_TOPIC

    if args.topic:
        # Explicit topic — research fresh, skip the file
        pass
    elif args.no_research:
        # Force the 3-agent Researcher path with default topic
        pass
    else:
        existing_research = load_research()
        if not existing_research:
            # No saved research — run script 01 first, then use its output
            existing_research = run_research_agent(args.model)

    if existing_research:
        first_line = existing_research.splitlines()[0][:80]
        age = ""
        if RESEARCH_FILE.exists():
            import time
            secs = time.time() - RESEARCH_FILE.stat().st_mtime
            age = f" ({int(secs // 60)} min ago)" if secs < 3600 else f" ({int(secs // 3600)}h ago)"
        print(f"\n[Research loaded from {RESEARCH_FILE.name}{age}]")
        print(f"  {first_line}...")
        print("  (Pipeline: Writer → Editor  |  pass --no-research to run fresh Researcher instead)")
    else:
        print(f"\nTopic: {topic}")
        print("  (Pipeline: Researcher → Writer → Editor)")

    format_label, format_instruction = pick_format(args.headless)

    print(f"\nFormat: {format_label}")
    print("-" * 70)

    crew = build_crew(args.model, topic, format_label, format_instruction, existing_research)
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL PIECE:")
    print("=" * 70)
    print(result.raw if hasattr(result, "raw") else str(result))
    print("\n✓ Pipeline complete")


if __name__ == "__main__":
    main()
