#!/usr/bin/env python3
"""
Demo 5: Storyboard to Pixel Art Pipeline
Frameworks: CrewAI (Stage 1) + smolagents ToolCallingAgent (Stage 2)
Models: Llama-3.3-70B (Stage 1: creative narrative) → Qwen3-32B (Stage 2: structured output)

Demonstrates:
  - Model selection by task: 70B for open-ended creativity, 32B for structured output
  - Server lifecycle from Python: docker stop → tt-smi -r → start new model
  - JSON artifact handoff between model stages (storyboard.json → pixelart_prompts.json)
  - Chip targeting: --device-id 0,1 vs --device-id 2,3
  - Dual-server mode: two models on separate chip pairs simultaneously

Pipeline:
  Theme input
    → Stage 1: CrewAI (ConceptDirector + StoryboardWriter + ArtDirector, 70B)
    → storyboard.json
    → Model switch (docker stop / tt-smi -r / run.py) — unless --simulate or --single-model
    → Stage 2: smolagents ToolCallingAgent (32B, tools: refine_scene_prompt + check_palette_consistency)
    → pixelart_prompts.json
    → pipeline_summary.txt

Usage:
  python3 05_storyboard_to_pixelart.py --simulate             (no server ops, use current server)
  python3 05_storyboard_to_pixelart.py --single-model         (skip model switch)
  python3 05_storyboard_to_pixelart.py --theme "space opera"
  python3 05_storyboard_to_pixelart.py --dual-server          (chips 0,1 + chips 2,3 simultaneously)
  python3 05_storyboard_to_pixelart.py --cpu-orchestrator     (Stage 2 on CPU Qwen3-0.6B, no chip switch)
  python3 05_storyboard_to_pixelart.py --cpu-orchestrator --cpu-model "Qwen/Qwen3-1.7B"
  python3 05_storyboard_to_pixelart.py --creative-model "meta-llama/Llama-3.3-70B-Instruct"
  python3 05_storyboard_to_pixelart.py --prompt-model "Qwen/Qwen3-32B"
  python3 05_storyboard_to_pixelart.py --ansi                 (Stage 3: render each scene as ANSI block art)
  python3 05_storyboard_to_pixelart.py --svg                  (Stage 3: SVG storyboard panels + storyboard.html)
  python3 05_storyboard_to_pixelart.py --ansi --svg           (both — ANSI pixel art + SVG illustrated panels)

Install deps:
  pip install crewai smolagents openai
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from crewai import Agent, Crew, LLM, Process, Task
except ImportError:
    print("ERROR: crewai not installed. Run: pip install crewai")
    sys.exit(1)

try:
    from smolagents import OpenAIServerModel, ToolCallingAgent, tool
    from smolagents.monitoring import AgentLogger, LogLevel
except ImportError:
    print("ERROR: smolagents not installed. Run: pip install smolagents")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

try:
    from rich.panel import Panel
except ImportError:
    print("ERROR: rich not installed. Run: pip install rich")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL_PRIMARY   = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
BASE_URL_SECONDARY = "http://localhost:8001/v1"   # dual-server mode only
BASE_URL_CPU       = "http://localhost:8002/v1"   # cpu-orchestrator mode only

# CPU orchestrator: small Qwen3 running on host CPU via transformers, no TT chips used.
# Proven pattern from ~/code/tt-local-generator/app/prompt_server.py.
CPU_SERVER_PORT   = 8002
CPU_SERVER_SCRIPT = Path(os.environ.get(
    "TT_LOCAL_GENERATOR_DIR",
    os.path.expanduser("~/code/tt-local-generator"),
)) / "app" / "prompt_server.py"
DEFAULT_CPU_MODEL = os.environ.get("CPU_ORCHESTRATOR_MODEL", "Qwen/Qwen3-0.6B")

# tt-inference-server location — override via TT_INFERENCE_DIR env var
TT_INFERENCE_DIR = Path(os.environ.get(
    "TT_INFERENCE_DIR",
    os.path.expanduser("~/.local/lib/tt-inference-server"),
))

DEFAULT_CREATIVE_MODEL = os.environ.get("VLLM_CREATIVE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
DEFAULT_PROMPT_MODEL   = os.environ.get("VLLM_PROMPT_MODEL", "Qwen/Qwen3-32B")

STORYBOARD_FILE    = Path(__file__).parent / "storyboard.json"
PROMPTS_FILE       = Path(__file__).parent / "pixelart_prompts.json"
SUMMARY_FILE       = Path(__file__).parent / "pipeline_summary.txt"
STORYBOARD_HTML    = Path(__file__).parent / "storyboard.html"

# Module-level state shared across Stage 2 tools (mirrors 04_dungeon_master.py pattern)
_storyboard: dict = {}
_refined_prompts: dict[int, str] = {}

# ── Theme suggestions ───────────────────────────────────────────────────────────

THEMES = [
    # Literary registers — specific, weird, with strong visual logic
    "1983 arcade at closing time: a janitor sweeps around the one cabinet still lit",
    "a Kafka civil servant discovers the department of forms has no exit",
    "Saturday morning cartoons, Hanna-Barbera backgrounds, cereal going soggy in the bowl",
    "the last bait shop before the county road ends at the reservoir",
    # Retro hardware / broken electronics — pixel art plays these as strength not limitation
    "a Game Boy with a cracked screen still running Tetris, battery low",
    "a Harryhausen skeleton sits at a grand piano and plays something slow",
    # Americana nostalgia — warm limited palettes, strong silhouettes
    "harvest festival in a small town that almost doesn't exist anymore",
    "after school Tuesday 1987: bikes on the sidewalk, four channels, nothing good on",
]


def pick_theme(headless: bool, theme_override: str | None) -> str:
    if theme_override:
        return theme_override
    daily_idx = datetime.date.today().toordinal() % len(THEMES)
    theme = THEMES[daily_idx]
    if headless:
        print(f"[headless] Auto-selected theme: {theme}")
    else:
        print(f"\n[Auto-selected theme: {theme}  (pass --theme to override)]\n")
    return theme


# ── Server lifecycle helpers (stdlib only, no extra deps) ───────────────────────

def _parser_for(model: str) -> str:
    """Return the correct vLLM tool-call parser for this model family."""
    return "hermes" if "qwen" in model.lower() else "llama3_json"


def _find_server_container(name_filter: str = "tt-inference-server") -> str | None:
    """Return the first running tt-inference-server container name, or None."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return out.split("\n")[0] if out else None
    except Exception:
        return None


def _stop_server(simulate: bool = False) -> None:
    name = _find_server_container()
    if name:
        print(f"  [docker stop {name}]")
        if not simulate:
            subprocess.run(["docker", "stop", name], check=True, capture_output=True)
    else:
        print("  [no running tt-inference-server container — skipping stop]")


def _reset_chips(targets: list[int] | None = None, simulate: bool = False) -> None:
    """Run tt-smi -r to flush chip state between model loads.
    This is required: loading a second model without reset can cause hardware faults.
    """
    cmd = ["tt-smi", "-r"] + ([str(t) for t in targets] if targets else [])
    print(f"  [{' '.join(cmd)}]")
    if not simulate:
        subprocess.run(cmd, check=True, capture_output=True)


def _start_server(model: str, port: int = 8000, device_ids: list[int] | None = None,
                  simulate: bool = False) -> None:
    """Launch tt-inference-server in the background. Call _wait_for_server() after."""
    run_py = TT_INFERENCE_DIR / "run.py"
    vllm_args = {"enable_auto_tool_choice": True, "tool_call_parser": _parser_for(model)}
    cmd = [
        "python3", str(run_py),
        "--model", model,
        "--tt-device", "p300x2",
        "--workflow", "server",
        "--docker-server",
        "--no-auth",
        "--service-port", str(port),
        "--vllm-override-args", json.dumps(vllm_args),
    ]
    if device_ids is not None:
        cmd += ["--device-id", ",".join(str(d) for d in device_ids)]

    device_str = f" --device-id {','.join(str(d) for d in device_ids)}" if device_ids else ""
    print(f"  [python3 run.py --model {model} --service-port {port}{device_str}]")

    if not simulate:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _detect_running_model(base_url: str) -> str | None:
    """Return the model ID currently loaded on the server, or None if no server is up."""
    try:
        url = f"{base_url.rstrip('/')}/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return None


def _same_model_family(a: str, b: str) -> bool:
    """True if two model IDs belong to the same family (e.g. both Qwen3, both Llama)."""
    al, bl = a.lower(), b.lower()
    for family in ("qwen", "llama", "mistral", "gemma", "phi"):
        if family in al and family in bl:
            return True
    return al == bl


def _wait_for_server(port: int = 8000, timeout: int = 900) -> str:
    """Poll /v1/models until the server is ready. Returns the loaded model ID."""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    print(f"  [waiting for server on port {port} (up to {timeout}s)...]", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                model_id = json.loads(r.read())["data"][0]["id"]
                print(f"  [server ready: {model_id}]")
                return model_id
        except Exception:
            time.sleep(10)
    raise TimeoutError(f"Server on port {port} not ready after {timeout}s")


# ── CPU orchestrator helpers ────────────────────────────────────────────────────
# A small Qwen3 model on host CPU handles Stage 2 while TT chips run Stage 1.
# This eliminates the docker-stop → tt-smi-r → restart cycle entirely.
# Source pattern: ~/code/tt-local-generator/app/prompt_server.py

def _start_cpu_server(model: str, simulate: bool = False) -> "subprocess.Popen | None":
    """Launch prompt_server.py on CPU. Returns the Popen handle, or None in simulate mode."""
    if not CPU_SERVER_SCRIPT.exists():
        print(f"  [NOTE: {CPU_SERVER_SCRIPT} not found]")
        print(f"  [To use --cpu-orchestrator, start prompt_server.py manually on port {CPU_SERVER_PORT}]")
        print(f"  [  git clone tt-local-generator, then: python3 app/prompt_server.py --port {CPU_SERVER_PORT}]")
        return None
    cmd = ["python3", str(CPU_SERVER_SCRIPT), "--model", model, "--port", str(CPU_SERVER_PORT)]
    print(f"  [CPU: python3 app/prompt_server.py --model {model} --port {CPU_SERVER_PORT}]")
    if simulate:
        return None
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_cpu_server(timeout: int = 300) -> None:
    """Poll /health until the CPU model is loaded. CPU server uses /health, not /v1/models."""
    url = f"http://localhost:{CPU_SERVER_PORT}/health"
    deadline = time.time() + timeout
    print(f"  [waiting for CPU server on port {CPU_SERVER_PORT} (~30-90s for 0.6B)...]", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("model_ready"):
                    print(f"  [CPU server ready: {data.get('model')}]")
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"CPU server on port {CPU_SERVER_PORT} not ready after {timeout}s")


def _stop_cpu_server(proc: "subprocess.Popen | None", simulate: bool = False) -> None:
    """Terminate the CPU server subprocess."""
    if proc is not None and not simulate:
        print(f"  [stopping CPU server (pid {proc.pid})]")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Quiet logger (mirrors 04_dungeon_master.py) ─────────────────────────────────

class _QuietLogger(AgentLogger):
    """Shows only tool names; suppresses JSON argument dumps and spurious warnings."""

    _MUTE = "If you want to return an answer"

    def __init__(self):
        super().__init__(level=LogLevel.INFO)

    def log(self, *args, level=LogLevel.INFO, **kwargs):
        for arg in args:
            if isinstance(arg, Panel):
                try:
                    text = getattr(getattr(arg, "renderable", None), "plain", "") or ""
                    m = re.match(r"Calling tool: '(\w+)'", text)
                    if m and m.group(1) != "final_answer":
                        print(f"  [{m.group(1)}]", flush=True)
                except Exception:
                    pass
                return
        if args and isinstance(args[0], str) and "Observations:" in args[0]:
            return
        if int(level) <= LogLevel.ERROR:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        if self._MUTE in error_message:
            return
        super().log_error(error_message)


# ── Stage 1: CrewAI storyboard pipeline ────────────────────────────────────────

def run_stage1(theme: str, model_id: str, base_url: str) -> dict:
    """Run the 3-agent CrewAI storyboard pipeline. Returns parsed storyboard dict."""
    print(f"\n[Stage 1: Storyboard — {model_id}]")

    llm = LLM(
        model=f"openai/{model_id}",
        base_url=base_url,
        api_key="none",
        temperature=0.8,
        max_tokens=2048,
    )

    concept_director = Agent(
        role="Visual Concept Director",
        goal="Define the hardware-appropriate visual language, palette, and era reference for the story",
        backstory=(
            "You are an art director who has shipped pixel art games since the Commodore 64. "
            "You believe pixel art's power comes from constraint, not decoration — a 4-color "
            "Game Boy palette, a Looney Tunes flat background, an arcade CRT flicker. "
            "You anchor every project in a specific real hardware era: NES (54 colors, 4/tile), "
            "Game Boy (4-shade green), Amiga (32-color EHB), CGA (4 colors), SNES (256-color), "
            "arcade CRT (dithered scanlines), or Atari 2600 (15 colors, single-screen). "
            "You avoid generic 'neon on dark' directions. You think in silhouettes, tile patterns, "
            "and what dithering can do for texture."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    storyboard_writer = Agent(
        role="Storyboard Writer",
        goal="Write 4-6 distinct scene descriptions that carry the story arc",
        backstory=(
            "You write storyboards for animated shorts and games. Each scene you write "
            "has a clear visual focus (what you see first), a mood, and something that "
            "changes from the last scene. You think in images, not words."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    art_director = Agent(
        role="Pixel Art Director",
        goal="Translate each scene into pixel art constraints and output a JSON storyboard",
        backstory=(
            "You are the pixel art lead on this project. You take the storyboard and the "
            "concept direction and produce a technical brief for each scene: pixel density, "
            "dithering style, which palette colors dominate. You output structured JSON only."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    concept_task = Task(
        description=(
            f"Define the visual concept for a pixel art story with this theme: '{theme}'.\n\n"
            "Deliver:\n"
            "1. The specific hardware era this lives in — e.g. NES (4 colors/tile, 54 total), "
            "Game Boy (4-shade grey-green), Amiga OCS (32-color EHB), arcade CRT (dithered scanlines), "
            "SNES (256-color, Mode 7 floor), or Atari 2600 (15 colors, flat single-screen).\n"
            "2. A palette of 4-8 named colors drawn from that hardware's actual constraints "
            "(e.g. 'NES tan #E8C070', 'Game Boy dark #0F380F') — no generic neon.\n"
            "3. The primary pixel art technique this story uses: flat tiles, dithered gradients, "
            "silhouette + backlighting, scanline overlay, sprite animation, or isometric.\n"
            "4. A one-sentence visual metaphor that connects all scenes.\n\n"
            "Avoid: neon on dark, generic fantasy, glowing outlines. "
            "Lean into: limited palettes with emotional weight, specific dithering, "
            "recognizable silhouettes that read at 16x16 or 32x32 pixels."
        ),
        agent=concept_director,
        expected_output="Hardware-specific concept brief with era, palette, technique, and unifying metaphor",
    )

    storyboard_task = Task(
        description=(
            f"Using the visual concept above, write a 4-6 scene storyboard for: '{theme}'.\n\n"
            "For each scene:\n"
            "- Scene number and title\n"
            "- What the player/viewer sees first (the establishing shot)\n"
            "- The mood in one word\n"
            "- What changes from the previous scene\n"
            "- One visual detail that rewards close attention\n\n"
            "Keep each scene to 3-5 sentences. Scenes should flow as a coherent arc."
        ),
        agent=storyboard_writer,
        expected_output="4-6 scene storyboard with establishing shots, moods, and transitions",
        context=[concept_task],
    )

    art_task = Task(
        description=(
            "You have the visual concept and storyboard above.\n\n"
            "Output a single valid JSON object (no markdown fences, no prose) in this exact format:\n\n"
            '{\n'
            '  "theme": "<the theme string>",\n'
            '  "hardware_era": "<e.g. NES, Game Boy, Amiga OCS, SNES, arcade CRT, Atari 2600>",\n'
            '  "palette": "<palette name and primary colors with hex codes>",\n'
            '  "lighting": "<lighting model: e.g. sodium-yellow streetlight, CRT phosphor glow, flat overcast>",\n'
            '  "technique": "<primary pixel technique: flat tiles / dithered gradients / silhouette / scanline / isometric>",\n'
            '  "scenes": [\n'
            '    {\n'
            '      "id": 1,\n'
            '      "title": "Scene title",\n'
            '      "description": "What the viewer sees — 2 sentences max.",\n'
            '      "mood": "one word",\n'
            '      "pixel_constraints": "sprite size, color count per tile, dithering detail"\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            f'Set "theme" to exactly: "{theme}"\n'
            "Include all scenes from the storyboard. Output ONLY the JSON — no explanation."
        ),
        agent=art_director,
        expected_output="Valid JSON storyboard with theme, palette, lighting, and scenes array",
        context=[storyboard_task],
    )

    crew = Crew(
        agents=[concept_director, storyboard_writer, art_director],
        tasks=[concept_task, storyboard_task, art_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    raw = result.raw if hasattr(result, "raw") else str(result)

    # Strip markdown code fences if the model added them, then parse JSON
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
        storyboard = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                storyboard = json.loads(m.group(0))
            except json.JSONDecodeError:
                storyboard = None
        else:
            storyboard = None

    if storyboard is None:
        print("  [Warning: could not parse JSON from Stage 1 — using fallback structure]")
        storyboard = {
            "theme": theme,
            "hardware_era": "NES",
            "palette": "NES tan #E8C070, dark brown #3C2005, sky blue #5BC8F5, black #000000",
            "lighting": "flat overcast, no harsh shadows",
            "technique": "flat tiles, 4 colors per tile",
            "scenes": [
                {
                    "id": 1,
                    "title": "Opening",
                    "description": raw[:200].strip(),
                    "mood": "quiet",
                    "pixel_constraints": "16x16 sprites, 4 colors per tile, NES hardware limits",
                },
            ],
        }

    STORYBOARD_FILE.write_text(json.dumps(storyboard, indent=2))
    n_scenes = len(storyboard.get("scenes", []))
    print(f"  [storyboard.json saved — {n_scenes} scene{'s' if n_scenes != 1 else ''}]")
    return storyboard


# ── Stage 2 tools (operate on module-level storyboard state) ───────────────────

@tool
def refine_scene_prompt(scene_id: int, draft_prompt: str) -> str:
    """Refine a draft pixel art image-generation prompt for a scene.
    Adds explicit style tags: pixel density, palette color count, dithering technique,
    era reference, and rendering constraints. Stores the result for consistency checking.

    Args:
        scene_id: The scene number (matches the id field in storyboard.json).
        draft_prompt: Your draft description of the image (2-4 sentences).
    """
    scene = next((s for s in _storyboard.get("scenes", []) if s["id"] == scene_id), None)
    if not scene:
        ids = [s["id"] for s in _storyboard.get("scenes", [])]
        return f"Scene {scene_id} not found. Available IDs: {ids}"

    palette = _storyboard.get("palette", "16-color")
    constraints = scene.get("pixel_constraints", "pixel art style")
    mood = scene.get("mood", "neutral")

    lighting = _storyboard.get("lighting", "")
    technique = _storyboard.get("technique", "")
    refined = (
        f"pixel art, {constraints}, {palette}, "
        + (f"{technique}, " if technique else "")
        + (f"lighting: {lighting}, " if lighting else "")
        + f"mood: {mood}, "
        + f"{draft_prompt.strip().rstrip('.')}. "
        + "hard pixel edges, no anti-aliasing, limited-palette dithering, retro hardware aesthetic."
    )
    _refined_prompts[scene_id] = refined
    return f"Scene {scene_id} refined: {refined}"


@tool
def check_palette_consistency(scene_ids: list[int]) -> str:
    """Check that refined prompts for the given scenes share a coherent visual style.
    Returns a consistency score (0-10) and flags scenes that break the palette or
    style language established by the concept direction.

    Args:
        scene_ids: List of scene IDs to compare (e.g. [1, 2, 3, 4]).
    """
    missing = [sid for sid in scene_ids if sid not in _refined_prompts]
    if missing:
        return f"Missing refined prompts for scenes: {missing}. Call refine_scene_prompt first."

    palette = _storyboard.get("palette", "")
    palette_keywords = [w.lower() for w in palette.split() if len(w) > 3]

    inconsistent = []
    for sid in scene_ids:
        prompt_lower = _refined_prompts[sid].lower()
        if palette_keywords and not any(kw in prompt_lower for kw in palette_keywords):
            inconsistent.append(sid)

    score = max(0, min(10, 10 - len(inconsistent) * 2))
    report = {
        "consistency_score": score,
        "scenes_checked": scene_ids,
        "palette_reference": palette,
        "issues": (
            [f"Scene {sid} may not reference the established palette" for sid in inconsistent]
            if inconsistent else ["All scenes reference the palette consistently"]
        ),
        "verdict": "consistent" if score >= 7 else "needs_revision",
    }
    return json.dumps(report, indent=2)


# ── Stage 2: smolagents prompt engineering ──────────────────────────────────────

def run_stage2(storyboard: dict, model_id: str, base_url: str) -> list[dict]:
    """Run the smolagents prompt engineering stage. Returns list of scene prompt dicts."""
    global _storyboard, _refined_prompts
    _storyboard = storyboard
    _refined_prompts = {}

    print(f"\n[Stage 2: Pixel Art Prompts — {model_id}]")

    model = OpenAIServerModel(
        model_id=model_id,
        api_base=base_url,
        api_key="none",
    )

    scenes = storyboard.get("scenes", [])
    scene_ids = [s["id"] for s in scenes]
    scenes_summary = json.dumps(
        [{"id": s["id"], "title": s["title"], "description": s["description"],
          "mood": s["mood"], "pixel_constraints": s["pixel_constraints"]}
         for s in scenes],
        indent=2,
    )

    task = (
        f"You are a pixel art prompt engineer. You have a storyboard with {len(scenes)} scenes.\n\n"
        f"HARDWARE ERA: {storyboard.get('hardware_era', 'NES')}\n"
        f"PALETTE: {storyboard.get('palette', '16-color')}\n"
        f"LIGHTING: {storyboard.get('lighting', 'flat overcast')}\n"
        f"TECHNIQUE: {storyboard.get('technique', 'flat tiles')}\n\n"
        f"SCENES:\n{scenes_summary}\n\n"
        "For EACH scene in order:\n"
        "  1. Write a 2-sentence draft describing what you see — specific subjects, "
        "actions, composition. Name actual pixel art details (sprite size, tile pattern, "
        "how dithering creates texture). Do NOT use neon or glowing outlines.\n"
        "  2. Call refine_scene_prompt(scene_id=<id>, draft_prompt=<your draft>) to finalize it\n\n"
        f"After all {len(scenes)} scenes are refined, call check_palette_consistency({scene_ids}) "
        "to verify they form a coherent visual set.\n\n"
        "Return a final answer confirming all scenes were processed and the consistency verdict."
    )

    agent = ToolCallingAgent(
        tools=[refine_scene_prompt, check_palette_consistency],
        model=model,
        max_steps=len(scenes) * 2 + 4,
        logger=_QuietLogger(),
    )
    agent.run(task)

    # Collect from module state (more reliable than parsing the agent's final answer text)
    results = []
    for scene in scenes:
        sid = scene["id"]
        results.append({
            "id": sid,
            "title": scene["title"],
            "prompt": _refined_prompts.get(sid, f"[prompt not generated for scene {sid}]"),
        })

    output = {
        "model": model_id,
        "theme": storyboard.get("theme", ""),
        "palette": storyboard.get("palette", ""),
        "scenes": results,
    }
    PROMPTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"  [pixelart_prompts.json saved — {len(results)} prompts]")
    return results


# ── Pipeline orchestration ──────────────────────────────────────────────────────

def switch_models(prompt_model: str, simulate: bool) -> None:
    """Stop current server, reset chips, start the prompt-engineering model."""
    print("\n[Model Switch: stopping server → resetting chips → starting prompt model]")
    print("  (tt-smi -r is required between model loads to flush chip state)")
    _stop_server(simulate=simulate)
    _reset_chips(simulate=simulate)
    _start_server(prompt_model, port=8000, simulate=simulate)
    if not simulate:
        _wait_for_server(port=8000)


def setup_dual_server(creative_model: str, prompt_model: str, simulate: bool) -> None:
    """Start both models simultaneously on separate chip pairs."""
    print("\n[Dual-Server Setup]")
    print(f"  chips 0,1 → port 8000 → {creative_model}")
    print(f"  chips 2,3 → port 8001 → {prompt_model}")
    _stop_server(simulate=simulate)
    _reset_chips(simulate=simulate)
    _start_server(creative_model, port=8000, device_ids=[0, 1], simulate=simulate)
    _start_server(prompt_model, port=8001, device_ids=[2, 3], simulate=simulate)
    if not simulate:
        _wait_for_server(port=8000)
        _wait_for_server(port=8001)


def teardown_dual_server(simulate: bool) -> None:
    """Stop both containers started in dual-server mode."""
    print("\n[Dual-Server Teardown]")
    for _ in range(2):
        name = _find_server_container()
        if name:
            print(f"  [docker stop {name}]")
            if not simulate:
                subprocess.run(["docker", "stop", name], check=True, capture_output=True)


def write_summary(theme: str, creative_model: str, prompt_model: str,
                  storyboard: dict, prompts: list[dict],
                  t1: float, t2: float, t_total: float,
                  cpu_mode: bool = False) -> None:
    stage2_label = f"{prompt_model} (CPU)" if cpu_mode else prompt_model
    lines = [
        "=" * 70,
        "tt-agents Demo 5: Storyboard to Pixel Art Pipeline",
        "=" * 70,
        f"Theme:           {theme}",
        f"Stage 1 model:   {creative_model}  (TT hardware)",
        f"Stage 2 model:   {stage2_label}",
        f"Stage 1 time:    {t1:.1f}s",
        f"Stage 2 time:    {t2:.1f}s",
        f"Total time:      {t_total:.1f}s",
        "",
        f"Hardware era: {storyboard.get('hardware_era', '')}",
        f"Palette:      {storyboard.get('palette', '')}",
        f"Technique:    {storyboard.get('technique', '')}",
        f"Lighting:     {storyboard.get('lighting', '')}",
        "",
        "─" * 70,
        "PIXEL ART PROMPTS",
        "─" * 70,
    ]
    for p in prompts:
        lines.append(f"\nScene {p['id']}: {p['title']}")
        lines.append(f"  {p['prompt']}")
    lines += [
        "",
        "─" * 70,
        "Artifacts: storyboard.json  |  pixelart_prompts.json  |  pipeline_summary.txt",
    ]
    SUMMARY_FILE.write_text("\n".join(lines))


# ── Stage 3: ANSI + SVG rendering ─────────────────────────────────────────────
# The LLM generates a compact palette-indexed pixel grid; we convert it to
# ANSI truecolor block chars (terminal) and/or SVG rects (file).

_GRID_W = 16       # columns (ANSI pixel grid)
_GRID_H = 12       # rows (ANSI pixel grid)
_SVG_CELL = 20     # pixels per cell in SVG pixel grid output
_SVG_PANEL_W = 320  # storyboard panel width
_SVG_PANEL_H = 180  # storyboard panel height


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to (r, g, b). Returns mid-gray on any bad input."""
    try:
        h = hex_color.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            return (128, 128, 128)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, AttributeError):
        return (128, 128, 128)


def _grid_to_ansi(grid: list[str], palette: dict[str, str]) -> str:
    """Render a palette-indexed grid as ANSI truecolor block characters."""
    rgb = {k: _hex_to_rgb(v) for k, v in palette.items()}
    # Fall back to the darkest palette color for unrecognised chars rather than space,
    # so a key-mismatch from the LLM produces a visible (if wrong-colored) image.
    fallback = next(iter(rgb.values()), (64, 64, 64))
    rows = []
    for row in grid:
        line = ""
        for ch in row:
            r, g, b = rgb.get(ch, fallback)
            line += f"\033[38;2;{r};{g};{b}m█\033[0m"
        rows.append(line)
    return "\n".join(rows)


def _grid_to_svg(grid: list[str], palette: dict[str, str]) -> str:
    """Render a palette-indexed grid as an SVG with crisp pixel rects."""
    cols = max((len(r) for r in grid), default=_GRID_W)
    rows = len(grid)
    w, h = cols * _SVG_CELL, rows * _SVG_CELL
    rects = []
    for ri, row in enumerate(grid):
        for ci, ch in enumerate(row):
            fill = palette.get(ch, "#888888")
            x, y = ci * _SVG_CELL, ri * _SVG_CELL
            rects.append(f'  <rect x="{x}" y="{y}" width="{_SVG_CELL}" height="{_SVG_CELL}" fill="{fill}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" shape-rendering="crispEdges">\n'
        + "\n".join(rects)
        + "\n</svg>"
    )


def _parse_grid(raw: str) -> tuple[dict[str, str], list[str]] | tuple[None, None]:
    """Extract (palette, grid) from LLM response. Returns (None, None) on failure."""
    # Strip Qwen3 thinking blocks before searching for JSON
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    # Find the last (outermost) JSON object — avoids matching inner fragments
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None, None
    try:
        data = json.loads(m.group(0))
        palette = {str(k): str(v) for k, v in data.get("palette", {}).items()}
        grid = [str(r) for r in data.get("grid", [])]
        return (palette, grid) if palette and grid else (None, None)
    except (json.JSONDecodeError, TypeError):
        return None, None


# ── Stage 3 helpers: SVG storyboard panels ─────────────────────────────────────

def _parse_svg(raw: str) -> str | None:
    """Extract and lightly validate SVG markup from an LLM response."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:svg|xml)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned.strip())
    m = re.search(r"(<svg\b[^>]*?>.*?</svg>)", cleaned, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    svg = m.group(1)
    try:
        import xml.etree.ElementTree as ET
        ET.fromstring(svg)
        return svg
    except Exception:
        # Accept if it has recognisable shape elements — the model may emit minor XML quirks
        if any(tag in svg for tag in ("<rect", "<path", "<polygon", "<circle", "<ellipse")):
            return svg
        return None


def _fallback_svg_panel(sid: int, title: str, hex_colors: list[str]) -> str:
    """Simple placeholder SVG used when LLM panel generation fails for a scene."""
    w, h  = _SVG_PANEL_W, _SVG_PANEL_H
    bg    = hex_colors[0] if hex_colors else "#1a1a2e"
    mid   = hex_colors[len(hex_colors) // 2] if hex_colors else "#444466"
    light = hex_colors[-1] if hex_colors else "#ccccdd"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" fill="{bg}"/>'
        f'<rect y="{int(h * 0.65)}" width="{w}" height="{int(h * 0.35)}" fill="{mid}"/>'
        f'<text x="{w // 2}" y="{int(h * 0.46)}" text-anchor="middle" '
        f'dominant-baseline="middle" fill="{light}" font-family="monospace" font-size="11">'
        f'Scene {sid}</text>'
        f'</svg>'
    )


def _generate_svg_panel(
    client: "OpenAI",
    model_id: str,
    sid: int,
    title: str,
    description: str,
    mood: str,
    era: str,
    hex_colors: list[str],
) -> str | None:
    """Ask the LLM to draw an SVG storyboard panel illustration for one scene."""
    colors   = hex_colors[:6] if hex_colors else ["#1a1a2e", "#2a2a4e", "#6a6a9e", "#aaaacc"]
    c_dark   = colors[0]
    c_mid    = colors[len(colors) // 2]
    c_light  = colors[-1]
    c_ground = colors[min(1, len(colors) - 1)]
    color_list = ", ".join(colors)

    task = (
        f"Draw a storyboard panel SVG ({_SVG_PANEL_W}×{_SVG_PANEL_H}px) for:\n"
        f"Scene {sid}: \"{title}\"\n"
        f"Description: {description}\n"
        f"Mood: {mood}  |  Hardware era: {era}\n\n"
        f"Palette (use ONLY these colors): {color_list}\n\n"
        f"Build the SVG with these layers in order:\n"
        f"1. In <defs>: a linearGradient id='sky' from {c_dark} (top) to {c_mid} (bottom)\n"
        f"2. Sky: <rect width='{_SVG_PANEL_W}' height='{_SVG_PANEL_H}' fill='url(#sky)'/>\n"
        f"3. Ground: a <rect> or <polygon> in {c_ground} covering the bottom 35-50px\n"
        f"4. Silhouettes: 1-3 shapes in {c_dark} representing the scene subjects — "
        f"buildings, figures, trees, machines, whatever fits the scene. "
        f"Use <polygon points=...> for blocky shapes or <path d=...> for organic ones. "
        f"Make them recognizable outlines, not just rectangles.\n"
        f"5. Accent: one small element in {c_light} for mood — a lit window, "
        f"moon disc, spark, or glowing spot. Use <circle> or <ellipse>.\n\n"
        f"SVG must start: <svg xmlns=\"http://www.w3.org/2000/svg\" "
        f"width=\"{_SVG_PANEL_W}\" height=\"{_SVG_PANEL_H}\">\n"
        f"SVG must end: </svg>\n"
        f"No <text>, no <image>, no external hrefs, no XML comments in the output.\n"
        f"Output ONLY the complete SVG. No explanation, no markdown."
    )

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": task}],
            max_tokens=2048,
            temperature=0.5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        return _parse_svg(raw)
    except Exception as e:
        print(f"    [warning: SVG panel LLM call failed — {e}]")
        return None


def _assemble_storyboard_html(storyboard: dict, panels: list[dict]) -> Path:
    """Embed all scene SVG panels into a single storyboard.html."""
    theme       = storyboard.get("theme", "Storyboard")
    hardware    = storyboard.get("hardware_era", "")
    palette_str = storyboard.get("palette", "")
    lighting    = storyboard.get("lighting", "")
    technique   = storyboard.get("technique", "")

    hex_colors = re.findall(r"#[0-9A-Fa-f]{6}", palette_str)
    swatches = "".join(
        f'<div class="swatch" style="background:{h}" title="{h}"></div>'
        for h in hex_colors
    )

    meta_parts = [p for p in [
        f"<strong>{hardware}</strong>" if hardware else "",
        lighting, technique,
    ] if p]

    panels_html_parts = []
    for p in panels:
        desc = p.get("description", "")
        if len(desc) > 180:
            desc = desc[:177] + "…"
        svg = p["svg"]
        # Inject viewBox so the inline SVG scales properly to the column width
        if 'viewBox' not in svg and f'width="{_SVG_PANEL_W}"' in svg:
            svg = svg.replace(
                f'width="{_SVG_PANEL_W}" height="{_SVG_PANEL_H}"',
                f'width="{_SVG_PANEL_W}" height="{_SVG_PANEL_H}" '
                f'viewBox="0 0 {_SVG_PANEL_W} {_SVG_PANEL_H}" preserveAspectRatio="xMidYMid meet"',
            )
        panels_html_parts.append(
            f'<div class="panel">\n'
            f'  <div class="panel-svg">{svg}</div>\n'
            f'  <div class="panel-info">\n'
            f'    <div class="scene-num">Scene {p["id"]}</div>\n'
            f'    <div class="scene-title">{p["title"]}</div>\n'
            f'    <div class="scene-mood">{p["mood"]}</div>\n'
            f'    <div class="scene-desc">{desc}</div>\n'
            f'  </div>\n'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Storyboard: {theme}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0e0e0e; color: #bbb; font-family: 'Courier New', monospace; padding: 32px; }}
h1 {{ color: #eee; font-size: 1.3rem; margin-bottom: 10px; font-weight: normal; letter-spacing: 0.02em; }}
.meta {{ color: #666; font-size: 0.78rem; line-height: 1.9; margin-bottom: 14px; }}
.palette {{ display: flex; gap: 5px; align-items: center; margin-bottom: 28px; }}
.swatch {{ width: 16px; height: 16px; border-radius: 2px; flex-shrink: 0; }}
.panels {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 860px; }}
.panel {{ background: #161616; border: 1px solid #282828; border-radius: 3px; overflow: hidden; }}
.panel-svg {{ display: block; line-height: 0; }}
.panel-svg svg {{ display: block; width: 100%; height: auto; }}
.panel-info {{ padding: 10px 12px 12px; border-top: 1px solid #222; }}
.scene-num {{ color: #555; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 3px; }}
.scene-title {{ color: #ddd; font-size: 0.92rem; font-weight: bold; margin-bottom: 4px; }}
.scene-mood {{ color: #888; font-size: 0.73rem; font-style: italic; margin-bottom: 6px; }}
.scene-desc {{ color: #666; font-size: 0.7rem; line-height: 1.55; }}
</style>
</head>
<body>
<h1>{theme}</h1>
<div class="meta">{' &nbsp;·&nbsp; '.join(meta_parts)}</div>
<div class="palette">{swatches}</div>
<div class="panels">
{''.join(panels_html_parts)}
</div>
</body>
</html>"""

    STORYBOARD_HTML.write_text(html)
    return STORYBOARD_HTML


def run_stage3(storyboard: dict, prompts: list[dict], model_id: str, base_url: str,
               make_ansi: bool, make_svg: bool) -> None:
    """Stage 3: visual rendering — two independent modes, both optional.

    --ansi  Palette-indexed 16×12 pixel grid from LLM → ANSI truecolor block art.
            Good for quick terminal preview; kept as the lightweight fast path.

    --svg   Full SVG illustration per scene (320×180, paths + gradients + silhouettes)
            → individual scene_N_title.svg files + assembled storyboard.html.
            This is the proper storyboard view: panels in a 2-column grid with
            scene numbers, titles, moods, and descriptions.

    Both modes can run together; each uses a separate LLM call with its own prompt.
    """
    if not make_ansi and not make_svg:
        return

    modes = " + ".join(filter(None, ["ANSI" if make_ansi else "", "SVG" if make_svg else ""]))
    print(f"\n[Stage 3: Storyboard Rendering ({modes}) — {model_id}]")
    if make_ansi:
        print("  ANSI: palette-indexed pixel grid (truecolor terminal required)")
    if make_svg:
        print("  SVG: illustrated storyboard panels → storyboard.html")

    client      = OpenAI(base_url=base_url, api_key="none")
    era         = storyboard.get("hardware_era", "pixel art")
    technique   = storyboard.get("technique", "flat tiles")
    ref_palette = storyboard.get("palette", "")
    hex_colors  = re.findall(r"#[0-9A-Fa-f]{6}", ref_palette)

    svg_panels:   list[dict] = []   # collected for HTML assembly at the end
    ansi_created: list[str]  = []
    svg_created:  list[str]  = []

    for p in prompts:
        sid   = p["id"]
        title = p["title"]
        slug  = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        scene = next((s for s in storyboard.get("scenes", []) if s["id"] == sid), {})
        description = scene.get("description", "")
        mood        = scene.get("mood", "neutral")
        print(f"  [scene {sid}: {title}]", flush=True)

        # ── ANSI: palette-indexed pixel grid → block art ────────────────────
        if make_ansi:
            key_letters = "ABCDEFGHIJ"
            key_assignments = "\n".join(
                f"  {key_letters[i]} → {h}"
                for i, h in enumerate(hex_colors[:len(key_letters)])
            ) or "  A → #888888"
            valid_keys  = ", ".join(key_letters[:len(hex_colors)] or ["A"])
            example_row = (key_letters[0] * 4 + key_letters[min(1, len(hex_colors)-1)] * 4) * (_GRID_W // 8)

            grid_task = (
                f"You are a pixel artist. Render scene {sid}: \"{title}\" as a "
                f"{_GRID_W}×{_GRID_H} pixel art grid.\n\n"
                f"Hardware era: {era}\n"
                f"Technique: {technique}\n"
                f"Scene: {p['prompt']}\n\n"
                f"PALETTE — use ONLY these exact uppercase letter keys:\n"
                f"{key_assignments}\n\n"
                "OUTPUT — a JSON object, no prose, no markdown fences:\n"
                "{\n"
                f'  "palette": {{"{key_letters[0]}": "{hex_colors[0] if hex_colors else "#888888"}", ...}},\n'
                f'  "grid": [\n'
                f'    "{example_row}",\n'
                f'    ... exactly {_GRID_H} rows total\n'
                "  ]\n"
                "}\n\n"
                f"RULES:\n"
                f"- Every character must be one of: {valid_keys}\n"
                f"- NO spaces, NO dots, NO other characters\n"
                f"- Each row: exactly {_GRID_W} characters\n"
                f"- Total rows: exactly {_GRID_H}\n"
                "- Sky fills top rows, ground fills bottom rows\n"
                "- Alternate keys for dithering where the technique calls for it"
            )
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": grid_task}],
                    max_tokens=1024,
                    temperature=0.3,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = resp.choices[0].message.content or ""
                grid_palette, grid = _parse_grid(raw)
                if grid_palette and grid:
                    ansi_art = _grid_to_ansi(grid, grid_palette)
                    ans_path = Path(__file__).parent / f"scene_{sid}_{slug}.ans"
                    ans_path.write_text(ansi_art)
                    print(f"\n  Scene {sid}: {title}")
                    print(ansi_art)
                    print()
                    ansi_created.append(ans_path.name)
                else:
                    print(f"    [warning: could not parse pixel grid for scene {sid}]")
            except Exception as e:
                print(f"    [warning: ANSI render failed for scene {sid} — {e}]")

        # ── SVG: illustrated storyboard panel ───────────────────────────────
        if make_svg:
            svg = _generate_svg_panel(
                client, model_id, sid, title, description, mood, era, hex_colors
            )
            if svg is None:
                print(f"    [using fallback panel for scene {sid}]")
                svg = _fallback_svg_panel(sid, title, hex_colors)
            svg_path = Path(__file__).parent / f"scene_{sid}_{slug}.svg"
            svg_path.write_text(svg)
            print(f"  [saved → {svg_path.name}]")
            svg_created.append(svg_path.name)
            svg_panels.append({
                "id": sid, "title": title, "mood": mood,
                "description": description, "svg": svg,
            })

    # Assemble the HTML storyboard from all illustrated panels
    if make_svg and svg_panels:
        html_path = _assemble_storyboard_html(storyboard, svg_panels)
        print(f"\n  [storyboard.html — {len(svg_panels)} panels assembled]")
        print(f"  open in browser: file://{html_path.resolve()}")
        svg_created.append(html_path.name)

    total = len(ansi_created) + len(svg_created)
    if total:
        print(f"\n[Stage 3 complete — {total} file(s) created]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Storyboard to pixel art pipeline (CrewAI + smolagents)"
    )
    parser.add_argument("--theme", default=None,
                        help="Creative theme (default: daily rotation)")
    parser.add_argument("--simulate", action="store_true",
                        help="Print server commands but don't run them; use current server for both stages")
    parser.add_argument("--single-model", action="store_true",
                        help="Use the same model for both stages (skip model switch)")
    parser.add_argument("--dual-server", action="store_true",
                        help="Start Stage 1 on chips 0,1 (port 8000) and Stage 2 on chips 2,3 (port 8001)")
    parser.add_argument("--creative-model", default=DEFAULT_CREATIVE_MODEL,
                        help="Stage 1 model for storyboard creativity (default: Llama-3.3-70B)")
    parser.add_argument("--prompt-model", default=DEFAULT_PROMPT_MODEL,
                        help="Stage 2 model for prompt engineering on TT hardware (default: Qwen3-32B)")
    parser.add_argument("--headless", action="store_true",
                        help="Suppress interactive prompts")
    # Advanced: CPU orchestrator mode
    parser.add_argument("--cpu-orchestrator", action="store_true",
                        help="[Advanced] Run Stage 2 on a small Qwen3 model on host CPU — no model switch, "
                             "no chip reset. TT hardware stays fully occupied with Stage 1.")
    parser.add_argument("--cpu-model", default=DEFAULT_CPU_MODEL,
                        help="CPU orchestrator model (default: Qwen/Qwen3-0.6B; try Qwen/Qwen3-1.7B for quality)")
    # Stage 3: pixel art rendering
    parser.add_argument("--ansi", action="store_true",
                        help="Stage 3: render each scene as ANSI truecolor block art (prints to terminal)")
    parser.add_argument("--svg", action="store_true",
                        help="Stage 3: render each scene as an SVG pixel grid (saved to scene_N_title.svg)")
    args = parser.parse_args()

    # Auto-detect single-model: if the running model is the same family as --prompt-model
    # (e.g. Qwen3-32B is loaded and prompt-model is also Qwen), skip the switch entirely.
    # Explicit --single-model always wins; dual-server and cpu-orchestrator manage their own routing.
    _auto_detected = False
    if not args.single_model and not args.dual_server and not args.cpu_orchestrator:
        running = _detect_running_model(BASE_URL_PRIMARY)
        if running and _same_model_family(running, args.prompt_model):
            args.single_model = True
            args.creative_model = running   # use exact loaded ID, not the default string
            _auto_detected = True

    # Manual --single-model (not auto-detected): normalise creative model to prompt model
    if args.single_model and not _auto_detected:
        args.creative_model = args.prompt_model

    print("=" * 70)
    print("tt-agents Demo 5: Storyboard to Pixel Art Pipeline")
    print("=" * 70)
    print(f"Stage 1 model: {args.creative_model}  (creative storyboard, TT hardware)")
    if args.cpu_orchestrator:
        print(f"Stage 2 model: {args.cpu_model}  (pixel art prompts, CPU — no TT chips)")
    else:
        print(f"Stage 2 model: {args.prompt_model}  (pixel art prompts, TT hardware)")
    print(f"Endpoint:      {BASE_URL_PRIMARY}")
    if _auto_detected:
        print(f"[auto: {args.creative_model} detected — using for both stages, no model switch needed]")
    if args.simulate:
        print("[--simulate: server operations printed but not executed]")
    if args.dual_server:
        print("[--dual-server: Stage 1 → port 8000 (chips 0,1) | Stage 2 → port 8001 (chips 2,3)]")
    if args.cpu_orchestrator:
        print(f"[--cpu-orchestrator: Stage 2 runs on CPU (port {CPU_SERVER_PORT}) — no model switch needed]")
    print()

    theme = pick_theme(headless=args.headless, theme_override=args.theme)
    print(f"Theme: {theme}")
    print("-" * 70)

    cpu_proc = None

    # Determine base URLs for each stage
    if args.cpu_orchestrator:
        # CPU orchestrator: Stage 1 on TT, Stage 2 on host CPU. No chip switch needed.
        stage1_url = BASE_URL_PRIMARY
        stage2_url = BASE_URL_CPU
        print("\n[CPU Orchestrator Setup]")
        print("  Stage 1 (70B storyboard) runs on TT hardware — chips fully occupied")
        print("  Stage 2 (prompt engineering) runs on CPU Qwen3 — zero chip overhead")
        print("  No docker stop / tt-smi reset / model reload required")
        cpu_proc = _start_cpu_server(model=args.cpu_model, simulate=args.simulate)
        if not args.simulate and cpu_proc is not None:
            _wait_for_cpu_server()
    elif args.dual_server:
        setup_dual_server(args.creative_model, args.prompt_model, simulate=args.simulate)
        stage1_url = BASE_URL_PRIMARY    # port 8000, chips 0,1
        stage2_url = BASE_URL_SECONDARY  # port 8001, chips 2,3
    else:
        stage1_url = BASE_URL_PRIMARY
        stage2_url = BASE_URL_PRIMARY

    t_start = time.time()

    # ── Stage 1: Storyboard ────────────────────────────────────────────────────
    t1_start = time.time()
    storyboard = run_stage1(theme, args.creative_model, stage1_url)
    t1 = time.time() - t1_start
    print(f"[Stage 1 complete in {t1:.1f}s]")

    # ── Model switch (if needed) ───────────────────────────────────────────────
    # Skipped for --cpu-orchestrator (Stage 2 already running on CPU, no TT chips needed)
    # Skipped for --dual-server (both models already loaded on separate chip pairs)
    if not args.single_model and not args.dual_server and not args.cpu_orchestrator:
        switch_models(args.prompt_model, simulate=args.simulate)

    # ── Stage 2: Pixel art prompts ─────────────────────────────────────────────
    # In single-model mode, creative_model holds the actual running model ID
    # (either auto-detected or the same as prompt_model). Use it so the API
    # call matches the server's loaded model name exactly.
    if args.cpu_orchestrator:
        stage2_model = args.cpu_model
    elif args.single_model:
        stage2_model = args.creative_model
    else:
        stage2_model = args.prompt_model
    t2_start = time.time()
    prompts = run_stage2(storyboard, stage2_model, stage2_url)
    t2 = time.time() - t2_start
    print(f"[Stage 2 complete in {t2:.1f}s]")

    t_total = time.time() - t_start

    # ── Stage 3: Pixel art rendering ──────────────────────────────────────────
    # Uses the Stage 2 model (already loaded) — no server change needed.
    if args.ansi or args.svg:
        run_stage3(storyboard, prompts, stage2_model, stage2_url,
                   make_ansi=args.ansi, make_svg=args.svg)

    # ── Teardown ───────────────────────────────────────────────────────────────
    if args.dual_server:
        teardown_dual_server(simulate=args.simulate)
    if args.cpu_orchestrator:
        print("\n[CPU Orchestrator Teardown]")
        _stop_cpu_server(cpu_proc, simulate=args.simulate)

    # ── Output ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIXEL ART PROMPTS")
    print("=" * 70)
    for p in prompts:
        print(f"\nScene {p['id']}: {p['title']}")
        print(f"  {p['prompt']}")

    write_summary(theme, args.creative_model, stage2_model,
                  storyboard, prompts, t1, t2, t_total,
                  cpu_mode=args.cpu_orchestrator)

    print(f"\n✓ Pipeline complete in {t_total:.1f}s")
    print(f"  storyboard.json       → {STORYBOARD_FILE.name}")
    print(f"  pixelart_prompts.json → {PROMPTS_FILE.name}")
    print(f"  pipeline_summary.txt  → {SUMMARY_FILE.name}")
    if args.ansi:
        print(f"  scene_*.ans           → ANSI block art (cat to terminal)")
    if args.svg:
        print(f"  scene_*.svg           → SVG storyboard panels")
        print(f"  storyboard.html       → assembled storyboard (open in browser)")
    if not args.ansi and not args.svg:
        print()
        print("Next: render the prompts.")
        print("  LLM-rendered (no extra model): re-run with --ansi or --svg")
        print("  Image generation (Flux on TT): cd ~/code/tt-local-generator && ./bin/start_flux.sh")


if __name__ == "__main__":
    main()
