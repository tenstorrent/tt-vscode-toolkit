#!/usr/bin/env python3
"""
Demo 6: Generative Landscape SVG
Framework: none — direct OpenAI client call
Model: any local model (auto-detected from server)

Demonstrates:
  - No agent framework needed for structured creative output
  - Parameterized prompts: CLI flags shape the generated scene
  - LLM writes proper SVG with <defs> gradients, layered <polygon> terrain,
    cloud ellipse groups, atmospheric effects — full SVG primitive generation
  - Parse + validate LLM SVG output before saving

Usage:
  python3 06_landscape_svg.py                              (sunset palette, mountains)
  python3 06_landscape_svg.py --palette blue --no-mountains --stars
  python3 06_landscape_svg.py --palette purple --mountains --clouds --stars
  python3 06_landscape_svg.py --palette red --clouds
  python3 06_landscape_svg.py --simulate                   (print prompt, skip LLM)

Palette choices: sunset  blue  purple  red  orange

Install deps:
  pip install openai
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────────

BASE_URL    = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
OUTPUT_FILE = Path(__file__).parent / "landscape.svg"

SVG_W = 800
SVG_H = 450

# ── Palettes ────────────────────────────────────────────────────────────────────
# Each palette defines a complete color system for sky, terrain, and atmosphere.
# Colors are chosen for internal harmony — the gradients should feel cohesive.

PALETTES: dict[str, dict[str, str]] = {
    "sunset": {
        "name":           "Sunset",
        "adjective":      "warm, dramatic, cinematic",
        "sky_top":        "#1A0A2E",   # deep violet-black
        "sky_mid":        "#8B1A4A",   # magenta-dark
        "sky_bottom":     "#FF6B35",   # orange horizon
        "sun":            "#FFD700",   # golden disc
        "cloud":          "#FF9966",   # warm peach cloud
        "atmosphere":     "#FF4500",   # horizon ember glow
        "mountain_far":   "#4A2040",   # purple silhouette
        "mountain_mid":   "#2A1020",   # deeper purple
        "mountain_near":  "#1A0A15",   # near black
        "ground":         "#0F0805",   # very dark warm earth
    },
    "blue": {
        "name":           "Midnight Blue",
        "adjective":      "cool, serene, nocturnal",
        "sky_top":        "#000B1E",   # near black
        "sky_mid":        "#0A2A5E",   # deep navy
        "sky_bottom":     "#1A6090",   # steel blue
        "sun":            "#E8F4FF",   # pale moon
        "cloud":          "#2A4A7A",   # dark slate cloud
        "atmosphere":     "#4682B4",   # steel blue glow
        "mountain_far":   "#0A2A4A",   # dark blue ridge
        "mountain_mid":   "#061828",   # deeper navy
        "mountain_near":  "#040F1A",   # almost black
        "ground":         "#030A10",   # night ground
    },
    "purple": {
        "name":           "Violet Dusk",
        "adjective":      "mystical, dreamy, otherworldly",
        "sky_top":        "#120020",   # deep violet-black
        "sky_mid":        "#3B1060",   # purple
        "sky_bottom":     "#7B2D8B",   # bright violet
        "sun":            "#F0C0FF",   # pale lavender disc
        "cloud":          "#9A4DBB",   # purple cloud
        "atmosphere":     "#CC66FF",   # violet glow
        "mountain_far":   "#5A2080",   # purple ridge
        "mountain_mid":   "#3A1050",   # mid purple
        "mountain_near":  "#1A0528",   # near black purple
        "ground":         "#0A0215",   # very dark ground
    },
    "red": {
        "name":           "Ember",
        "adjective":      "intense, volcanic, apocalyptic",
        "sky_top":        "#1A0000",   # near black red
        "sky_mid":        "#6B0000",   # deep red
        "sky_bottom":     "#CC2200",   # bright red horizon
        "sun":            "#FF8800",   # orange sun
        "cloud":          "#8B2000",   # dark red cloud
        "atmosphere":     "#FF4400",   # ember glow
        "mountain_far":   "#4A0A00",   # dark red ridge
        "mountain_mid":   "#2A0500",   # deeper
        "mountain_near":  "#150200",   # near black
        "ground":         "#0A0100",   # very dark
    },
    "orange": {
        "name":           "Golden Hour",
        "adjective":      "warm, golden, autumnal",
        "sky_top":        "#1A0F00",   # warm dark
        "sky_mid":        "#8B4500",   # deep amber
        "sky_bottom":     "#FFB300",   # golden
        "sun":            "#FFEE00",   # bright yellow disc
        "cloud":          "#CC7722",   # amber cloud
        "atmosphere":     "#FF8C00",   # orange glow
        "mountain_far":   "#4A2800",   # amber ridge
        "mountain_mid":   "#2A1500",   # darker amber
        "mountain_near":  "#150A00",   # near black
        "ground":         "#0A0500",   # very dark
    },
}


# ── Prompt builder ──────────────────────────────────────────────────────────────

def _build_prompt(palette: dict, has_mountains: bool, has_clouds: bool,
                  has_stars: bool) -> str:
    """Build a structured LLM prompt that produces a layered landscape SVG."""
    p = palette
    w, h = SVG_W, SVG_H
    horizon_y = int(h * 0.50)
    ground_y  = int(h * 0.76)

    # Collect the ordered feature list (back to front)
    features: list[str] = [
        f"Sky: full-background <rect> using a 3-stop linearGradient id='sky' "
        f"({p['sky_top']} → {p['sky_mid']} → {p['sky_bottom']}, vertical)",
    ]
    if has_stars:
        features.append(
            f"Stars: 35-50 <circle> elements, r=0.5-2, fill={p['sun']}, "
            f"opacity 0.5-0.9, scattered above y={int(h*0.48)}"
        )
    features.append(
        f"Atmospheric glow: one wide <ellipse> centered near y={horizon_y}, "
        f"fill={p['atmosphere']}, opacity 0.15-0.30, no stroke"
    )
    if has_clouds:
        features.append(
            f"Clouds: 4-6 groups, each group = 3-5 overlapping <ellipse> in {p['cloud']}, "
            f"placed at varied x, y={int(h*0.10)}-{int(h*0.40)}"
        )
    if has_mountains:
        features += [
            f"Far mountains: <polygon> peaks at y={int(h*0.28)}-{int(h*0.48)}, "
            f"fill={p['mountain_far']} — must start 0,{h} and end {w},{h}",
            f"Mid mountains: <polygon> peaks at y={int(h*0.40)}-{int(h*0.58)}, "
            f"fill={p['mountain_mid']} — must start 0,{h} and end {w},{h}",
            f"Near mountains: <polygon> peaks at y={int(h*0.52)}-{int(h*0.68)}, "
            f"fill={p['mountain_near']} — must start 0,{h} and end {w},{h}",
        ]
    features += [
        f"Sun or moon: <circle> r=28-44, center near x={int(w*0.25)}-{int(w*0.75)}, "
        f"y={int(h*0.38)}-{int(h*0.54)}, fill={p['sun']}",
        f"Ground: <rect x='0' y='{ground_y}' width='{w}' height='{h - ground_y}' "
        f"fill={p['ground']}>",
    ]

    feature_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(features))

    mountain_rule = (
        "\nMOUNTAIN POLYGON RULE: Each polygon must span the full canvas width. "
        f"The points string must begin with '0,{h}' and end with '{w},{h}' "
        "so the shape closes along the bottom edge. "
        "Add 8-12 irregular peaks between those anchors for natural ridgelines.\n"
    ) if has_mountains else ""

    defs_hint = (
        "Required in <defs>:\n"
        "  - linearGradient id='sky' (gradientUnits='objectBoundingBox', "
        "x1='0' y1='0' x2='0' y2='1') with 3 stops\n"
        + ("  - radialGradient id='glow' (cx=0.5 cy=0.5 r=0.5) for atmospheric glow\n"
           if has_mountains else "")
    )

    return (
        f"Generate a layered landscape SVG ({w}×{h}px).\n"
        f"Mood: {p['adjective']}\n\n"
        f"PALETTE (use ONLY these colors):\n"
        f"  sky_top={p['sky_top']}  sky_mid={p['sky_mid']}  sky_bottom={p['sky_bottom']}\n"
        f"  sun/moon={p['sun']}  atmosphere={p['atmosphere']}  ground={p['ground']}\n"
        + (f"  clouds={p['cloud']}\n" if has_clouds else "")
        + (f"  mountains: far={p['mountain_far']}  mid={p['mountain_mid']}  near={p['mountain_near']}\n"
           if has_mountains else "")
        + f"\nLAYERS (back to front):\n{feature_block}\n"
        + mountain_rule
        + f"\n{defs_hint}\n"
        f"RULES:\n"
        f"  - Use ONLY colors from the PALETTE above\n"
        f"  - No <text>, no <image>, no <use>, no external hrefs\n"
        f"  - SVG root: <svg xmlns=\"http://www.w3.org/2000/svg\" "
        f"width=\"{w}\" height=\"{h}\">\n\n"
        f"Output ONLY the complete SVG, starting with <svg and ending with </svg>. "
        f"No explanation, no markdown, no comments."
    )


# ── SVG parsing ─────────────────────────────────────────────────────────────────

def _parse_svg(raw: str) -> str | None:
    """Extract and lightly validate SVG markup from an LLM response."""
    # Strip thinking blocks (Qwen3 with enable_thinking=True would prepend these)
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
        # Accept despite minor XML quirks if core shape elements are present
        if any(tag in svg for tag in ("<rect", "<path", "<polygon", "<circle", "<ellipse")):
            return svg
        return None


# ── Server helpers ───────────────────────────────────────────────────────────────

def _detect_model(base_url: str) -> str | None:
    """Return the model ID currently loaded on the server, or None."""
    try:
        url = f"{base_url.rstrip('/')}/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return None


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generative landscape SVG via direct LLM → SVG generation"
    )
    parser.add_argument(
        "--palette", choices=list(PALETTES), default="sunset",
        help="Color palette (default: sunset)",
    )
    parser.add_argument("--mountains", dest="mountains", action="store_true", default=True,
                        help="Include layered mountain ridges (default: on)")
    parser.add_argument("--no-mountains", dest="mountains", action="store_false",
                        help="Generate flat terrain without mountains")
    parser.add_argument("--clouds", dest="clouds", action="store_true", default=False,
                        help="Add cloud formations (default: off)")
    parser.add_argument("--no-clouds", dest="clouds", action="store_false")
    parser.add_argument("--stars", dest="stars", action="store_true", default=False,
                        help="Scatter stars in the sky (default: off)")
    parser.add_argument("--no-stars", dest="stars", action="store_false")
    parser.add_argument(
        "--output", default=str(OUTPUT_FILE),
        help=f"Output SVG path (default: {OUTPUT_FILE.name})",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model ID to use (default: auto-detect from server)",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Print the prompt but skip the LLM call",
    )
    args = parser.parse_args()

    palette = PALETTES[args.palette]
    feature_tags = [
        *( ["mountains"] if args.mountains else ["flat terrain"]),
        *( ["clouds"]    if args.clouds    else []),
        *( ["stars"]     if args.stars     else []),
    ]

    print("=" * 60)
    print("tt-agents Demo 6: Generative Landscape SVG")
    print("=" * 60)
    print(f"Palette:   {palette['name']} — {palette['adjective']}")
    print(f"Features:  {', '.join(feature_tags)}")
    print(f"Output:    {args.output}")
    print(f"Endpoint:  {BASE_URL}")
    print()

    # Auto-detect model unless --model given or --simulate
    model_id = args.model
    if not args.simulate and model_id is None:
        model_id = _detect_model(BASE_URL)
        if model_id:
            print(f"[auto-detected model: {model_id}]")
        else:
            print("[WARNING: no model detected — start a model server or use --model / --simulate]")
            sys.exit(1)
    if args.simulate:
        model_id = model_id or "simulated-model"

    prompt = _build_prompt(palette, args.mountains, args.clouds, args.stars)

    if args.simulate:
        print("[--simulate: LLM call skipped]\n")
        print("PROMPT:")
        print("─" * 60)
        print(prompt)
        return

    print(f"[Generating landscape SVG via {model_id}...]", flush=True)

    client = OpenAI(base_url=BASE_URL, api_key="none")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.6,
            # Qwen3 /no_think for direct structured SVG output
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"ERROR: LLM call failed — {e}")
        sys.exit(1)

    svg = _parse_svg(raw)
    if svg is None:
        raw_path = Path(args.output).with_suffix(".raw.txt")
        raw_path.write_text(raw)
        print(f"WARNING: could not extract valid SVG from response.")
        print(f"  Raw output saved to: {raw_path}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.write_text(svg)
    print(f"\n[SVG saved → {out_path.name}]")
    print(f"  open in browser: file://{out_path.resolve()}")
    print(f"\n[Done]")


if __name__ == "__main__":
    main()
