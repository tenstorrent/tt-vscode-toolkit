#!/usr/bin/env python3
"""
Proof 4: Dungeon Master Agent
Framework: smolagents ToolCallingAgent (with persistent JSON state)
Model: Llama-3.3-70B-Instruct (recommended for narrative quality)
       Qwen3-32B also works, slightly faster

Demonstrates:
  - Stateful multi-turn agents (world.json persists between sessions)
  - Tool-backed narrative consistency (DM can't invent things)
  - Creating new tools by naming them and writing a docstring
  - The 70B quality difference on creative/reasoning tasks

The tools here range from mechanical (roll_dice, update_player_hp) to
creative (manage_lore, examine_item). Each new tool is ~10-15 lines:
a name, a docstring the model reads, and a function body that touches
world state. The model figures out when and how to use each one.

Usage:
  python3 04_dungeon_master.py
  python3 04_dungeon_master.py --model "Qwen/Qwen3-32B"
  python3 04_dungeon_master.py --reset     (start fresh from world.json)

Controls:
  Type your action at the > prompt
  Type 'status'  — see your current stats and spell slots
  Type 'lore'    — see everything the DM has recorded about the world
  Type 'quit'    — exit and save
  Ctrl+C         — emergency exit

Install deps:
  pip install smolagents
"""
import argparse
import json
import os
import re
import random
import sys
import urllib.request
from pathlib import Path

try:
    from smolagents import OpenAIServerModel, ToolCallingAgent, tool
    from smolagents.monitoring import AgentLogger, LogLevel
except ImportError:
    print("ERROR: smolagents not installed. Run: pip install smolagents")
    sys.exit(1)

try:
    from rich.panel import Panel
except ImportError:
    print("ERROR: rich not installed. Run: pip install rich")
    sys.exit(1)


class _QuietLogger(AgentLogger):
    """Shows only tool names (not arguments or observations); suppresses the
    spurious 'If you want to return an answer' warning."""

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
                return  # suppress all other Panel content (step summaries, etc.)
        if args and isinstance(args[0], str) and "Observations:" in args[0]:
            return
        if int(level) <= LogLevel.ERROR:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        if self._MUTE in error_message:
            return
        super().log_error(error_message)

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
_FALLBACK_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _detect_model() -> str:
    """Query the vLLM endpoint and return the first loaded model ID."""
    if model_env := os.environ.get("VLLM_MODEL"):
        return model_env
    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=3) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["id"]
    except Exception:
        return _FALLBACK_MODEL


DEFAULT_MODEL = _detect_model()
WORLD_FILE = Path(__file__).parent / "world.json"
SESSION_FILE = Path(__file__).parent / "world_session.json"

# Module-level world state — shared across all @tool functions
_world_state: dict = {}


def _is_context_error(exc: Exception) -> bool:
    return "maximum context length" in str(exc) or "context_length_exceeded" in str(exc)


def _prune_memory(agent, keep_turns: int = 4) -> int:
    """Drop the oldest conversation turns from the agent's memory.

    Finds TaskStep boundaries and slices off everything before the
    (len - keep_turns)th turn, so recent context is preserved.
    Returns the number of steps removed.
    """
    from smolagents.memory import TaskStep
    steps = agent.memory.steps
    task_indices = [i for i, s in enumerate(steps) if isinstance(s, TaskStep)]
    if len(task_indices) <= keep_turns:
        return 0
    cut_at = task_indices[-keep_turns]
    agent.memory.steps = steps[cut_at:]
    return cut_at


# ── World state helpers ───────────────────────────────────────────────────────

def load_world(reset: bool = False) -> dict:
    source = WORLD_FILE if (reset or not SESSION_FILE.exists()) else SESSION_FILE
    with open(source) as f:
        return json.load(f)


def save_world(world: dict) -> None:
    with open(SESSION_FILE, "w") as f:
        json.dump(world, f, indent=2)


# ── Core state tools ──────────────────────────────────────────────────────────

@tool
def get_player_status() -> str:
    """Get the player's current HP, gold, inventory, spell slots, and location with exits and items present.
    Call this at the start of every turn before narrating anything.
    """
    p = _world_state["player"]
    loc = _world_state["locations"].get(p["location"], {})
    return json.dumps({
        "name": p["name"],
        "hp": f"{p['hp']}/{p['max_hp']}",
        "gold": p["gold"],
        "inventory": p["inventory"],
        "spellbook": p.get("spellbook", []),
        "spell_slots": p.get("spell_slots", {}),
        "location": loc.get("name", p["location"]),
        "location_description": loc.get("description", ""),
        "exits": loc.get("exits", []),
        "items_here": loc.get("items", []),
        "npcs_here": loc.get("npcs", []),
    }, indent=2)


@tool
def move_player(destination: str) -> str:
    """Move the player to a new location. Must be one of the current location's exits.

    Args:
        destination: The location ID to move to (e.g. 'rusty_axe', 'library', 'crypt').
    """
    p = _world_state["player"]
    current_loc = _world_state["locations"].get(p["location"], {})
    exits = current_loc.get("exits", [])
    if destination not in exits:
        available = ", ".join(exits) if exits else "none"
        return f"Cannot move to '{destination}'. Available exits: {available}"
    p["location"] = destination
    new_loc = _world_state["locations"].get(destination, {})
    _world_state["turn"] += 1
    save_world(_world_state)
    return json.dumps({
        "moved_to": new_loc.get("name", destination),
        "description": new_loc.get("description", ""),
        "items_here": new_loc.get("items", []),
        "npcs_here": new_loc.get("npcs", []),
        "exits": new_loc.get("exits", []),
    }, indent=2)


@tool
def pick_up_item(item_name: str) -> str:
    """Pick up an item from the current location and add it to the player's inventory.

    Args:
        item_name: The exact name of the item to pick up (must be present in current location).
    """
    p = _world_state["player"]
    loc = _world_state["locations"].get(p["location"], {})
    items = loc.get("items", [])
    if item_name not in items:
        available = ", ".join(items) if items else "none"
        return f"Item '{item_name}' not found here. Items available: {available}"
    items.remove(item_name)
    p["inventory"].append(item_name)
    _world_state["turn"] += 1
    save_world(_world_state)
    return f"Picked up '{item_name}'. Inventory: {p['inventory']}"


@tool
def roll_dice(sides: int, count: int = 1, modifier: int = 0) -> str:
    """Roll dice for combat, skill checks, or any random outcome. Returns individual rolls and total.

    Args:
        sides: Number of sides on the die. Must be 4, 6, 8, 10, 12, 20, or 100.
        count: Number of dice to roll (default 1, max 10).
        modifier: Flat bonus or penalty added to the total (default 0).
    """
    if sides not in (4, 6, 8, 10, 12, 20, 100):
        return f"Invalid dice: d{sides}. Use d4, d6, d8, d10, d12, d20, or d100."
    rolls = [random.randint(1, sides) for _ in range(max(1, min(count, 10)))]
    total = sum(rolls) + modifier
    mod_str = f" + {modifier}" if modifier > 0 else (f" - {abs(modifier)}" if modifier < 0 else "")
    return f"Rolled {count}d{sides}{mod_str}: {rolls} = {total}"


@tool
def update_player_hp(change: int, reason: str) -> str:
    """Modify player HP. Negative values deal damage, positive values heal.

    Args:
        change: HP change amount (negative = damage, positive = healing).
        reason: Short description of why HP changed (e.g. 'goblin attack', 'healing potion').
    """
    p = _world_state["player"]
    old_hp = p["hp"]
    p["hp"] = max(0, min(p["max_hp"], p["hp"] + change))
    save_world(_world_state)
    status = "alive" if p["hp"] > 0 else "DEAD"
    return f"HP: {old_hp} → {p['hp']}/{p['max_hp']} ({reason}) [{status}]"


@tool
def update_player_gold(change: int, reason: str) -> str:
    """Add or remove gold from the player's purse.

    Args:
        change: Gold change amount (negative = spending, positive = earning).
        reason: Short description of the transaction (e.g. 'bought potion', 'found treasure').
    """
    p = _world_state["player"]
    old_gold = p["gold"]
    p["gold"] = max(0, p["gold"] + change)
    save_world(_world_state)
    return f"Gold: {old_gold} → {p['gold']} ({reason})"


# ── Creative / generative tools ───────────────────────────────────────────────
# These tools show the second half of the pattern: tools that CREATE new material.
# The model calls them to generate content it then weaves into the narrative.
# Adding a new one takes ~10 lines: name it, describe it, touch world state.

@tool
def cast_spell(spell_name: str, target: str) -> str:
    """Cast a spell from the player's spellbook at a target. Consumes a spell slot.
    Returns the spell's mechanical effect — narrate it vividly.

    Args:
        spell_name: Name of the spell (e.g. 'magic_missile', 'healing_word', 'shield').
        target: What or who the spell is targeting (e.g. 'the goblin', 'myself').
    """
    p = _world_state["player"]
    spellbook = p.get("spellbook", [])
    if spell_name not in spellbook:
        known = ", ".join(spellbook) if spellbook else "none"
        return f"'{spell_name}' is not in your spellbook. Known spells: {known}"

    spells = _world_state.get("spells", {})
    spell = spells.get(spell_name)
    if not spell:
        return f"Spell '{spell_name}' has no definition in the world. Cannot cast."

    slot_level = str(spell["slot"])
    slots = p.get("spell_slots", {})
    if slots.get(slot_level, 0) <= 0:
        return f"No level-{slot_level} spell slots remaining. Rest to recover slots."

    slots[slot_level] -= 1
    p["spell_slots"] = slots
    _world_state["turn"] += 1
    save_world(_world_state)

    effect_parts = [f"Cast '{spell_name}' at {target}.", spell["description"]]
    if "damage" in spell:
        effect_parts.append(f"Deals up to {spell['damage']} damage.")
    if "heal" in spell:
        effect_parts.append(f"Restores up to {spell['heal']} HP.")
    if "effect" in spell:
        effect_parts.append(f"Effect: {spell['effect']}.")
    effect_parts.append(f"Slots remaining: {slots}")
    return " ".join(effect_parts)


@tool
def examine_item(item_name: str) -> str:
    """Examine an item closely to reveal its full description and hidden properties.
    Call this when a player inspects an item carefully — it reveals more than a glance would.

    Args:
        item_name: Name of the item to examine (must be in inventory or present in the current location).
    """
    p = _world_state["player"]
    loc = _world_state["locations"].get(p["location"], {})
    accessible = p["inventory"] + loc.get("items", [])
    if item_name not in accessible:
        return f"'{item_name}' is not in your inventory or in this location."

    details = _world_state.get("item_details", {}).get(item_name)
    if not details:
        return f"You examine the {item_name}. It appears to be exactly what it looks like."

    return json.dumps({
        "item": item_name,
        "description": details.get("description", ""),
        "hidden_properties": details.get("hidden", "Nothing unusual revealed."),
    }, indent=2)


@tool
def manage_lore(subject: str, description: str = "") -> str:
    """Record or retrieve lore about any person, place, object, or event.

    The DM should call this to:
    - RECORD: When inventing backstory, NPC personality, history, or secrets — pass both subject and description.
    - RETRIEVE: When recalling previously established facts — pass only subject.

    This keeps the world consistent across turns. If you invented something, record it.

    Args:
        subject: The person, place, or thing this lore describes (e.g. 'barkeep', 'the rusty key', 'crypt runes').
        description: The lore to record (1-3 sentences). Leave empty to look up existing lore.
    """
    lore = _world_state.setdefault("lore", {})
    key = subject.lower().strip()

    if description:
        lore[key] = description
        save_world(_world_state)
        return f"Lore recorded — {subject}: {description}"

    existing = lore.get(key)
    if existing:
        return f"Established lore — {subject}: {existing}"
    return f"No lore recorded for '{subject}' yet. Invent it now and call manage_lore to save it."


@tool
def check_rules(action: str) -> str:
    """Validate whether a player action is possible given current world state.
    Call this before resolving any unusual, risky, or potentially invalid action.
    Returns what's mechanically possible and any relevant constraints.

    Args:
        action: The action the player wants to take (e.g. 'pick the lock', 'cast fireball', 'bribe the guard').
    """
    p = _world_state["player"]
    loc = _world_state["locations"].get(p["location"], {})

    if p["hp"] <= 0:
        return "Player is dead. No actions possible."

    action_lower = action.lower()
    notes = []

    if any(w in action_lower for w in ("cast", "spell", "magic")):
        slots = p.get("spell_slots", {})
        total_slots = sum(slots.values())
        known = p.get("spellbook", [])
        notes.append(f"Spells known: {', '.join(known) or 'none'}")
        notes.append(f"Spell slots remaining: {slots} (total: {total_slots})")
        if total_slots == 0:
            notes.append("WARNING: No spell slots. Player must rest to cast again.")

    if any(w in action_lower for w in ("attack", "fight", "strike", "hit")):
        npcs = loc.get("npcs", [])
        notes.append(f"Entities present: {', '.join(npcs) or 'none'}")
        has_weapon = any(w in p["inventory"] for w in ("short sword", "dagger", "axe", "staff"))
        notes.append(f"Has melee weapon: {has_weapon}")

    if any(w in action_lower for w in ("pick up", "take", "grab")):
        items_here = loc.get("items", [])
        notes.append(f"Items available here: {', '.join(items_here) or 'none'}")

    if any(w in action_lower for w in ("move", "go", "travel", "walk")):
        notes.append(f"Available exits: {', '.join(loc.get('exits', [])) or 'none'}")

    result = f"Action '{action}' is mechanically possible."
    if notes:
        result += "\nContext:\n" + "\n".join(f"  • {n}" for n in notes)
    return result


# ── System prompt ─────────────────────────────────────────────────────────────

DM_SYSTEM_PROMPT = """You are a creative, atmospheric Dungeon Master running a text adventure set in the town of Millhaven.

CORE RULES:
1. Call get_player_status() ONCE per turn, before narrating any scene change, movement, combat,
   or item interaction. Call it exactly once — do not repeat it within a single turn.
   For pure dialogue or simple verbal responses, you may skip it.
2. Call move_player() with the exact location ID when the player travels.
3. Call pick_up_item() with the exact item name when picking up items.
4. For combat: call roll_dice() for attacks (d20), call update_player_hp() for all HP changes.
5. Call update_player_gold() for all gold transactions.
6. Never invent locations, items, or exits not in the world state.

CREATIVE TOOLS — use these to build living, consistent world content:
7. cast_spell(spell_name, target) — when player casts; consumes a slot, returns mechanical effect.
8. examine_item(item_name) — when player inspects closely; reveals hidden properties.
9. manage_lore(subject, description) — RECORD lore when you invent backstory, NPC personality, or
   secrets; RETRIEVE it on later turns by calling with only subject. Keeps the world consistent.
10. check_rules(action) — call before resolving unusual or ambiguous actions.

LORE WORKFLOW:
- When you give an NPC a personality or backstory, call manage_lore('npc_name', '...') immediately.
- Before describing an NPC you've met before, call manage_lore('npc_name') to retrieve what you recorded.
- Limit yourself to 1-2 lore calls per turn — record the most important new fact, not every detail.

STYLE:
- Write vivid, atmospheric descriptions (2-4 sentences per response).
- NPCs should feel like real people with histories — use manage_lore to make them consistent.
- Make combat tense and specific: describe the attack, the hit, the consequence.
- Items with hidden properties (examine_item) should feel like small discoveries.
- The world has secrets — the crypt runes, the mysterious coin, the torn journal all connect.
- Match tone to situation: warm in the tavern, ominous in the crypt, tense in combat.

Begin by calling get_player_status(), then describe the opening scene."""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _world_state

    parser = argparse.ArgumentParser(description="Dungeon master agent demo (smolagents)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID on vLLM")
    parser.add_argument("--reset", action="store_true", help="Reset to fresh world state")
    args = parser.parse_args()

    print("=" * 70)
    print("tt-agents Proof 4: Dungeon Master Agent (smolagents)")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Endpoint: {BASE_URL}")
    state_src = "world.json (fresh)" if args.reset or not SESSION_FILE.exists() else "world_session.json (saved)"
    print(f"State:    {state_src}")
    print("\nTip: 70B gives richer narrative; 32B responds in ~8s. Override with --model.")
    print("Commands: 'status' | 'lore' | 'quit'")
    print("-" * 70)

    _world_state = load_world(reset=args.reset)

    model = OpenAIServerModel(
        model_id=args.model,
        api_base=BASE_URL,
        api_key="none",
    )

    # ToolCallingAgent uses JSON tool calls (not CodeAgent's Python code generation).
    # More predictable for interactive sessions, easier to audit turn by turn.
    agent = ToolCallingAgent(
        tools=[
            get_player_status,
            move_player,
            pick_up_item,
            roll_dice,
            update_player_hp,
            update_player_gold,
            cast_spell,
            examine_item,
            manage_lore,
            check_rules,
        ],
        model=model,
        instructions=DM_SYSTEM_PROMPT,
        max_steps=15,
        logger=_QuietLogger(),
    )

    print("\nThe DM is setting the scene...\n")
    response = agent.run(
        "Describe the starting area and greet the player. Check their status first.",
        reset=True,
    )
    print(f"\nDM: {response}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nFarewell, adventurer!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Farewell, adventurer!")
            break

        if user_input.lower() in ("status", "stats"):
            try:
                s = json.loads(get_player_status())
                slots = s.get("spell_slots", {})
                slot_str = "  ".join(f"L{k}:{v}" for k, v in slots.items()) if slots else "none"
                print(f"\n[Status] HP: {s['hp']} | Gold: {s['gold']} g | Location: {s['location']}")
                print(f"[Inventory] {', '.join(s['inventory']) or 'empty'}")
                print(f"[Spells] {', '.join(s['spellbook']) or 'none'}  |  Slots: {slot_str}")
                print(f"[Exits] {', '.join(s['exits']) or 'none'}\n")
            except Exception as e:
                print(f"[Status error: {e}]\n")
            continue

        if user_input.lower() == "lore":
            lore = _world_state.get("lore", {})
            if lore:
                print("\n[World Lore — recorded by the DM this session]")
                for subject, text in lore.items():
                    print(f"  {subject}: {text}")
                print()
            else:
                print("[No lore recorded yet — explore and interact to build the world]\n")
            continue

        print()
        try:
            response = agent.run(f"Player action: {user_input}", reset=False)
            print(f"\nDM: {response}\n")
        except Exception as e:
            if _is_context_error(e):
                dropped = _prune_memory(agent, keep_turns=4)
                if dropped:
                    print(f"  [Session history trimmed ({dropped} old steps removed) — retrying...]\n")
                    try:
                        response = agent.run(f"Player action: {user_input}", reset=False)
                        print(f"\nDM: {response}\n")
                    except Exception as e2:
                        print(f"[Still over context after pruning. Type 'quit' and restart with --reset.]\n")
                else:
                    print("[Context full and nothing left to trim. Type 'quit' and restart with --reset.]\n")
            else:
                print(f"[Error: {e}]\n")
            continue

        if _world_state["player"]["hp"] <= 0:
            print("Your adventure ends here. Game over.")
            break

    save_world(_world_state)
    print(f"\n[World state saved to {SESSION_FILE.name}]")
    lore = _world_state.get("lore", {})
    if lore:
        print(f"[Lore entries created this session: {len(lore)}]")
    print(f"[Turns played: {_world_state['turn']}]")


if __name__ == "__main__":
    main()
