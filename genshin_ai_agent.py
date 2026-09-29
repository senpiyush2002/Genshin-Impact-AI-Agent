"""
Context-Aware AI Agent for the Genshin Impact Multi-Agent Combat Arena.

Scenario: CHARACTER vs CHARACTER (team vs team), supporting 1v1 / 2v2 / 3v3 /
4v4 / NvN. Two teams are drafted from the same character pool and fight each
other turn by turn inside a procedurally generated, partially observable arena.

Data sources
------------
- genshin_impact_-_Adjusted.csv  -> used for ALL combat math (this is the
  "adjusted" file with usage/popularity noise removed, per the project rules:
  "for calcs use the adjusted document").
- genshin_impact.csv             -> used ONLY for the "holistic" roster view
  printed alongside the battle report (region, release date, popularity...),
  never for scoring or team selection ("for a holistic look, use the
  original").

ascension_stat_calc toggle (0 or 1)
------------------------------------
The adjusted CSV carries two parallel stat sets:
  - lvl_90_HP / lvl_90_ATK / lvl_90_DEF   -> plain level-90 base stats
  - final_hp  / final_atk  / final_def    -> stats after folding in each
    character's ascension stat bonus (ATK%/DEF%/HP%/etc.)
`ascension_stat_calc = 1` makes the whole simulation use final_atk/final_def/
final_hp (plus the dmg_bonus / crit_ascension / Heals / em ascension flags as
small combat modifiers). `ascension_stat_calc = 0` ignores all of that and
uses the plain lvl_90_HP/ATK/DEF stats, matching the original prototype.

Hidden objective rules:
  Objective A - Elimination   : favor higher ATK. Prefer exactly one
                                 On-Field character with the rest Off-Field;
                                 Off-Field + Support is prioritized highest.
  Objective B - Control zones : same elimination signal as A, PLUS a bonus
                                 for the Survivability role.
  Objective C - Relic cores   : identical rule to Objective B.
  Objective D - Elemental orbs: no role weighting at all -- raw HP/ATK/DEF.
  Objective E - Survive longest: only HP/DEF (+ Healing ascension flag)
                                 matter; ATK is ignored.
  Objective F - Escort        : identical rule to Objective D.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------- Global toggles ----------------------------
ASCENSION_STAT_CALC = 1
CONSTELLATION_EFFECT = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADJUSTED_CSV_DEFAULT = os.path.join(SCRIPT_DIR, "genshin_impact_-_Adjusted.csv")
ORIGINAL_CSV_DEFAULT = os.path.join(SCRIPT_DIR, "genshin_impact.csv")

ELEMENTS = ["Pyro", "Hydro", "Cryo", "Electro", "Anemo", "Geo", "Dendro"]
WEAPONS = ["Sword", "Bow", "Catalyst", "Claymore", "Polearm"]

# Hidden objective codes as given in the problem statement.
OBJECTIVE_CODES: Dict[str, str] = {
    "A": "elimination",
    "B": "control",
    "C": "relic",
    "D": "orbs",
    "E": "survival",
    "F": "escort",
}
OBJECTIVE_LABELS: Dict[str, str] = {
    v: f"Objective {k} ({v})" for k, v in OBJECTIVE_CODES.items()
}
OBJECTIVES = list(OBJECTIVE_CODES.values())

HAZARDS = [
    "Lava eruption",
    "Flood",
    "Corruption zone",
    "Gravity inversion",
    "Element suppression",
    "Energy blackout",
    "Shield nullification",
    "Time distortion",
    "None",
]

WEATHER_EFFECTS = {
    "Thunderstorm": {"Electro": 1.30, "Hydro": 1.10, "visibility": 0.80},
    "Sandstorm": {"Bow": 0.75, "Catalyst": 0.85, "Claymore": 1.10},
    "Blizzard": {"Cryo": 1.20, "movement": 0.80},
    "Monsoon": {"Hydro": 1.15, "Pyro": 1.10, "Electro": 1.10},
    "Solar Flare": {"Pyro": 1.20, "Cryo": 0.85},
    "Clear": {},
}

TERRAIN_EFFECTS = {
    "Floating Islands": {"Bow": 1.15, "Catalyst": 1.10, "Claymore": 0.90},
    "Crystalline Caverns": {"Geo": 1.20, "defense": 1.15},
    "Dense Forest": {"Dendro": 1.20},
    "Ancient Ruins": {"energy": 0.75},
    "Plains": {},
}


# ---------------------------- Data classes ----------------------------


@dataclass
class Fighter:
    name: str
    element: str
    weapon: str
    roles: List[str]
    rarity: int
    lvl90_hp: float
    lvl90_atk: float
    lvl90_def: float
    final_hp: float
    final_atk: float
    final_def: float
    dmg_bonus: bool
    crit_ascension: bool
    heals: bool
    em: bool
    avg_copies_per_player: float
    constellation_pulls: (
        List[float]
    )  # index 0 = constellation_1_pull ... index 6 = constellation_7_pull


@dataclass
class ArenaState:
    weather: str
    terrain: str
    elemental_field: str
    visibility: str
    energy_flux: str
    hazard: str
    objective: str
    turn: int = 1


@dataclass
class TeamState:
    label: str
    team: List[Fighter]
    hp: Dict[str, float]
    max_hp: Dict[str, float]
    atk: Dict[str, float]
    defense: Dict[str, float]
    energy: Dict[str, float]
    cooldown: Dict[str, int]
    objective_progress: float = 0.0
    resources: float = 100.0
    history: List[str] = field(default_factory=list)


# ---------------------------- Data preparation ----------------------------


def load_characters(csv_path: str = ADJUSTED_CSV_DEFAULT) -> pd.DataFrame:
    """Load the ADJUSTED csv -- this is the only file used for combat math."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    constellation_cols = [f"constellation_{i}_pull" for i in range(1, 8)]
    required = [
        "name",
        "rarity",
        "weapon",
        "element",
        "roles",
        "lvl_90_HP",
        "lvl_90_ATK",
        "lvl_90_DEF",
        "final_hp",
        "final_atk",
        "final_def",
        "dmg_bonus",
        "crit_ascension",
        "Heals",
        "em",
        "avg_copies_per_player",
        *constellation_cols,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    numeric_cols = [
        "rarity",
        "lvl_90_HP",
        "lvl_90_ATK",
        "lvl_90_DEF",
        "final_hp",
        "final_atk",
        "final_def",
        "dmg_bonus",
        "crit_ascension",
        "Heals",
        "em",
        "avg_copies_per_player",
        *constellation_cols,
    ]
    df = df.copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["roles"] = df["roles"].fillna("").astype(str)
    return df


def load_holistic_profile(csv_path: str = ORIGINAL_CSV_DEFAULT) -> pd.DataFrame:
    """Load the ORIGINAL csv -- descriptive/reporting only, never used in scoring."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    keep = [
        c
        for c in [
            "name",
            "region",
            "release_date",
            "days_since_release",
            "num_banners",
            "is_standard_banner",
            "is_archon",
            "pulled_count",
        ]
        if c in df.columns
    ]
    return df[keep].copy()


def parse_roles(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def to_fighters(df: pd.DataFrame) -> List[Fighter]:
    fighters = []
    for _, r in df.iterrows():
        fighters.append(
            Fighter(
                name=str(r["name"]),
                element=str(r["element"]),
                weapon=str(r["weapon"]),
                roles=parse_roles(r["roles"]),
                rarity=int(r["rarity"]),
                lvl90_hp=float(r["lvl_90_HP"]),
                lvl90_atk=float(r["lvl_90_ATK"]),
                lvl90_def=float(r["lvl_90_DEF"]),
                final_hp=float(r["final_hp"]),
                final_atk=float(r["final_atk"]),
                final_def=float(r["final_def"]),
                dmg_bonus=bool(r["dmg_bonus"]),
                crit_ascension=bool(r["crit_ascension"]),
                heals=bool(r["Heals"]),
                em=bool(r["em"]),
                avg_copies_per_player=float(r["avg_copies_per_player"]),
                constellation_pulls=[
                    float(r[f"constellation_{i}_pull"]) for i in range(1, 8)
                ],
            )
        )
    return fighters


def effective_stats(
    f: Fighter, ascension_stat_calc: int
) -> Tuple[float, float, float]:
    if ascension_stat_calc == 1:
        return f.final_hp, f.final_atk, f.final_def
    return f.lvl90_hp, f.lvl90_atk, f.lvl90_def


# ---------------------------- Arena generation ----------------------------


def generate_arena(
    rng: random.Random,
    objective: Optional[str] = None,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    hazard: Optional[str] = None,
    elemental_field: Optional[str] = None,
    visibility: Optional[str] = None,
    energy_flux: Optional[str] = None,
) -> ArenaState:
    """Generate arena conditions. Standard parameters are chosen randomly if not specified."""
    chosen_weather = weather if weather and weather != "random" else rng.choice(list(WEATHER_EFFECTS))
    chosen_terrain = terrain if terrain and terrain != "random" else rng.choice(list(TERRAIN_EFFECTS))
    chosen_elemental = elemental_field if elemental_field and elemental_field != "random" else rng.choice(ELEMENTS + ["Neutral"])
    chosen_visibility = visibility if visibility and visibility != "random" else rng.choice(["Clear", "Clear", "Reduced"])
    chosen_energy = energy_flux if energy_flux and energy_flux != "random" else rng.choice(["Normal", "Normal", "Unstable", "Surge"])
    chosen_hazard = hazard if hazard and hazard != "random" else rng.choice(HAZARDS)

    if not objective or objective == "random":
        obj_code = rng.choice(list(OBJECTIVE_CODES))
        chosen_objective = OBJECTIVE_CODES[obj_code]
    elif objective in OBJECTIVE_CODES:
        chosen_objective = OBJECTIVE_CODES[objective]
    else:
        chosen_objective = objective.lower()

    return ArenaState(
        weather=chosen_weather,
        terrain=chosen_terrain,
        elemental_field=chosen_elemental,
        visibility=chosen_visibility,
        energy_flux=chosen_energy,
        hazard=chosen_hazard,
        objective=chosen_objective,
    )


# ---------------------------- Team-selection scoring ----------------------------


def hp_component(hp: float) -> float:
    return hp / 13000.0


def atk_component(atk: float) -> float:
    return atk / 450.0


def def_component(defn: float) -> float:
    return defn / 1100.0


def constellation_score_bonus(f: Fighter) -> float:
    pulls = f.constellation_pulls
    mean_pull = sum(pulls) / len(pulls) if pulls else 0.0
    if mean_pull <= 0:
        return 0.0

    normalized = [(p / mean_pull) * 0.10 for p in pulls]
    owned = int(f.avg_copies_per_player)
    owned = max(0, min(owned, len(normalized)))

    return sum(normalized[:owned])


def usage_rate_bonus(f: Fighter) -> float:
    extra_copies = max(0.0, f.avg_copies_per_player - 1.0)
    normalized = min(1.0, extra_copies / 7.0)
    return normalized * 0.15


def objective_character_value(
    f: Fighter, objective: str, ascension_stat_calc: int
) -> float:
    hp, atk, defn = effective_stats(f, ascension_stat_calc)
    hp_c, atk_c, def_c = (
        hp_component(hp),
        atk_component(atk),
        def_component(defn),
    )
    roles = set(f.roles)

    if objective == "elimination":
        value = 1.6 * atk_c + 0.5 * hp_c + 0.4 * def_c
        if "off-field" in roles:
            value += 0.6
            if "support" in roles:
                value += 1.3
        if ascension_stat_calc == 1:
            if f.dmg_bonus:
                value += 0.4
            if f.crit_ascension:
                value += 0.5

    elif objective in ("control", "relic"):
        value = 1.6 * atk_c + 0.5 * hp_c + 0.4 * def_c
        if "off-field" in roles:
            value += 0.6
            if "support" in roles:
                value += 1.3
        if "survivability" in roles:
            value += 1.4
        if ascension_stat_calc == 1:
            if f.em:
                value += 0.5
            if f.dmg_bonus:
                value += 0.4
            if f.crit_ascension:
                value += 0.5

    elif objective in ("orbs", "escort"):
        value = 1.0 * hp_c + 1.0 * atk_c + 1.0 * def_c

    elif objective == "survival":
        value = 1.4 * hp_c + 1.2 * def_c
        if ascension_stat_calc == 1 and f.heals:
            value += 1.5

    else:
        value = hp_c + atk_c + def_c

    value += 0.3 * (f.rarity / 5.0)
    value *= 1 + usage_rate_bonus(f)

    return value


def composition_bonus(
    f: Fighter, selected: List[Fighter], objective: str
) -> float:
    if objective not in ("elimination", "control", "relic"):
        return 0.0
    roles = set(f.roles)
    on_field_selected = sum(1 for s in selected if "on-field" in s.roles)

    bonus = 0.0
    if "on-field" in roles:
        bonus += 1.6 if on_field_selected == 0 else -1.2
    if "off-field" in roles:
        bonus += 1.0
        if "support" in roles:
            bonus += 1.3
    return bonus


def context_multiplier(f: Fighter, arena: ArenaState) -> float:
    mult = 1.0
    weather = WEATHER_EFFECTS.get(arena.weather, {})
    terrain = TERRAIN_EFFECTS.get(arena.terrain, {})

    mult *= weather.get(f.element, 1.0)
    mult *= weather.get(f.weapon, 1.0)
    mult *= terrain.get(f.element, 1.0)
    mult *= terrain.get(f.weapon, 1.0)

    if "defense" in terrain:
        mult *= 1.05 + (0.05 if "survivability" in f.roles else 0)

    if arena.elemental_field == f.element:
        mult *= 1.10

    if arena.visibility == "Reduced" and f.weapon in ("Bow", "Catalyst"):
        mult *= 0.90

    if arena.energy_flux == "Unstable" and "dps" in f.roles:
        mult *= 0.97

    if arena.hazard == "Energy blackout":
        mult *= 0.82 if "dps" in f.roles else 1.02
    if arena.hazard == "Shield nullification":
        mult *= 0.75 if "survivability" in f.roles else 1.0
    if arena.hazard == "Element suppression":
        mult *= 0.88

    return mult


def role_synergy(a: Fighter, b: Fighter) -> float:
    ra, rb = set(a.roles), set(b.roles)
    score = 0.0
    if ("support" in ra and "dps" in rb) or ("support" in rb and "dps" in ra):
        score += 2.0
    if ("survivability" in ra and "dps" in rb) or (
        "survivability" in rb and "dps" in ra
    ):
        score += 1.6
    if ("off-field" in ra and "on-field" in rb) or (
        "off-field" in rb and "on-field" in ra
    ):
        score += 1.6
    if ra == rb and "dps" in ra:
        score -= 0.5
    return score


def select_team(
    fighters: List[Fighter],
    arena: ArenaState,
    team_size: int,
    ascension_stat_calc: int,
    exclude: Optional[Set[str]] = None,
    pre_selected: Optional[List[Fighter]] = None,
) -> Tuple[List[Fighter], List[Tuple[str, float]]]:
    if team_size < 1:
        raise ValueError("team_size must be >= 1")
    exclude = exclude or set()
    pre_selected = pre_selected or []

    selected: List[Fighter] = list(pre_selected)
    explanation: List[Tuple[str, float]] = [
        (f.name, 0.0) for f in pre_selected
    ]

    needed = team_size - len(selected)
    if needed <= 0:
        return selected[:team_size], explanation[:team_size]

    pool = [
        f
        for f in fighters
        if f.name not in exclude and f.name not in {s.name for s in selected}
    ]
    scored = []
    for f in pool:
        value = objective_character_value(
            f, arena.objective, ascension_stat_calc
        )
        value *= context_multiplier(f, arena)
        scored.append((f, value))

    for _ in range(min(needed, len(scored))):
        candidates = []
        for f, value in scored:
            if f in selected:
                continue
            comp = composition_bonus(f, selected, arena.objective)

            roles = set(f.roles)
            current_roles = set(r for x in selected for r in x.roles)
            diversity = 0.0
            if "support" in roles and "support" not in current_roles:
                diversity += 1.0
            if "survivability" in roles and "survivability" not in current_roles:
                diversity += 1.0

            total = value + comp + diversity
            candidates.append((total, f, value))

        if not candidates:
            break
        _, chosen, raw_value = max(candidates, key=lambda x: x[0])
        selected.append(chosen)
        explanation.append((chosen.name, round(raw_value, 3)))

    return selected, explanation


def build_matchup(
    fighters: List[Fighter],
    arena: ArenaState,
    team_size: int,
    ascension_stat_calc: int,
    custom_chars_a: Optional[List[str]] = None,
    custom_chars_b: Optional[List[str]] = None,
    randomize_teams: bool = False,
    rng: Optional[random.Random] = None,
) -> Tuple[
    List[Fighter],
    List[Fighter],
    List[Tuple[str, float]],
    List[Tuple[str, float]],
]:
    rng = rng or random.Random()
    fighter_map = {f.name.lower(): f for f in fighters}

    pre_a = [fighter_map[name.lower()] for name in (custom_chars_a or []) if name.lower() in fighter_map]
    pre_b = [fighter_map[name.lower()] for name in (custom_chars_b or []) if name.lower() in fighter_map]

    if randomize_teams:
        available = list(fighters)
        rng.shuffle(available)
        
        # Pick completely random teams
        team_a = available[:team_size]
        team_b = available[team_size : team_size * 2]
        expl_a = [(f.name, 0.0) for f in team_a]
        expl_b = [(f.name, 0.0) for f in team_b]
        return team_a, team_b, expl_a, expl_b

    # Optimal selection using AI heuristics, incorporating user pre-selected characters
    team_a, expl_a = select_team(
        fighters, arena, team_size, ascension_stat_calc, pre_selected=pre_a
    )
    exclude = {f.name for f in team_a}
    
    # Exclude already picked character for B unless specifically passed
    for f in pre_b:
        exclude.discard(f.name)

    team_b, expl_b = select_team(
        fighters,
        arena,
        team_size,
        ascension_stat_calc,
        exclude=exclude,
        pre_selected=pre_b,
    )
    return team_a, team_b, expl_a, expl_b


# ---------------------------- Team state / combat ----------------------------


def power_adjusted_stats(
    f: Fighter, ascension_stat_calc: int, constellation_effect: int
) -> Tuple[float, float, float]:
    hp, atk, defn = effective_stats(f, ascension_stat_calc)
    if constellation_effect == 1:
        boost = 1 + constellation_score_bonus(f)
        hp *= boost
        atk *= boost
        defn *= boost
    return hp, atk, defn


def make_team_state(
    label: str,
    team: List[Fighter],
    ascension_stat_calc: int,
    constellation_effect: int,
) -> TeamState:
    hp, max_hp, atk, defense = {}, {}, {}, {}
    for f in team:
        h, a, d = power_adjusted_stats(
            f, ascension_stat_calc, constellation_effect
        )
        hp[f.name] = h
        max_hp[f.name] = h
        atk[f.name] = a
        defense[f.name] = d
    return TeamState(
        label=label,
        team=team,
        hp=hp,
        max_hp=max_hp,
        atk=atk,
        defense=defense,
        energy={f.name: 30.0 for f in team},
        cooldown={f.name: 0 for f in team},
    )


def alive_fighters(state: TeamState) -> List[Fighter]:
    return [f for f in state.team if state.hp[f.name] > 0]


def team_health(state: TeamState) -> float:
    ratios = [
        max(0.0, state.hp[f.name] / max(state.max_hp[f.name], 1))
        for f in state.team
    ]
    return float(np.mean(ratios)) if ratios else 0.0


def choose_action(
    state: TeamState, arena: ArenaState, rng: random.Random
) -> Tuple[str, str]:
    health = team_health(state)

    if arena.objective == "survival" and health < 0.55:
        return "Defend", "Survival objective + low team health"

    if health < 0.35:
        return "Retreat", "Critical team health"

    if arena.objective in ("control", "relic", "orbs", "escort"):
        if state.objective_progress < 0.75:
            if health < 0.65:
                return (
                    "Coordinate",
                    "Protect objective progress while recovering",
                )
            return (
                "Capture Objective",
                "Objective score has higher value than raw damage",
            )

    if arena.hazard != "Energy blackout":
        ready = [
            f
            for f in alive_fighters(state)
            if state.energy[f.name] >= 60 and state.cooldown[f.name] == 0
        ]
        if ready and arena.objective == "elimination":
            return (
                "Burst",
                "Burst available and elimination objective favors damage",
            )

    if any(state.cooldown[f.name] == 0 for f in alive_fighters(state)):
        return (
            "Skill",
            "Efficient damage/support without spending all burst energy",
        )

    if state.resources < 20:
        return "Gather Resource", "Resource reserve is low"

    return rng.choice(
        [
            ("Attack", "Default pressure"),
            ("Move", "Reposition under uncertainty"),
            ("Coordinate", "Maintain team coordination"),
        ]
    )


def apply_action(
    state: TeamState, arena: ArenaState, action: str, ascension_stat_calc: int
) -> None:
    if action == "Gather Resource":
        state.resources = min(100, state.resources + 18)
    elif action == "Capture Objective":
        state.objective_progress = min(1.0, state.objective_progress + 0.18)
        state.resources = max(0, state.resources - 5)
    elif action == "Defend":
        for f in state.team:
            if state.hp[f.name] <= 0:
                continue
            heal_mult = 1.5 if (ascension_stat_calc == 1 and f.heals) else 1.0
            state.hp[f.name] = min(
                state.max_hp[f.name],
                state.hp[f.name] + 0.04 * state.max_hp[f.name] * heal_mult,
            )
    elif action == "Retreat":
        for f in state.team:
            if state.hp[f.name] <= 0:
                continue
            state.hp[f.name] = min(
                state.max_hp[f.name],
                state.hp[f.name] + 0.02 * state.max_hp[f.name],
            )
        state.objective_progress = max(0, state.objective_progress - 0.03)
    elif action == "Move":
        state.resources = max(0, state.resources - 3)
    elif action == "Coordinate":
        state.resources = max(0, state.resources - 4)
        state.objective_progress = min(1.0, state.objective_progress + 0.04)
    elif action == "Skill":
        state.resources = max(0, state.resources - 5)
        for f in alive_fighters(state):
            state.energy[f.name] = min(100, state.energy[f.name] + 12)
            state.cooldown[f.name] = 1
    elif action == "Burst":
        state.resources = max(0, state.resources - 15)
        for f in alive_fighters(state):
            state.energy[f.name] = max(0, state.energy[f.name] - 70)
            state.cooldown[f.name] = 3
    elif action == "Attack":
        state.resources = max(0, state.resources - 4)
        for f in alive_fighters(state):
            state.energy[f.name] = min(100, state.energy[f.name] + 7)

    hazard_damage = {
        "Lava eruption": 0.07,
        "Flood": 0.04,
        "Corruption zone": 0.06,
        "Gravity inversion": 0.05,
        "Time distortion": 0.03,
    }.get(arena.hazard, 0.0)
    if hazard_damage:
        for f in state.team:
            state.hp[f.name] = max(0.0, state.hp[f.name] * (1 - hazard_damage))

    if arena.hazard == "Energy blackout":
        for f in state.team:
            state.energy[f.name] *= 0.75

    for f in state.team:
        state.cooldown[f.name] = max(0, state.cooldown[f.name] - 1)
        state.energy[f.name] = min(100, state.energy[f.name] + 3)


def resolve_damage(
    attacker: TeamState,
    defender: TeamState,
    action: str,
    rng: random.Random,
    ascension_stat_calc: int,
) -> None:
    action_power = {"Attack": 0.5, "Skill": 0.8, "Burst": 1.6}.get(action, 0.0)
    if action_power == 0.0:
        return

    alive_att = alive_fighters(attacker)
    alive_def = alive_fighters(defender)
    if not alive_att or not alive_def:
        return

    total_atk = 0.0
    for f in alive_att:
        weight = 1.0 if "on-field" in f.roles else 0.5
        ascension_mult = 1.0
        if ascension_stat_calc == 1:
            if f.dmg_bonus:
                ascension_mult *= 1.10
            if f.crit_ascension:
                ascension_mult *= 1.15
        total_atk += attacker.atk[f.name] * weight * ascension_mult

    avg_def = float(np.mean([defender.defense[f.name] for f in alive_def]))
    mitigation = avg_def / (avg_def + 600.0)
    raw_damage = total_atk * action_power * (1 - mitigation)
    raw_damage *= rng.uniform(0.85, 1.15)

    alive_def_sorted = sorted(
        alive_def,
        key=lambda f: defender.hp[f.name] / max(defender.max_hp[f.name], 1),
    )
    primary = alive_def_sorted[0]
    defender.hp[primary.name] = max(
        0.0, defender.hp[primary.name] - raw_damage * 0.7
    )
    if len(alive_def_sorted) > 1:
        splash = raw_damage * 0.3 / (len(alive_def_sorted) - 1)
        for f in alive_def_sorted[1:]:
            defender.hp[f.name] = max(0.0, defender.hp[f.name] - splash)


# ---------------------------- Scoring ----------------------------


def score_match(
    state: TeamState,
    opponent: TeamState,
    arena: ArenaState,
    ascension_stat_calc: int,
) -> Dict[str, float]:
    survival = team_health(state)
    objective = state.objective_progress
    resource = state.resources / 100

    alive = alive_fighters(state)
    if len(alive) > 1:
        pairs = list(combinations(alive, 2))
        synergy = min(
            1.0,
            float(np.mean([max(0.0, role_synergy(a, b)) for a, b in pairs]))
            / 4.0,
        )
    else:
        synergy = 0.0

    damage = round(max(0.0, 1 - team_health(opponent)), 3)

    adaptability = (
        0.75
        if (
            arena.hazard in ("Energy blackout", "Element suppression")
            and any(
                "support" in f.roles or "survivability" in f.roles
                for f in state.team
            )
        )
        else 0.55
    )
    if ascension_stat_calc == 1 and any(f.em for f in state.team):
        adaptability = min(1.0, adaptability + 0.10)

    risk_efficiency = max(0.0, survival * 0.7 + resource * 0.3)

    total = (
        0.25 * survival
        + 0.20 * objective
        + 0.15 * resource
        + 0.15 * synergy
        + 0.10 * damage
        + 0.10 * adaptability
        + 0.05 * risk_efficiency
    )
    return {
        "Survival": round(survival, 3),
        "ObjectiveControl": round(objective, 3),
        "ResourceManagement": round(resource, 3),
        "TeamSynergy": round(synergy, 3),
        "DamageContribution": round(damage, 3),
        "Adaptability": round(adaptability, 3),
        "RiskEfficiency": round(risk_efficiency, 3),
        "TotalScore": round(total, 3),
    }


def determine_winner(
    state_a: TeamState,
    state_b: TeamState,
    score_a: Dict,
    score_b: Dict,
    arena: ArenaState,
) -> str:
    alive_a, alive_b = bool(alive_fighters(state_a)), bool(
        alive_fighters(state_b)
    )
    if alive_a and not alive_b:
        return state_a.label
    if alive_b and not alive_a:
        return state_b.label
    if not alive_a and not alive_b:
        return "Draw (double knockout)"

    if arena.objective in ("control", "relic", "orbs", "escort"):
        if state_a.objective_progress != state_b.objective_progress:
            return (
                state_a.label
                if state_a.objective_progress > state_b.objective_progress
                else state_b.label
            )

    if score_a["TotalScore"] != score_b["TotalScore"]:
        return (
            state_a.label
            if score_a["TotalScore"] > score_b["TotalScore"]
            else state_b.label
        )
    return "Draw"


# ---------------------------- Simulation ----------------------------


def run_battle(
    adjusted_csv: str = ADJUSTED_CSV_DEFAULT,
    team_size: int = 4,
    turns: int = 20,
    seed: int = 42,
    ascension_stat_calc: int = ASCENSION_STAT_CALC,
    constellation_effect: int = CONSTELLATION_EFFECT,
    objective: Optional[str] = None,
    custom_chars_a: Optional[List[str]] = None,
    custom_chars_b: Optional[List[str]] = None,
    randomize_arena: bool = True,
    randomize_teams: bool = False,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    hazard: Optional[str] = None,
):
    rng = random.Random(seed)
    np.random.seed(seed)

    fighters = to_fighters(load_characters(adjusted_csv))

    if randomize_arena:
        arena = generate_arena(
            rng,
            objective=objective,
            weather=weather,
            terrain=terrain,
            hazard=hazard,
        )
    else:
        arena = generate_arena(
            rng,
            objective=objective or "elimination",
            weather=weather or "Clear",
            terrain=terrain or "Plains",
            hazard=hazard or "None",
            elemental_field="Neutral",
            visibility="Clear",
            energy_flux="Normal",
        )

    team_a, team_b, expl_a, expl_b = build_matchup(
        fighters,
        arena,
        team_size,
        ascension_stat_calc,
        custom_chars_a=custom_chars_a,
        custom_chars_b=custom_chars_b,
        randomize_teams=randomize_teams,
        rng=rng,
    )
    state_a = make_team_state(
        "Team A", team_a, ascension_stat_calc, constellation_effect
    )
    state_b = make_team_state(
        "Team B", team_b, ascension_stat_calc, constellation_effect
    )

    log = []
    log.append(
        f"Arena: weather={arena.weather}, terrain={arena.terrain}, "
        f"elemental_field={arena.elemental_field}, hazard={arena.hazard}, "
        f"objective={OBJECTIVE_LABELS.get(arena.objective, arena.objective)}"
    )
    log.append("Team A: " + ", ".join(f.name for f in team_a))
    log.append("Team B: " + ", ".join(f.name for f in team_b))

    for turn in range(1, turns + 1):
        arena.turn = turn

        if turn > 1 and rng.random() < 0.15 and randomize_arena:
            arena.hazard = rng.choice(HAZARDS)
            log.append(f"Turn {turn}: hidden event changed -> {arena.hazard}")

        for attacker, defender in ((state_a, state_b), (state_b, state_a)):
            if not alive_fighters(attacker):
                continue
            action, reason = choose_action(attacker, arena, rng)
            apply_action(attacker, arena, action, ascension_stat_calc)
            resolve_damage(
                attacker, defender, action, rng, ascension_stat_calc
            )
            log.append(
                f"Turn {turn} [{attacker.label}]: {action} | {reason} | "
                f"self_health={team_health(attacker):.2f}, opp_health={team_health(defender):.2f}, "
                f"objective={attacker.objective_progress:.2f}, resources={attacker.resources:.1f}"
            )

        if not alive_fighters(state_a) or not alive_fighters(state_b):
            log.append(
                f"Turn {turn}: one team has been fully knocked out -- battle ends early."
            )
            break
        if arena.objective in ("control", "relic", "orbs", "escort") and (
            state_a.objective_progress >= 1.0
            or state_b.objective_progress >= 1.0
        ):
            log.append(
                f"Turn {turn}: objective completed -- battle ends early."
            )
            break

    score_a = score_match(state_a, state_b, arena, ascension_stat_calc)
    score_b = score_match(state_b, state_a, arena, ascension_stat_calc)
    winner = determine_winner(state_a, state_b, score_a, score_b, arena)

    return (
        arena,
        state_a,
        state_b,
        score_a,
        score_b,
        winner,
        log,
        expl_a,
        expl_b,
    )


def prompt_user_inputs(fighters: List[Fighter]) -> dict:
    """Interactive user configuration when launched directly without CLI args."""
    print("==================================================")
    print("    Genshin Combat Simulation Setup Configuration ")
    print("==================================================\n")

    # 1. Team Size
    ts_in = input("Enter Team Size [1-8] (default 4): ").strip()
    team_size = int(ts_in) if ts_in.isdigit() and 1 <= int(ts_in) <= 8 else 4

    # 2. Objective Selection
    print("\nSelect Objective:")
    print("  [A] Elimination (default)")
    print("  [B] Control zones")
    print("  [C] Relic cores")
    print("  [D] Elemental orbs")
    print("  [E] Survive longest")
    print("  [F] Escort")
    print("  [R] Randomize Objective")
    obj_in = input("Objective choice (A/B/C/D/E/F/R): ").strip().upper()
    objective = (
        OBJECTIVE_CODES.get(obj_in, "random")
        if obj_in in OBJECTIVE_CODES
        else "random"
    )

    # 3. Field & Arena Randomization
    rand_arena_in = (
        input("\nRandomize field conditions (Weather/Terrain/Hazards)? [Y/n]: ")
        .strip()
        .lower()
    )
    randomize_arena = rand_arena_in != "n"

    # 4. Character Selection or Randomization
    print("\nTeam Selection Options:")
    print("  1. Optimal Selection by AI (default)")
    print("  2. Fully Randomize Teams")
    print("  3. Pick Custom Characters")
    team_opt = input("Choice (1/2/3): ").strip()

    custom_a, custom_b = [], []
    randomize_teams = False

    if team_opt == "2":
        randomize_teams = True
    elif team_opt == "3":
        print(f"\nEnter up to {team_size} comma-separated names for Team A")
        print("(e.g., 'Raiden Shogun, Bennett, Xiangling'):")
        chars_a_str = input("Team A Characters: ").strip()
        if chars_a_str:
            custom_a = [c.strip() for c in chars_a_str.split(",")]

        print(f"\nEnter up to {team_size} comma-separated names for Team B:")
        chars_b_str = input("Team B Characters: ").strip()
        if chars_b_str:
            custom_b = [c.strip() for c in chars_b_str.split(",")]

    return {
        "team_size": team_size,
        "objective": objective,
        "randomize_arena": randomize_arena,
        "randomize_teams": randomize_teams,
        "custom_chars_a": custom_a,
        "custom_chars_b": custom_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjusted-csv", default=ADJUSTED_CSV_DEFAULT)
    parser.add_argument("--original-csv", default=ORIGINAL_CSV_DEFAULT)
    parser.add_argument("--team-size", type=int, default=None)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective",
        choices=["A", "B", "C", "D", "E", "F", "random"],
        default=None,
    )
    parser.add_argument(
        "--chars-a",
        nargs="*",
        default=None,
        help="Specific character names for Team A",
    )
    parser.add_argument(
        "--chars-b",
        nargs="*",
        default=None,
        help="Specific character names for Team B",
    )
    parser.add_argument(
        "--randomize-arena",
        action="store_true",
        default=None,
        help="Randomize weather/terrain/hazards",
    )
    parser.add_argument(
        "--randomize-teams", action="store_true", help="Randomize character selection"
    )
    parser.add_argument(
        "--ascension-stat-calc",
        type=int,
        choices=[0, 1],
        default=ASCENSION_STAT_CALC,
    )
    parser.add_argument(
        "--constellation-effect",
        type=int,
        choices=[0, 1],
        default=CONSTELLATION_EFFECT,
    )
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt mode")
    args = parser.parse_args()

    fighters = to_fighters(load_characters(args.adjusted_csv))

    # Trigger interactive prompt mode if requested or if no arguments were explicitly passed
    if args.interactive or (
        args.team_size is None and args.objective is None and args.chars_a is None
    ):
        config = prompt_user_inputs(fighters)
        team_size = config["team_size"]
        objective = config["objective"]
        randomize_arena = config["randomize_arena"]
        randomize_teams = config["randomize_teams"]
        custom_a = config["custom_chars_a"]
        custom_b = config["custom_chars_b"]
    else:
        team_size = args.team_size if args.team_size is not None else 4
        objective = (
            OBJECTIVE_CODES.get(args.objective, "random")
            if args.objective in OBJECTIVE_CODES
            else "random"
        )
        randomize_arena = (
            args.randomize_arena if args.randomize_arena is not None else True
        )
        randomize_teams = args.randomize_teams
        custom_a = args.chars_a or []
        custom_b = args.chars_b or []

    (
        arena,
        state_a,
        state_b,
        score_a,
        score_b,
        winner,
        log,
        expl_a,
        expl_b,
    ) = run_battle(
        adjusted_csv=args.adjusted_csv,
        team_size=team_size,
        turns=args.turns,
        seed=args.seed,
        ascension_stat_calc=args.ascension_stat_calc,
        constellation_effect=args.constellation_effect,
        objective=objective,
        custom_chars_a=custom_a,
        custom_chars_b=custom_b,
        randomize_arena=randomize_arena,
        randomize_teams=randomize_teams,
    )

    print("\n=== GENSHIN IMPACT CHARACTER-VS-CHARACTER ARENA ===\n")
    print(
        f"ascension_stat_calc = {args.ascension_stat_calc} "
        f"({'final_atk/final_def/final_hp + ascension flags' if args.ascension_stat_calc else 'plain lvl_90_HP/ATK/DEF'})"
    )
    print(
        f"constellation_effect = {args.constellation_effect} "
        f"({'constellation power-up applied to combat HP/ATK/DEF' if args.constellation_effect else 'ignored'})\n"
    )

    print("Arena:")
    print(f"  Weather          : {arena.weather}")
    print(f"  Terrain          : {arena.terrain}")
    print(f"  Elemental Field  : {arena.elemental_field}")
    print(f"  Hazard           : {arena.hazard}")
    print(
        f"  Objective        : {OBJECTIVE_LABELS.get(arena.objective, arena.objective)}"
    )

    for label, state in (("Team A", state_a), ("Team B", state_b)):
        print(f"\n{label}:")
        for f in state.team:
            hp, atk, defn = power_adjusted_stats(
                f, args.ascension_stat_calc, args.constellation_effect
            )
            print(
                f"  {f.name:18s} | {f.element:7s} | {f.weapon:9s} | "
                f"{','.join(f.roles):30s} | HP={hp:.0f} ATK={atk:.0f} DEF={defn:.0f}"
            )

    print("\nTurn Log:")
    for line in log:
        print(" ", line)

    for label, score in (("Team A", score_a), ("Team B", score_b)):
        print(f"\nFinal Score ({label}):")
        for k, v in score.items():
            print(f"  {k:20s}: {v}")

    print(f"\nWinner: {winner}")


if __name__ == "__main__":
    main()