# Genshin-Impact-AI-Agent
Based on case study to build a combat simulator on Genshin Impact characters. Note: Does not simulte actual Genshin combat, and uses a character vs character scenario with preselected objectives and random field environments. 

# Prerequisites

Make sure Python 3.8+ and the required packages are installed: 
```bash pip install pandas numpy 
```
Ensure 'genshin_impact_-_Adjusted.csv' and 'genshin_impact.csv' are in the same folder as the script. 

## Assumptions made:
Meta Information:
•	release_date
•	days_since_release
•	months_since_release
•	num_banners
•	is_standard_banner
•	is_archon
### None of these were used for analysis as these are ownership details and character banner details, both of which aren't needed for a combat scenario.

Popularity / Ownership Signals
•	pulled_count
•	duplicate_rate
•	avg_copies_per_player
•	c6_rate 
### Avg copies per player was used to simulate the average constellation owned by the character,and adjust power level according to average constellation. Others details weren't used as those are ownership details.

### Arena Mechanics: Increase or Decrease coming from the arena were quantified using assumed values as data doesn't have quantification for the same.

### Team synergy requirements weren't implemented as the case lacked data for the same.

## Assumptions based on Ascension stats:

Genshin's reaction system wasn't used, instead having EM gave a passive boost to characters when the teams had characters of different elements.

Characters with ER ascension or Electro element are reliant on burst, and thus affected by energy field conditions.

Characters with Atk, Def or HP ascension gain 28.8% more of that stat, added to final_atk, final_def or final_HP.

Characters with Healing bonus ascension gain higher score on survivaability based scenarios.

# How to Run

## Option 1: Interactive Terminal Mode (Recommended)


Launch the program without arguments to trigger interactive configuration prompts:

```bash
python genshin_ai_agent_v3_2.py

```

You will be prompted to choose:

* Team size (1v1 to 8v8)
* Battle objective (A–F or Random)
* Field conditions (Randomized or Static)
* Drafting strategy (Optimal AI selection, Fully Random, or Custom Characters)

---
## Option 2: Command-Line Interface (CLI)

Run directly with arguments for automated or custom execution:

```bash
# Example 1: Standard 4v4 battle with randomized arena and objective
python genshin_ai_agent_v3_2.py --team-size 4 --randomize-arena

# Example 2: Custom character team draft for Objective A (Elimination)
python genshin_ai_agent_v3_2.py --team-size 3 --objective A --chars-a "Raiden Shogun" Bennett Xiangling --chars-b Zhongli Xingqiu Nahida

# Example 3: Fully randomized teams and arena conditions
python genshin_ai_agent_v3_2.py --team-size 4 --randomize-teams --randomize-arena

```

---

## Features

* **Dynamic Team Drafting**: Optimal heuristic selection based on objective requirements, role complementarities, and popularity signals.
* **Interactive & CLI Setup**: Choose team sizes, picks, objectives, and field toggles via interactive prompts or CLI flags.
* **Turn-Based Combat Logic**: Decision-making based on team health ratios, energy levels, skill/burst cooldowns, and objective priorities.
* **Procedural Arenas**: Dynamic hazard events during battle, weather, terrain bonuses, and field condition adjustments.
* **Holistic & Calculation Splitting**: Combat and selection calculations strictly use adjusted dataset values, while descriptive profiles use the original dataset.

---

## Parameters Affecting Final Results

The final combat score and battle outcome are directly influenced by the following key parameters:

## 1. Simulation & Toggle Settings

* **`ascension_stat_calc` (0 or 1)**:
* `1`: Uses `final_hp`, `final_atk`, `final_def` along with ascension flags (`dmg_bonus`, `crit_ascension`, `Heals`, `em`).
* `0`: Uses unadjusted base Level 90 stats (`lvl_90_HP`, `lvl_90_ATK`, `lvl_90_DEF`).


* **`constellation_effect` (0 or 1)**:
* `1`: Applies a normalized stat multiplier based on `avg_copies_per_player` and `constellation_1_pull` .. `constellation_7_pull` to combat HP/ATK/DEF.
* `0`: Ignores constellation stat scaling in combat.


* **`turns`**: Total maximum turns permitted before evaluating victory conditions (default: 20).
* **`seed`**: Integer random seed controlling battlefield RNG, damage variance (±15%), and mid-combat hazard events.

---

## 2. Team & Composition Parameters

* **`team_size`**: Number of fighters per side (1–8).
* **`custom_chars_a` / `custom_chars_b**`: Pre-selected character names forcing specific picks into Team A/B.
* **`randomize_teams`**: Bypasses AI optimization logic to pick completely randomized character rosters.
* **Role Synergy**: Complementary role pairs (e.g., `Support` + `DPS`, `Survivability` + `DPS`, `Off-Field` + `On-Field`) grant score multipliers and selection bonuses.

---

## 3. Battlefield & Environmental Parameters

* **`objective`**: Defines character selection heuristics and combat decision priorities:
* **A (Elimination)**: Prioritizes ATK, single On-Field carry + Off-Field support structure.
* **B (Control) / C (Relic)**: Elimination rules + Survivability role bonuses.
* **D (Orbs) / F (Escort)**: Balanced raw HP / ATK / DEF scoring.
* **E (Survival)**: Focuses exclusively on HP / DEF / Healing flags (ATK ignored).


* **`weather`**: Elements/weapons receive multipliers (e.g., *Thunderstorm* boosts Electro +30%, Hydro +10%).
* **`terrain`**: Grants weapon/elemental bonuses (e.g., *Floating Islands* boosts Bow +15%, Claymore -10%).
* **`hazard`**: Environmental field effects (e.g., *Lava Eruption* causes HP drain, *Energy Blackout* reduces energy, *Shield Nullification* suppresses shields).
* **`elemental_field`**: Matching element characters receive a +10% contribution bonus.
* **`visibility` & `energy_flux**`: Affect accuracy/range weapons and burst energy decay rates.




```

```
