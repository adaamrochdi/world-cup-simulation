You are building a Monte Carlo simulation of the 2026 FIFA World Cup focused on Morocco's trajectory.

## Project context

- Stack: Python, numpy, pandas, scipy, joblib, pyarrow
- Main entry point: `python monte_carlo.py`
- All module names use camelCase (`matchEngine.py`, `groupStage.py`, etc.)

### Data
- `teams.json` — 48 teams across 12 groups (A→L), 4 teams each
- Each team has: `name`, `group`, `hostBonus` (bool), `eloRating`, `attackStrength`, `defenseStrength`
- `attackStrength` and `defenseStrength` are a 50/50 blend of ELO-derived strength and normalized historical weighted match stats
- Higher `defenseStrength` = stronger defense (fewer goals conceded)
- Some teams (Curacao) have negative raw stats — these are clamped to 0.01 in the engine

**`buildTeams.py`** — Regenerates `teams.json` from `114/results.csv`
- Run with: `python buildTeams.py`
- Filters matches from 2021-01-01 onward
- Opponent weighting: `(opponentELO / avgELO) ** 2` — goals against weak opponents count less
- Tournament tiers (combined with opponent weight as composite match weight):
  - **Tier 1 — 1.0**: FIFA World Cup, UEFA Euro, Copa América, AFCON, AFC Asian Cup, Oceania Nations Cup, UEFA Nations League, all qualifications, CONMEBOL–UEFA Cup of Champions
  - **Tier 2 — 0.6**: Gold Cup, Gold Cup qualification, CONCACAF Nations League, Arab Cup, Gulf Cup, EAFF/AFF/SAFF championships + qualifications
  - **Tier 3 — 0.2**: COSAFA Cup, CAFA Nations Cup, CONCACAF Series, FIFA Series, Intercontinental Cup
  - **Friendlies — 0.35**
  - **Default (catch-all) — 0.1**
- Teams with fewer than 5 matches fall back to pure ELO-derived strength
- Final strengths: `0.5 * eloToStrength(elo) + 0.5 * normalizedHistoricalStat`

### Already built

**`knockoutStage.py`** — Full knockout bracket simulator
- `simulateKnockoutStage(groupStageResult, teams)` → `{"winner": str, "moroccoPath": [...]}`
- Seeding: seeds 1–12 = group winners (A→L), 13–24 = runners-up, 25–32 = best 8 third-place teams
- R32 pairs: seed k vs seed (33-k), with greedy same-group conflict avoidance
- Subsequent rounds: consecutive bracket pairing (winner of match N vs winner of match N+1)
- 5-round knockout bracket: R32 (32→16) → R16 (16→8) → QF (8→4) → SF (4→2) → Final
- SF losers play 3rd-place match, labeled `"third_place"`; championship match labeled `"final"`
- `winner` = championship match winner; Morocco is `"champion"` iff `moroccoPath` has a `"final"` win AND `winner == "Morocco"`

**`matchEngine.py`** — Dixon-Coles bivariate Poisson simulator
- `simulateMatch(home, away, knockout=False)` → `(home_goals, away_goals)`
- `home` and `away` are full team dicts from `teams.json`
- λ = `home.attack / away.defense * homeAdvantage` (division, not multiplication — defense is strength not weakness)
- μ = `away.attack / home.defense`
- `homeAdvantage`: 1.25 for USA/Canada/Mexico (hostBonus=True), 1.0 for Morocco (neutral), 1.10 default
- tau Dixon-Coles correction applied for low-score outcomes (0,0), (1,0), (0,1), (1,1); rho = -0.13
- Knockout mode: draw → 30-min extra time (λ/3, μ/3) → if still draw → penalty shootout (Bernoulli p=0.5)
- Max goals per team: 8

**`groupStage.py`** — Full group stage simulator
- `simulateGroupStage(teams)` → dict
- Round-robin within each group (6 matches), tiebreakers: pts → GD → GF → H2H pts → H2H GD → random
- Qualifiers: top 2 per group (24) + best 8 third-place teams = 32 total
- Returns: `{"qualified": [32 names], "thirdPlace": [8 names], "groupResults": {group: [ranked names]}, "stats": {name: {pts,gd,gf}}}`

**`monte_carlo.py`** — Runner
- Load teams.json
- Run N=100,000 simulations using joblib Parallel (n_jobs=-1)
- Each simulation: run groupStage → knockoutStage → return Morocco's stage reached + full path
- Stage encoding: `"group_stage"`, `"round_of_32"`, `"round_of_16"`, `"quarter_final"`, `"semi_final"`, `"final"`, `"champion"`
- `stage` in the parquet = furthest championship round Morocco *played in* (not last win); `"third_place"` entries are excluded from stage and opponent columns
- Saves to `results.parquet` (requires pyarrow); prints stage probabilities, top 5 opponents per round, top 10 paths

**`analysis.py`** — Analysis + Plotly Sankey
- Load `results.parquet`, infer correct `stageReached` from opponent columns
- Print full stats table: stage probabilities (exact + cumulative), top 5 opponents per round
- Generate Plotly Sankey: nodes = stage+opponent combos, links filtered to > 0.5% frequency
- Export to `morocco_sankey.html`

---

## Conventions
- camelCase for variables/functions, PascalCase for classes
- No verbose comments
- Each module is independently runnable
- Seed numpy random for reproducibility where needed (but NOT inside parallel workers — each worker gets unique seed via joblib)
