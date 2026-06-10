Monte Carlo simulation of the 2026 FIFA World Cup focused on Morocco's trajectory.

## Project context

- Stack: Python, numpy, pandas, scipy, joblib, pyarrow
- Main entry point: `python monteCarlo.py`
- All module names use camelCase (`matchEngine.py`, `groupStage.py`, etc.)

The model is a **Dixon-Coles** system end to end: `buildTeams.py` fits the
parameters by penalized Poisson maximum likelihood, and `matchEngine.py` consumes
them. The two are consistent by construction: `λ = exp(μ0 + att_home − def_away + home)`
is the same thing as `baseGoals · attack_home / defense_away · homeAdv`.

### Data
- `114/results.csv` — historical international results (input to the fit). Dataset
  spells some names differently (e.g. "Curaçao"); `NAME_FIXES` in `buildTeams.py`
  maps them to the names used in `GROUPS`.
- `teams.json` — 48 teams across 12 groups (A→L), 4 teams each.
  Each team has: `name`, `group`, `hostBonus` (bool), `eloRating`, `attackStrength`, `defenseStrength`.
  `attackStrength` / `defenseStrength` are **multiplicative, centered on 1.0**: a value
  > 1.0 means above-average attack / above-average defense. Higher `defenseStrength`
  = stronger defense (fewer goals conceded, since the engine *divides* by it).
- `modelConfig.json` — the three fit-level parameters the engine reads:
  `baseGoals`, `homeAdvantage`, `rho`.

### Modules

**`buildTeams.py`** — Regenerates `teams.json` + `modelConfig.json` from `114/results.csv`
- Run with: `python buildTeams.py` (~5 s)
- Penalized (MAP) Poisson Dixon-Coles fit via `scipy.optimize.minimize` (L-BFGS-B,
  numerical gradient — no hand-written `jac`)
- `negLogLik` is a flat top-level function; the match arrays are passed via `args=`
- Reparametrization: `overall = (att+def)/2`, `balance = (att-def)/2`, with `att = overall+balance`, `def = overall−balance`
  - `overall` (team level) gets a **strong** ELO shrinkage prior (`KAPPA_OVERALL = 35`): ELO is cross-confederation calibrated, so it anchors level
  - `balance` (attack-vs-defense lean) gets a **weak** prior centered on 0 (`KAPPA_BALANCE = 6`): the goal data sets the split
  - This stops confederation-isolated goal volume from inflating a team's overall level while still letting it come out attack- or defense-leaning
- ELO enters only as the prior mean (`ELO_PRIOR_SCALE = 0.30` log-strength spread). Teams with many matches are data-driven; low-sample teams fall toward ELO; zero-match teams sit exactly on the prior
- Opponent strength is handled **endogenously** — every team in the data gets its own attack/defense param, so scoring against weak sides is automatically discounted by the opponent's fitted defense
- Recency: Dixon-Coles time down-weighting, `HALF_LIFE_DAYS = 540` (`xi = ln2 / halfLife`)
- Tournament-tier match weighting (`tournamentWeight`), combined with recency as `matchWeight = tourWeight · timeWeight`:
  - **Tier 1 — 1.0**: FIFA World Cup, UEFA Euro, Copa América, AFCON, AFC Asian Cup, Oceania Nations Cup, UEFA Nations League, all qualifications, CONMEBOL–UEFA Cup of Champions
  - **Tier 2 — 0.6**: Gold Cup (+qual), CONCACAF Nations League, Arab Cup (+qual), Gulf Cup, EAFF/AFF/SAFF championships (+qual)
  - **Tier 3 — 0.2**: COSAFA Cup, CAFA Nations Cup, CONCACAF Series, FIFA Series, Intercontinental Cup
  - **Friendlies — 0.35**
  - **Default (catch-all) — 0.1**
- `rho` (low-score correction) is estimated in a **second stage** via `minimize_scalar` (bounded, `RHO_BOUNDS = (-0.25, 0.10)`) given the fitted attack/defense/home
- Matches filtered from `HISTORY_START = "2021-01-01"` onward
- `GROUPS`, `ELO_RATING`, `HOST_TEAMS`, `NAME_FIXES` are hardcoded constants in the file
- Outputs: `teams.json` (engine schema) and `modelConfig.json` (`baseGoals = exp(μ0)`, `homeAdvantage = exp(home)`, `rho`)

**`matchEngine.py`** — Dixon-Coles bivariate Poisson simulator
- `simulateMatch(teamA, teamB, knockout=False)` → `(goalsA, goalsB)`; team args are full dicts from `teams.json`
- `λ_A = baseGoals · advA · attackA / defenseB`, `λ_B = baseGoals · advB · attackB / defenseA`, floored at 0.02
- `advX = homeAdvantage` only if team X is an actual host (`hostBonus=True`: USA/Canada/Mexico), else `1.0`. There is **no positional home advantage** — every non-host WC match is on neutral ground, so the result is order-independent (Morocco is always neutral)
- `baseGoals`, `homeAdvantage`, `rho` read from `modelConfig.json` (required — `buildTeams.py` generates it)
- Poisson pmf computed inline (`exp(−λ)·λ^k / k!`, `_poissonPmf`) — no scipy dependency
- `_sampleScore` builds the `(MAX_GOALS+1)²` joint score matrix per match, applies the
  Dixon-Coles `tau` correction to (0,0), (0,1), (1,0), (1,1), and samples by inverse CDF
  (`MAX_GOALS = 10`). No caching — simple recompute per match, ~6 core-min for 100k tournaments
- Knockout mode: draw → 30-min extra time (plain `np.random.poisson` at `λ/3`, `μ/3`, no DC correction — it's calibrated for 90 min) → if still draw → penalty shootout (`binomial(5, 0.5)` best-of-5 then sudden death, team-agnostic)
- `matchExpectedGoals(teamA, teamB)` exposes `(λ_A, λ_B)` for validation

**`groupStage.py`** — Full group stage simulator
- `simulateGroupStage(teams)` → dict
- Round-robin within each group (6 matches); `_rankGroup` sorts by pts → GD → GF, then
  breaks exact ties (via `itertools.groupby`) by H2H pts → H2H GD → random
- Qualifiers: top 2 per group (24) + best 8 third-place teams = 32 total
- Returns: `{"qualified": [32], "thirdPlace": [8], "groupResults": {group: [ranked names]}, "standings": {group: [rows]}, "stats": {name: {pts,gd,gf}}}`

**`knockoutStage.py`** — Full knockout bracket simulator
- `simulateKnockoutStage(groupStageResult, teams)` → `{"winner": str, "moroccoPath": [...]}`
- Seeding: seeds 1–12 = group winners (A→L), 13–24 = runners-up, 25–32 = best 8 third-place teams
- R32 pairs: seed k vs seed (33−k), with greedy same-group conflict avoidance
- Subsequent rounds: consecutive bracket pairing via `_nextRoundPairs` (winner of match N vs winner of match N+1)
- `_playRound` returns `(winners, losers)`; SF losers play `"third_place"`, championship match labeled `"final"`
- 5 rounds: R32 (32→16) → R16 → QF → SF → Final
- `winner` = championship match winner. Each `moroccoPath` entry: `{round, opponent, result, scoreHome, scoreAway}`

**`monteCarlo.py`** — Runner
- Loads `teams.json`, runs `N_SIMULATIONS = 100,000` via `joblib.Parallel(n_jobs=-1)`
- Each sim: groupStage → (if Morocco qualified) knockoutStage → Morocco's `stage` + full `path`
- Stage encoding: `group_stage`, `round_of_32`, `round_of_16`, `quarter_final`, `semi_final`, `final`, `champion`
- `stage` = furthest championship round Morocco *played in*; `"third_place"` entries excluded from stage and opponent columns. `champion` iff Morocco reached `final` AND `winner == "Morocco"`
- Saves `results.parquet` (requires pyarrow); builds one summary text (`_buildSummary`) that is both printed and written to `morocco_summary.txt`: stage probabilities, top opponents per round, top winners, top paths

---

## Conventions
- camelCase for variables/functions, PascalCase for classes
- No verbose comments
- Each module is independently runnable
- Seed numpy random for reproducibility where needed (but NOT inside parallel workers — each worker gets a unique seed via joblib)
