# 2026 FIFA World Cup Simulation

Monte Carlo simulation of the 2026 FIFA World Cup. Runs 100,000 tournaments to estimate each team's probability of winning, powered by a Dixon-Coles statistical model fitted on historical international results.

## How it works

### Model

The engine uses a **Dixon-Coles bivariate Poisson** model — the standard approach for football score prediction:

- Each team has an `attackStrength` and `defenseStrength` fitted from historical match data (2021–present).
- Expected goals for a match: `λ = baseGoals × attack_home / defense_away`, with a host-nation advantage multiplier for USA, Canada, and Mexico.
- A low-score correction (`rho`) adjusts the joint probability of 0–0, 0–1, 1–0, and 1–1 scorelines, which plain Poisson underestimates.
- Knockout matches that end level go to 30-min extra time, then penalties if still tied.

### Parameter fitting (`buildTeams.py`)

Parameters are estimated by penalized maximum likelihood (L-BFGS-B) on `114/results.csv`:

- **Recency weighting**: matches decay with a 540-day half-life.
- **Tournament weighting**: World Cup / major continental tournaments (1.0) down to friendlies (0.35) and minor cups (0.1).
- **ELO prior**: each team's overall strength is anchored to its ELO rating, preventing data-sparse confederations from producing unrealistic estimates.
- `rho` is estimated in a second stage via `minimize_scalar` given the fitted attack/defense parameters.

Output: `teams.json` (48 teams with fitted strengths) and `modelConfig.json` (global parameters).

## Project structure

```
buildTeams.py       Fits the Dixon-Coles model, writes teams.json + modelConfig.json
matchEngine.py      Simulates a single match (group or knockout)
groupStage.py       Runs the full group stage for all 12 groups
knockoutStage.py    Runs the knockout bracket (R32 → Final)
monteCarlo.py       Orchestrates 100,000 parallel simulations, writes results.txt
teams.json          48 teams with attack/defense strengths (generated)
modelConfig.json    Fitted global parameters: baseGoals, homeAdvantage, rho (generated)
114/results.csv     Historical international results used for fitting
```

## Setup

```bash
pip install numpy pandas scipy joblib pyarrow
```

## Usage

**Run the simulation** (uses the pre-fitted `teams.json`):

```bash
python monteCarlo.py
```

Writes `results.txt` with each team's win probability, e.g.:

```
Spain: 14.32%
France: 12.81%
Brazil: 11.04%
...
```

**Re-fit the model** from raw results data:

```bash
python buildTeams.py
```

Regenerates `teams.json` and `modelConfig.json` from `114/results.csv`. Takes ~5 seconds.

## Groups

| Group | Teams |
|-------|-------|
| A | Mexico, South Africa, South Korea, Czech Republic |
| B | Canada, Bosnia and Herzegovina, Qatar, Switzerland |
| C | Brazil, Morocco, Haiti, Scotland |
| D | United States, Paraguay, Australia, Turkey |
| E | Germany, Curaçao, Ivory Coast, Ecuador |
| F | Netherlands, Japan, Sweden, Tunisia |
| G | Belgium, Egypt, Iran, New Zealand |
| H | Spain, Cape Verde, Saudi Arabia, Uruguay |
| I | France, Senegal, Iraq, Norway |
| J | Argentina, Algeria, Austria, Jordan |
| K | Portugal, DR Congo, Uzbekistan, Colombia |
| L | England, Croatia, Ghana, Panama |

Host nations (USA, Canada, Mexico) receive a home-advantage multiplier in all their matches.

## Qualification format

- Top 2 from each of the 12 groups qualify automatically (24 teams).
- Best 8 third-place teams across all groups also advance.
- Total: 32 teams enter the knockout stage.

Knockout seeding: group winners (seeds 1–12) vs. runners-up (13–24) vs. best third-place (25–32). Seed *k* faces seed *33−k* in the Round of 32, with conflict avoidance for same-group opponents.
