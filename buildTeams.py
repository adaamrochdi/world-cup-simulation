"""
buildTeams.py

Dixon-Coles attack/defense estimation by penalized (MAP) Poisson MLE.

Design decisions (vs. the previous heuristic):
  * attackStrength / defenseStrength are MULTIPLICATIVE and centered on 1.0
    (log-params centered on 0). They are consumed by matchEngine as
        lam = baseGoals * attack_home / defense_away * hostAdv
    so a value > 1 = above-average attack / above-average defence.
  * Opponent strength is handled ENDOGENOUSLY: every team that appears in the
    match data gets its own attack/defence param, so scoring against weak sides
    is automatically discounted by the opponent's fitted defence. No more
    opponent-Elo reweighting hack and no confederation-qualifying inflation.
  * Elo enters as a GAUSSIAN SHRINKAGE PRIOR (ridge toward the Elo-implied
    log-strength), weighted against the data likelihood by sample size. Teams
    with many matches are data-driven; low-sample teams fall back toward Elo;
    teams with zero matches sit exactly on the prior. This subsumes the old
    matchCount < 5 -> usedElo branch.
  * Dixon-Coles time down-weighting (xi) plus tournament-tier weighting.
  * rho (low-score correction) is estimated in a second stage and written to
    modelConfig.json together with baseGoals (exp(intercept)) and the fitted
    home advantage (exp(home term)).

Outputs:
  teams.json        - engine schema, unchanged shape (name/group/hostBonus/
                      eloRating/attackStrength/defenseStrength)
  modelConfig.json  - {baseGoals, homeAdvantage, rho, ...} read by matchEngine
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

# --------------------------------------------------------------------------- #
# Tournament metadata                                                         #
# --------------------------------------------------------------------------- #
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOST_TEAMS = {"United States", "Canada", "Mexico"}

ELO_RATING = {
    "France": 2062, "Brazil": 1988, "England": 2020, "Spain": 2155, "Argentina": 2113,
    "Portugal": 1984, "Belgium": 1888, "Netherlands": 1944, "Germany": 1925,
    "Croatia": 1908, "Uruguay": 1892, "Colombia": 1977, "Mexico": 1875,
    "United States": 1733, "Morocco": 1824, "Senegal": 1867, "Japan": 1906, "South Korea": 1758,
    "Turkey": 1906, "Canada": 1793, "Ecuador": 1935, "Switzerland": 1894, "Australia": 1774,
    "Ghana": 1510, "Algeria": 1760, "Ivory Coast": 1695, "Iran": 1772,
    "Sweden": 1712, "Saudi Arabia": 1566, "Austria": 1830, "Czech Republic": 1740,
    "Norway": 1917, "Paraguay": 1833, "Scotland": 1770, "Egypt": 1699,
    "Bosnia and Herzegovina": 1591, "Qatar": 1423, "Iraq": 1618,
    "Uzbekistan": 1718, "DR Congo": 1661, "Panama": 1734, "Tunisia": 1633,
    "New Zealand": 1563, "South Africa": 1518, "Cape Verde": 1576,
    "Curacao": 1433, "Jordan": 1685, "Haiti": 1554,
}

TIER1_TOURNAMENTS = {
    "FIFA World Cup", "FIFA World Cup qualification",
    "UEFA Euro", "UEFA Euro qualification", "UEFA Nations League",
    "Copa América", "Copa América qualification",
    "African Cup of Nations", "African Cup of Nations qualification",
    "AFC Asian Cup", "AFC Asian Cup qualification",
    "Oceania Nations Cup", "Oceania Nations Cup qualification",
    "CONMEBOL–UEFA Cup of Champions",
}
TIER2_TOURNAMENTS = {
    "Gold Cup", "Gold Cup qualification", "CONCACAF Nations League",
    "Arab Cup", "Arab Cup qualification", "Gulf Cup",
    "EAFF Championship", "EAFF Championship qualification",
    "AFF Championship", "AFF Championship qualification", "SAFF Cup",
}
TIER3_TOURNAMENTS = {
    "COSAFA Cup", "CAFA Nations Cup", "CONCACAF Series",
    "FIFA Series", "Intercontinental Cup",
}


def tournamentWeight(tournament):
    if tournament in TIER1_TOURNAMENTS:
        return 1.0
    if tournament in TIER2_TOURNAMENTS:
        return 0.6
    if isinstance(tournament, str) and "Friendly" in tournament:
        return 0.35
    if tournament in TIER3_TOURNAMENTS:
        return 0.2
    return 0.1


# --------------------------------------------------------------------------- #
# Fit configuration                                                           #
# --------------------------------------------------------------------------- #
HISTORY_START = "2021-01-01"
HALF_LIFE_DAYS = 540.0      # Dixon-Coles recency down-weighting half-life
ELO_PRIOR_SCALE = 0.30      # log-strength spread implied by Elo z-scores
# overall = (attack+defence)/2 is the team's level -> Elo cross-calibrates it,
# so it gets a STRONG prior. balance = (attack-defence)/2 is the attack-vs-
# defence lean -> goals carry this honestly, so it gets a WEAK prior. This is
# what stops confederation-isolated goal volume (e.g. Japan vs weak AFC sides)
# from inflating a team's overall level above genuinely stronger UEFA/CONMEBOL
# sides, while still letting Japan come out attack-leaning.
KAPPA_OVERALL = 35.0        # strong: pins level to Elo
KAPPA_BALANCE = 6.0         # weak: lets goal data set the attack/defence split
LOG_CLIP = (-6.0, 4.0)      # numerical guard on log-rates
RHO_BOUNDS = (-0.25, 0.10)  # plausible low-score correlation range


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def loadMatches(csvPath):
    df = pd.read_csv(csvPath, parse_dates=["date"])
    df = df[df["date"] >= HISTORY_START].copy()
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    refDate = df["date"].max()
    xi = np.log(2.0) / HALF_LIFE_DAYS
    daysAgo = (refDate - df["date"]).dt.days.to_numpy(dtype=float)
    timeWeight = np.exp(-xi * daysAgo)
    tourWeight = df["tournament"].apply(tournamentWeight).to_numpy(dtype=float)
    df["matchWeight"] = tourWeight * timeWeight
    return df


def buildTeamIndex(df):
    teams = set(df["home_team"]) | set(df["away_team"])
    teams |= {t for groupTeams in GROUPS.values() for t in groupTeams}
    ordered = sorted(teams)
    return {team: idx for idx, team in enumerate(ordered)}


def buildPriors(teamIndex):
    """Elo z-score -> log-strength prior on a team's OVERALL level. Elo is
    cross-confederation calibrated (long history, all opponents), so it is the
    right anchor for level; the attack/defence split is left to the goal data
    via a separate weak prior on balance (centered on 0)."""
    elos = np.array(list(ELO_RATING.values()), dtype=float)
    meanElo, stdElo = elos.mean(), elos.std()

    n = len(teamIndex)
    overallPrior = np.zeros(n)
    for team, idx in teamIndex.items():
        elo = ELO_RATING.get(team)
        if elo is None:
            continue
        overallPrior[idx] = ELO_PRIOR_SCALE * (elo - meanElo) / stdElo
    return overallPrior


# --------------------------------------------------------------------------- #
# Penalized Poisson Dixon-Coles fit                                           #
#   parameters: overall = (att+def)/2  (Elo-pinned level)                      #
#               balance = (att-def)/2  (data-driven attack/defence lean)       #
#   att = overall + balance ,  def = overall - balance                        #
# --------------------------------------------------------------------------- #
def fitDixonColes(df, teamIndex, overallPrior):
    n = len(teamIndex)
    homeIdx = df["home_team"].map(teamIndex).to_numpy()
    awayIdx = df["away_team"].map(teamIndex).to_numpy()
    homeGoals = df["home_score"].to_numpy(dtype=float)
    awayGoals = df["away_score"].to_numpy(dtype=float)
    weight = df["matchWeight"].to_numpy(dtype=float)

    def unpack(params):
        overall = params[:n]
        balance = params[n:2 * n]
        mu0 = params[2 * n]
        home = params[2 * n + 1]
        att = overall + balance
        defn = overall - balance
        return overall, balance, att, defn, mu0, home

    def rates(params):
        overall, balance, att, defn, mu0, home = unpack(params)
        logLam = np.clip(mu0 + att[homeIdx] - defn[awayIdx] + home, *LOG_CLIP)
        logMu = np.clip(mu0 + att[awayIdx] - defn[homeIdx], *LOG_CLIP)
        return overall, balance, att, defn, logLam, logMu

    def negLogLik(params):
        overall, balance, att, defn, logLam, logMu = rates(params)
        lam, mu = np.exp(logLam), np.exp(logMu)
        nll = np.sum(weight * (lam - homeGoals * logLam))
        nll += np.sum(weight * (mu - awayGoals * logMu))
        nll += 0.5 * KAPPA_OVERALL * np.sum((overall - overallPrior) ** 2)
        nll += 0.5 * KAPPA_BALANCE * np.sum(balance ** 2)
        return nll

    def gradient(params):
        overall, balance, att, defn, logLam, logMu = rates(params)
        lam, mu = np.exp(logLam), np.exp(logMu)
        rHome = weight * (lam - homeGoals)   # d/d(logLam)
        rAway = weight * (mu - awayGoals)     # d/d(logMu)

        gAttData = np.bincount(homeIdx, rHome, n) + np.bincount(awayIdx, rAway, n)
        gDefData = -np.bincount(awayIdx, rHome, n) - np.bincount(homeIdx, rAway, n)
        # chain rule: overall = att+def lever, balance = att-def lever
        gOverall = (gAttData + gDefData) + KAPPA_OVERALL * (overall - overallPrior)
        gBalance = (gAttData - gDefData) + KAPPA_BALANCE * balance
        gMu0 = np.sum(rHome) + np.sum(rAway)
        gHome = np.sum(rHome)
        return np.concatenate([gOverall, gBalance, [gMu0, gHome]])

    x0 = np.concatenate([overallPrior, np.zeros(n), [np.log(1.4)], [np.log(1.15)]])
    res = minimize(
        negLogLik, x0, jac=gradient, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    _, _, att, defn, mu0, home = unpack(res.x)
    return att, defn, float(mu0), float(home), res


def estimateRho(df, teamIndex, att, defn, mu0, home):
    homeIdx = df["home_team"].map(teamIndex).to_numpy()
    awayIdx = df["away_team"].map(teamIndex).to_numpy()
    hg = df["home_score"].to_numpy()
    ag = df["away_score"].to_numpy()
    w = df["matchWeight"].to_numpy(dtype=float)

    lam = np.exp(np.clip(mu0 + att[homeIdx] - defn[awayIdx] + home, *LOG_CLIP))
    mu = np.exp(np.clip(mu0 + att[awayIdx] - defn[homeIdx], *LOG_CLIP))

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def negLogLikRho(rho):
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))

    res = minimize_scalar(negLogLikRho, bounds=RHO_BOUNDS, method="bounded")
    return float(res.x)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def buildTeamsJson(csvPath, teamsOut="teams.json", configOut="modelConfig.json"):
    df = loadMatches(csvPath)
    teamIndex = buildTeamIndex(df)
    overallPrior = buildPriors(teamIndex)

    att, defn, mu0, home, res = fitDixonColes(df, teamIndex, overallPrior)
    rho = estimateRho(df, teamIndex, att, defn, mu0, home)

    teamList = []
    for group, teams in GROUPS.items():
        for team in teams:
            idx = teamIndex[team]
            teamList.append({
                "name": team,
                "group": group,
                "hostBonus": team in HOST_TEAMS,
                "eloRating": ELO_RATING.get(team, 1700),
                "attackStrength": round(float(np.exp(att[idx])), 4),
                "defenseStrength": round(float(np.exp(defn[idx])), 4),
            })

    config = {
        "baseGoals": round(float(np.exp(mu0)), 5),
        "homeAdvantage": round(float(np.exp(home)), 5),
        "rho": round(rho, 5),
        "halfLifeDays": HALF_LIFE_DAYS,
        "kappaOverall": KAPPA_OVERALL,
        "kappaBalance": KAPPA_BALANCE,
        "eloPriorScale": ELO_PRIOR_SCALE,
    }

    with open(teamsOut, "w") as f:
        json.dump(teamList, f, indent=2)
    with open(configOut, "w") as f:
        json.dump(config, f, indent=2)

    _printSummary(teamList, config, att, defn, teamIndex, res)
    return teamList, config


def _printSummary(teamList, config, att, defn, teamIndex, res):
    qualifiedIdx = [teamIndex[t["name"]] for t in teamList]
    attQ = att[qualifiedIdx]
    defQ = defn[qualifiedIdx]
    elo = np.array([t["eloRating"] for t in teamList], dtype=float)

    print(f"converged={res.success}  iterations={res.nit}  nll={res.fun:.2f}")
    print(f"baseGoals={config['baseGoals']}  homeAdvantage={config['homeAdvantage']}  rho={config['rho']}")
    print(f"attack:  mean={np.exp(attQ).mean():.3f}  centered_log_mean={attQ.mean():+.3f}")
    print(f"defense: mean={np.exp(defQ).mean():.3f}  centered_log_mean={defQ.mean():+.3f}")
    print(f"corr(attack_log, elo)   = {np.corrcoef(attQ, elo)[0, 1]:+.3f}")
    print(f"corr(defense_log, elo)  = {np.corrcoef(defQ, elo)[0, 1]:+.3f}")
    print(f"corr(attack, defense)   = {np.corrcoef(attQ, defQ)[0, 1]:+.3f}  "
          f"(old 50/50 blend was ~1.0)")

    byName = {t["name"]: t for t in teamList}

    def attackLean(name):
        t = byName[name]
        return np.log(t["attackStrength"]) - np.log(t["defenseStrength"])

    print("\nattack-vs-defence lean  (log att - log def; >0 = attack-leaning):")
    for name in ("Japan", "France", "Morocco"):
        t = byName[name]
        print(f"  {name:<10} att={t['attackStrength']:.3f}  def={t['defenseStrength']:.3f}  "
              f"lean={attackLean(name):+.3f}  elo={t['eloRating']}")

    jpn, fra = byName["Japan"], byName["France"]
    verdict = "Japan < France in attack (level fixed)" if fra["attackStrength"] > jpn["attackStrength"] \
        else "Japan STILL >= France in attack -> raise KAPPA_OVERALL"
    print(f"  -> {verdict}; Japan more attack-leaning: {attackLean('Japan') > attackLean('France')}")

    ranked = sorted(teamList, key=lambda t: t["attackStrength"], reverse=True)
    print("\ntop attack:")
    for t in ranked[:6]:
        print(f"  {t['name']:<22} att={t['attackStrength']:.3f}  def={t['defenseStrength']:.3f}")
    morocco = next(t for t in teamList if t["name"] == "Morocco")
    print(f"\nMorocco: att={morocco['attackStrength']:.3f}  def={morocco['defenseStrength']:.3f}")


if __name__ == "__main__":
    buildTeamsJson("114/results.csv", "teams.json", "modelConfig.json")