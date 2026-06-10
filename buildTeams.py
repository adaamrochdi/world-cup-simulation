import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

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

# results.csv spellings that differ from the names used in GROUPS
NAME_FIXES = {"Curaçao": "Curacao"}

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


HISTORY_START = "2021-01-01"
HALF_LIFE_DAYS = 540.0
ELO_PRIOR_SCALE = 0.30
KAPPA_OVERALL = 35.0
KAPPA_BALANCE = 6.0
LOG_CLIP = (-6.0, 4.0)
RHO_BOUNDS = (-0.25, 0.10)


def loadMatches(csvPath):
    df = pd.read_csv(csvPath, parse_dates=["date"])
    df = df[df["date"] >= HISTORY_START].copy()
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_team"] = df["home_team"].replace(NAME_FIXES)
    df["away_team"] = df["away_team"].replace(NAME_FIXES)
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


def negLogLik(params, n, homeIdx, awayIdx, homeGoals, awayGoals, weight, overallPrior):
    overall = params[:n]
    balance = params[n:2 * n]
    mu0, home = params[2 * n], params[2 * n + 1]
    att = overall + balance
    defn = overall - balance

    logLam = np.clip(mu0 + att[homeIdx] - defn[awayIdx] + home, *LOG_CLIP)
    logMu = np.clip(mu0 + att[awayIdx] - defn[homeIdx], *LOG_CLIP)

    nll = np.sum(weight * (np.exp(logLam) - homeGoals * logLam))
    nll += np.sum(weight * (np.exp(logMu) - awayGoals * logMu))
    nll += 0.5 * KAPPA_OVERALL * np.sum((overall - overallPrior) ** 2)
    nll += 0.5 * KAPPA_BALANCE * np.sum(balance ** 2)
    return nll


def fitDixonColes(df, teamIndex, overallPrior):
    n = len(teamIndex)
    homeIdx = df["home_team"].map(teamIndex).to_numpy()
    awayIdx = df["away_team"].map(teamIndex).to_numpy()
    homeGoals = df["home_score"].to_numpy(dtype=float)
    awayGoals = df["away_score"].to_numpy(dtype=float)
    weight = df["matchWeight"].to_numpy(dtype=float)

    x0 = np.concatenate([overallPrior, np.zeros(n), [np.log(1.4)], [np.log(1.15)]])
    res = minimize(
        negLogLik, x0,
        args=(n, homeIdx, awayIdx, homeGoals, awayGoals, weight, overallPrior),
        method="L-BFGS-B", options={"maxiter": 500},
    )

    overall, balance = res.x[:n], res.x[n:2 * n]
    att = overall + balance
    defn = overall - balance
    mu0, home = res.x[2 * n], res.x[2 * n + 1]
    return att, defn, float(mu0), float(home), res


def negLogLikRho(rho, lam, mu, hg, ag, w):
    tau = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))


def estimateRho(df, teamIndex, att, defn, mu0, home):
    homeIdx = df["home_team"].map(teamIndex).to_numpy()
    awayIdx = df["away_team"].map(teamIndex).to_numpy()
    hg = df["home_score"].to_numpy()
    ag = df["away_score"].to_numpy()
    w = df["matchWeight"].to_numpy(dtype=float)

    lam = np.exp(np.clip(mu0 + att[homeIdx] - defn[awayIdx] + home, *LOG_CLIP))
    mu = np.exp(np.clip(mu0 + att[awayIdx] - defn[homeIdx], *LOG_CLIP))

    res = minimize_scalar(negLogLikRho, args=(lam, mu, hg, ag, w), bounds=RHO_BOUNDS, method="bounded")
    return float(res.x)


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
    }

    with open(teamsOut, "w") as f:
        json.dump(teamList, f, indent=2)
    with open(configOut, "w") as f:
        json.dump(config, f, indent=2)

    return teamList, config


if __name__ == "__main__":
    buildTeamsJson("114/results.csv", "teams.json", "modelConfig.json")
