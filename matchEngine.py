import json
import math

import numpy as np

with open("modelConfig.json") as f:
    _config = json.load(f)
BASE_GOALS = _config["baseGoals"]
HOST_ADVANTAGE = _config["homeAdvantage"]
RHO = _config["rho"]

MAX_GOALS = 10
_GOALS = np.arange(MAX_GOALS + 1)
_FACTORIALS = np.array([math.factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)


def _poissonPmf(rate):
    return np.exp(-rate) * rate ** _GOALS / _FACTORIALS


def matchExpectedGoals(teamA, teamB):
    advA = HOST_ADVANTAGE if teamA.get("hostBonus") else 1.0
    advB = HOST_ADVANTAGE if teamB.get("hostBonus") else 1.0
    lam = max(0.02, BASE_GOALS * advA * teamA["attackStrength"] / teamB["defenseStrength"])
    mu = max(0.02, BASE_GOALS * advB * teamB["attackStrength"] / teamA["defenseStrength"])
    return lam, mu


def _sampleScore(lam, mu):
    probs = np.outer(_poissonPmf(lam), _poissonPmf(mu))
    probs[0, 0] *= 1.0 - lam * mu * RHO
    probs[0, 1] *= 1.0 + lam * RHO
    probs[1, 0] *= 1.0 + mu * RHO
    probs[1, 1] *= 1.0 - RHO

    cdf = probs.ravel().cumsum()
    idx = int(np.searchsorted(cdf, np.random.random() * cdf[-1]))
    return divmod(idx, MAX_GOALS + 1)


def _penaltyShootout():
    while True:
        pA, pB = np.random.binomial(5, 0.5, size=2)
        if pA != pB:
            return int(pA), int(pB)


def simulateMatch(teamA, teamB, knockout=False):
    lam, mu = matchExpectedGoals(teamA, teamB)
    ga, gb = _sampleScore(lam, mu)

    if not knockout or ga != gb:
        return int(ga), int(gb)

    # 30-min extra time: plain Poisson at a third of the 90-min rate
    ga += np.random.poisson(lam / 3.0)
    gb += np.random.poisson(mu / 3.0)
    if ga != gb:
        return int(ga), int(gb)

    pA, pB = _penaltyShootout()
    return (int(ga) + 1, int(gb)) if pA > pB else (int(ga), int(gb) + 1)
