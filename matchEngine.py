"""
matchEngine.py

Dixon-Coles match simulator consuming attack/defence centered on 1.0.

Goal model (neutral-venue tournament):
    lam_A = baseGoals * advA * attack_A / defense_B
    lam_B = baseGoals * advB * attack_B / defense_A
where advX = homeAdvantage if team X is an actual host (USA/Canada/Mexico),
else 1.0. There is NO positional "home" advantage: at a World Cup every
non-host match is on neutral ground, so the result must not depend on which
team is passed first. baseGoals (exp of the fitted intercept) sets the scoring
level; without it a ratio-only model collapses average matchups to ~1.0 goals.

baseGoals / homeAdvantage / rho are read from modelConfig.json (written by
buildTeams.py). Fallback defaults are used if the file is absent.

Extra time uses PLAIN Poisson at 1/3 of the regulation rate: the Dixon-Coles
low-score correction is calibrated for 90 minutes and would over-inflate 0-0
over a 30-minute window, biasing the extra-time vs. penalties split.
"""

import json
from functools import lru_cache

import numpy as np
from scipy.stats import poisson

# --------------------------------------------------------------------------- #
# Model configuration                                                          #
# --------------------------------------------------------------------------- #
_DEFAULT_CONFIG = {"baseGoals": 1.40, "homeAdvantage": 1.15, "rho": -0.13}


def _loadConfig(path="modelConfig.json"):
    try:
        with open(path) as f:
            cfg = json.load(f)
        return (
            float(cfg.get("baseGoals", _DEFAULT_CONFIG["baseGoals"])),
            float(cfg.get("homeAdvantage", _DEFAULT_CONFIG["homeAdvantage"])),
            float(cfg.get("rho", _DEFAULT_CONFIG["rho"])),
        )
    except (FileNotFoundError, ValueError, KeyError):
        d = _DEFAULT_CONFIG
        return d["baseGoals"], d["homeAdvantage"], d["rho"]


BASE_GOALS, HOST_ADVANTAGE, RHO = _loadConfig()

MAX_GOALS = 10
_X = np.arange(MAX_GOALS + 1)
_KEY_DECIMALS = 3   # quantization for the score-matrix cache


# --------------------------------------------------------------------------- #
# Scoreline distribution                                                       #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8192)
def _scoreCdfCached(lamQ, muQ, correct):
    """Return the cumulative distribution over the flattened (MAX_GOALS+1)^2
    scoreline grid. Inputs are pre-quantized so the cache actually hits: with
    fixed team params there are only ~48^2 distinct (lam, mu) pairs."""
    px = poisson.pmf(_X, lamQ)
    py = poisson.pmf(_X, muQ)
    matrix = np.outer(px, py)
    if correct:
        matrix[0, 0] *= 1.0 - lamQ * muQ * RHO
        matrix[0, 1] *= 1.0 + lamQ * RHO
        matrix[1, 0] *= 1.0 + muQ * RHO
        matrix[1, 1] *= 1.0 - RHO
    flat = matrix.ravel()
    return np.cumsum(flat / flat.sum())


def _scoreCdf(lam, mu, correct=True):
    return _scoreCdfCached(round(lam, _KEY_DECIMALS), round(mu, _KEY_DECIMALS), correct)


def _sampleScore(cdf):
    idx = int(np.searchsorted(cdf, np.random.random()))
    return divmod(idx, MAX_GOALS + 1)


def _penaltyShootout():
    """Team-agnostic 50/50 shootout (best-of-5 then sudden death). Shootouts
    are empirically close to random; swap in an attack-aware conversion model
    here if you want shooter quality to matter."""
    while True:
        pH = np.random.binomial(5, 0.5)
        pA = np.random.binomial(5, 0.5)
        if pH != pA:
            return pH, pA


def matchExpectedGoals(teamA, teamB):
    """Expected (lam_A, lam_B) for validation/debugging."""
    advA = HOST_ADVANTAGE if teamA.get("hostBonus") else 1.0
    advB = HOST_ADVANTAGE if teamB.get("hostBonus") else 1.0
    lam = max(0.02, BASE_GOALS * advA * teamA["attackStrength"] / teamB["defenseStrength"])
    mu = max(0.02, BASE_GOALS * advB * teamB["attackStrength"] / teamA["defenseStrength"])
    return lam, mu


# --------------------------------------------------------------------------- #
# Match simulation                                                             #
# --------------------------------------------------------------------------- #
def simulateMatch(teamA, teamB, knockout=False):
    """Simulate one match. Order-independent except for genuine host advantage.
    Returns (goalsA, goalsB), guaranteed non-draw when knockout=True."""
    lam, mu = matchExpectedGoals(teamA, teamB)

    ga, gb = _sampleScore(_scoreCdf(lam, mu, correct=True))

    if not knockout or ga != gb:
        return int(ga), int(gb)

    # Extra time: plain Poisson, 1/3 of regulation rate, no low-score correction.
    eta, etb = _sampleScore(_scoreCdf(lam / 3.0, mu / 3.0, correct=False))
    ga += eta
    gb += etb
    if ga != gb:
        return int(ga), int(gb)

    pA, pB = _penaltyShootout()
    if pA > pB:
        return int(ga) + 1, int(gb)
    return int(ga), int(gb) + 1