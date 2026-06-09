import json, numpy as np
from collections import Counter
import matchEngine
from matchEngine import simulateMatch
from groupStage import simulateGroupStage

teams = json.load(open("teams.json"))
tmap = {t["name"]: t for t in teams}

print("=== BUG #1: Curacao match outcomes (current engine) ===")
ger, cur = tmap["Germany"], tmap["Curacao"]
scores = Counter()
np.random.seed(1)
for _ in range(5000):
    scores[simulateMatch(ger, cur)] += 1
print("Germany(home) vs Curacao top scorelines:", scores.most_common(5))
# avg goals
gg = ag_ = 0
np.random.seed(2)
for _ in range(5000):
    h,a = simulateMatch(ger,cur); gg+=h; ag_+=a
print(f"avg: Germany {gg/5000:.2f} - {ag_/5000:.2f} Curacao  (Germany attack 1.53 vs Curacao def {cur['defenseStrength']})")

print("\n=== BUG #1 impact: Group E qualification over 3000 group-stage sims ===")
qual = Counter(); place = Counter()
for _ in range(3000):
    r = simulateGroupStage(teams)
    ranked = r["groupResults"]["E"]
    place[ranked[0]] += 0  # ensure key
    if "Curacao" in r["qualified"]: qual["Curacao"] += 1
    for nm in ranked[:2]: qual[nm] += 1
for nm in ["Germany","Ecuador","Ivory Coast","Curacao"]:
    print(f"  {nm:12s} top-2 finish: {qual[nm]/3000*100:5.1f}%")

print("\n=== BUG #2: does _homeAdvantage neutralize Morocco? ===")
print("  Morocco homeAdv =", matchEngine._homeAdvantage("Morocco", False), "(spec says 1.0)")

print("\n=== BUG #3: home-order bias (same two teams, swap home/away) ===")
a,b = tmap["Morocco"], tmap["Scotland"]
def winrate(home, away, n=8000):
    w=0
    for _ in range(n):
        h,ag=simulateMatch(home,away); w += h>ag
    return w/n
np.random.seed(3); wr_home = winrate(a,b)
np.random.seed(3); wr_away = 1-winrate(b,a)  # morocco as away, its winrate
print(f"  Morocco win% as HOME vs Scotland: {wr_home*100:.1f}%")
print(f"  Morocco win% as AWAY vs Scotland: {wr_away*100:.1f}%  (gap = list-order bias)")
