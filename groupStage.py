import json
import random
from itertools import combinations
from matchEngine import simulateMatch


def _buildGroupMap(teams):
    groups = {}
    for team in teams:
        groups.setdefault(team["group"], []).append(team)
    return groups


def _simulateGroup(groupTeams):
    names = [t["name"] for t in groupTeams]
    stats = {name: {"pts": 0, "gd": 0, "gf": 0} for name in names}
    h2h = {
        name: {other: {"pts": 0, "gd": 0} for other in names if other != name}
        for name in names
    }

    for home, away in combinations(groupTeams, 2):
        hg, ag = simulateMatch(home, away)
        hn, an = home["name"], away["name"]

        stats[hn]["gf"] += hg
        stats[an]["gf"] += ag
        stats[hn]["gd"] += hg - ag
        stats[an]["gd"] += ag - hg
        h2h[hn][an]["gd"] += hg - ag
        h2h[an][hn]["gd"] += ag - hg

        if hg > ag:
            stats[hn]["pts"] += 3
            h2h[hn][an]["pts"] += 3
        elif ag > hg:
            stats[an]["pts"] += 3
            h2h[an][hn]["pts"] += 3
        else:
            stats[hn]["pts"] += 1
            stats[an]["pts"] += 1
            h2h[hn][an]["pts"] += 1
            h2h[an][hn]["pts"] += 1

    def primaryKey(name):
        s = stats[name]
        return (-s["pts"], -s["gd"], -s["gf"])

    sortedNames = sorted(names, key=primaryKey)

    ranked = []
    i = 0
    while i < len(sortedNames):
        j = i + 1
        while j < len(sortedNames) and primaryKey(sortedNames[j]) == primaryKey(sortedNames[i]):
            j += 1
        tied = sortedNames[i:j]
        if len(tied) > 1:
            def h2hKey(name, tied=tied):
                h2hPts = sum(h2h[name][o]["pts"] for o in tied if o != name)
                h2hGd = sum(h2h[name][o]["gd"] for o in tied if o != name)
                return (-h2hPts, -h2hGd, random.random())
            tied.sort(key=h2hKey)
        ranked.extend(tied)
        i = j

    return ranked, stats


def simulateGroupStage(teams):
    groups = _buildGroupMap(teams)
    groupResults = {}
    allStats = {}

    for groupLetter in sorted(groups):
        ranked, stats = _simulateGroup(groups[groupLetter])
        groupResults[groupLetter] = ranked
        allStats.update(stats)

    qualified = []
    thirdPlaceCandidates = []

    for g in sorted(groupResults):
        ranked = groupResults[g]
        qualified.append(ranked[0])
        qualified.append(ranked[1])
        thirdPlaceCandidates.append(ranked[2])

    def thirdKey(name):
        s = allStats[name]
        return (-s["pts"], -s["gd"], -s["gf"], random.random())

    thirdPlaceCandidates.sort(key=thirdKey)
    best8Third = thirdPlaceCandidates[:8]
    qualified.extend(best8Third)

    standings = {
        g: [
            {"rank": i + 1, "name": name, **allStats[name]}
            for i, name in enumerate(groupResults[g])
        ]
        for g in sorted(groupResults)
    }

    return {
        "qualified": qualified,
        "thirdPlace": best8Third,
        "groupResults": groupResults,
        "standings": standings,
        "stats": allStats,
    }


if __name__ == "__main__":
    with open("teams.json") as f:
        teams = json.load(f)
    result = simulateGroupStage(teams)
    print("Third-place qualifiers (8):", result["thirdPlace"])
    print("\nGroup standings:")
    for g, rows in result["standings"].items():
        print(f"  Group {g}:")
        for r in rows:
            print(f"    {r['rank']}. {r['name']:30s} pts={r['pts']} gd={r['gd']:+d} gf={r['gf']}")
