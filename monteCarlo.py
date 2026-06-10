import json
from collections import defaultdict, Counter
import pandas as pd
from joblib import Parallel, delayed
from groupStage import simulateGroupStage
from knockoutStage import simulateKnockoutStage

N_SIMULATIONS = 100_000
STAGE_ORDER = ["group_stage", "round_of_32", "round_of_16", "quarter_final", "semi_final", "final", "champion"]
KNOCKOUT_ROUNDS = STAGE_ORDER[1:-1]
ROUND_LABELS = {
    "round_of_32": "R32", "round_of_16": "R16", "quarter_final": "QF",
    "semi_final": "SF", "final": "Final", "third_place": "3rd Place",
}
ROUND_ORDER = ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final", "third_place"]


def _runSimulation(teams):
    gsResult = simulateGroupStage(teams)
    ksResult = simulateKnockoutStage(gsResult, teams)
    moroccoPath = ksResult["moroccoPath"]

    stage = "group_stage"
    if "Morocco" in gsResult["qualified"]:
        for entry in moroccoPath:
            if entry["round"] == "third_place":
                continue
            stage = entry["round"]
            if entry["result"] == "loss":
                break
        if stage == "final" and ksResult["winner"] == "Morocco":
            stage = "champion"

    return {"stage": stage, "path": moroccoPath, "winner": ksResult["winner"], "allMatches": ksResult["allMatches"]}


def _extractOpponents(path):
    return {
        entry["round"]: entry["opponent"]
        for entry in path
        if entry["round"] != "third_place"
    }


def _buildSummary(df, teams, total):
    lines = []
    lines.append("2026 FIFA WORLD CUP — MOROCCO MONTE CARLO SIMULATION RESULTS")
    lines.append(f"Simulations: {total:,}")
    lines.append("")

    moroccoTeam = next(t for t in teams if t["name"] == "Morocco")
    lines.append(f"Morocco ELO: {moroccoTeam['eloRating']}  Attack: {moroccoTeam['attackStrength']}  Defense: {moroccoTeam['defenseStrength']}  Group: {moroccoTeam['group']}")
    lines.append("")

    groupH = [t for t in teams if t["group"] == moroccoTeam["group"]]
    lines.append(f"Morocco's group ({moroccoTeam['group']}):")
    for t in sorted(groupH, key=lambda x: -x["eloRating"]):
        lines.append(f"  {t['name']:30s}  ELO {t['eloRating']}  Atk {t['attackStrength']}  Def {t['defenseStrength']}")
    lines.append("")

    lines.append("STAGE PROBABILITIES (cumulative = reach that round or further)")
    lines.append(f"{'Stage':<22} {'Exact':>8} {'Cumulative':>12}")
    lines.append("-" * 44)
    stageCounts = {s: (df["stage"] == s).sum() for s in STAGE_ORDER}
    cumulative = 0
    cumByStage = {}
    for s in reversed(STAGE_ORDER):
        cumulative += stageCounts.get(s, 0)
        cumByStage[s] = cumulative
    for s in STAGE_ORDER:
        exact = stageCounts.get(s, 0) / total * 100
        cum = cumByStage[s] / total * 100
        label = s.replace("_", " ").title()
        lines.append(f"  {label:<20} {exact:>7.2f}%  {cum:>10.2f}%")
    lines.append("")

    lines.append("TOP OPPONENTS PER KNOCKOUT ROUND (% of all simulations)")
    for rnd in KNOCKOUT_ROUNDS:
        if rnd not in df.columns:
            continue
        col = df[rnd].dropna()
        if col.empty:
            continue
        label = rnd.replace("_", " ").title()
        lines.append(f"\n  {label}:")
        for opp, cnt in col.value_counts().head(8).items():
            lines.append(f"    {opp:<30} {cnt / total * 100:6.2f}%")
    lines.append("")

    lines.append("TOP 10 TOURNAMENT WINNERS")
    if "winner" in df.columns:
        for team, cnt in df["winner"].dropna().value_counts().head(10).items():
            lines.append(f"  {team:<30} {cnt / total * 100:.2f}%")
    lines.append("")

    lines.append("TOP 10 MOST COMMON MOROCCO PATHS")
    for path, cnt in df["path"].value_counts().head(10).items():
        lines.append(f"  {cnt / total * 100:.2f}%  {path}")
    lines.append("")

    lines.append("ALL TEAMS BY GROUP (for context)")
    groups = {}
    for t in teams:
        groups.setdefault(t["group"], []).append(t)
    for g in sorted(groups):
        lines.append(f"  Group {g}: " + ", ".join(f"{t['name']} ({t['eloRating']})" for t in sorted(groups[g], key=lambda x: -x["eloRating"])))

    return "\n".join(lines)


def exportBracket(bracketData, outPath="bracket_results.md"):
    h2hWins = Counter()
    for slots in bracketData.values():
        for slot in slots.values():
            for (winner, loser), count in slot.items():
                h2hWins[(winner, loser)] += count

    def favored(a, b):
        wa, wb = h2hWins[(a, b)], h2hWins[(b, a)]
        if wa + wb == 0:
            return a, b, 0.5
        if wb > wa:
            a, b, wa, wb = b, a, wb, wa
        return a, b, wa / (wa + wb)

    used = set()
    pairs = []
    r32 = bracketData["round_of_32"]
    for slotIdx in sorted(r32):
        pairCounts = Counter()
        for (winner, loser), count in r32[slotIdx].items():
            pairCounts[frozenset({winner, loser})] += count
        fresh = [p for p in pairCounts if not (set(p) & used)]
        best = max(fresh or pairCounts, key=pairCounts.__getitem__)
        used.update(best)
        pairs.append(tuple(sorted(best)))

    lines = ["# 2026 FIFA World Cup — Bracket Results", ""]
    sfWinners, sfLosers, champion = [], [], None
    for roundName in ROUND_ORDER:
        if roundName == "final":
            pairs = [tuple(sfWinners)]
        elif roundName == "third_place":
            pairs = [tuple(sfLosers)]
        lines.append(f"## {ROUND_LABELS[roundName]}")
        winners, losers = [], []
        for a, b in pairs:
            w, l, p = favored(a, b)
            winners.append(w)
            losers.append(l)
            lines.append(f"**{w}** {p*100:.0f}% — {l} {(1-p)*100:.0f}%")
        lines.append("")
        if roundName == "semi_final":
            sfWinners, sfLosers = winners, losers
        elif roundName == "final":
            champion = winners[0]
        pairs = list(zip(winners[::2], winners[1::2]))

    lines.append("## Champion")
    lines.append(champion)

    with open(outPath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Bracket results written to {outPath}")


def main():
    with open("teams.json") as f:
        teams = json.load(f)

    results = Parallel(n_jobs=-1)(
        delayed(_runSimulation)(teams) for _ in range(N_SIMULATIONS)
    )

    rows = []
    for r in results:
        opponents = _extractOpponents(r["path"])
        pathStr = " -> ".join(
            f"{e['round']}({e['opponent']})"
            for e in r["path"]
            if e["round"] != "third_place"
        )
        rows.append({"stage": r["stage"], "path": pathStr, "winner": r.get("winner"), **opponents})

    df = pd.DataFrame(rows)
    df.to_parquet("results.parquet", index=False)

    summary = _buildSummary(df, teams, len(df))
    print(summary)
    with open("morocco_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("\nFull summary written to morocco_summary.txt")

    bracketData = defaultdict(lambda: defaultdict(Counter))
    for r in results:
        for roundName, matches in r.get("allMatches", {}).items():
            for slotIdx, (winner, loser) in enumerate(matches):
                bracketData[roundName][slotIdx][(winner, loser)] += 1
    exportBracket(bracketData)


if __name__ == "__main__":
    main()
