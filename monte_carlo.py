import json
import pandas as pd
from joblib import Parallel, delayed
from groupStage import simulateGroupStage
from knockoutStage import simulateKnockoutStage

N_SIMULATIONS = 100_000
STAGE_ORDER = ["group_stage", "round_of_32", "round_of_16", "quarter_final", "semi_final", "final", "champion"]
ROUND_LABELS = {"round_of_32", "round_of_16", "quarter_final", "semi_final", "final"}


def _runSimulation(teams):
    gsResult = simulateGroupStage(teams)

    if "Morocco" not in gsResult["qualified"]:
        return {"stage": "group_stage", "path": []}

    ksResult = simulateKnockoutStage(gsResult, teams)
    moroccoPath = ksResult["moroccoPath"]

    stage = "group_stage"
    for entry in moroccoPath:
        if entry["round"] == "third_place":
            continue
        stage = entry["round"]
        if entry["result"] == "loss":
            break

    if stage == "final" and ksResult["winner"] == "Morocco":
        stage = "champion"

    return {"stage": stage, "path": moroccoPath, "winner": ksResult["winner"]}


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
    for rnd in ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]:
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


def main():
    with open("teams.json") as f:
        teams = json.load(f)

    results = Parallel(n_jobs=-1)(
        delayed(_runSimulation)(teams) for _ in range(N_SIMULATIONS)
    )

    rows = []
    for r in results:
        opponents = _extractOpponents(r["path"])
        pathStr = " → ".join(
            f"{e['round']}({e['opponent']})"
            for e in r["path"]
            if e["round"] != "third_place" and (e["result"] == "win" or r["stage"] != "champion")
        )
        rows.append({"stage": r["stage"], "path": pathStr, "winner": r.get("winner"), **opponents})

    df = pd.DataFrame(rows)
    df.to_parquet("results.parquet", index=False)

    total = len(df)
    print("=== Morocco Stage Probabilities ===")
    reachedOrBetter = {}
    for s in STAGE_ORDER:
        count = (df["stage"] == s).sum()
        reachedOrBetter[s] = count
    cumulative = 0
    for s in reversed(STAGE_ORDER):
        cumulative += reachedOrBetter.get(s, 0)
        pct = cumulative / total * 100
        label = s.replace("_", " ").title()
        print(f"  Reach {label:20s}: {pct:6.2f}%")

    print("\n=== Top 5 Opponents per Knockout Round ===")
    for rnd in ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]:
        if rnd not in df.columns:
            continue
        col = df[rnd].dropna()
        if col.empty:
            continue
        top5 = col.value_counts().head(5)
        label = rnd.replace("_", " ").title()
        print(f"\n  {label}:")
        for opp, cnt in top5.items():
            print(f"    {opp:30s} {cnt / total * 100:.2f}%")

    print("\n=== Top 10 Tournament Winners ===")
    if "winner" in df.columns:
        winnerCounts = df["winner"].dropna().value_counts().head(10)
        for team, cnt in winnerCounts.items():
            print(f"  {team:30s} {cnt / total * 100:.2f}%")

    print("\n=== Top 10 Most Frequent Full Paths ===")
    pathCounts = df["path"].value_counts().head(10)
    for path, cnt in pathCounts.items():
        print(f"  {cnt / total * 100:.2f}%  {path}")

    summary = _buildSummary(df, teams, total)
    with open("morocco_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n→ Full summary written to morocco_summary.txt")


if __name__ == "__main__":
    main()
