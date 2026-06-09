import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

STAGE_ORDER = ["group_stage", "round_of_32", "round_of_16", "quarter_final", "semi_final", "final", "champion"]
ROUND_COLS = ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]
ROUND_SHORT = {
    "round_of_32": "R32",
    "round_of_16": "R16",
    "quarter_final": "QF",
    "semi_final": "SF",
    "final": "Final",
}
THRESHOLD = 0.005  # 0.5%


def _inferStage(df):
    # monte_carlo.py stage = last WINNING round; re-derive furthest stage reached from opponent columns
    conditions = [
        df["round_of_32"].isna(),
        df["round_of_16"].isna(),
        df["quarter_final"].isna(),
        df["semi_final"].isna(),
        df["final"].isna(),
        df["stage"] == "champion",
    ]
    choices = ["group_stage", "round_of_32", "round_of_16", "quarter_final", "semi_final", "champion"]
    df = df.copy()
    df["stageReached"] = np.select(conditions, choices, default="final")
    return df


def printStats(df):
    total = len(df)
    print(f"=== Morocco 2026 World Cup Analysis  (N={total:,}) ===\n")

    counts = df["stageReached"].value_counts()
    print("Stage probabilities:")
    cumulative = 0
    for s in reversed(STAGE_ORDER):
        cnt = int(counts.get(s, 0))
        cumulative += cnt
        label = s.replace("_", " ").title()
        print(f"  {label:20s}  exact: {cnt/total*100:6.2f}%   reached+: {cumulative/total*100:.2f}%")

    print()
    for col, lbl in ROUND_SHORT.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        top5 = series.value_counts().head(5)
        print(f"Top 5 opponents in {lbl}:")
        for opp, cnt in top5.items():
            print(f"  {opp:30s} {cnt / total * 100:.2f}%")
        print()


def _addLink(links, src, tgt, count):
    if count > 0:
        links[(src, tgt)] += count


def buildSankey(df, threshold=THRESHOLD):
    total = len(df)
    minCount = total * threshold
    links = defaultdict(int)

    _addLink(links, "Morocco", "Group Stage Exit", int(df["round_of_32"].isna().sum()))

    r32 = df[df["round_of_32"].notna()]
    for opp, cnt in r32["round_of_32"].value_counts().items():
        _addLink(links, "Morocco", f"R32 vs {opp}", int(cnt))

    for opp in r32["round_of_32"].unique():
        sub = r32[r32["round_of_32"] == opp]
        won = sub["round_of_16"].notna()
        _addLink(links, f"R32 vs {opp}", "Eliminated R32", int((~won).sum()))
        for opp2, cnt2 in sub.loc[won, "round_of_16"].value_counts().items():
            _addLink(links, f"R32 vs {opp}", f"R16 vs {opp2}", int(cnt2))

    r16 = df[df["round_of_16"].notna()]
    for opp in r16["round_of_16"].unique():
        sub = r16[r16["round_of_16"] == opp]
        won = sub["quarter_final"].notna()
        _addLink(links, f"R16 vs {opp}", "Eliminated R16", int((~won).sum()))
        for opp2, cnt2 in sub.loc[won, "quarter_final"].value_counts().items():
            _addLink(links, f"R16 vs {opp}", f"QF vs {opp2}", int(cnt2))

    qf = df[df["quarter_final"].notna()]
    for opp in qf["quarter_final"].unique():
        sub = qf[qf["quarter_final"] == opp]
        won = sub["semi_final"].notna()
        _addLink(links, f"QF vs {opp}", "Eliminated QF", int((~won).sum()))
        for opp2, cnt2 in sub.loc[won, "semi_final"].value_counts().items():
            _addLink(links, f"QF vs {opp}", f"SF vs {opp2}", int(cnt2))

    sf = df[df["semi_final"].notna()]
    for opp in sf["semi_final"].unique():
        sub = sf[sf["semi_final"] == opp]
        won = sub["final"].notna()
        _addLink(links, f"SF vs {opp}", "Eliminated SF", int((~won).sum()))
        for opp2, cnt2 in sub.loc[won, "final"].value_counts().items():
            _addLink(links, f"SF vs {opp}", f"Final vs {opp2}", int(cnt2))

    fin = df[df["final"].notna()]
    for opp in fin["final"].unique():
        sub = fin[fin["final"] == opp]
        isChamp = sub["stageReached"] == "champion"
        _addLink(links, f"Final vs {opp}", "Champion", int(isChamp.sum()))
        _addLink(links, f"Final vs {opp}", "Eliminated Final", int((~isChamp).sum()))

    filteredLinks = [(s, t, v) for (s, t), v in links.items() if v >= minCount]

    nodeNames = []
    nodeIndex = {}
    for s, t, _ in filteredLinks:
        for n in (s, t):
            if n not in nodeIndex:
                nodeIndex[n] = len(nodeNames)
                nodeNames.append(n)

    def nodeColor(name):
        if name == "Morocco":
            return "#C1272D"
        if name == "Champion":
            return "#FFD700"
        if "Eliminated" in name or "Group Stage Exit" in name:
            return "#AAAAAA"
        if "R32" in name:
            return "#76B7B2"
        if "R16" in name:
            return "#4E79A7"
        if "QF" in name:
            return "#59A14F"
        if "SF" in name:
            return "#F28E2B"
        if "Final" in name:
            return "#E15759"
        return "#BAB0AC"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=nodeNames,
            color=[nodeColor(n) for n in nodeNames],
            pad=15,
            thickness=20,
        ),
        link=dict(
            source=[nodeIndex[s] for s, _, _ in filteredLinks],
            target=[nodeIndex[t] for _, t, _ in filteredLinks],
            value=[v for _, _, v in filteredLinks],
        ),
    ))

    fig.update_layout(
        title_text="Morocco's 2026 World Cup Trajectory — Monte Carlo Simulation (N=100,000)",
        font_size=11,
        height=900,
    )

    fig.write_html("morocco_sankey.html")
    print("Sankey saved → morocco_sankey.html")


def main():
    df = pd.read_parquet("results.parquet")
    df = _inferStage(df)
    printStats(df)
    buildSankey(df)


if __name__ == "__main__":
    main()
