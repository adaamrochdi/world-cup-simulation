import json
from collections import Counter
from joblib import Parallel, delayed
from groupStage import simulateGroupStage
from knockoutStage import simulateKnockoutStage

N_SIMULATIONS = 100_000


def runSimulation(teams):
    gsResult = simulateGroupStage(teams)
    ksResult = simulateKnockoutStage(gsResult, teams)
    return {"winner": ksResult["winner"]}


def main():
    with open("teams.json") as f:
        teams = json.load(f)

    results = Parallel(n_jobs=-1)(
        delayed(runSimulation)(teams) for _ in range(N_SIMULATIONS)
    )

    counts = Counter(r["winner"] for r in results)
    lines = [f"{team}: {count / N_SIMULATIONS * 100:.2f}%" for team, count in counts.most_common()]
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
