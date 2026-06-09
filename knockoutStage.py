import json
from matchEngine import simulateMatch

MOROCCO = "Morocco"


def _buildTeamMap(teams):
    return {t["name"]: t for t in teams}


def _groupOf(name, groupResults):
    for g, ranked in groupResults.items():
        if name in ranked:
            return g
    return None


def _seedTeams(groupResults, thirdPlace):
    groups = sorted(groupResults.keys())
    winners = [groupResults[g][0] for g in groups]      # seeds 1-12
    runnersUp = [groupResults[g][1] for g in groups]    # seeds 13-24
    return winners + runnersUp + list(thirdPlace)        # seeds 25-32


def _buildR32Pairs(seeds, groupResults):
    paired = list(seeds)
    n = len(paired)

    for i in range(n // 2):
        j = n - 1 - i
        if _groupOf(paired[i], groupResults) == _groupOf(paired[j], groupResults):
            for k in range(j - 1, i, -1):
                if _groupOf(paired[k], groupResults) != _groupOf(paired[i], groupResults):
                    paired[j], paired[k] = paired[k], paired[j]
                    break

    return [(paired[i], paired[n - 1 - i]) for i in range(n // 2)]


def _playRound(pairs, teamMap, roundName, moroccoPath):
    winners = []
    for homeN, awayN in pairs:
        hg, ag = simulateMatch(teamMap[homeN], teamMap[awayN], knockout=True)
        winner = homeN if hg > ag else awayN

        if MOROCCO in (homeN, awayN):
            moroccoPath.append({
                "round": roundName,
                "opponent": awayN if homeN == MOROCCO else homeN,
                "result": "win" if winner == MOROCCO else "loss",
                "scoreHome": hg,
                "scoreAway": ag,
            })

        winners.append(winner)
    return winners


def simulateKnockoutStage(groupStageResult, teams):
    teamMap = _buildTeamMap(teams)
    groupResults = groupStageResult["groupResults"]
    thirdPlace = groupStageResult["thirdPlace"]

    seeds = _seedTeams(groupResults, thirdPlace)
    r32Pairs = _buildR32Pairs(seeds, groupResults)

    moroccoPath = []

    # Round of 32: 32 → 16
    r32Winners = _playRound(r32Pairs, teamMap, "round_of_32", moroccoPath)

    # Round of 16: 16 → 8
    r16Pairs = [(r32Winners[i], r32Winners[i + 1]) for i in range(0, len(r32Winners), 2)]
    r16Winners = _playRound(r16Pairs, teamMap, "round_of_16", moroccoPath)

    # Quarter-finals: 8 → 4
    qfPairs = [(r16Winners[i], r16Winners[i + 1]) for i in range(0, len(r16Winners), 2)]
    qfWinners = _playRound(qfPairs, teamMap, "quarter_final", moroccoPath)

    # Semi-finals: 4 → 2
    sfPairs = [(qfWinners[0], qfWinners[1]), (qfWinners[2], qfWinners[3])]
    sfWinners = _playRound(sfPairs, teamMap, "semi_final", moroccoPath)
    sfLosers = [sfPairs[i][1] if sfWinners[i] == sfPairs[i][0] else sfPairs[i][0] for i in range(2)]

    # Championship final + 3rd-place match
    champWinners = _playRound([(sfWinners[0], sfWinners[1])], teamMap, "final", moroccoPath)
    _playRound([(sfLosers[0], sfLosers[1])], teamMap, "third_place", moroccoPath)

    return {
        "winner": champWinners[0],
        "moroccoPath": moroccoPath,
    }


if __name__ == "__main__":
    from groupStage import simulateGroupStage

    with open("teams.json") as f:
        teams = json.load(f)

    gsResult = simulateGroupStage(teams)
    result = simulateKnockoutStage(gsResult, teams)

    print(f"Tournament winner: {result['winner']}")
    print("\nMorocco's path:")
    for entry in result["moroccoPath"]:
        print(f"  {entry['round']}: vs {entry['opponent']} — {entry['result']} ({entry['scoreHome']}-{entry['scoreAway']})")
