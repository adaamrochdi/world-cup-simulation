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
    winners = [groupResults[g][0] for g in groups]
    runnersUp = [groupResults[g][1] for g in groups]
    return winners + runnersUp + list(thirdPlace)


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
    winners, losers, matches = [], [], []
    for homeN, awayN in pairs:
        hg, ag = simulateMatch(teamMap[homeN], teamMap[awayN], knockout=True)
        winner, loser = (homeN, awayN) if hg > ag else (awayN, homeN)
        matches.append((winner, loser))

        if MOROCCO in (homeN, awayN):
            moroccoPath.append({
                "round": roundName,
                "opponent": awayN if homeN == MOROCCO else homeN,
                "result": "win" if winner == MOROCCO else "loss",
                "scoreHome": hg,
                "scoreAway": ag,
            })

        winners.append(winner)
        losers.append(loser)
    return winners, losers, matches


def _nextRoundPairs(winners):
    return list(zip(winners[::2], winners[1::2]))


def simulateKnockoutStage(groupStageResult, teams):
    teamMap = _buildTeamMap(teams)
    groupResults = groupStageResult["groupResults"]
    thirdPlace = groupStageResult["thirdPlace"]

    seeds = _seedTeams(groupResults, thirdPlace)
    pairs = _buildR32Pairs(seeds, groupResults)

    moroccoPath = []
    allMatches = {}

    winners, _, allMatches["round_of_32"] = _playRound(pairs, teamMap, "round_of_32", moroccoPath)
    winners, _, allMatches["round_of_16"] = _playRound(_nextRoundPairs(winners), teamMap, "round_of_16", moroccoPath)
    winners, _, allMatches["quarter_final"] = _playRound(_nextRoundPairs(winners), teamMap, "quarter_final", moroccoPath)
    finalists, sfLosers, allMatches["semi_final"] = _playRound(_nextRoundPairs(winners), teamMap, "semi_final", moroccoPath)
    champion, _, allMatches["final"] = _playRound([tuple(finalists)], teamMap, "final", moroccoPath)
    _, _, allMatches["third_place"] = _playRound([tuple(sfLosers)], teamMap, "third_place", moroccoPath)

    return {
        "winner": champion[0],
        "moroccoPath": moroccoPath,
        "allMatches": allMatches,
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
