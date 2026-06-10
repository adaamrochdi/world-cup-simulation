from matchEngine import simulateMatch


def buildTeamMap(teams):
    return {t["name"]: t for t in teams}


def groupOf(name, groupResults):
    for g, ranked in groupResults.items():
        if name in ranked:
            return g
    return None


def seedTeams(groupResults, thirdPlace):
    groups = sorted(groupResults.keys())
    winners = [groupResults[g][0] for g in groups]
    runnersUp = [groupResults[g][1] for g in groups]
    return winners + runnersUp + list(thirdPlace)


def buildR32Pairs(seeds, groupResults):
    paired = list(seeds)
    n = len(paired)

    for i in range(n // 2):
        j = n - 1 - i
        if groupOf(paired[i], groupResults) == groupOf(paired[j], groupResults):
            for k in range(j - 1, i, -1):
                if groupOf(paired[k], groupResults) != groupOf(paired[i], groupResults):
                    paired[j], paired[k] = paired[k], paired[j]
                    break

    return [(paired[i], paired[n - 1 - i]) for i in range(n // 2)]


def playRound(pairs, teamMap):
    winners, losers, matches = [], [], []
    for homeN, awayN in pairs:
        hg, ag = simulateMatch(teamMap[homeN], teamMap[awayN], knockout=True)
        winner, loser = (homeN, awayN) if hg > ag else (awayN, homeN)
        matches.append((winner, loser))
        winners.append(winner)
        losers.append(loser)
    return winners, losers, matches


def nextRoundPairs(winners):
    return list(zip(winners[::2], winners[1::2]))


def simulateKnockoutStage(groupStageResult, teams):
    teamMap = buildTeamMap(teams)
    groupResults = groupStageResult["groupResults"]
    thirdPlace = groupStageResult["thirdPlace"]

    seeds = seedTeams(groupResults, thirdPlace)
    pairs = buildR32Pairs(seeds, groupResults)

    allMatches = {}

    winners, _, allMatches["round_of_32"] = playRound(pairs, teamMap)
    winners, _, allMatches["round_of_16"] = playRound(nextRoundPairs(winners), teamMap)
    winners, _, allMatches["quarter_final"] = playRound(nextRoundPairs(winners), teamMap)
    finalists, sfLosers, allMatches["semi_final"] = playRound(nextRoundPairs(winners), teamMap)
    champion, _, allMatches["final"] = playRound([tuple(finalists)], teamMap)
    _, _, allMatches["third_place"] = playRound([tuple(sfLosers)], teamMap)

    return {
        "winner": champion[0],
        "allMatches": allMatches,
    }

