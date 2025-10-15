import json
from pathlib import Path
from typing import Dict
from .models import SeasonData, Player, Team, PlayerWeek, ScoringRules
from .scoring import ScoringEngine


def load_data(path: str) -> SeasonData:
    j = json.loads(Path(path).read_text())
    players = {p['id']: Player(**{k: v for k, v in p.items() if k != 'weekly'}) for p in j['players']}
    player_weeks = []
    for p in j['players']:
        for w in p.get('weekly', []):
            player_weeks.append(PlayerWeek(player_id=p['id'], week=w['week'], stats=w['stats']))
            players[p['id']].weekly.append(player_weeks[-1])
    teams = {t['id']: Team(**t) for t in j['teams']}
    weeks = j.get('weeks', 17)
    return SeasonData(players=players, teams=teams, weeks=weeks, player_weeks=player_weeks)


def load_scoring(path: str) -> ScoringRules:
    j = json.loads(Path(path).read_text())
    return ScoringRules(rules=j)


def run_simulation(data: SeasonData, scoring: ScoringRules) -> Dict:
    engine = ScoringEngine(scoring)
    # compute player weekly points (week keys are strings)
    player_points: Dict[str, Dict[str, float]] = {}
    for pid, player in data.players.items():
        player_points[pid] = engine.score_player_weeks(player.weekly)

    # compute team points per week using starting lineup
    team_week_points: Dict[str, Dict[str, float]] = {tid: {} for tid in data.teams}
    for tid, team in data.teams.items():
        for week in range(1, data.weeks + 1):
            wk_key = str(week)
            points = 0.0
            for pid in team.starting:
                points += player_points.get(pid, {}).get(wk_key, 0.0)
            team_week_points[tid][wk_key] = points

    # simple round-robin pairing: team i vs team i+1 each week rotated (naive)
    team_ids = list(data.teams.keys())
    standings = {tid: {'wins': 0, 'losses': 0, 'ties': 0, 'points_for': 0.0} for tid in team_ids}
    n = len(team_ids)
    for week in range(1, data.weeks + 1):
        # pair sequentially
        for i in range(0, n, 2):
            if i+1 >= n:
                continue
            t1 = team_ids[i]
            t2 = team_ids[i+1]
            wk = str(week)
            p1 = team_week_points[t1].get(wk, 0.0)
            p2 = team_week_points[t2].get(wk, 0.0)
            standings[t1]['points_for'] += p1
            standings[t2]['points_for'] += p2
            if p1 > p2:
                standings[t1]['wins'] += 1
                standings[t2]['losses'] += 1
            elif p2 > p1:
                standings[t2]['wins'] += 1
                standings[t1]['losses'] += 1
            else:
                standings[t1]['ties'] += 1
                standings[t2]['ties'] += 1

    # build result
    result = {
        'player_points': player_points,
        'team_week_points': team_week_points,
        'standings': standings,
    }
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--scoring', required=True)
    args = parser.parse_args()
    data = load_data(args.data)
    scoring = load_scoring(args.scoring)
    res = run_simulation(data, scoring)
    print(json.dumps(res, indent=2))