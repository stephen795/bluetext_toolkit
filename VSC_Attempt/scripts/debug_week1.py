from src.cli import build_report
from src.sleeper_api import SleeperClient

def main():
    league_id = '1260098143609966592'
    season = 2025
    week = 1
    r, pts, meta = build_report(league_id, season, week, True, source='sleeper', cache_dir='.cache')
    print('teams', len(r))
    team_with_ck = None
    for team_id, t in r.items():
        if 'Christian Kirk' in t.get('players', {}):
            team_with_ck = team_id
            break
    print('team_with_ck', team_with_ck)
    client = SleeperClient()
    players = client.get_players()
    ckpids = [k for k,v in players.items() if isinstance(v, dict) and (v.get('full_name')=='Christian Kirk' or v.get('name')=='Christian Kirk')]
    print('ck_pids', ckpids[:3])
    m = client.get_matchups(league_id, week)
    in_m = bool(ckpids and any(ckpids[0] in (mm.get('players') or []) for mm in (m or [])))
    print('christian_kirk_in_week1_matchups', in_m)
    if team_with_ck:
        names = sorted(list(r[team_with_ck]['players'].keys()))
        print('names_on_team_sample', names[:30])
    # Also show two random team names list lengths
    for i,(tid,t) in enumerate(r.items()):
        print('team', tid, 'players_count', len(t.get('players', {})))
        if i>2:
            break

if __name__ == '__main__':
    main()
