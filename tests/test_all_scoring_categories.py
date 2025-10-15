import io, sys, json
from unittest import mock
from src.cli import import_sleeper_league

# Simulate a league with kicker and defensive scoring plus offensive

def _setup_all_scoring_mocks(mock_client):
    mock_client.get_players.return_value = {
        'qb1': {'full_name': 'Test QB', 'position': 'QB'},
        'k1': {'full_name': 'Test Kicker', 'position': 'K'},
        'dst1': {'full_name': 'Test Defense', 'position': 'D/ST'},
    }
    mock_client.get_rosters.return_value = [
        {'owner_id': 'teamA', 'players': ['qb1','k1','dst1']}
    ]
    mock_client.get_league.return_value = {
        'scoring_settings': {
            'pass_yd': 0.04,
            'pass_td': 4.0,
            'pass_int': -1.0,
            'rush_yd': 0.1,
            'rec_yd': 0.1,
            'rec': 1.0,
            'fgm_0_19': 3.0,
            'fgm_30_39': 3.0,
            'fgm_40_49': 4.0,
            'fgm_50p': 5.0,
            'xpm': 1.0,
            'fgmiss': -1.0,
            'sack': 1.0,
            'def_td': 6.0,
            'fum_rec': 2.0,
            'safe': 2.0,
            'int': 2.0,  # defensive interceptions
        }
    }
    mock_client.get_week_stats.return_value = {
        'qb1': {'player_id':'qb1','pass_yd':250,'pass_td':2,'pass_int':1},
        'k1': {'player_id':'k1','fgm_0_19':1,'fgm_30_39':2,'fgm_40_49':1,'fgm_50p':1,'xpm':3,'fgmiss':1},
        'dst1': {'player_id':'dst1','sack':4,'def_td':1,'fum_rec':2,'safe':1,'int':1},
    }

@mock.patch('src.cli.SleeperClient')
def test_all_scoring_copied_and_scored(mock_client_cls):
    mock_client = mock_client_cls.return_value
    _setup_all_scoring_mocks(mock_client)
    with mock.patch('src.cli.build_parsed_pbp_by_name', side_effect=Exception('no pbp')):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            import_sleeper_league('leagueX', 2025, 1, infer=True, source='auto', match_threshold=0.5, scoring_path=None, assume_buckets='infer')
        finally:
            sys.stdout = old
        out = buf.getvalue()
    # Quick sanity: ensure categories appear in JSON
    assert '"fgm_50p"' in out
    assert '"sack"' in out
    assert 'Test QB' in out
    # Compute expected points for participants (approx):
    # QB: 250*0.04=10 + 2*4=8 -1 =17
    assert 'Test QB' in out and '17.00' in out
    # Kicker: (1*3)+(2*3)+(1*4)+(1*5)+(3*1)+(-1)=3+6+4+5+3-1=20
    assert 'Test Kicker' in out and '20.00' in out
    # Defense (D/ST) scoring not asserted here because toolkit currently filters to individual rostered players; extend later for team defense aggregation.
