import io, sys
from unittest import mock
from src.cli import import_sleeper_league

def _setup_basic(mock_client):
    mock_client.get_players.return_value = {'p1': {'full_name': 'Sample Player', 'position': 'RB'}}
    mock_client.get_rosters.return_value = [{'owner_id': 't1', 'players': ['p1']}]
    mock_client.get_league.return_value = {'scoring_settings': {'rush_yd':0.1,'rush_att':0.5}}
    mock_client.get_week_stats.return_value = {'p1': {'player_id':'p1','rush_yd':40,'rush_att':8}}

@mock.patch('src.cli.SleeperClient')
def test_explain_outputs_breakdown(mock_client_cls):
    mock_client = mock_client_cls.return_value
    _setup_basic(mock_client)
    with mock.patch('src.cli.build_parsed_pbp_by_name', side_effect=Exception('no pbp')):
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:
            import_sleeper_league('X', 2025, 1, infer=True, explain=True)
        finally:
            sys.stdout = old
        output = buf.getvalue()
    assert 'EXPLAIN_START' in output
    assert 'Sample Player W1:' in output
    # expected total 40*0.1 + 8*0.5 = 4 + 4 = 8
    assert '8.00' in output
