import unittest
from unittest import mock
from src.cli import import_sleeper_league
import io
import sys


def _setup_mocks(mock_client):
    # players map with two players
    mock_client.get_players.return_value = {
        'pid1': {'full_name': 'Josh Allen', 'position': 'QB'},
        'pid2': {'full_name': 'Juwan Johnson', 'position': 'TE'},
    }
    mock_client.get_rosters.return_value = [
        {'owner_id': 'team1', 'players': ['pid1', 'pid2']}
    ]
    mock_client.get_league.return_value = {
        'scoring_settings': {
            'pass_yd': 0.04,
            'pass_td': 4,
            'rush_yd': 0.1,
            'rush_td': 6,
            'rec': 0,
            'rec_yd': 0.1,
            'rec_td': 6,
            'rush_att': 0.5,
            'rec_0_4': 0.5,
            'rec_5_9': 0.75,
            'rec_10_19': 1.0,
            'rec_20_29': 1.0,
            'rec_30_39': 1.25,
            'rec_40p': 1.5,
            'bonus_rec_te': 0.75,
        }
    }
    mock_client.get_week_stats.return_value = {
        'pid1': {'player_id': 'pid1', 'pass_yd': 100, 'pass_td': 1, 'rush_att': 2, 'rush_yd': 10},
        'pid2': {'player_id': 'pid2', 'rec': 2, 'rec_yd': 20},
    }


class TestWeeklyStatsFallback(unittest.TestCase):
    @mock.patch('src.cli.SleeperClient')
    def test_weekly_stats_used_when_pbp_missing(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        _setup_mocks(mock_client)
        with mock.patch('src.cli.build_parsed_pbp_by_name', side_effect=Exception('pbp missing')) as mock_pbp:
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                import_sleeper_league('leagueX', 2025, 1, infer=True, match_threshold=0.5, source='auto', scoring_path=None, assume_buckets='infer')
            finally:
                sys.stdout = old_stdout
            output = buf.getvalue()
            self.assertTrue(mock_pbp.called, 'PBP path should be attempted first')
            self.assertTrue(mock_client.get_week_stats.called, 'Weekly stats fallback should be invoked')
            # Simple point assertions based on mocked stats & scoring: Josh Allen -> pass_yd 100*0.04=4 + pass_td 1*4=4 + rush_att 2*0.5=1 + rush_yd 10*0.1=1 => 10.0
            self.assertIn('Josh Allen', output)
            self.assertIn('10.00', output)
            # Juwan Johnson: rec buckets inferred 2 receptions for 20 yards -> naive distribution yields rec_0_4=1 rec_20_29=1 plus bonus_rec_te 0.75*2=1.5 and yardage 20*0.1=2 => bucket points 0.5 + 1.0 + 1.5 + 2 = 5.0
            self.assertIn('Juwan Johnson', output)
            self.assertIn('5.00', output)


@mock.patch('src.cli.SleeperClient')
def test_weekly_stats_fallback_function(mock_client_cls):
    mock_client = mock_client_cls.return_value
    _setup_mocks(mock_client)
    with mock.patch('src.cli.build_parsed_pbp_by_name', side_effect=Exception('pbp missing')) as mock_pbp:
        import_sleeper_league('leagueX', 2025, 1, infer=True, match_threshold=0.5, source='auto', scoring_path=None, assume_buckets='infer')
        assert mock_pbp.called
        assert mock_client.get_week_stats.called


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
