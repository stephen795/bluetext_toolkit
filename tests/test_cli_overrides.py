import json
import tempfile
import io
import sys
from unittest import mock, TestCase

from src.cli import import_sleeper_league


class TestCLIOverrides(TestCase):
    @mock.patch('src.cli.SleeperClient')
    def test_overrides_mapping(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        # League data
        mock_client.get_league.return_value = { 'scoring_settings': { 'pass_yd': 0.04 } }
        # Players in Sleeper have an odd full_name that won't fuzzy match purposely
        mock_client.get_players.return_value = {
            'pX': {'full_name': 'J AllenQ', 'position': 'QB'}
        }
        # Roster referencing the ID
        mock_client.get_rosters.return_value = [ {'owner_id': 'team1', 'players': ['pX']} ]
        # PBP fetch forced to fail so we go weekly path easily (or we can just mock pbp output). We'll supply weekly stats.
        mock_client.get_week_stats.return_value = {
            'pX': {'player_id': 'pX', 'pass_yd': 300}
        }

        overrides = { 'Josh Allen': 'J AllenQ' }
        with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.json') as f:
            json.dump(overrides, f)
            f.flush()
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                import_sleeper_league('leagueY', 2025, 1, infer=True, match_threshold=0.95, source='sleeper', scoring_path=None, assume_buckets='infer', overrides_path=f.name)
            finally:
                sys.stdout = old_stdout
        out = buf.getvalue()
        # Ensure the canonical desired name appears despite high threshold that would normally fail
        assert 'Josh Allen' in out
        assert 'team1' in out
