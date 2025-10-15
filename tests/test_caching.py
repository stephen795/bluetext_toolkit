import os
import io
import sys
import json
import tempfile
from unittest import mock, TestCase
from pathlib import Path

from src.cli import import_sleeper_league


class TestCaching(TestCase):
    @mock.patch('src.cli.SleeperClient')
    def test_weekly_stats_caching(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.get_league.return_value = {'scoring_settings': {'pass_yd': 0.04}}
        mock_client.get_players.return_value = {'p1': {'full_name': 'Test QB', 'position': 'QB'}}
        mock_client.get_rosters.return_value = [{'owner_id': 'team1', 'players': ['p1']}]
        mock_client.get_week_stats.return_value = {'p1': {'player_id': 'p1', 'pass_yd': 100}}
        with tempfile.TemporaryDirectory() as tmp:
            # First run populates cache
            buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
            try:
                import_sleeper_league('L1', 2025, 1, infer=True, source='sleeper', match_threshold=0.5, scoring_path=None, assume_buckets='infer', cache_dir=tmp)
            finally:
                sys.stdout = old
            # Modify client return to raise if called again to prove cache used
            mock_client.get_week_stats.side_effect = AssertionError('Should not refetch when cached')
            buf2 = io.StringIO(); old = sys.stdout; sys.stdout = buf2
            try:
                import_sleeper_league('L1', 2025, 1, infer=True, source='sleeper', match_threshold=0.5, scoring_path=None, assume_buckets='infer', cache_dir=tmp)
            finally:
                sys.stdout = old
            out2 = buf2.getvalue()
            assert 'SUMMARY_TABLE_START' in out2
