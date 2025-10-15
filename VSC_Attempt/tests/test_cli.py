import json
import io
import sys
from unittest import mock, TestCase

from src import cli

class TestCLIImportSleeper(TestCase):
    def setUp(self):
        # Patch SleeperClient methods and pbp_to_playerweeks
        self.patcher_client = mock.patch('src.cli.SleeperClient')
        self.mock_client_cls = self.patcher_client.start()
        self.mock_client = self.mock_client_cls.return_value
        self.mock_client.get_players.return_value = {
            'p1': {'full_name': 'Justin Jefferson'},
            'p2': {'full_name': 'Josh Allen'},
            'p3': {'full_name': 'Stefon Diggs'},
        }
        self.mock_client.get_rosters.return_value = [
            {'owner_id': 'teamA', 'players': ['p1', 'p2']},
            {'owner_id': 'teamB', 'players': ['p3']},
        ]
        self.patcher_pbp = mock.patch('src.cli.pbp_to_playerweeks')
        self.mock_pbp = self.patcher_pbp.start()
        # Provide PBP stats with slight name variation to test fuzzy matching
        self.mock_pbp.return_value = [
            ('Justin Jefferson', 1, {'rec': 5, 'rec_yds': 85}),
            ('Joshua Allen', 1, {'pass_yds': 300, 'pass_td': 2}),
            ('S Diggs', 1, {'rec': 7, 'rec_yds': 95}),
        ]

    def tearDown(self):
        self.patcher_client.stop()
        self.patcher_pbp.stop()

    def run_cli(self, extra_args=None):
        args = ['import-sleeper', '--league', '123', '--season', '2024']
        if extra_args:
            args.extend(extra_args)
        buf = io.StringIO()
        with mock.patch('sys.stdout', buf):
            cli.main(args)
        output = buf.getvalue()
        lines = output.splitlines()
        # Validate summary table sentinel
        self.assertIn('SUMMARY_TABLE_START', lines, 'Summary table sentinel missing')
        # Extract JSON
        try:
            json_idx = lines.index('REPORT_JSON_START')
        except ValueError:
            raise AssertionError(f'Report sentinel not found in output:\n{output}')
        json_str = '\n'.join(lines[json_idx+1:])
        report = json.loads(json_str)
        # Basic summary table sanity: at least one player row
        table_start = lines.index('SUMMARY_TABLE_START')
        header_line = lines[table_start+1]
        self.assertIn('Player', header_line)
        # Collect rows until REPORT_JSON_START
        table_rows = []
        for i in range(table_start+2, json_idx):
            row = lines[i].strip()
            if not row:
                continue
            table_rows.append(row)
        self.assertGreaterEqual(len(table_rows), 1, 'No data rows in summary table')
        # Ensure numeric points column parses
        for r in table_rows:
            if set(r) == {'-'}:
                continue  # skip separator
            parts = r.split()
            # last column should be points
            try:
                float(parts[-1])
            except Exception:
                self.fail(f'Points column not numeric in row: {r}')
        return report

    def test_import_basic(self):
        report = self.run_cli()
        # Ensure teams present
        self.assertIn('teamA', report)
        self.assertIn('teamB', report)
        # Justin Jefferson exact
        self.assertIn('Justin Jefferson', report['teamA']['players'])
        # Josh Allen fuzzy matches Joshua Allen
        self.assertIn('Josh Allen', report['teamA']['players'])
        # Stefon Diggs fuzzy matches S Diggs
        self.assertIn('Stefon Diggs', report['teamB']['players'])

    def test_threshold_too_high(self):
        # Set threshold extremely high to force misses
        report = self.run_cli(['--match-threshold', '0.99'])
        # Justin Jefferson still matches (exact)
        self.assertIn('Justin Jefferson', report['teamA']['players'])
        # Josh Allen should fail fuzzy
        self.assertEqual(report['teamA']['players']['Josh Allen'], {})
        # Stefon Diggs fails fuzzy
        self.assertEqual(report['teamB']['players']['Stefon Diggs'], {})

if __name__ == '__main__':
    import unittest
    unittest.main()
