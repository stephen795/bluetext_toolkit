import unittest
import responses
import gzip
import io
from src.play_by_play import fetch_pbp_gz_for_season, parse_pbp_bytes_to_player_week_stats, PBP_BASE_RAW
from src.sleeper_mapper import map_pbp_stats_to_sleeper_full

TEST_CSV = (
    "week,receiver_player_name,complete_pass,yards_gained\n"
    "1,Player Agg,1,3\n"
    "1,Player Agg,1,8\n"
    "1,Player Agg,1,12\n"
    "1,Player Agg,1,25\n"
    "1,Player Agg,1,35\n"
    "1,Player Agg,1,50\n"
)


class TestSleeperFullMapping(unittest.TestCase):
    @responses.activate
    def test_full_mapping(self):
        season = 2024
        url = f"{PBP_BASE_RAW}/play_by_play_{season}.csv.gz"
        bio = io.BytesIO()
        with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
            gz.write(TEST_CSV.encode("utf-8"))
        bio.seek(0)
        responses.add(responses.GET, url, body=bio.read(), status=200)
        raw = fetch_pbp_gz_for_season(season)
        parsed = parse_pbp_bytes_to_player_week_stats(raw)
        stats = parsed['Player Agg']['1']
        full = map_pbp_stats_to_sleeper_full(stats)
        # expect counts for each bucket to be 1, and rec_10p=4 (12,25,35,50), rec_20p=3 (25,35,50)
        self.assertEqual(full.get('rec_0_4', 0), 1)
        self.assertEqual(full.get('rec_5_9', 0), 1)
        self.assertEqual(full.get('rec_10_19', 0), 1)
        self.assertEqual(full.get('rec_20_29', 0), 1)
        self.assertEqual(full.get('rec_30_39', 0), 1)
        self.assertEqual(full.get('rec_40p', 0), 1)
        self.assertEqual(full.get('rec_10p', 0), 4)
        self.assertEqual(full.get('rec_20p', 0), 3)


if __name__ == '__main__':
    unittest.main()
