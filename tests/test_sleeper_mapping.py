import unittest
import responses
import gzip
import io
from src.play_by_play import fetch_pbp_gz_for_season, parse_pbp_bytes_to_player_week_stats, PBP_BASE_RAW
from src.sleeper_mapper import map_pbp_stats_to_sleeper, map_all_players_pbp_to_sleeper

TEST_CSV = (
    "week,receiver_player_name,complete_pass,yards_gained\n"
    "1,Player One,1,12\n"
    "1,Player One,1,35\n"
    "1,Player One,1,45\n"
)


class TestSleeperMapping(unittest.TestCase):
    @responses.activate
    def test_map_single_player(self):
        season = 2024
        url = f"{PBP_BASE_RAW}/play_by_play_{season}.csv.gz"
        bio = io.BytesIO()
        with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
            gz.write(TEST_CSV.encode("utf-8"))
        bio.seek(0)
        responses.add(responses.GET, url, body=bio.read(), status=200)
        raw = fetch_pbp_gz_for_season(season)
        parsed = parse_pbp_bytes_to_player_week_stats(raw)
        mapped = map_all_players_pbp_to_sleeper(parsed)
        # Player One had a 12-yd catch (rec_10_19), a 35-yd catch (rec_30_39), and a 45-yd catch (rec_40p)
        self.assertIn('Player One', mapped)
        wk = mapped['Player One']['1']
        self.assertEqual(wk.get('rec_10_19', 0), 1)
        self.assertEqual(wk.get('rec_30_39', 0), 1)
        self.assertEqual(wk.get('rec_40p', 0), 1)


if __name__ == '__main__':
    unittest.main()
