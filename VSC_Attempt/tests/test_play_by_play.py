import unittest
import unittest
import responses
import gzip
import io
from src.play_by_play import (
    fetch_pbp_gz_for_season,
    parse_pbp_bytes_to_player_week_stats,
    PBP_BASE_RAW,
)

TEST_CSV = (
    "week,passer_player_name,passing_yards,pass_touchdown,interception,"
    "rusher_player_name,rushing_yards,rush_touchdown,receiver_player_name,"
    "receiving_yards,receiving_touchdown\n"
    "1,Tom Brady,300,1,0,,0,0,Player One,50,1\n"
)


class TestPBP(unittest.TestCase):
    @responses.activate
    def test_parse_simple_csv(self):
        season = 2024
        url = f"{PBP_BASE_RAW}/play_by_play_{season}.csv.gz"

        # gzip the TEST_CSV
        bio = io.BytesIO()
        with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
            gz.write(TEST_CSV.encode("utf-8"))
        bio.seek(0)

        responses.add(responses.GET, url, body=bio.read(), status=200)

        raw = fetch_pbp_gz_for_season(season)
        parsed = parse_pbp_bytes_to_player_week_stats(raw)

        # expect Tom Brady pass_yds 300 and pass_tds 1
        self.assertIn("Tom Brady", parsed)
        self.assertIn("1", parsed["Tom Brady"])
        self.assertAlmostEqual(parsed["Tom Brady"]["1"].get("pass_yds", 0), 300)
        self.assertAlmostEqual(parsed["Tom Brady"]["1"].get("pass_tds", 0), 1)

        # receiver Player One rec_yds 50 and rec_tds 1
        self.assertIn("Player One", parsed)
        self.assertAlmostEqual(parsed["Player One"]["1"].get("rec_yds", 0), 50)
        self.assertAlmostEqual(parsed["Player One"]["1"].get("rec_tds", 0), 1)


if __name__ == "__main__":
    unittest.main()
import responses
