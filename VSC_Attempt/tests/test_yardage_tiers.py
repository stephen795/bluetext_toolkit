import unittest
import responses
import gzip
import io
from src.play_by_play import fetch_pbp_gz_for_season, parse_pbp_bytes_to_player_week_stats, PBP_BASE_RAW
from src.scoring import ScoringEngine
from src.models import ScoringRules, PlayerWeek

TEST_CSV_BUCKETS = (
    "week,receiver_player_name,complete_pass,yards_gained\n"
    "1,Player Long,1,35\n"
    "1,Player Long,1,45\n"
)


class TestYardageBucketsAndTiers(unittest.TestCase):
    @responses.activate
    def test_buckets_recorded(self):
        season = 2024
        url = f"{PBP_BASE_RAW}/play_by_play_{season}.csv.gz"
        bio = io.BytesIO()
        with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
            gz.write(TEST_CSV_BUCKETS.encode("utf-8"))
        bio.seek(0)
        responses.add(responses.GET, url, body=bio.read(), status=200)
        raw = fetch_pbp_gz_for_season(season)
        parsed = parse_pbp_bytes_to_player_week_stats(raw)
        # Player Long had one 35-yd catch (rec_30_39) and one 45-yd catch (rec_40p)
        self.assertIn("Player Long", parsed)
        wk = parsed["Player Long"]["1"]
        self.assertEqual(wk.get("rec_30_39", 0), 1)
        self.assertEqual(wk.get("rec_40p", 0), 1)

    def test_non_stacking_yardage_tiers(self):
        # create a playerweek with 120 rec_yds
        pw = PlayerWeek(player_id="pX", week=1, stats={"rec": 6, "rec_yds": 120, "rec_tds": 1})
        scoring_rules = ScoringRules(rules={
            "rec": 1,
            "rec_yds": 0.1,
            "rec_tds": 6,
            "stat_tiers": [
                {"stat": "rec_yds", "min": 100, "points": 3},
                {"stat": "rec_yds", "min": 150, "points": 5}
            ]
        })
        engine = ScoringEngine(scoring_rules)
        pts = engine.score_player_week(pw)
        # base: rec 6*1 = 6, rec_yds 120*0.1=12, rec_tds 1*6=6 => 24. plus tier: 100+ => +3 => 27
        self.assertAlmostEqual(pts, 27)


if __name__ == '__main__':
    unittest.main()
