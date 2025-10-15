import unittest
from src.scoring import ScoringEngine
from src.models import PlayerWeek, ScoringRules

LEAGUE_RULES = {
    'pass_yds': 0.04,
    'pass_tds': 4.0,
    'int': -1.0,
    'rush_yds': 0.1,
    'rush_tds': 6.0,
    'rush_att': 0.5,
    'rec': 0.0,
    'rec_yds': 0.1,
    'rec_tds': 6.0,
    'fum_lost': -2.0,
    'rec_0_4': 0.5,
    'rec_5_9': 0.75,
    'rec_10_19': 1.0,
    'rec_20_29': 1.0,
    'rec_30_39': 1.25,
    'rec_40p': 1.5,
    'bonus_rec_te': 0.75,
}

class TestCustomLeagueScoring(unittest.TestCase):
    def setUp(self):
        self.engine = ScoringEngine(ScoringRules(rules=LEAGUE_RULES))

    def test_josh_allen_week(self):
        # 394 pass yds, 2 pass TD, 30 rush yds, 14 rush att, 2 rush TD
        stats = {
            'pass_yds': 394.0,
            'pass_tds': 2.0,
            'rush_yds': 30.0,
            'rush_att': 14.0,
            'rush_tds': 2.0
        }
        pw = PlayerWeek(player_id='Josh Allen', week=1, stats=stats)
        pts = self.engine.score_player_week(pw)
        expected = 394*0.04 + 2*4 + 30*0.1 + 14*0.5 + 2*6
        self.assertAlmostEqual(pts, expected, places=2)
        self.assertAlmostEqual(pts, 45.76, places=2)

    def test_derrick_henry_week(self):
        # Provided scoring: 169 rush yds, 18 rush att, 2 rush TD, 13 rec yds, buckets: one 10-19, one 0-4, no 5-9 shown (example total 38.2)
        # Using buckets: rec_10_19=1 (1.0), rec_0_4=0? Example includes only 10-19 and receiving yards; total given: 38.2 -> compute needed bucket points.
        # Recompute from provided breakdown (12 rush TD + 9 rush att bonus + 16.9 rush yds + 1.3 rec yds -2 fumble + 1.0 bucket = 38.2)
        stats = {
            'rush_yds': 169.0,
            'rush_att': 18.0,
            'rush_tds': 2.0,
            'rec_yds': 13.0,
            'rec_10_19': 1.0,
            'fum_lost': 1.0
        }
        pw = PlayerWeek(player_id='Derrick Henry', week=1, stats=stats)
        pts = self.engine.score_player_week(pw)
        expected = (169*0.1) + (18*0.5) + (2*6) + (13*0.1) + (1*1.0) + (-2)
        self.assertAlmostEqual(pts, expected, places=2)
        self.assertAlmostEqual(round(pts,2), 38.2)

    def test_zay_flowers_week(self):
        stats = {
            'rec_0_4': 1.0,
            'rec_5_9': 1.0,
            'rec_10_19': 1.0,
            'rec_20_29': 2.0,
            'rec_30_39': 2.0,
            'rec_tds': 1.0,
            'rec_yds': 143.0,
            'rush_att': 2.0,
            'rush_yds': 8.0
        }
        pw = PlayerWeek(player_id='Zay Flowers', week=1, stats=stats)
        pts = self.engine.score_player_week(pw)
        expected = (1*0.5)+(1*0.75)+(1*1.0)+(2*1.0)+(2*1.25)+(1*6)+(143*0.1)+(2*0.5)+(8*0.1)
        self.assertAlmostEqual(round(pts,2), 28.85)

    def test_juwan_johnson_week(self):
        stats = {
            'rec_0_4': 1.0,
            'rec_5_9': 3.0,
            'rec_10_19': 3.0,
            'rec_20_29': 1.0,
            'rec_yds': 76.0,
            'rec': 8.0,  # total receptions for TE bonus
            '_pos': 'TE'
        }
        pw = PlayerWeek(player_id='Juwan Johnson', week=1, stats=stats)
        pts = self.engine.score_player_week(pw)
        expected = (1*0.5)+(3*0.75)+(3*1.0)+(1*1.0)+(76*0.1)+(8*0.75)
        self.assertAlmostEqual(round(pts,2), 20.35)

if __name__ == '__main__':
    unittest.main()
