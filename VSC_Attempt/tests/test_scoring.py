import unittest
import json
from pathlib import Path
from src.simulate import load_data, load_scoring, run_simulation

class TestScoring(unittest.TestCase):
    def test_sample_simulation(self):
        data = load_data(str(Path('data/sample_season.json').resolve()))
        scoring = load_scoring(str(Path('data/sample_scoring.json').resolve()))
        res = run_simulation(data, scoring)
        # check player points for p1 week1: 250*0.04 + 2*4 + 1*-2 = 10 + 8 -2 = 16
        p1_w1 = res['player_points']['p1']['1']
        self.assertAlmostEqual(p1_w1, 16.0)
        # check team points week1
        t1_w1 = res['team_week_points']['t1']['1']
        self.assertAlmostEqual(t1_w1, 16.0 + (80*0.1 + 1*6))
        # check reception tier for p3 week1: rec=5 -> base rec points (5*1) + rec_yds (70*0.1) + rec_tds (1*6) = 18
        # with reception_tiers defined in sample_scoring.json (min 5 -> +1), expected 19
        p3_w1 = res['player_points']['p3']['1']
        self.assertAlmostEqual(p3_w1, 19.0)

if __name__ == '__main__':
    unittest.main()