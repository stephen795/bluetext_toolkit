import json
from src.sleeper_mapper import map_pbp_stats_to_sleeper_full
from src.scoring import ScoringEngine
from src.models import PlayerWeek, ScoringRules

def test_rush_att_preserved_and_scored():
    raw = {
        'rush_att': 8,
        'rush_yds': 40,
        'rush_tds': 1,
        'rec': 0
    }
    mapped = map_pbp_stats_to_sleeper_full(raw)
    assert 'rush_att' in mapped, 'rush_att should be preserved in mapped stats'
    rules = ScoringRules(rules={
        'rush_att': 0.5,
        'rush_yds': 0.1,
        'rush_tds': 6
    })
    engine = ScoringEngine(rules)
    pw = PlayerWeek(player_id='Test RB', week=1, stats=mapped)
    pts = engine.score_player_week(pw)
    # expected: rush attempts 8 * .5 =4, rush yards 40*.1=4, rush TD 6 => total 14
    assert abs(pts - 14.0) < 1e-6, f'Expected 14.0 got {pts}'
