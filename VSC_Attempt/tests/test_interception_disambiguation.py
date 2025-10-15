from src.cli import _transform_scoring
from src.scoring import ScoringEngine
from src.models import PlayerWeek, ScoringRules

def test_pass_int_overrides_def_int_positive():
    raw_scoring = {
        'pass_int': -1.0,
        'int': 2.0,  # defensive interceptions made
        'pass_yd': 0.04,
        'pass_td': 4.0
    }
    transformed = _transform_scoring(raw_scoring)
    # Should have 'int' == -1.0 and preserve defensive under 'def_int'
    assert transformed['int'] == -1.0
    assert transformed.get('def_int') == 2.0
    rules = ScoringRules(rules=transformed)
    engine = ScoringEngine(rules)
    # Simulate QB week: 2 INT, 180 pass yards, 2 pass TD
    stats = {'pass_yds': 180.0, 'pass_tds': 2.0, 'int': 2.0}
    pw = PlayerWeek(player_id='QB', week=1, stats=stats)
    pts = engine.score_player_week(pw)
    # 180*.04=7.2 + 2*4=8 - 2*1= -2 => 13.2
    assert abs(pts - 13.2) < 1e-6, pts
