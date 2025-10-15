from typing import Dict, Any
from .models import PlayerWeek, ScoringRules


class ScoringEngine:
    def __init__(self, rules: ScoringRules):
        self.rules = rules

    def score_player_week(self, pw: PlayerWeek) -> float:
        total = 0.0
        # base multipliers
        for stat, value in pw.stats.items():
            factor = self.rules.rules.get(stat, 0.0)
            try:
                total += factor * value
            except Exception:
                # skip non-numeric rules
                pass

        # receptions can have tier bonuses
        rec = pw.stats.get('rec') or pw.stats.get('receptions') or 0
        if rec:
            tiers = self.rules.rules.get('reception_tiers') or []
            # tiers may be list of {min: X, points: Y} or {min:X, add:Z}
            # reception tiers default to stacking (sum all matching).
            for tier in tiers:
                try:
                    if rec >= tier.get('min', 0):
                        if 'points' in tier:
                            total += tier['points']
                        elif 'mult' in tier:
                            total += tier['mult'] * rec
                        elif 'add' in tier:
                            total += tier['add']
                except Exception:
                    continue

        # non-stacking yardage/stat tiers: look for 'stat_tiers' or 'yardage_tiers'
        stat_tiers = self.rules.rules.get('stat_tiers') or self.rules.rules.get('yardage_tiers') or []
        if stat_tiers:
            # group tiers by stat and pick highest applicable tier per stat
            tiers_by_stat: Dict[str, list] = {}
            for t in stat_tiers:
                s = t.get('stat')
                if not s:
                    continue
                tiers_by_stat.setdefault(s, []).append(t)

            for s, tiers in tiers_by_stat.items():
                try:
                    stat_val = pw.stats.get(s, 0)
                    # find highest tier where stat_val >= min
                    best_points = 0
                    best_min = -1
                    for t in tiers:
                        mn = t.get('min', 0)
                        pts = t.get('points', 0)
                        if stat_val >= mn and mn > best_min:
                            best_min = mn
                            best_points = pts
                    total += best_points
                except Exception:
                    continue

    # two_pt conversions, fumbles, etc. are handled automatically by base multipliers if their
    # keys exist in rules (e.g., 'two_pt': 2). No special logic required.

        # TE specific per-reception bonus (bonus_rec_te) if position info carried in stats as _pos
        try:
            if self.rules.rules.get('bonus_rec_te') and pw.stats.get('_pos') == 'TE':
                rec_val = pw.stats.get('rec') or 0
                total += self.rules.rules['bonus_rec_te'] * rec_val
        except Exception:
            pass

        # Threshold bonuses (non-stacking within each category)
        try:
            rush_yds = float(pw.stats.get('rush_yds', 0) or 0)
            if rush_yds >= 200 and self.rules.rules.get('bonus_rush_200p'):
                total += float(self.rules.rules['bonus_rush_200p'])
            elif rush_yds >= 100 and self.rules.rules.get('bonus_rush_100_199'):
                total += float(self.rules.rules['bonus_rush_100_199'])
        except Exception:
            pass
        try:
            rec_yds = float(pw.stats.get('rec_yds', 0) or 0)
            if rec_yds >= 200 and self.rules.rules.get('bonus_rec_200p'):
                total += float(self.rules.rules['bonus_rec_200p'])
            elif rec_yds >= 100 and self.rules.rules.get('bonus_rec_100_199'):
                total += float(self.rules.rules['bonus_rec_100_199'])
        except Exception:
            pass
        try:
            pass_yds = float(pw.stats.get('pass_yds', 0) or 0)
            if pass_yds >= 400 and self.rules.rules.get('bonus_pass_400p'):
                total += float(self.rules.rules['bonus_pass_400p'])
            elif pass_yds >= 300 and self.rules.rules.get('bonus_pass_300_399'):
                total += float(self.rules.rules['bonus_pass_300_399'])
        except Exception:
            pass
        try:
            comb_yds = float(pw.stats.get('rush_yds', 0) or 0) + float(pw.stats.get('rec_yds', 0) or 0)
            if comb_yds >= 200 and self.rules.rules.get('bonus_comb_200p'):
                total += float(self.rules.rules['bonus_comb_200p'])
            elif comb_yds >= 100 and self.rules.rules.get('bonus_comb_100_199'):
                total += float(self.rules.rules['bonus_comb_100_199'])
        except Exception:
            pass
        try:
            pass_cmp = float(pw.stats.get('pass_cmp', 0) or 0)
            if pass_cmp >= 25 and self.rules.rules.get('bonus_pass_cmp_25p'):
                total += float(self.rules.rules['bonus_pass_cmp_25p'])
        except Exception:
            pass
        try:
            carries = float(pw.stats.get('rush_att', 0) or 0)
            if carries >= 20 and self.rules.rules.get('bonus_carries_20p'):
                total += float(self.rules.rules['bonus_carries_20p'])
        except Exception:
            pass

        return total

    def score_player_weeks(self, pws: list[PlayerWeek]) -> Dict[str, float]:
        # return mapping week (string) -> points
        res: Dict[str, float] = {}
        for pw in pws:
            res[str(pw.week)] = self.score_player_week(pw)
        return res