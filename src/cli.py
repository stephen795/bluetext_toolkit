import argparse
import json
from typing import Dict, Any, List

from .sleeper_api import SleeperClient
from .play_by_play import pbp_to_playerweeks
from .sleeper_mapper import map_all_players_pbp_to_sleeper, map_pbp_stats_to_sleeper_full
from .scoring import ScoringEngine
from .models import PlayerWeek, ScoringRules
from .name_matcher import find_best_match, normalize_name as nm_normalize


def normalize_name(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum() or ch.isspace()).strip()


def build_parsed_pbp_by_name(season: int, week: int | None = None):
    # pbp_to_playerweeks returns list of tuples (player_name, wk, stats)
    lst = pbp_to_playerweeks(season, week=week)
    parsed = {}
    for player_name, wk, stats in lst:
        parsed.setdefault(player_name, {})[str(wk)] = stats
    return parsed


def _transform_scoring(sc: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Sleeper league scoring_settings keys to our internal stat keys.

    We accept both native sleeper keys (pass_yd, rush_td, rec_yd, etc.) and our internal *_yds, *_tds forms.
    """
    mapping = {
        'pass_yd': 'pass_yds',
        'rush_yd': 'rush_yds',
        'rec_yd': 'rec_yds',
        'pass_td': 'pass_tds',
        'rush_td': 'rush_tds',
        'rec_td': 'rec_tds',
    }
    out: Dict[str, Any] = {}
    # First pass: apply simple mapping
    for k, v in sc.items():
        key = mapping.get(k, k)
        out[key] = v

    # Interception disambiguation:
    # Sleeper may have both pass_int (offensive interceptions thrown, usually negative)
    # and int (defensive interceptions made, usually positive). Our player offensive stats
    # normalize to a single 'int' count for passes intercepted. If both exist and have
    # conflicting signs (pass_int negative, int positive), prefer pass_int value for 'int'.
    if 'pass_int' in sc and 'int' in sc:
        try:
            raw_pass = sc.get('pass_int')
            raw_def = sc.get('int')
            if raw_pass is None or raw_def is None:
                raise ValueError('Missing interception values')
            v_pass = float(raw_pass)
            v_def = float(raw_def)
            # Heuristic: if pass_int is negative (penalty) and int is positive (IDP reward),
            # set 'int' to offensive penalty and preserve defensive under a different key.
            if v_pass < 0 <= v_def:
                out['int'] = v_pass  # ensure penalty applied
                # keep defensive interception points separately if needed later
                out['def_int'] = v_def
            # If pass_int exists but int missing or same sign, still map offensive to 'int'
            elif 'int' not in out:
                out['int'] = v_pass
        except Exception:
            pass
    elif 'pass_int' in sc and 'int' not in sc:
        # Only offensive present; ensure we expose via 'int'
        try:
            raw_pass = sc.get('pass_int')
            if raw_pass is not None:
                out['int'] = float(raw_pass)
        except Exception:
            pass
    return out


def score_breakdown(stats: Dict[str, float], scoring_rules: Dict[str, Any]) -> Dict[str, float]:
    """Return per-stat contribution given current scoring_rules (includes tier bonuses separately labeled)."""
    contributions: Dict[str, float] = {}
    # base stats
    for stat, val in stats.items():
        if stat.startswith('_'):
            continue
        factor = scoring_rules.get(stat)
        if isinstance(factor, (int, float)) and isinstance(val, (int, float)) and factor != 0:
            contributions[stat] = factor * val
    # reception tiers (stacking)
    rec = stats.get('rec') or stats.get('receptions') or 0
    tiers = scoring_rules.get('reception_tiers') or []
    tier_total = 0.0
    for tier in tiers:
        try:
            if rec >= tier.get('min', 0):
                if 'points' in tier:
                    tier_total += tier['points']
                elif 'mult' in tier:
                    tier_total += tier['mult'] * rec
                elif 'add' in tier:
                    tier_total += tier['add']
        except Exception:
            continue
    if tier_total:
        contributions['reception_tiers'] = tier_total
    # yardage/stat non-stacking tiers
    stat_tiers = scoring_rules.get('stat_tiers') or scoring_rules.get('yardage_tiers') or []
    if stat_tiers:
        tiers_by = {}
        for t in stat_tiers:
            s = t.get('stat')
            if s:
                tiers_by.setdefault(s, []).append(t)
        for s, tiers_list in tiers_by.items():
            try:
                stat_val = stats.get(s, 0)
                best_points = 0
                best_min = -1
                for t in tiers_list:
                    mn = t.get('min', 0)
                    pts = t.get('points', 0)
                    if stat_val >= mn and mn > best_min:
                        best_min = mn
                        best_points = pts
                if best_points:
                    contributions[f'{s}_tier'] = best_points
            except Exception:
                continue
    # TE bonus
    try:
        if scoring_rules.get('bonus_rec_te') and stats.get('_pos') == 'TE' and stats.get('rec'):
            contributions['bonus_rec_te'] = scoring_rules['bonus_rec_te'] * stats.get('rec', 0)
    except Exception:
        pass

    # Threshold bonuses (non-stacking within each category)
    try:
        rush_yds = float(stats.get('rush_yds', 0) or 0)
        if rush_yds >= 200 and scoring_rules.get('bonus_rush_200p'):
            contributions['bonus_rush_200p'] = float(scoring_rules['bonus_rush_200p'])
        elif rush_yds >= 100 and scoring_rules.get('bonus_rush_100_199'):
            contributions['bonus_rush_100_199'] = float(scoring_rules['bonus_rush_100_199'])
    except Exception:
        pass
    try:
        rec_yds = float(stats.get('rec_yds', 0) or 0)
        if rec_yds >= 200 and scoring_rules.get('bonus_rec_200p'):
            contributions['bonus_rec_200p'] = float(scoring_rules['bonus_rec_200p'])
        elif rec_yds >= 100 and scoring_rules.get('bonus_rec_100_199'):
            contributions['bonus_rec_100_199'] = float(scoring_rules['bonus_rec_100_199'])
    except Exception:
        pass
    try:
        pass_yds = float(stats.get('pass_yds', 0) or 0)
        if pass_yds >= 400 and scoring_rules.get('bonus_pass_400p'):
            contributions['bonus_pass_400p'] = float(scoring_rules['bonus_pass_400p'])
        elif pass_yds >= 300 and scoring_rules.get('bonus_pass_300_399'):
            contributions['bonus_pass_300_399'] = float(scoring_rules['bonus_pass_300_399'])
    except Exception:
        pass
    try:
        comb_yds = float(stats.get('rush_yds', 0) or 0) + float(stats.get('rec_yds', 0) or 0)
        if comb_yds >= 200 and scoring_rules.get('bonus_comb_200p'):
            contributions['bonus_comb_200p'] = float(scoring_rules['bonus_comb_200p'])
        elif comb_yds >= 100 and scoring_rules.get('bonus_comb_100_199'):
            contributions['bonus_comb_100_199'] = float(scoring_rules['bonus_comb_100_199'])
    except Exception:
        pass
    try:
        pass_cmp = float(stats.get('pass_cmp', 0) or 0)
        if pass_cmp >= 25 and scoring_rules.get('bonus_pass_cmp_25p'):
            contributions['bonus_pass_cmp_25p'] = float(scoring_rules['bonus_pass_cmp_25p'])
    except Exception:
        pass
    try:
        carries = float(stats.get('rush_att', 0) or 0)
        if carries >= 20 and scoring_rules.get('bonus_carries_20p'):
            contributions['bonus_carries_20p'] = float(scoring_rules['bonus_carries_20p'])
    except Exception:
        pass
    return contributions


def build_report(
    league_id: str,
    season: int,
    week: int | None,
    infer: bool,
    match_threshold: float = 0.78,
    source: str = 'auto',
    scoring_path: str | None = None,
    assume_buckets: str = 'infer',
    cache_dir: str | None = None,
    debug_source: bool = False,
    overrides_path: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """Programmatic API to generate the same report as the CLI.

    Returns (report, player_points, meta) where meta includes scoring_rules, used_source, league_info, name_pos.
    """
    client = SleeperClient()
    players_map = client.get_players()
    rosters = client.get_rosters(league_id)
    # Map owner_id -> manager display name, and owner_id -> team_name
    owner_to_manager: Dict[str, str] = {}
    owner_to_team_name: Dict[str, str] = {}
    try:
        users = client.get_league_users(league_id) or []
        for u in users:
            if isinstance(u, dict):
                uid = u.get('user_id')
                dn = u.get('display_name') or u.get('username') or uid
                if uid and dn:
                    owner_to_manager[str(uid)] = dn
                # Prefer explicit team_name from user metadata, else fall back to display name
                tname = None
                meta_u = u.get('metadata') or {}
                if isinstance(meta_u, dict):
                    tname = meta_u.get('team_name') or meta_u.get('team_name_update')
                if uid:
                    if isinstance(tname, str) and tname:
                        owner_to_team_name[str(uid)] = tname
                    elif isinstance(dn, str) and dn:
                        owner_to_team_name[str(uid)] = dn
    except Exception:
        pass
    # Backfill team names from roster metadata if present
    try:
        for r in rosters or []:
            oid = r.get('owner_id')
            if oid is None:
                continue
            meta_r = r.get('metadata') or {}
            if isinstance(meta_r, dict):
                tname = meta_r.get('team_name') or meta_r.get('team_name_update')
                if tname and str(oid) not in owner_to_team_name:
                    owner_to_team_name[str(oid)] = tname
            # Ensure we always have at least manager display name as team name
            if str(oid) not in owner_to_team_name and str(oid) in owner_to_manager:
                owner_to_team_name[str(oid)] = owner_to_manager[str(oid)]
    except Exception:
        pass
    league_info = None
    try:
        league_info = client.get_league(league_id)
    except Exception:
        league_info = None

    # build name -> player_id map
    name_to_id: Dict[str, str] = {}
    for pid, pdata in players_map.items():
        fullname = pdata.get('full_name') or pdata.get('name')
        if not fullname:
            continue
        name_to_id[nm_normalize(fullname)] = pid

    # fetch PBP and map to sleeper categories
    print(f"Fetching play-by-play for season {season} (this may be a few MB)...")
    parsed = {}
    used_source = None
    # simple caching helpers (file-based)
    import os, hashlib, json as _json
    def _cache_path(key: str) -> str:
        return os.path.join(cache_dir, hashlib.sha256(key.encode()).hexdigest()+'.json') if cache_dir else ''
    def cache_load(key: str):
        if not cache_dir:
            return None
        try:
            path = _cache_path(key)
            if os.path.exists(path):
                with open(path,'r',encoding='utf-8') as fh:
                    return _json.load(fh)
        except Exception:
            return None
        return None
    def cache_save(key: str, obj):
        if not cache_dir:
            return
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(_cache_path(key),'w',encoding='utf-8') as fh:
                _json.dump(obj, fh)
        except Exception:
            pass

    if source in ('pbp', 'auto'):
        try:
            parsed = build_parsed_pbp_by_name(season, week)
            used_source = 'pbp'
        except Exception as e:
            # Non-fatal; fall back unless explicitly requested pbp
            if source == 'pbp':
                raise
    parsed_all_for_meta = None
    if (not parsed) and source in ('sleeper', 'auto') and week is not None:
        # fetch weekly stats from Sleeper
        try:
            cache_key = f"sleeper-week-{season}-{week}"
            week_stats_raw = cache_load(cache_key)
            if week_stats_raw is None:
                week_stats_raw = client.get_week_stats(season, week)
                cache_save(cache_key, week_stats_raw)
            wk_key = str(week)
            # Recognize shapes: list of dicts OR dict of player_id -> stats
            if isinstance(week_stats_raw, dict):
                iterable = []
                for pid, stat_block in week_stats_raw.items():
                    # unify item dict
                    entry = {'player_id': pid}
                    if isinstance(stat_block, dict):
                        entry.update(stat_block)
                    iterable.append(entry)
            elif isinstance(week_stats_raw, list):
                iterable = week_stats_raw
            else:
                iterable = []
            # Build roster player_id set for filtering.
            # Prefer week-specific matchups (historical rosters) when a week is provided; fall back to current rosters.
            roster_player_ids = set()
            if week is not None:
                try:
                    matchups = client.get_matchups(league_id, week)
                    # Each matchup has 'players' (all rostered that week) and 'starters' (lineup)
                    for m in matchups or []:
                        for pid in (m.get('players') or []):
                            if pid:
                                roster_player_ids.add(pid)
                        for pid in (m.get('starters') or []):
                            if pid:
                                roster_player_ids.add(pid)
                except Exception:
                    pass
            # Fallback to current rosters only if no specific week was requested
            if not roster_player_ids and week is None:
                for r in rosters or []:
                    for pid in r.get('players', []) or []:
                        if pid:
                            roster_player_ids.add(pid)
            # map Sleeper player object provides full_name in players_map
            # Determine dynamic extra keys based on league scoring settings (beyond core mapping)
            league_scoring_keys = set()
            if league_info and isinstance(league_info, dict):
                league_scoring_keys = set((league_info.get('scoring_settings') or {}).keys())
            # Keys we already explicitly map or will synthesize
            known_mapped = {
                'pass_yd','rush_yd','rec_yd','pass_td','rush_td','rec_td','rush_att','rec','fum_lost','pass_int','int',
                # buckets handled separately
                'rec_0_4','rec_5_9','rec_10_19','rec_20_29','rec_30_39','rec_40p'
            }
            # Exclude aggregate team-level or unsupported categories for player lines we don't currently compute: pts_allow_*, def_st_*, st_ff, st_td etc. We'll still copy if the player stat block has them (e.g., IDP stats) but not attempt inference.
            # Build list of extra keys to attempt copying verbatim if present in a player's weekly item.
            extra_copy_keys = [k for k in league_scoring_keys if k not in known_mapped]

            # We'll build two maps: one for all players, one filtered to rostered players
            parsed_all = {}
            parsed_rostered = {}
            for item in iterable:
                if not isinstance(item, dict):
                    continue
                pid = item.get('player_id') or item.get('playerId') or item.get('pid')
                name = None
                if pid and pid in players_map:
                    pdata = players_map[pid]
                    name = pdata.get('full_name') or pdata.get('name') or pdata.get('first_name')
                if not name:
                    name = item.get('player') or pid
                if not name:
                    continue
                stats = {}
                def copy(src_key, dest_key):
                    v = item.get(src_key)
                    if v is not None:
                        try:
                            stats[dest_key] = float(v)
                        except Exception:
                            pass
                seen_int = False
                for src, dest in [
                    ('pass_yd','pass_yds'),('rush_yd','rush_yds'),('rec_yd','rec_yds'),
                    ('pass_td','pass_tds'),('rush_td','rush_tds'),('rec_td','rec_tds'),
                    ('rush_att','rush_att'),('rec','rec'),('fum_lost','fum_lost'),
                    ('pass_int','int'),('int','int')
                ]:
                    if src in ('pass_int','int'):
                        if seen_int:
                            continue
                        # mark once if either key present
                        if item.get(src) is not None:
                            seen_int = True
                        else:
                            continue
                    copy(src, dest)
                # Copy bucket stats directly if present in weekly stats
                bucket_keys = ['rec_0_4','rec_5_9','rec_10_19','rec_20_29','rec_30_39','rec_40p']
                any_bucket_present = False
                # stat_block (alias) is the raw item for direct key lookup
                stat_block = item
                def _sb_get(k):
                    v = stat_block.get(k) if isinstance(stat_block, dict) else None
                    return v
                for bk in bucket_keys:
                    raw_val = _sb_get(bk)
                    if raw_val is None:
                        continue
                    try:
                        stats[bk] = int(raw_val)
                        any_bucket_present = True
                    except Exception:
                        pass
                # Copy any extra scoring-related keys verbatim if present in item and numeric
                for extra_key in extra_copy_keys:
                    if extra_key in known_mapped:
                        continue
                    if extra_key in stats:  # already set by earlier logic
                        continue
                    val = item.get(extra_key)
                    if val is None:
                        continue
                    # Skip obviously team-only categories (heuristic) if no player context: pts_allow_ etc.
                    if extra_key.startswith('pts_allow_'):
                        continue
                    try:
                        stats[extra_key] = float(val)
                    except Exception:
                        pass

                # Ensure important offensive keys are copied even if not in league_scoring_keys
                def copy_alias(src_key: str, dest_key: str):
                    if dest_key in stats:
                        return
                    v = item.get(src_key)
                    if v is None:
                        return
                    try:
                        stats[dest_key] = float(v)
                    except Exception:
                        pass

                # Completions/attempts common aliases
                copy_alias('pass_cmp', 'pass_cmp')
                copy_alias('cmp', 'pass_cmp')
                copy_alias('complete_pass', 'pass_cmp')
                copy_alias('completions', 'pass_cmp')
                copy_alias('pass_comp', 'pass_cmp')
                copy_alias('pass_completions', 'pass_cmp')
                copy_alias('comp', 'pass_cmp')
                copy_alias('passcomp', 'pass_cmp')
                copy_alias('pass_att', 'pass_att')
                copy_alias('att', 'pass_att')
                # First downs and sacks taken
                copy_alias('pass_fd', 'pass_fd')
                copy_alias('rush_fd', 'rush_fd')
                copy_alias('rec_fd', 'rec_fd')
                copy_alias('sack_taken', 'sack_taken')
                copy_alias('sacked', 'sack_taken')
                # Specific two-point conversions
                copy_alias('pass_2pt', 'pass_2pt')
                copy_alias('rec_2pt', 'rec_2pt')
                copy_alias('rush_2pt', 'rush_2pt')

                # Aggregate any 2pt conversion style keys (e.g., rush_2pt, pass_2pt, rec_2pt, two_pt)
                two_pt_total = 0.0
                if isinstance(stat_block, dict):
                    for ksrc, kval in stat_block.items():
                        if '2pt' in ksrc.lower():
                            try:
                                two_pt_total += float(kval)
                            except Exception:
                                pass
                if two_pt_total:
                    stats['two_pt'] = two_pt_total
                # If bucket bonuses used but no buckets and assume_buckets=='infer', attempt inference
                if not any_bucket_present and assume_buckets == 'infer' and 'rec' in stats and 'rec_yds' in stats:
                    from .sleeper_mapper import infer_buckets_from_rec_and_yards, SLEEPER_BUCKET_KEYS
                    buckets = infer_buckets_from_rec_and_yards(int(stats.get('rec',0)), float(stats.get('rec_yds',0)))
                    for bk, val in buckets.items():
                        stats[bk] = val
                if stats:
                    # Always include in the all-players map
                    parsed_all.setdefault(name, {})[wk_key] = stats
                    # Include in rostered-only map if player was on a roster that week
                    if not roster_player_ids or (pid and pid in roster_player_ids):
                        parsed_rostered.setdefault(name, {})[wk_key] = stats
            # Use rostered-only map for the main report
            parsed = parsed_rostered
            # Stash the all-players parsed for FA view later (mapped below)
            parsed_all_for_meta = parsed_all
            used_source = 'sleeper'
        except Exception as e:
            # If sleeper also fails, we'll return empty parsed
            pass
    mapped_full = {}
    for pname, weeks in parsed.items():
        mapped_full[pname] = {}
        for wk, stats in weeks.items():
            # map full (includes inferred buckets)
            mapped_full[pname][wk] = map_pbp_stats_to_sleeper_full(stats)
    # Also build a full mapped view for all players (including free agents) when available
    mapped_full_all = None
    if 'parsed_all_for_meta' in locals() and isinstance(parsed_all_for_meta, dict):
        mapped_full_all = {}
        for pname, weeks in parsed_all_for_meta.items():
            mapped_full_all[pname] = {}
            for wk, stats in weeks.items():
                mapped_full_all[pname][wk] = map_pbp_stats_to_sleeper_full(stats)

    # build report
    report: Dict[str, Any] = {}
    # allow overrides file at ./overrides.json if present (used only in non-weekly roster mode)
    overrides = {}
    reverse_overrides = {}  # sleeper name -> canonical desired display name
    override_file_candidates = [overrides_path, 'overrides.json'] if overrides_path else ['overrides.json']
    for ov_path in override_file_candidates:
        try:
            with open(ov_path, 'r', encoding='utf-8') as fh:
                raw_ov = json.load(fh)
                for canonical, sleeper_name in raw_ov.items():
                    overrides[nm_normalize(canonical)] = sleeper_name
                    reverse_overrides[sleeper_name] = canonical
        except Exception:
            continue

    starters_names_by_team: Dict[str, set] = {}
    players_names_by_team: Dict[str, set] = {}
    matchups_by_id: Dict[str, list] = {}
    if week is not None:
        # Build weekly report using matchups, avoiding fuzzy matching and excluding free agents
        roster_id_to_owner: Dict[int, str] = {}
        for r in rosters or []:
            rid = r.get('roster_id')
            if rid is not None:
                owner_id = r.get('owner_id')
                roster_id_to_owner[rid] = str(owner_id) if owner_id is not None else str(rid)
        try:
            matchups = client.get_matchups(league_id, week)
        except Exception:
            matchups = []
        for m in matchups or []:
            rid = m.get('roster_id')
            owner_id = roster_id_to_owner.get(rid)
            team_id = owner_id or str(rid)
            if team_id is None:
                team_id = 'unknown'
            manager = owner_to_manager.get(str(owner_id)) if owner_id is not None else None
            team_name = owner_to_team_name.get(str(owner_id)) if owner_id is not None else None
            report.setdefault(team_id, {'players': {}, 'manager': manager, 'team_name': team_name})
            starters_set = starters_names_by_team.setdefault(team_id, set())
            players_set = players_names_by_team.setdefault(team_id, set())
            # Track matchup pairing
            mid = m.get('matchup_id')
            if mid is None:
                mid = 'unknown'
            try:
                mid_str = str(mid)
            except Exception:
                mid_str = 'unknown'
            lst = matchups_by_id.setdefault(mid_str, [])
            if team_id not in lst:
                lst.append(team_id)
            for pid in (m.get('players') or []):
                if not pid:
                    continue
                pdata = players_map.get(pid, {})
                fullname = pdata.get('full_name') or pdata.get('name') or pid
                # Attach stats only if present in mapped_full; else leave empty mapping
                stats_weeks = mapped_full.get(fullname, {})
                report[team_id]['players'][fullname] = stats_weeks
                players_set.add(fullname)
            for pid in (m.get('starters') or []):
                if not pid:
                    continue
                pdata = players_map.get(pid, {})
                fullname = pdata.get('full_name') or pdata.get('name') or pid
                starters_set.add(fullname)
    else:
        # Season/aggregate mode: use current rosters and fuzzy matching
        for roster in rosters or []:
            team_id = roster.get('owner_id') or roster.get('roster_id') or str(roster.get('team_id'))
            manager = owner_to_manager.get(str(team_id))
            team_name = owner_to_team_name.get(str(team_id))
            report.setdefault(team_id, {'players': {}, 'manager': manager, 'team_name': team_name})
            for pid in roster.get('players', []) or []:
                pdata = players_map.get(pid, {})
                fullname = pdata.get('full_name') or pdata.get('name') or pid
                candidates = list(mapped_full.keys())
                match = find_best_match(fullname, candidates, overrides=overrides, threshold=match_threshold)
                if match:
                    display_name = reverse_overrides.get(match, fullname)
                    report[team_id]['players'][display_name] = mapped_full[match]
                else:
                    report[team_id]['players'][fullname] = {}

    # build scoring rules (priority: explicit path > league scoring_settings > sample fallback)
    scoring_rules = None
    if scoring_path:
        try:
            with open(scoring_path, 'r', encoding='utf-8') as fh:
                scoring_rules = json.load(fh)
        except Exception as e:
            scoring_rules = None
    if scoring_rules is None and league_info and isinstance(league_info, dict):
        league_sc = league_info.get('scoring_settings') or {}
        scoring_rules = _transform_scoring(league_sc)
    if scoring_rules is None:
        scoring_rules = {
            'pass_yds': 0.04,
            'pass_tds': 4,
            'int': -1,  # align with provided league data (-1 per INT)
            'rush_yds': 0.1,
            'rush_tds': 6,
            'rec': 0,
            'rec_yds': 0.1,
            'rec_tds': 6,
            'rush_att': 0.5,  # league uses 0.5 per rush attempt (all positions observed)
            'two_pt': 2,
        }
    engine = ScoringEngine(ScoringRules(rules=scoring_rules))

    # compute points per player name aggregated across weeks (using mapped stats)
    player_points: Dict[str, float] = {}
    # build name->position from players_map
    name_pos = {}
    for pid, pdata in players_map.items():
        fullname = pdata.get('full_name') or pdata.get('name')
        if fullname:
            name_pos[fullname] = pdata.get('position') or pdata.get('pos')
    for team_id, tdata in report.items():
        for pname, weeks in tdata['players'].items():
            total = 0.0
            pos = name_pos.get(pname)
            for wk, wkstats in weeks.items():
                if not wkstats:
                    continue
                stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                if pos:
                    stats_numeric['_pos'] = pos
                pw = PlayerWeek(player_id=pname, week=int(wk), stats=stats_numeric)
                total += engine.score_player_week(pw)
            player_points[pname] = total

    # Build convenience map for the selected week across all players (for FA view purposes)
    week_all_player_stats: Dict[str, Dict[str, float]] = {}
    # Prefer the all-players mapped view if available; else fall back to mapped_full
    _source_for_week_all = mapped_full_all if mapped_full_all is not None else mapped_full
    if isinstance(_source_for_week_all, dict) and week is not None:
        for pname, weeks in _source_for_week_all.items():
            st = weeks.get(str(week))
            if isinstance(st, dict):
                # ensure numeric-only for scoring
                try:
                    week_all_player_stats[pname] = {k: float(v) for k, v in st.items() if isinstance(v, (int, float))}
                except Exception:
                    pass

    meta = {
        'scoring_rules': scoring_rules,
        'used_source': used_source,
        'league_info': league_info,
        'name_pos': name_pos,
        'starters_names_by_team': starters_names_by_team,
        'players_names_by_team': players_names_by_team,
        'owner_to_manager': owner_to_manager,
        'owner_to_team_name': owner_to_team_name,
        'matchups_by_id': matchups_by_id,
        'roster_id_to_owner': {str(k): v for k, v in (locals().get('roster_id_to_owner') or {}).items()},
        'week_all_player_stats': week_all_player_stats,
    }
    return report, player_points, meta


def import_sleeper_league(league_id: str, season: int, week: int | None, infer: bool, match_threshold: float = 0.78, source: str = 'auto', scoring_path: str | None = None, assume_buckets: str = 'infer', cache_dir: str | None = None, debug_source: bool = False, overrides_path: str | None = None, explain: bool = False):
    print(f"Fetching players and rosters for league {league_id}...")
    try:
        report, player_points, meta = build_report(
            league_id,
            season,
            week,
            infer,
            match_threshold=match_threshold,
            source=source,
            scoring_path=scoring_path,
            assume_buckets=assume_buckets,
            cache_dir=cache_dir,
            debug_source=debug_source,
            overrides_path=overrides_path,
        )
    except Exception as e:
        print(f"Error building report: {e}")
        report, player_points, meta = {}, {}, {'scoring_rules': {}, 'used_source': None, 'league_info': None, 'name_pos': {}}

    used_source = meta.get('used_source')
    if used_source:
        print(f"Data source used: {used_source}")

    # print summary table
    print('SUMMARY_TABLE_START')
    header = f"{'Player':30} {'Team':10} {'Weeks':5} {'Points':8}"
    print(header)
    print('-' * len(header))
    for team_id, tdata in report.items():
        for pname, weeks in tdata['players'].items():
            weeks_played = len([1 for _wk, _st in weeks.items() if _st])
            pts = player_points.get(pname, 0.0)
            print(f"{pname:30} {team_id:10} {weeks_played:5d} {pts:8.2f}")

    # print JSON report with sentinel so tests can reliably capture
    print('REPORT_JSON_START')
    print(json.dumps(report, indent=2))
    if explain:
        print('EXPLAIN_START')
        for team_id, tdata in report.items():
            for pname, weeks in tdata['players'].items():
                for wk, wkstats in weeks.items():
                    if not wkstats:
                        continue
                    stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                    # attach position if known for TE bonus
                    pos = meta.get('name_pos', {}).get(pname)
                    if pos:
                        stats_numeric['_pos'] = pos
                    breakdown = score_breakdown(stats_numeric, meta.get('scoring_rules', {}))
                    total_from_breakdown = sum(breakdown.values())
                    print(f"{pname} W{wk}: {total_from_breakdown:.2f} -> " + ', '.join(f"{k}={v:.2f}" for k,v in sorted(breakdown.items())))
        print('EXPLAIN_END')
    # Explanation sentinel appended only if explain flag used later (implemented in main parser)

from typing import Optional

def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(prog='fantasy_toolkit')
    sub = parser.add_subparsers(dest='cmd')

    imp = sub.add_parser('import-sleeper')
    imp.add_argument('--league', required=True, help='Sleeper league_id')
    imp.add_argument('--season', type=int, required=True, help='Season year to fetch PBP for')
    imp.add_argument('--week', type=int, default=None, help='Optional single week to import')
    imp.add_argument('--no-infer', dest='infer', action='store_false', help='Disable bucket inference')
    imp.add_argument('--match-threshold', type=float, default=0.78, help='Fuzzy match threshold for player name matching (0-1, default 0.78)')
    imp.add_argument('--source', choices=['auto','pbp','sleeper'], default='auto', help='Data source preference: pbp full play-by-play, sleeper weekly stats, or auto fallback (default).')
    imp.add_argument('--scoring', dest='scoring_path', help='Path to custom scoring JSON (overrides league scoring).')
    imp.add_argument('--assume-buckets', choices=['infer','none'], default='infer', help='When using Sleeper weekly stats without buckets, infer bucket counts or leave absent.')
    imp.add_argument('--cache-dir', help='Directory to cache downloaded data')
    imp.add_argument('--debug-source', action='store_true', help='Print raw weekly stats source snippet when using sleeper source')
    imp.add_argument('--overrides', dest='overrides_path', help='Path to name overrides JSON')
    imp.add_argument('--explain', action='store_true', help='Print per-player stat contribution breakdown after report')
    imp.set_defaults(infer=True)

    args = parser.parse_args(argv)
    if args.cmd == 'import-sleeper':
        import_sleeper_league(
            args.league,
            args.season,
            args.week,
            args.infer,
            match_threshold=args.match_threshold,
            source=args.source,
            scoring_path=args.scoring_path,
            assume_buckets=args.assume_buckets,
            cache_dir=args.cache_dir,
            debug_source=args.debug_source,
            overrides_path=args.overrides_path,
            explain=args.explain,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
