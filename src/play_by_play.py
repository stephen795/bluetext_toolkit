import requests
import gzip
import io
import csv
from typing import Dict, Optional, Any

PBP_BASE_RAW = 'https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/data'


class PlayByPlayError(Exception):
    pass


def fetch_pbp_gz_for_season(season: int) -> bytes:
    """Fetch the season play-by-play gz CSV from nflfastR-data GitHub.

    Returns raw bytes of the gz file.
    """
    url = f"{PBP_BASE_RAW}/play_by_play_{season}.csv.gz"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        raise PlayByPlayError(f"Failed to fetch pbp for {season}: {resp.status_code}")
    return resp.content


def parse_pbp_bytes_to_player_week_stats(pbp_bytes: bytes, week: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Parse a nflfastR play_by_play CSV (gz bytes) and aggregate basic per-player stats by week.

    Returns mapping of player_key -> {week: {stat_key: value}}

    player_key will be the player's name from the play-by-play file (e.g., 'Tom Brady').
    The function attempts to find common columns used by nflfastR CSVs but will gracefully skip missing columns.
    """
    result: Dict[str, Dict[str, Any]] = {}
    bio = io.BytesIO(pbp_bytes)
    with gzip.open(bio, mode='rt', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                row_week = int(row.get('week') or 0)
            except Exception:
                row_week = 0
            row_week_key = str(row_week)
            if week is not None and row_week != week:
                continue

            # helpers to accumulate
            def add_stat(player_name: Optional[str], stat_key: str, amount: float):
                if not player_name:
                    return
                pk = player_name.strip()
                if pk == '':
                    return
                wk_stats = result.setdefault(pk, {}).setdefault(row_week_key, {})
                wk_stats[stat_key] = wk_stats.get(stat_key, 0.0) + amount

            # Passing
            passer = row.get('passer_player_name') or row.get('passer') or row.get('passer_player')
            # yards: common names
            pass_yards = None
            for k in ('passing_yards', 'pass_yards', 'pass_yds', 'yds_gained', 'yards_gained'):
                v = row.get(k)
                if v:
                    try:
                        pass_yards = float(v)
                        break
                    except Exception:
                        pass
            if passer and pass_yards:
                add_stat(passer, 'pass_yds', pass_yards)

            # pass TD
            pass_td_flag = None
            for k in ('pass_touchdown', 'pass_td', 'pass_touchdowns'):
                v = row.get(k)
                if v in ('1', 'True', 'true', 'TRUE', 't'):
                    pass_td_flag = True
                    break
            if passer and pass_td_flag:
                add_stat(passer, 'pass_tds', 1)

            # Interception (credited to passer)
            int_flag = None
            for k in ('interception', 'int', 'interception_player_id'):
                v = row.get(k)
                if v in ('1', 'True', 'true', 'TRUE', 't'):
                    int_flag = True
                    break
            if passer and int_flag:
                add_stat(passer, 'int', 1)

            # Rushing
            rusher = row.get('rusher_player_name') or row.get('rusher')
            if rusher:
                # count every rushing attempt (even 0 yard) for attempt-based scoring
                add_stat(rusher, 'rush_att', 1)
            rush_yards = None
            for k in ('rushing_yards', 'rush_yards', 'rush_yds', 'yds_rushed', 'yds_gained'):
                v = row.get(k)
                if v:
                    try:
                        rush_yards = float(v)
                        break
                    except Exception:
                        pass
            if rusher and rush_yards:
                add_stat(rusher, 'rush_yds', rush_yards)
            # rush TD
            rush_td_flag = None
            for k in ('rush_touchdown', 'rush_td', 'rushing_touchdown'):
                v = row.get(k)
                if v in ('1', 'True', 'true', 'TRUE', 't'):
                    rush_td_flag = True
                    break
            if rusher and rush_td_flag:
                add_stat(rusher, 'rush_tds', 1)

            # Receiving
            receiver = row.get('receiver_player_name') or row.get('receiver')
            rec_yards = None
            for k in ('receiving_yards', 'rec_yards', 'rec_yds', 'yds_receiving'):
                v = row.get(k)
                if v:
                    try:
                        rec_yards = float(v)
                        break
                    except Exception:
                        pass
            if receiver and rec_yards:
                add_stat(receiver, 'rec_yds', rec_yards)

            # reception count
            reception = None
            for k in ('reception', 'receptions', 'complete_pass'):
                v = row.get(k)
                if v and v not in ('0', 'False', 'false', ''):
                    try:
                        # some CSVs use 1/0
                        if v in ('1', 'True', 'true', 'TRUE', 't'):
                            reception = 1
                        else:
                            reception = float(v)
                        break
                    except Exception:
                        pass
            if receiver and reception:
                add_stat(receiver, 'rec', reception)

                # also record per-catch yardage bucket if yards for that reception exist
                # Many pbp CSVs include 'yards_gained' or 'complete_pass' yards in the same row
                yards_for_play = None
                for yk in ('yards_gained', 'yards', 'complete_pass_yards', 'rec_yards', 'receiving_yards'):
                    vy = row.get(yk)
                    if vy:
                        try:
                            yards_for_play = float(vy)
                            break
                        except Exception:
                            pass
                if yards_for_play is not None:
                    # decide bucket
                    bucket_key = None
                    if yards_for_play <= 4:
                        bucket_key = 'rec_0_4'
                    elif yards_for_play <= 9:
                        bucket_key = 'rec_5_9'
                    elif yards_for_play <= 19:
                        bucket_key = 'rec_10_19'
                    elif yards_for_play <= 29:
                        bucket_key = 'rec_20_29'
                    elif yards_for_play <= 39:
                        bucket_key = 'rec_30_39'
                    else:
                        bucket_key = 'rec_40p'
                    add_stat(receiver, bucket_key, 1)

            # rec TD
            rec_td_flag = None
            for k in ('rec_touchdown', 'rec_td', 'receiving_touchdown'):
                v = row.get(k)
                if v in ('1', 'True', 'true', 'TRUE', 't'):
                    rec_td_flag = True
                    break
            if receiver and rec_td_flag:
                add_stat(receiver, 'rec_tds', 1)

            # return TDs (kickoff/punt returns)
            returner = row.get('returner_player_name') or row.get('returner')
            return_td_flag = None
            for k in ('return_touchdown', 'return_td'):
                v = row.get(k)
                if v in ('1', 'True', 'true', 'TRUE', 't'):
                    return_td_flag = True
                    break
            if returner and return_td_flag:
                add_stat(returner, 'return_tds', 1)

    return result


def pbp_to_playerweeks(season: int, week: Optional[int] = None):
    """High-level convenience: fetch pbp for season, parse, and return list of (player_name, week, stats) tuples."""
    raw = fetch_pbp_gz_for_season(season)
    parsed = parse_pbp_bytes_to_player_week_stats(raw, week=week)
    out = []
    for player_name, weeks in parsed.items():
        for wk, stats in weeks.items():
            out.append((player_name, wk, stats))
    return out
