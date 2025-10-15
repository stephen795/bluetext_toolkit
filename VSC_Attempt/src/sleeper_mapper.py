"""Map play-by-play bucket stats to Sleeper scoring category keys.

This module provides a small function to convert PBP-derived bucket counts
and other stats into the Sleeper-style category keys such as `rec_0_4`,
`rec_30_39`, `rec_40p`, and also counts for `rec_10p` style aggregates if desired.
"""
from typing import Dict, Any, Optional, Mapping


SLEEPER_BUCKET_KEYS = [
    'rec_0_4', 'rec_5_9', 'rec_10_19', 'rec_20_29', 'rec_30_39', 'rec_40p'
]


def map_pbp_stats_to_sleeper(player_week_stats: Dict[str, Any]) -> Dict[str, int]:
    """Given a player-week PBP stats dict, return a dict with Sleeper category counts.

    Input example: {'rec':6, 'rec_yds':120, 'rec_30_39':1, 'rec_40p':1}
    Output example: {'rec_0_4':0, 'rec_5_9':0, 'rec_10_19':0, 'rec_20_29':0, 'rec_30_39':1, 'rec_40p':1}
    """
    out: Dict[str, int] = {k: 0 for k in SLEEPER_BUCKET_KEYS}
    for k in SLEEPER_BUCKET_KEYS:
        if k in player_week_stats:
            try:
                out[k] = int(player_week_stats.get(k, 0))
            except Exception:
                out[k] = 0

    # Some providers report aggregated keys like rec_40p directly under a different naming.
    # We also ensure the total rec (count) matches the sum of buckets if available.
    # If rec is present but buckets aren't, we leave buckets as zero — caller can choose how to handle.

    # If buckets are all zero but rec and rec_yds exist, infer a distribution
    if sum(out.values()) == 0:
        try:
            rec = int(player_week_stats.get('rec', 0) or 0)
            rec_yds = float(player_week_stats.get('rec_yds', 0) or 0)
            if rec > 0 and rec_yds >= 0:
                inferred = infer_buckets_from_rec_and_yards(rec, rec_yds)
                out.update(inferred)
        except Exception:
            pass

    return out


def infer_buckets_from_rec_and_yards(rec: int, rec_yds: float, priors: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Infer per-reception bucket counts given total rec and rec_yds.

    Heuristic method:
      - Start with prior proportions for each bucket (league-average priors)
      - Compute expected yards from these priors using bucket mean yards
      - Scale counts so expected yards match rec_yds
      - Round counts to integers while keeping total rec

    Returns mapping bucket_key -> int count that sum to `rec`.
    """
    # default priors (approximate typical distribution)
    default_priors = {
        'rec_0_4': 0.10,
        'rec_5_9': 0.20,
        'rec_10_19': 0.30,
        'rec_20_29': 0.20,
        'rec_30_39': 0.12,
        'rec_40p': 0.08,
    }
    pri = priors or default_priors

    # mean yards per bucket (representative)
    means = {
        'rec_0_4': 2.0,
        'rec_5_9': 7.0,
        'rec_10_19': 14.5,
        'rec_20_29': 24.5,
        'rec_30_39': 34.5,
        'rec_40p': 50.0,
    }

    # initial fractional counts
    frac_counts = {k: pri.get(k, 0) * rec for k in SLEEPER_BUCKET_KEYS}

    # estimated yards
    est_yards = sum(frac_counts[k] * means[k] for k in SLEEPER_BUCKET_KEYS)
    if est_yards <= 0:
        # fallback: equal split
        base = rec // len(SLEEPER_BUCKET_KEYS)
        res = {k: base for k in SLEEPER_BUCKET_KEYS}
        rem = rec - base * len(SLEEPER_BUCKET_KEYS)
        i = 0
        keys = SLEEPER_BUCKET_KEYS
        while rem > 0:
            res[keys[i]] += 1
            rem -= 1
            i = (i + 1) % len(keys)
        return res

    scale = rec_yds / est_yards if est_yards > 0 else 1.0
    adj_counts = {k: frac_counts[k] * scale for k in SLEEPER_BUCKET_KEYS}

    # round while preserving sum == rec
    floored = {k: int(adj_counts[k]) for k in SLEEPER_BUCKET_KEYS}
    current_sum = sum(floored.values())
    rem = rec - current_sum

    # greedy allocate leftover counts (or remove extras) to minimize yardage error
    def estimated_yards(counts: Mapping[str, float | int]) -> float:
        return sum(counts[k] * means[k] for k in SLEEPER_BUCKET_KEYS)

    # start with floored counts
    counts = floored.copy()

    # if we need to add counts, choose bucket that reduces |est - rec_yds| most
    while rem > 0:
        best_k = None
        best_err = None
        for k in SLEEPER_BUCKET_KEYS:
            trial = counts.copy()
            trial[k] += 1
            err = abs(estimated_yards(trial) - rec_yds)
            if best_err is None or err < best_err:
                best_err = err
                best_k = k
        if best_k is None:
            break
        counts[best_k] += 1
        rem -= 1

    # if we have too many (rem < 0), remove counts from buckets that improve error
    while rem < 0:
        best_k = None
        best_err = None
        for k in SLEEPER_BUCKET_KEYS:
            if counts[k] <= 0:
                continue
            trial = counts.copy()
            trial[k] -= 1
            err = abs(estimated_yards(trial) - rec_yds)
            if best_err is None or err < best_err:
                best_err = err
                best_k = k
        if best_k is None:
            break
        counts[best_k] -= 1
        rem += 1

    # final sanity: if rounding left any gap due to ties, distribute evenly
    final_sum = sum(counts.values())
    i = 0
    while final_sum < rec:
        k = SLEEPER_BUCKET_KEYS[i % len(SLEEPER_BUCKET_KEYS)]
        counts[k] += 1
        final_sum += 1
        i += 1
    while final_sum > rec:
        # remove from smallest bucket with count>0
        for k in reversed(SLEEPER_BUCKET_KEYS):
            if counts[k] > 0:
                counts[k] -= 1
                final_sum -= 1
                break

    return counts


def map_all_players_pbp_to_sleeper(parsed_pbp: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Convert parsed PBP mapping (player_name -> week_key -> stats) into
    mapping player_name -> week_key -> sleeper category counts.
    """
    res: Dict[str, Dict[str, Dict[str, int]]] = {}
    for player_name, weeks in parsed_pbp.items():
        res[player_name] = {}
        for wk, stats in weeks.items():
            res[player_name][wk] = map_pbp_stats_to_sleeper(stats)
    return res


def map_pbp_stats_to_sleeper_full(player_week_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Map PBP-derived player-week stats into an expanded set of Sleeper scoring
    categories and base numeric stats so scoring can be computed even when the
    Sleeper API hasn't provided categories yet.

    Returns a dict that includes:
      - base stats: rec, rec_yds, rec_tds, rush_yds, rush_tds, pass_yds, pass_tds, int, fumbles, two_pt
      - bucket keys: rec_0_4 ... rec_40p
      - aggregated bucket keys: rec_10p (>=10), rec_20p (>=20)
    """
    out: Dict[str, Any] = {}

    # Generic pass-through: copy any primitive numeric stat so scoring rules can apply to
    # the wide range of Sleeper categories (kicker, defense, IDP, bonuses, etc.). We still
    # provide explicit buckets and aggregated categories below.
    for k, v in player_week_stats.items():
        if isinstance(v, (int, float)):
            out[k] = v

    # Normalize common aliases to engine-friendly keys
    def _copy_alias(src: str, dest: str):
        if dest in out:
            return
        val = player_week_stats.get(src)
        if isinstance(val, (int, float)):
            out[dest] = val
    # pass completions/attempts
    _copy_alias('cmp', 'pass_cmp')
    _copy_alias('complete_pass', 'pass_cmp')
    _copy_alias('completions', 'pass_cmp')
    _copy_alias('pass_comp', 'pass_cmp')
    _copy_alias('pass_completions', 'pass_cmp')
    _copy_alias('comp', 'pass_cmp')
    _copy_alias('passcomp', 'pass_cmp')
    _copy_alias('att', 'pass_att')
    # sacks taken
    _copy_alias('sacked', 'sack_taken')

    # buckets
    buckets = map_pbp_stats_to_sleeper(player_week_stats)
    out.update(buckets)

    # aggregated categories
    try:
        rec_10p = 0
        rec_20p = 0
        for bk, val in buckets.items():
            if not isinstance(val, int):
                try:
                    val = int(val)
                except Exception:
                    val = 0
            # parse bucket name to decide inclusion
            if bk == 'rec_10_19' or bk == 'rec_10_19':
                pass
            # easier: map by ranges
            if bk in ('rec_10_19', 'rec_20_29', 'rec_30_39', 'rec_40p'):
                rec_10p += val
            if bk in ('rec_20_29', 'rec_30_39', 'rec_40p'):
                rec_20p += val
        out['rec_10p'] = rec_10p
        out['rec_20p'] = rec_20p
    except Exception:
        out['rec_10p'] = 0
        out['rec_20p'] = 0

    return out
