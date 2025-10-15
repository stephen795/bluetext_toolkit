import os
import math
import itertools as it
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import streamlit as st
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # fallback handled later

from src.cli import build_report, score_breakdown
from src.sleeper_api import SleeperClient

# Simple, separate Streamlit tool to analyze trade values and recommend trade packages.
# Leverages existing Sleeper integration and scoring engine from this repo.

CACHE_VERSION = "v1-trade-tool"
DEFAULT_LEAGUE = os.environ.get("FT_LEAGUE_ID", "1260098143609966592")
DEFAULT_SEASON = int(os.environ.get("FT_SEASON", "2025"))
DEFAULT_WEEK = int(os.environ.get("FT_WEEK", "6"))

st.set_page_config(page_title="Trade Analyzer", page_icon="🔁", layout="wide", initial_sidebar_state="expanded")

# Theme-light CSS
st.markdown(
    """
    <style>
    h2, h3, h4 { margin-top: 0.6rem; }
    .stDataFrame table { font-size: 0.92rem; }
    .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_week(league_id: str, season: int, week: Optional[int], source: str, cache_dir: str):
    return build_report(
        league_id, season, week,
        infer=True,
        match_threshold=0.78,
        source=source,
        scoring_path=None,
        assume_buckets='infer',
        cache_dir=cache_dir,
        debug_source=False,
        overrides_path=None,
    )

# Determine last completed week (matchups present and positive PF)
def detect_last_completed_week(league_id: str, season: int, source: str, cache_dir: str) -> Optional[int]:
    last = None
    for w in range(1, 19):
        try:
            r, pp, m = load_week(league_id, season, w, source, cache_dir)
        except Exception:
            continue
        matchups_by_id = (m or {}).get('matchups_by_id', {}) or {}
        try:
            pf_total = sum(float(v) for v in (pp or {}).values())
        except Exception:
            pf_total = 0.0
        if matchups_by_id and pf_total > 0:
            last = w
    return last

# Aggregate season-to-date player points (sum), games played count, and weekly series for volatility
@st.cache_data(show_spinner=False)
def aggregate_points_season_to_date(league_id: str, season: int, upto_week: int, source: str, cache_dir: str):
    cum_points: Dict[str, float] = {}
    gp: Dict[str, int] = {}
    weekly_points: Dict[str, List[float]] = {}
    meta_last = None
    for w in range(1, upto_week + 1):
        r_w, pp_w, m_w = load_week(league_id, season, w, source, cache_dir)
        meta_last = m_w
        for pname, pts in (pp_w or {}).items():
            val = float(pts or 0.0)
            if val != 0.0:
                gp[pname] = gp.get(pname, 0) + 1
            cum_points[pname] = cum_points.get(pname, 0.0) + val
            weekly_points.setdefault(pname, []).append(val)
    return cum_points, gp, (meta_last or {}), weekly_points


def _derive_lineup_slots(league_info: Optional[dict]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Parse Sleeper league roster_positions into per-position starter counts and raw slot counts.

    Returns:
      (pos_slots_effective, raw_counts)
      - pos_slots_effective: float counts for QB/RB/WR/TE where FLEX is split evenly across RB/WR/TE.
      - raw_counts: raw counts by slot: {'QB':n, 'RB':n, 'WR':n, 'TE':n, 'FLEX':n, 'SUPER_FLEX':n}
    """
    raw = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'SUPER_FLEX': 0}
    try:
        positions = (league_info or {}).get('roster_positions') or []
        for p in positions:
            p = str(p).upper()
            if p in raw:
                raw[p] += 1
            elif p in ('W/R/T', 'WRRBTE'):
                raw['FLEX'] += 1
            elif p in ('Q/W/R/T', 'SUPERFLEX', 'SUPER-FLEX', 'SUPER_FLEX'):
                raw['SUPER_FLEX'] += 1
            else:
                # ignore DEF/K/IDP/BN/IR
                continue
    except Exception:
        pass
    # Distribute FLEX across RB/WR/TE equally; SUPER_FLEX across QB and RB/WR/TE equally (approximate)
    flex_share = raw['FLEX'] / 3.0
    sflex_share_qb = raw['SUPER_FLEX'] / 4.0
    sflex_share_skill = (raw['SUPER_FLEX'] - sflex_share_qb) / 3.0 if raw['SUPER_FLEX'] else 0.0
    pos_slots = {
        'QB': float(raw['QB']) + float(sflex_share_qb),
        'RB': float(raw['RB']) + float(flex_share) + float(sflex_share_skill),
        'WR': float(raw['WR']) + float(flex_share) + float(sflex_share_skill),
        'TE': float(raw['TE']) + float(flex_share) + float(sflex_share_skill),
    }
    return pos_slots, raw

@dataclass
class ValueConfig:
    mode: str  # 'win_now' | 'balanced' | 'dynasty'
    use_per_game: bool
    scarcity_method: str  # 'nth_starter' | 'median_starter'
    start_slots_per_team: Dict[str, int]
    longevity_multiplier: Dict[str, float]


def compute_value_board(
    league_id: str,
    season: int,
    week: Optional[int],
    source: str,
    cache_dir: str,
    mode: str = 'balanced',
    use_per_game: bool = True,
    scarcity_method: str = 'nth_starter',
    z_boost: float = 0.25,
    vol_weight: float = 0.15,
    apply_injury: bool = True,
    short_injury_weight: float = 0.4,
    season_injury_weight: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return a DataFrame with player values and a dict of replacement baselines by position."""
    # Determine scope (up to selected week or full season as of last completed)
    if week is None:
        last = detect_last_completed_week(league_id, season, source, cache_dir) or 1
        upto = int(last)
    else:
        upto = int(week)

    cum_points, gp, meta_last, weekly_points = aggregate_points_season_to_date(league_id, season, upto, source, cache_dir)
    name_pos = (meta_last or {}).get('name_pos', {}) or {}
    # Injury status map from Sleeper players endpoint (best-effort)
    injury_status_by_name: Dict[str, str] = {}
    injury_note_by_name: Dict[str, str] = {}
    if apply_injury:
        try:
            client = SleeperClient()
            players_map = client.get_players() or {}
            for _pid, pdata in players_map.items():
                try:
                    nm = pdata.get('full_name') or pdata.get('name')
                    if not nm:
                        continue
                    st1 = str(pdata.get('injury_status') or '').upper()
                    st2 = str(pdata.get('status') or '').upper()
                    # Prefer explicit injury_status when present
                    status = st1 or st2
                    if status:
                        injury_status_by_name[nm] = status
                    note = pdata.get('injury_notes') or pdata.get('notes') or ''
                    if isinstance(note, str) and note:
                        injury_note_by_name[nm] = note
                except Exception:
                    continue
        except Exception:
            injury_status_by_name = {}
            injury_note_by_name = {}
    # Known ambiguous display-name fixes
    POS_OVERRIDES = {
        'Lamar Jackson': 'QB',
        'Michael Carter': 'RB',
    }
    owner_to_team_name = (meta_last or {}).get('owner_to_team_name', {}) or {}
    owner_to_manager = (meta_last or {}).get('owner_to_manager', {}) or {}

    # Build a roster membership map to tag team labels
    players_by_team = (meta_last or {}).get('players_names_by_team', {}) or {}
    team_label_by_player: Dict[str, str] = {}
    for tid, names in players_by_team.items():
        team_name = owner_to_team_name.get(str(tid)) or str(tid)
        manager = owner_to_manager.get(str(tid), "")
        label = f"{team_name} ({manager})" if manager else team_name
        for n in names:
            team_label_by_player[n] = label

    # Compute per-game points if desired
    data_rows = []
    VALID_POS = {'QB','RB','WR','TE'}
    for pname, total in cum_points.items():
        pos = POS_OVERRIDES.get(pname) or name_pos.get(pname)
        if not pos:
            continue
        # Skip non-offensive positions to avoid CB/DB collisions on common names
        if pos not in VALID_POS:
            continue
        games = max(1, gp.get(pname, upto)) if use_per_game else 1
        ppg = float(total) / float(games)
        wp = weekly_points.get(pname, [])
        vol = float(np.std(wp, ddof=0)) if len(wp) >= 2 else 0.0
        rel_vol = float(vol / max(ppg, 1e-6)) if ppg > 0 else 0.0
        inj_status = injury_status_by_name.get(pname, '') if apply_injury else ''
        inj_note = injury_note_by_name.get(pname, '') if apply_injury else ''
        data_rows.append({
            'player': pname,
            'team': team_label_by_player.get(pname, ''),
            'pos': pos,
            'points_total': round(total, 2),
            'games': int(games if use_per_game else upto),
            'ppg': round(ppg, 3) if use_per_game else round(total, 2),
            'vol': round(vol, 3),
            'rel_vol': round(rel_vol, 3),
            'injury_status': inj_status,
            'injury_note': inj_note,
        })
    df = pd.DataFrame(data_rows)
    if df.empty:
        return df, {}

    # Determine replacement levels by position using starters count (last meta)
    starters_map = (meta_last or {}).get('starters_names_by_team', {}) or {}
    n_teams = max(1, len(starters_map) or int((meta_last or {}).get('league_info', {}).get('total_rosters') or 1))
    # Starter slots from Sleeper league roster_positions (split FLEX evenly RB/WR/TE)
    pos_slots_effective, raw_counts = _derive_lineup_slots((meta_last or {}).get('league_info', {}))
    # Sensible fallback if lineup unavailable
    if not any(pos_slots_effective.values()):
        # Fallback to requested config: QB, RB, RB, WR, WR, WR, TE, 3 FLEX (RB/WR/TE)
        # Effective starters with 3 FLEX distributed equally (1 each to RB/WR/TE): RB=3, WR=4, TE=2, QB=1
        pos_slots_effective = {'QB': 1.0, 'RB': 3.0, 'WR': 4.0, 'TE': 2.0}
    # For replacement index, use nth best equal to starters across league
    rep_index = {
        'QB': int(math.ceil(n_teams * float(pos_slots_effective.get('QB', 1.0)))),
        'RB': int(math.ceil(n_teams * float(pos_slots_effective.get('RB', 2.0)))),
        'WR': int(math.ceil(n_teams * float(pos_slots_effective.get('WR', 2.0)))),
        'TE': int(math.ceil(n_teams * float(pos_slots_effective.get('TE', 1.0)))),
    }

    replacement: Dict[str, float] = {}
    for pos in ['QB','RB','WR','TE']:
        s = df[df['pos']==pos]['ppg'].sort_values(ascending=False).reset_index(drop=True)
        if s.empty:
            replacement[pos] = 0.0
        else:
            if scarcity_method == 'median_starter':
                rep_val = s.head(rep_index.get(pos, len(s))).median() if len(s) >= 1 else 0.0
            else:  # nth_starter (clamped)
                idx = max(1, min(rep_index.get(pos, len(s)), len(s))) - 1
                rep_val = float(s.iloc[idx])
            replacement[pos] = float(rep_val)

    # Longevity multipliers for dynasty tilt (very rough without ages)
    longevity_mult = {
        'QB': 1.10,
        'WR': 1.06,
        'TE': 1.04,
        'RB': 0.92,
    }
    if mode == 'win_now':
        mode_mult = 1.0
        pos_mult = {p: 1.0 for p in ['QB','RB','WR','TE']}
    elif mode == 'dynasty':
        mode_mult = 0.95  # slightly discount current-season variance
        pos_mult = longevity_mult
    else:  # balanced
        mode_mult = 1.0
        pos_mult = {p: (1.0 + longevity_mult.get(p,1.0)) / 2.0 for p in ['QB','RB','WR','TE']}

    # Compute value components: VOR + z_boost*max(0,z_ppg) - vol_weight*rel_vol, with multipliers
    vals = []
    vors = []
    zlist = []
    rep_used = []
    def _injury_category(status: str, note: str) -> str:
        s = (status or '').upper()
        n = (note or '').lower()
        if not s:
            return 'none'
        # Season-long bucket: clear season-ending or reserve designations without return
        if s in ('IR','INJURED_RESERVE','NFI'):
            # If note indicates designated to return, treat as short
            if 'return' in n or 'designated' in n or 'eligible' in n:
                return 'short'
            return 'season'
        if s.startswith('PUP'):
            # PUP-R treated as short, plain PUP as season by default
            if '-R' in s or 'R' == s[-1] or 'return' in n or 'eligible' in n:
                return 'short'
            return 'season'
        # Suspensions typically finite; treat as short
        if s in ('SUS','SUSPENDED'):
            return 'short'
        # COVID and typical weekly statuses short-term
        if s in ('COVID','COVID-19','Q','QUESTIONABLE','D','DOUBTFUL','O','OUT'):
            return 'short'
        return 'none'

    def _injury_severity(status: str) -> float:
        if not status:
            return 0.0
        s = status.upper()
        # Map common Sleeper statuses/severities
        if s in ('IR', 'INJURED_RESERVE'):
            return 0.8
        if s in ('PUP', 'PUP-R', 'PUP-P'):
            return 0.7
        if s in ('OUT', 'O'):
            return 0.6
        if s in ('SUS', 'SUSPENDED'):
            return 0.6
        if s in ('NFI',):
            return 0.7
        if s in ('D', 'DOUBTFUL'):
            return 0.4
        if s in ('Q', 'QUESTIONABLE'):
            return 0.2
        # Active/Healthy/None
        return 0.0

    for _, r in df.iterrows():
        pos = r['pos']
        rep = replacement.get(pos, 0.0)
        rep_used.append(rep)
        vor = max(0.0, float(r['ppg']) - rep)
        # Normalize within position: z-score on ppg (fallback if std==0)
        pos_series = df[df['pos']==pos]['ppg']
        mu = float(pos_series.mean()) if len(pos_series) else 0.0
        sd = float(pos_series.std(ddof=0)) if len(pos_series) else 0.0
        z = (float(r['ppg']) - mu) / sd if sd > 1e-9 else 0.0
        z_nonneg = max(0.0, z)
        rel_vol = float(r.get('rel_vol', 0.0) or 0.0)
        raw_value = vor * 1.0 + z_nonneg * float(z_boost) - rel_vol * float(vol_weight)
        # Injury adjustment as multiplicative factor using separate weights per category
        if apply_injury:
            status = str(r.get('injury_status', '') or '')
            note = str(r.get('injury_note', '') or '')
            cat = _injury_category(status, note)
            sev = _injury_severity(status)
            if cat == 'short':
                w = float(short_injury_weight)
            elif cat == 'season':
                w = float(season_injury_weight)
            else:
                w = 0.0
            inj_factor = max(0.0, 1.0 - w * float(sev))
        else:
            inj_factor = 1.0
        adj_value = raw_value * mode_mult * pos_mult.get(pos, 1.0)
        adj_value = adj_value * inj_factor
        vals.append(adj_value)
        vors.append(vor)
        zlist.append(z)
    df['value'] = np.round(vals, 3)
    df['rep_ppg'] = np.round(rep_used, 3)
    df['vor'] = np.round(vors, 3)
    df['z_ppg'] = np.round(zlist, 3)
    if apply_injury:
        # Recompute injury_factor per row for display
        def _row_inj_factor(row):
            status = str(row.get('injury_status', '') or '')
            note = str(row.get('injury_note', '') or '')
            cat = _injury_category(status, note)
            sev = _injury_severity(status)
            w = float(short_injury_weight) if cat == 'short' else (float(season_injury_weight) if cat == 'season' else 0.0)
            return max(0.0, 1.0 - w * float(sev))
        try:
            df['injury_factor'] = df.apply(_row_inj_factor, axis=1).round(3)
        except Exception:
            df['injury_factor'] = 1.0
        try:
            df['injury_category'] = df.apply(lambda r: _injury_category(str(r.get('injury_status','') or ''), str(r.get('injury_note','') or '')), axis=1)
        except Exception:
            df['injury_category'] = 'none'
    # Rank within position and overall
    df['pos_rank'] = df.groupby('pos')['value'].rank(method='min', ascending=False).astype(int)
    df['rank'] = df['value'].rank(method='min', ascending=False).astype(int)
    df = df.sort_values(['value','ppg'], ascending=[False, False]).reset_index(drop=True)

    return df, replacement


def _team_options(meta: dict) -> List[Tuple[str, str]]:
    owner_to_team_name = (meta or {}).get('owner_to_team_name', {}) or {}
    owner_to_manager = (meta or {}).get('owner_to_manager', {}) or {}
    opts = []
    for tid, team_name in owner_to_team_name.items():
        manager = owner_to_manager.get(str(tid)) or ""
        label = f"{team_name} ({manager})" if manager else str(team_name)
        opts.append((str(tid), label))
    # Fallback: if map is empty, attempt to derive from report later
    return sorted(opts, key=lambda x: x[1])


def _roster_names_for(meta: dict, team_id: str) -> List[str]:
    players_map = (meta or {}).get('players_names_by_team', {}) or {}
    return list(players_map.get(team_id, []))


def _build_value_lookup(df_values: pd.DataFrame) -> Dict[str, float]:
    return {r['player']: float(r['value']) for _, r in df_values.iterrows()}


def propose_trades(
    user_team: str,
    target_team: str,
    meta: dict,
    values_df: pd.DataFrame,
    aggressiveness: str = 'safe',  # 'safe' | 'balanced' | 'aggressive'
    max_candidates: int = 12,
    include_two_for_two: bool = False,
) -> pd.DataFrame:
    """Generate simple 1-for-1 (and optional 2-for-2) trade ideas using value deltas and acceptance heuristics."""
    if user_team == target_team:
        return pd.DataFrame(columns=['send','receive','user_delta','target_delta','acceptance','notes'])

    vals = _build_value_lookup(values_df)
    user_roster = _roster_names_for(meta, user_team)
    tgt_roster = _roster_names_for(meta, target_team)

    # Take top-N candidates by value (both sides)
    user_cands = sorted([(p, vals.get(p, 0.0)) for p in user_roster], key=lambda x: x[1], reverse=True)[:max_candidates]
    tgt_cands = sorted([(p, vals.get(p, 0.0)) for p in tgt_roster], key=lambda x: x[1], reverse=True)[:max_candidates]

    if aggressiveness == 'safe':
        imbalance_pct = 0.02  # 2%
    elif aggressiveness == 'aggressive':
        imbalance_pct = 0.15
    else:
        imbalance_pct = 0.07

    proposals: List[Dict[str, object]] = []

    # Helper to score acceptance based on target delta
    def acceptance_label(target_delta: float) -> str:
        if target_delta >= 0:
            return 'Likely'
        # Allow small negative within threshold to still be 'Possible'
        if target_delta >= -0.5:
            return 'Possible'
        return 'Unlikely'

    # 1-for-1 trades
    for (u_name, u_val) in user_cands:
        for (t_name, t_val) in tgt_cands:
            user_delta = t_val - u_val
            target_delta = u_val - t_val
            # Acceptance: target should not be giving up too much relative to package size
            total = max(1e-9, (u_val + t_val) / 2.0)
            if target_delta < -imbalance_pct * total:
                continue
            proposals.append({
                'send': u_name,
                'receive': t_name,
                'user_delta': round(user_delta, 3),
                'target_delta': round(target_delta, 3),
                'acceptance': acceptance_label(target_delta),
                'notes': '1-for-1',
            })

    # Optional: 2-for-2 trades (limited search)
    if include_two_for_two:
        user_pairs = list(it.combinations(user_cands[:8], 2))  # limit branching
        tgt_pairs = list(it.combinations(tgt_cands[:8], 2))
        for (u1, u2) in user_pairs:
            for (t1, t2) in tgt_pairs:
                u_val = u1[1] + u2[1]
                t_val = t1[1] + t2[1]
                user_delta = t_val - u_val
                target_delta = u_val - t_val
                total = max(1e-9, (u_val + t_val) / 2.0)
                if target_delta < -imbalance_pct * total:
                    continue
                proposals.append({
                    'send': f"{u1[0]} + {u2[0]}",
                    'receive': f"{t1[0]} + {t2[0]}",
                    'user_delta': round(user_delta, 3),
                    'target_delta': round(target_delta, 3),
                    'acceptance': acceptance_label(target_delta),
                    'notes': '2-for-2',
                })

    df_prop = pd.DataFrame(proposals)
    if df_prop.empty:
        return df_prop
    # Sort by user benefit (desc) but show likely first within similar bands
    df_prop['accept_rank'] = df_prop['acceptance'].map({'Likely':0,'Possible':1,'Unlikely':2})
    df_prop = df_prop.sort_values(['accept_rank','user_delta'], ascending=[True, False]).drop(columns=['accept_rank']).reset_index(drop=True)
    return df_prop


# --- Assets: Picks and FAAB (experimental) ---
def fetch_team_assets(league_id: str, season: int, league_info: Optional[dict]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
    """Return (picks_by_team, faab_by_team) using actual Sleeper data.

    - FAAB: current remaining budget from roster settings (waiver_budget - waiver_budget_used), fallback to 'faab' if present.
    - Picks: Build a baseline of one pick per round per roster for seasons >= current season and next season (plus any seasons implied by trades),
      then apply /traded_picks to move picks to their current owners. Output keyed by owner_id with pick dicts including season, round, and original_roster_id.
    """
    picks_by_team: Dict[str, List[Dict[str, Any]]] = {}
    faab_by_team: Dict[str, float] = {}
    try:
        client = SleeperClient()
        rosters = client.get_rosters(league_id) or []
        roster_id_to_owner: Dict[int, str] = {}
        # League-level waiver budget (max) from league settings; fallback to 200 if present or 0 otherwise
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None
        league_settings = (league_info or {}).get('settings', {}) if isinstance(league_info, dict) else {}
        league_budget = _num((league_settings or {}).get('waiver_budget'))
        if league_budget is None:
            # Some leagues keep it directly on league_info
            league_budget = _num((league_info or {}).get('waiver_budget'))
        if league_budget is None:
            # Default assumption per your note: 200
            league_budget = 200.0
        for r in rosters:
            rid = r.get('roster_id')
            oid = r.get('owner_id')
            if rid is not None:
                roster_id_to_owner[int(rid)] = str(oid) if oid is not None else str(rid)
            # Compute FAAB remaining
            tid = str(oid) if oid is not None else (str(rid) if rid is not None else '')
            if not tid:
                continue
            settings = r.get('settings', {}) or {}
            # Compute remaining as league max - used when possible; otherwise use remaining fields
            try:
                used = _num(settings.get('waiver_budget_used'))
                remaining_direct = None
                for key in ['waiver_budget_remaining','waiver_remaining','waiver_balance','faab','waiver_budget']:
                    v = _num(settings.get(key))
                    if v is not None:
                        remaining_direct = v
                        break
                if used is not None and league_budget is not None:
                    faab_by_team[tid] = max(0.0, float(league_budget) - float(used))
                elif remaining_direct is not None:
                    faab_by_team[tid] = float(remaining_direct)
                else:
                    faab_by_team[tid] = 0.0
            except Exception:
                faab_by_team[tid] = 0.0

        # Traded picks from Sleeper
        try:
            import requests
            resp = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/traded_picks", timeout=10)
            traded = resp.json() if resp.ok else []
        except Exception:
            traded = []

        # Determine rounds and seasons of interest
        try:
            draft_rounds = int((league_info or {}).get('draft_rounds') or 4)
        except Exception:
            draft_rounds = 4
        # Only future picks should be tradable; exclude current season from baseline
        seasons_set = {int(season)+1, int(season)+2, int(season)+3}
        for p in traded or []:
            try:
                s_val = int(p.get('season') or 0)
                if s_val >= int(season)+1:
                    seasons_set.add(s_val)
            except Exception:
                continue
        seasons_of_interest = sorted([s for s in seasons_set if s >= int(season)+1])

        # Build baseline owner mapping for each pick (season, round, original_roster_id)
        # Initialize owner as the current roster owner (by roster_id_to_owner at time of call)
        pick_owner_by_key: Dict[Tuple[int, int, int], int] = {}
        roster_ids = [int(r.get('roster_id')) for r in rosters if r.get('roster_id') is not None]
        for s in seasons_of_interest:
            for rnd in range(1, draft_rounds+1):
                for rid in roster_ids:
                    pick_owner_by_key[(int(s), int(rnd), int(rid))] = int(rid)

        # Apply traded pick movements
        for p in traded or []:
            try:
                s = int(p.get('season'))
                rnd = int(p.get('round'))
                orig_rid = int(p.get('roster_id'))
                new_owner_rid = int(p.get('owner_id')) if p.get('owner_id') is not None else None
                if s < int(season) or rnd <= 0 or orig_rid is None or new_owner_rid is None:
                    continue
                key = (s, rnd, orig_rid)
                pick_owner_by_key[key] = new_owner_rid
            except Exception:
                continue

        # Aggregate into picks_by_team keyed by owner_id
        for (s, rnd, orig_rid), owner_rid in pick_owner_by_key.items():
            owner_id = roster_id_to_owner.get(int(owner_rid)) or str(owner_rid)
            picks_by_team.setdefault(owner_id, []).append({'season': int(s), 'round': int(rnd), 'original_roster_id': int(orig_rid)})
    except Exception:
        # Best-effort only
        pass
    return picks_by_team, faab_by_team


@dataclass
class TradeSearchConfig:
    max_players_per_side: int = 2
    top_candidates_per_roster: int = 10
    include_picks: bool = True
    include_faab: bool = True
    faab_step: int = 5
    faab_value_per_dollar: float = 0.12
    pick_values: Optional[Dict[int, float]] = None  # round -> value
    max_proposals_per_team: int = 12
    max_results: int = 12
    only_mutual: bool = False
    fit_weight: float = 1.0
    per_receive_limit: int = 1
    per_send_limit: int = 2
    alpha_my_gain: float = 1.0
    beta_fit_balance: float = 1.0


def _compute_team_needs_z(meta: dict, values_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute per-team z-scores of positional strength using lineup-derived starter counts with FLEX.
    Uses fractional weighting for FLEX (e.g., 1/3 of each FLEX to RB/WR/TE). Returns team_id -> z-scores."""
    # Derive starter slots from league info
    pos_slots_effective, _raw = _derive_lineup_slots((meta or {}).get('league_info', {}))
    if not any(pos_slots_effective.values()):
        pos_slots_effective = {'QB':1.0,'RB':3.0,'WR':4.0,'TE':2.0}
    owner_to_team_name = (meta or {}).get('owner_to_team_name', {}) or {}
    roster_map_all = (meta or {}).get('players_names_by_team', {}) or {}
    # Build team rows
    rows = []
    for tid, team_players in roster_map_all.items():
        pvals = values_df[values_df['player'].isin(team_players)]
        pos_vals: Dict[str, float] = {}
        for pos, n in pos_slots_effective.items():
            # fractional count support: sum top floor(n) fully plus next value weighted by fractional remainder
            series = pvals[pvals['pos']==pos]['value'].sort_values(ascending=False).reset_index(drop=True)
            k = int(math.floor(float(n)))
            frac = float(n) - float(k)
            total = float(series.iloc[:k].sum()) if k>0 else 0.0
            if frac > 1e-9 and len(series) > k:
                total += frac * float(series.iloc[k])
            pos_vals[pos] = total
        total = sum(pos_vals.get(p,0.0) for p in pos_slots_effective.keys())
        rows.append({'tid': str(tid), **pos_vals, 'total': total})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: Dict[str, Dict[str, float]] = {}
    for col in ['QB','RB','WR','TE','total']:
        mu = float(df[col].mean()); sd = float(df[col].std(ddof=0) or 1.0)
        z = (df[col]-mu) / (sd if sd>1e-9 else 1.0)
        df[f'{col}_z'] = z
    for _, r in df.iterrows():
        out[str(r['tid'])] = {p: float(r[f'{p}_z']) for p in ['QB','RB','WR','TE']}
    return out


def value_of_assets(picks: List[Dict[str, Any]], faab_amount: float, pick_values: Dict[int, float]) -> float:
    v = 0.0
    for p in picks or []:
        rnd = int(p.get('round') or 0)
        v += float(pick_values.get(rnd, 0.0))
    v += float(faab_amount or 0.0) * float(pick_values.get(0, 0.0))  # allow FAAB via round 0 mapping if provided
    return v


def propose_trades_enhanced(
    user_team: str,
    target_team: str,
    meta: dict,
    values_df: pd.DataFrame,
    picks_by_team: Dict[str, List[Dict[str, Any]]],
    faab_by_team: Dict[str, float],
    cfg: TradeSearchConfig,
) -> pd.DataFrame:
    """Propose multi-asset trades including picks and FAAB.

    Strategy: start from 1-for-1 and 2-for-1 / 1-for-2 bases; if target_delta too negative per aggressiveness,
    add smallest sweetener(s) from user side: lowest round pick and/or small FAAB step until acceptable.
    """
    if user_team == target_team:
        return pd.DataFrame(columns=['send','receive','user_delta','target_delta','acceptance','notes'])

    vals = _build_value_lookup(values_df)
    user_roster = _roster_names_for(meta, user_team)
    tgt_roster = _roster_names_for(meta, target_team)
    # Top candidates by value
    user_cands = sorted([(p, vals.get(p, 0.0)) for p in user_roster], key=lambda x: x[1], reverse=True)[:cfg.top_candidates_per_roster]
    tgt_cands = sorted([(p, vals.get(p, 0.0)) for p in tgt_roster], key=lambda x: x[1], reverse=True)[:cfg.top_candidates_per_roster]

    # No acceptance thresholds anymore

    # Prepare asset inventories
    user_picks = sorted(picks_by_team.get(user_team, []), key=lambda p: (int(p.get('season', 0)), int(p.get('round', 10)))) if cfg.include_picks else []
    tgt_picks = sorted(picks_by_team.get(target_team, []), key=lambda p: (int(p.get('season', 0)), int(p.get('round', 10)))) if cfg.include_picks else []
    user_faab_avail = float(faab_by_team.get(user_team, 0.0) if cfg.include_faab else 0.0)
    tgt_faab_avail = float(faab_by_team.get(target_team, 0.0) if cfg.include_faab else 0.0)

    # Build pos map and team needs z-scores
    pos_map: Dict[str, str] = {row['player']: row['pos'] for _, row in values_df[['player','pos']].iterrows()}
    needs_z = _compute_team_needs_z(meta, values_df)
    user_z = needs_z.get(str(user_team), {k:0.0 for k in ['QB','RB','WR','TE']})
    target_z = needs_z.get(str(target_team), {k:0.0 for k in ['QB','RB','WR','TE']})

    def _fit_contrib(team_z: Dict[str, float], recv_names: List[str], send_names: List[str]) -> float:
        # value-weighted by position need/strength: reward filling needs (z<0), penalize depleting strengths (z>0)
        def _sum(names, sign: int):
            total = 0.0
            for n in names:
                pos = pos_map.get(n, '')
                if not pos:
                    continue
                v = float(vals.get(n, 0.0))
                z = float(team_z.get(pos, 0.0))
                if sign > 0:
                    # receiving: reward if need (negative z)
                    total += v * max(0.0, -z)
                else:
                    # sending: penalize if from strength (positive z)
                    total += v * max(0.0, +z)
            return total
        return _sum(recv_names, +1) - _sum(send_names, -1)

    proposals: List[Dict[str, object]] = []

    # Generate base player packages up to max_players_per_side (limited to 2 by default)
    def combos(cands, kmax):
        out = []
        for k in range(1, min(kmax, len(cands)) + 1):
            out.extend(list(it.combinations(cands, k)))
        return out

    left_pkgs = combos(user_cands, max(1, cfg.max_players_per_side))
    right_pkgs = combos(tgt_cands, max(1, cfg.max_players_per_side))

    # Minimal sweetener builders
    def smallest_pick(picks_list):
        # pick with lowest value (higher round number)
        best = None; best_val = math.inf
        pv_map = cfg.pick_values or {}
        for p in picks_list:
            rnd = int(p.get('round') or 0)
            val = float(pv_map.get(rnd, 0.0))
            if val < best_val and val > 0:
                best = p; best_val = val
        return best, (0.0 if best is None else best_val)

    def faab_positive_steps(max_amt):
        step = max(1, int(cfg.faab_step))
        # generate up to two positive steps only
        opts = []
        if int(max_amt) > 0:
            opts.append(min(step, int(max_amt)))
            if int(max_amt) >= 2*step:
                opts.append(min(2*step, int(max_amt)))
        return opts

    for u_pkg in left_pkgs:
        u_names = [x[0] for x in u_pkg]
        u_val = sum(x[1] for x in u_pkg)
        for t_pkg in right_pkgs:
            t_names = [x[0] for x in t_pkg]
            t_val = sum(x[1] for x in t_pkg)
            base_user_delta = t_val - u_val
            base_target_delta = u_val - t_val
            # Record base package with fit metrics
            proposals.append({
                  'send': ' + '.join(u_names),
                  'receive': ' + '.join(t_names),
                 'you_picks': '',
                 'you_faab': 0,
                 'they_picks': '',
                 'they_faab': 0,
                 'my_gain': round(base_user_delta, 3),
                 'their_gain': round(base_target_delta, 3),
                 'mutual_gain': round(min(base_user_delta, base_target_delta), 3) if base_user_delta>0 and base_target_delta>0 else 0.0,
                 'fit_user': round(_fit_contrib(user_z, t_names, u_names), 3),
                 'fit_target': round(_fit_contrib(target_z, u_names, t_names), 3),
                 'notes': f"players ({len(u_names)}v{len(t_names)})",
             })

            # Also try small sweeteners from either side to improve balance/fit
            add_pick, pick_val = smallest_pick(user_picks)
            # Build FAAB options: include 0 only if adding a pick, else only positive steps
            faab_options = []
            if add_pick is not None:
                faab_options.append(0)
            if cfg.include_faab:
                faab_options.extend(faab_positive_steps(user_faab_avail))
            for faab_add in faab_options:
                added_val = (pick_val if add_pick is not None else 0.0) + float(faab_add) * float(cfg.faab_value_per_dollar)
                # Skip if sweetener has no value (would duplicate base)
                if added_val <= 1e-9:
                    continue
                new_my = base_user_delta - added_val
                new_their = base_target_delta + added_val
                parts_send = list(u_names)
                you_picks_str = ''
                if add_pick:
                    rnd_val = add_pick.get('round')
                    try:
                        rnd_i = int(rnd_val) if rnd_val is not None else 0
                    except Exception:
                        rnd_i = 0
                    s_val = add_pick.get('season')
                    try:
                        s_i = int(s_val) if s_val is not None else None
                    except Exception:
                        s_i = None
                    label = f"{s_i} R{rnd_i}" if s_i else f"R{rnd_i}"
                    parts_send.append(f"Pick {label}")
                    you_picks_str = label
                if faab_add>0:
                    parts_send.append(f"FAAB ${int(faab_add)}")
                proposals.append({
                    'send': ' + '.join(parts_send),
                    'receive': ' + '.join(t_names),
                    'you_picks': you_picks_str,
                    'you_faab': int(faab_add),
                    'they_picks': '',
                    'they_faab': 0,
                    'my_gain': round(new_my, 3),
                    'their_gain': round(new_their, 3),
                    'mutual_gain': round(min(new_my, new_their), 3) if new_my>0 and new_their>0 else 0.0,
                    'fit_user': round(_fit_contrib(user_z, t_names, u_names), 3),
                    'fit_target': round(_fit_contrib(target_z, u_names, t_names), 3),
                    'notes': 'sweetened',
                })

            # Try opponent-side sweetener (they add a pick/FAAB to close the gap the other way)
            add_pick_t, pick_val_t = smallest_pick(tgt_picks)
            faab_options_t = []
            if add_pick_t is not None:
                faab_options_t.append(0)
            if cfg.include_faab:
                faab_options_t.extend(faab_positive_steps(tgt_faab_avail))
            for faab_add_t in faab_options_t:
                added_val_t = (pick_val_t if add_pick_t is not None else 0.0) + float(faab_add_t) * float(cfg.faab_value_per_dollar)
                if added_val_t <= 1e-9:
                    continue
                # They add value to your receive side
                new_my = base_user_delta + added_val_t
                new_their = base_target_delta - added_val_t
                parts_recv = list(t_names)
                they_picks_str = ''
                if add_pick_t:
                    rnd_val_t = add_pick_t.get('round')
                    try:
                        rnd_t = int(rnd_val_t) if rnd_val_t is not None else 0
                    except Exception:
                        rnd_t = 0
                    s_val_t = add_pick_t.get('season')
                    try:
                        s_t = int(s_val_t) if s_val_t is not None else None
                    except Exception:
                        s_t = None
                    label_t = f"{s_t} R{rnd_t}" if s_t else f"R{rnd_t}"
                    parts_recv.append(f"Pick {label_t}")
                    they_picks_str = label_t
                if faab_add_t>0:
                    parts_recv.append(f"FAAB ${int(faab_add_t)}")
                proposals.append({
                    'send': ' + '.join(u_names),
                    'receive': ' + '.join(parts_recv),
                    'you_picks': '',
                    'you_faab': 0,
                    'they_picks': they_picks_str,
                    'they_faab': int(faab_add_t),
                    'my_gain': round(new_my, 3),
                    'their_gain': round(new_their, 3),
                    'mutual_gain': round(min(new_my, new_their), 3) if new_my>0 and new_their>0 else 0.0,
                    'fit_user': round(_fit_contrib(user_z, t_names, u_names), 3),
                    'fit_target': round(_fit_contrib(target_z, u_names, t_names), 3),
                    'notes': 'sweetened (opponent)',
                })

            # No acceptance/steal labeling; "steal" shows implicitly when their_gain is very negative

    if not proposals:
        return pd.DataFrame(columns=['send','receive','my_gain','their_gain','mutual_gain','fit_user','fit_target','fit_score','fit_balance','notes'])
    df_prop = pd.DataFrame(proposals)
    # Drop duplicates on full trade components to avoid repeated rows
    try:
        df_prop = df_prop.drop_duplicates(subset=['send','receive','you_picks','you_faab','they_picks','they_faab'])
    except Exception:
        df_prop = df_prop.drop_duplicates()
    # Compute overall fit score (user + target scaled)
    df_prop['fit_score'] = (df_prop['fit_user'] + df_prop['fit_target']) * float(cfg.fit_weight)
    # Fit balance and optional filter
    df_prop['fit_balance'] = df_prop[['fit_user','fit_target']].min(axis=1)
    # Primary ranking score
    try:
        alpha = float(getattr(cfg, 'alpha_my_gain', 1.0) or 1.0)
        beta = float(getattr(cfg, 'beta_fit_balance', 1.0) or 1.0)
    except Exception:
        alpha, beta = 1.0, 1.0
    df_prop['trade_score'] = alpha * df_prop['my_gain'] + beta * df_prop['fit_balance']
    # Optional filter: only mutual-benefit
    if bool(cfg.only_mutual):
        df_prop = df_prop[(df_prop['my_gain']>0) & (df_prop['their_gain']>0)]
    df_prop = df_prop.sort_values(['trade_score','fit_balance','my_gain','mutual_gain','fit_score'], ascending=[False, False, False, False, False]).reset_index(drop=True)
    # Enforce per-receive limit (post-sort)
    try:
        k = int(getattr(cfg, 'per_receive_limit', 1) or 1)
        df_prop['__recv_rank'] = df_prop.groupby('receive').cumcount()
        df_prop = df_prop[df_prop['__recv_rank'] < k].drop(columns=['__recv_rank'])
    except Exception:
        pass
    # Enforce per-send limit (post-sort)
    try:
        ks = int(getattr(cfg, 'per_send_limit', 2) or 2)
        df_prop['__send_rank'] = df_prop.groupby('send').cumcount()
        df_prop = df_prop[df_prop['__send_rank'] < ks].drop(columns=['__send_rank'])
    except Exception:
        pass
    # Cap to max_results
    try:
        nmax = int(getattr(cfg, 'max_results', 12) or 12)
        df_prop = df_prop.head(nmax)
    except Exception:
        df_prop = df_prop.head(12)
    return df_prop




# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    st.caption("Trade analyzer using current-season data and rough scarcity/dynasty heuristics.")
    league_id = st.text_input("Sleeper league id", value=DEFAULT_LEAGUE)
    season = st.number_input("Season", min_value=2010, max_value=2100, value=DEFAULT_SEASON)
    _week_options = ["Full Season"] + list(range(1,19))
    _week_choice = st.selectbox("Scope", options=_week_options, index=0)
    week = None if _week_choice == "Full Season" else int(_week_choice)
    source = st.selectbox("Source", options=["auto","sleeper","pbp"], index=1)
    cache_dir = st.text_input("Cache dir", value=".cache")
    if st.button("Refresh data cache"):
        st.cache_data.clear()
    mode = st.selectbox("Methodology", options=["Win now","Balanced","Dynasty build"], index=1)
    use_per_game = st.checkbox("Use per-game (PPG) values", value=True)
    scarcity_method = st.selectbox("Replacement method", options=["nth_starter","median_starter"], index=0)
    z_boost = st.slider("Z boost weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05, help="Weight for position-normalized z-score boost (non-negative).")
    vol_weight = st.slider("Volatility penalty", min_value=0.0, max_value=1.0, value=0.15, step=0.05, help="Penalty applied to relative weekly volatility (std/ppg).")
    apply_injury = st.checkbox("Apply injury discount", value=True, help="Discount values for players with OUT/PUP/IR/etc. statuses.")
    short_injury_weight = st.slider("Short-term injury weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05, help="How strongly questionable/out/pup-return statuses reduce value.")
    season_injury_weight = st.slider("Season-long injury weight", min_value=0.0, max_value=1.0, value=0.8, step=0.05, help="How strongly full IR/PUP/NFI without return designation reduce value.")
    # Removed acceptance sliders and legacy 2-for-2 toggle
    st.divider()
    st.caption("Enhanced search (multi-asset)")
    league_wide = st.checkbox("Scan league-wide (all teams)", value=False)
    max_players_per_side = st.slider("Max players per side", min_value=1, max_value=3, value=2)
    top_candidates_per_roster = st.slider("Top candidates per roster", min_value=5, max_value=20, value=7, step=1)
    include_picks = st.checkbox("Include draft picks (sweetener)", value=True)
    include_faab = st.checkbox("Include FAAB (sweetener)", value=True)
    faab_step = st.slider("FAAB step ($)", min_value=1, max_value=50, value=5)
    faab_value_per_dollar = st.number_input("FAAB value per $", min_value=0.01, max_value=1.0, value=0.12, step=0.01)
    only_mutual = st.checkbox("Only show mutual-benefit trades (both sides gain)", value=False)
    fit_weight = st.slider("Position fit weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1, help="Boost proposals that improve each team's weakest positions.")
    max_results = st.slider("Max recommendations", min_value=5, max_value=30, value=12, step=1, help="Limits output to speed up searching.")
    per_receive_limit = st.slider("Max per identical receive", min_value=1, max_value=5, value=1, step=1, help="Limit how many proposals share the same receive package.")
    per_send_limit = st.slider("Max per identical send", min_value=1, max_value=5, value=2, step=1, help="Limit how many proposals share the same send package.")
    alpha_my_gain = st.slider("Alpha: my gain weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Importance of your net gain in ranking.")
    beta_fit_balance = st.slider("Beta: fit balance weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Importance of balanced positional fit (min of fit_user and fit_target).")
    st.caption("Ranking: fit balance (min of fit_user & fit_target) desc, then your gain desc.")
    st.caption("Draft pick round values (rough baseline)")
    pick_values_defaults = {1: 6.0, 2: 3.5, 3: 2.0, 4: 1.2}
    # Back these inputs with session_state so we can update them from calibration
    if 'pv1' not in st.session_state:
        st.session_state['pv1'] = pick_values_defaults[1]
        st.session_state['pv2'] = pick_values_defaults[2]
        st.session_state['pv3'] = pick_values_defaults[3]
        st.session_state['pv4'] = pick_values_defaults[4]
    pv1 = st.number_input("Round 1 value", 0.0, 20.0, key='pv1', step=0.1)
    pv2 = st.number_input("Round 2 value", 0.0, 20.0, key='pv2', step=0.1)
    pv3 = st.number_input("Round 3 value", 0.0, 20.0, key='pv3', step=0.1)
    pv4 = st.number_input("Round 4 value", 0.0, 20.0, key='pv4', step=0.1)
    # League has 4 draft rounds currently


# Load and compute values
with st.spinner("Loading and valuing players…"):
    try:
        values_df, rep = compute_value_board(
            league_id, int(season), week, source, cache_dir,
            mode=('win_now' if mode=="Win now" else ('dynasty' if mode=="Dynasty build" else 'balanced')),
            use_per_game=bool(use_per_game),
            scarcity_method=str(scarcity_method),
            z_boost=float(z_boost),
            vol_weight=float(vol_weight),
            apply_injury=bool(apply_injury),
            short_injury_weight=float(short_injury_weight),
            season_injury_weight=float(season_injury_weight),
        )
    except Exception as e:
        st.error(f"Failed to load/compute values: {e}")
        st.stop()

if values_df.empty:
    st.info("No player data available for this selection.")
    st.stop()

# Context expander
with st.expander("Context", expanded=False):
    last_completed = detect_last_completed_week(league_id, int(season), source, cache_dir)
    scope_label = f"Week {int(week)}" if week is not None else f"Full Season (through W{last_completed})"
    st.caption(f"League: {league_id} | Season: {season} | Scope: {scope_label} | Source: {source}")
    # Show lineup slots parsed from Sleeper
    try:
        _report_ctx, _pp_ctx, _meta_ctx = load_week(league_id, int(season), int(last_completed) if last_completed else None, source, cache_dir)
        pos_eff, raw_counts = _derive_lineup_slots((_meta_ctx or {}).get('league_info', {}))
        if any(pos_eff.values()):
            st.caption(
                f"Lineup: QB {pos_eff.get('QB',0):.2f}, RB {pos_eff.get('RB',0):.2f}, WR {pos_eff.get('WR',0):.2f}, TE {pos_eff.get('TE',0):.2f} (FLEX {raw_counts.get('FLEX',0)}, SFlex {raw_counts.get('SUPER_FLEX',0)})"
            )
    except Exception:
        pass

# Calibration removed per request

# Team selection
# Grab meta from the last aggregation call (approximate by re-running detect to fetch last)
last_completed = detect_last_completed_week(league_id, int(season), source, cache_dir) or (int(week) if week else 1)
_report, _pp, meta = load_week(league_id, int(season), int(last_completed), source, cache_dir)
team_opts = _team_options(meta)

st.title("🔁 Trade Analyzer")
colA, colB = st.columns(2)
with colA:
    _labels = [lbl for _, lbl in team_opts]
    try:
        _default_user_idx = next((i for i, lbl in enumerate(_labels) if isinstance(lbl, str) and 'stuswood' in lbl.lower()), 0)
    except Exception:
        _default_user_idx = 0
    user_team_label = st.selectbox("Your team", options=_labels, index=_default_user_idx)
with colB:
    target_team_label = st.selectbox("Target team", options=[lbl for _, lbl in team_opts])
# Map back to team ids
user_team_id = next((tid for tid, lbl in team_opts if lbl == user_team_label), None)
target_team_id = next((tid for tid, lbl in team_opts if lbl == target_team_label), None)
if not user_team_id or not target_team_id:
    st.info("Select teams to generate proposals.")

# Main tabs
tabs = st.tabs(["Value board", "Team needs", "Proposals", "Best trade available", "Trade evaluator", "Assets"])

with tabs[0]:
    st.subheader("Value board (by position)")
    show_cols = ['rank','player','team','pos','pos_rank','ppg','rep_ppg','vor','z_ppg','rel_vol','injury_status','injury_category','injury_factor','value']
    existing_cols = [c for c in show_cols if c in values_df.columns]
    st.dataframe(values_df[existing_cols], use_container_width=True, hide_index=True)
    st.caption("Value = VOR + z_boost*max(0,z_ppg) - vol_weight*rel_vol, with lineup-aware replacement and modest positional multipliers in Dynasty/Balanced modes.")
    st.markdown("#### Replacement baseline (PPG)")
    rep_rows = [{'pos': k, 'replacement_ppg': round(v,3)} for k,v in (rep or {}).items()]
    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Team needs (roster strength vs league)")
    # Compare each team's top starters-worth of players at each position to league average
    meta_last = meta
    starters_map = (meta_last or {}).get('starters_names_by_team', {}) or {}
    owner_to_team_name = (meta_last or {}).get('owner_to_team_name', {}) or {}
    owner_to_manager = (meta_last or {}).get('owner_to_manager', {}) or {}

    def _team_label(tid: str) -> str:
        return f"{owner_to_team_name.get(str(tid), str(tid))} ({owner_to_manager.get(str(tid), '')})".strip()

    # compute positional scores by summing top starter counts from lineup (with fractional flex)
    pos_slots, raw_lineup = _derive_lineup_slots((meta_last or {}).get('league_info', {}))
    if not any(pos_slots.values()):
        pos_slots = {'QB':1.0,'RB':2.0,'WR':2.0,'TE':1.0}
    team_pos_scores: List[Dict[str, object]] = []
    roster_map_all = (meta_last or {}).get('players_names_by_team', {}) or {}
    for tid in roster_map_all.keys():
        team_players = (meta_last or {}).get('players_names_by_team', {}).get(tid, [])
        pvals = values_df[values_df['player'].isin(team_players)]
        pos_vals: Dict[str, float] = {}
        for pos, n in pos_slots.items():
            s = pvals[pvals['pos']==pos]['value'].sort_values(ascending=False).reset_index(drop=True)
            k = int(math.floor(float(n))); frac = float(n) - float(k)
            total = float(s.iloc[:k].sum()) if k>0 else 0.0
            if frac > 1e-9 and len(s) > k:
                total += frac * float(s.iloc[k])
            pos_vals[pos] = round(total, 3)
        total_val = round(sum(pos_vals.get(p, 0.0) for p in pos_slots.keys()), 3)
        row: Dict[str, object] = {'team': _team_label(tid), **pos_vals, 'total': total_val}
        team_pos_scores.append(row)
    df_needs = pd.DataFrame(team_pos_scores)
    if not df_needs.empty:
        # Standardize vs league mean for a quick "need" indicator
        for pos in ['QB','RB','WR','TE','total']:
            mu = float(df_needs[pos].mean()); sd = float(df_needs[pos].std(ddof=0) or 1.0)
            df_needs[f'{pos}_z'] = ((df_needs[pos]-mu) / (sd if sd>1e-9 else 1.0)).round(2)
        st.dataframe(df_needs.sort_values('total', ascending=False), use_container_width=True, hide_index=True)
        st.caption("z-scores show relative strength (+) or need (-) vs league at each position, computed from full roster values.")
    else:
        st.caption("No starters data available to compute team needs.")

with tabs[2]:
    st.subheader("Recommended trade packages")
    if not user_team_id or not target_team_id:
        st.info("Select both teams to generate proposals.")
    else:
        # Config
        cfg = TradeSearchConfig(
            max_players_per_side=int(max_players_per_side),
            top_candidates_per_roster=int(top_candidates_per_roster),
            include_picks=bool(include_picks),
            include_faab=bool(include_faab),
            faab_step=int(faab_step),
            faab_value_per_dollar=float(faab_value_per_dollar),
            pick_values={1: float(pv1), 2: float(pv2), 3: float(pv3), 4: float(pv4), 0: float(faab_value_per_dollar)},
            max_proposals_per_team=int(max_results),
            max_results=int(max_results),
            only_mutual=bool(only_mutual),
            fit_weight=float(fit_weight),
            per_receive_limit=int(per_receive_limit),
            per_send_limit=int(per_send_limit),
            alpha_my_gain=float(alpha_my_gain),
            beta_fit_balance=float(beta_fit_balance),
        )
        # Fetch assets
        if cfg.include_picks or cfg.include_faab:
            picks_by_team, faab_by_team = fetch_team_assets(league_id, int(season), (meta or {}).get('league_info', {}))
        else:
            picks_by_team, faab_by_team = {}, {}

        if league_wide:
            # Scan all other teams as targets
            owner_to_team_name = (meta or {}).get('owner_to_team_name', {}) or {}
            all_tids = list(owner_to_team_name.keys()) or list((meta or {}).get('players_names_by_team', {}).keys())
            all_tids = [str(t) for t in all_tids]
            proposals_all: List[pd.DataFrame] = []
            for tid in all_tids:
                if str(tid) == str(user_team_id):
                    continue
                df_prop = propose_trades_enhanced(
                    user_team=str(user_team_id),
                    target_team=str(tid),
                    meta=meta,
                    values_df=values_df,
                    picks_by_team=picks_by_team,
                    faab_by_team=faab_by_team,
                    cfg=cfg,
                )
                if not df_prop.empty:
                    df_prop.insert(0, 'target_team', tid)
                    proposals_all.append(df_prop.head(cfg.max_proposals_per_team))
            if not proposals_all:
                st.info("No suitable packages found under the current settings.")
            else:
                out = pd.concat(proposals_all, ignore_index=True)
                desired_cols = ['trade_score','target_team','send','receive','you_picks','you_faab','they_picks','they_faab','my_gain','their_gain','mutual_gain','fit_user','fit_target','fit_balance','fit_score','notes']
                for col in desired_cols:
                    if col not in out.columns:
                        if col in ('target_team','send','receive','you_picks','they_picks','notes'):
                            out[col] = ''
                        elif col in ('you_faab','they_faab'):
                            out[col] = 0
                        else:
                            out[col] = 0.0
                st.dataframe(out[desired_cols].head(max_results), use_container_width=True, hide_index=True)
                st.caption("trade_score = alpha*my_gain + beta*fit_balance. my_gain = your net; their_gain = opponent net; mutual_gain = benefit to both; fit_* and fit_score reflect positional needs.")
        else:
            # Single target
            df_prop = propose_trades_enhanced(
                user_team=str(user_team_id),
                target_team=str(target_team_id),
                meta=meta,
                values_df=values_df,
                picks_by_team=picks_by_team,
                faab_by_team=faab_by_team,
                cfg=cfg,
            )
            if df_prop.empty:
                st.info("No suitable packages found under the current settings.")
            else:
                display_cols = [c for c in ['trade_score','send','receive','you_picks','you_faab','they_picks','they_faab','my_gain','their_gain','mutual_gain','fit_user','fit_target','fit_balance','fit_score','notes'] if c in df_prop.columns]
                st.dataframe(df_prop[display_cols].head(max_results), use_container_width=True, hide_index=True)
                st.caption("trade_score = alpha*my_gain + beta*fit_balance. my_gain = your net; their_gain = opponent net; mutual_gain = benefit to both; fit_* and fit_score reflect positional needs.")
                st.markdown("""
                ### Column glossary
                - send: What you would give up (players and any sweeteners).
                - receive: What you would get back (players and any sweeteners).
                - you_picks / they_picks: Draft pick rounds included by each side (e.g., R2 means a 2nd‑round pick).
                - you_faab / they_faab: FAAB dollars included by each side.
                - my_gain: Your net value gain (value of receive minus value of send). Higher is better for you.
                - their_gain: The other team’s net value gain. Positive means they benefit too.
                - mutual_gain: If both sides gain, this is the smaller of the two gains (a quick win‑win signal).
                - fit_user / fit_target: How much the trade improves each team’s weakest positions (more is better).
                - fit_balance: The lower of (fit_user, fit_target). Higher means the trade helps both teams’ needs more evenly.
                - trade_score: Combined ranking score = alpha*my_gain + beta*fit_balance (set by the sliders).
                - fit_score: Overall positional fit = (fit_user + fit_target) scaled by the Fit weight slider.
                - notes: Extra tags like “sweetened” to indicate picks/FAAB were added.

                ### Methodology (short)
                - Player value = VOR + z_boost*max(0, z_ppg) − vol_weight*rel_vol, then adjusted by position/mode and an injury factor.
                - Replacement is lineup‑aware (using your Sleeper roster settings) via nth‑starter or median‑starter.
                - Fit metrics reward adding value to weak positions and penalize taking from strengths.
                - Picks/FAAB are converted to value using your sliders, and proposals are ranked by trade_score.

                ### How to evaluate quickly
                - Use trade_score as the primary sort; it blends “how much you win” and “positional fit balance.”
                - Check my_gain > 0. If it’s small, ensure fit_balance is high so you’re solving the right roster problems.
                - If you’re targeting a specific need, sort or filter by fit_user to prioritize help to your roster.
                - Prefer proposals that don’t require large sweeteners unless the my_gain or fit_balance justify them.
                - Watch injury_factor on the Value board: big discounts can make “name” players look fairly priced or risky.
                - In Superflex, QB replacement is higher—compare QB-for-QB deals where possible.
                """)

with tabs[3]:
    st.subheader("Best trade available (league-wide)")
    if not user_team_id:
        st.info("Select your team to scan the league.")
    else:
        # Reuse enhanced config from sidebar
        cfg = TradeSearchConfig(
            max_players_per_side=int(max_players_per_side),
            top_candidates_per_roster=int(top_candidates_per_roster),
            include_picks=bool(include_picks),
            include_faab=bool(include_faab),
            faab_step=int(faab_step),
            faab_value_per_dollar=float(faab_value_per_dollar),
            pick_values={1: float(pv1), 2: float(pv2), 3: float(pv3), 4: float(pv4), 0: float(faab_value_per_dollar)},
            max_proposals_per_team=int(max_results),
            max_results=int(max_results),
            only_mutual=bool(only_mutual),
            fit_weight=float(fit_weight),
            per_receive_limit=int(per_receive_limit),
            per_send_limit=int(per_send_limit),
            alpha_my_gain=float(alpha_my_gain),
            beta_fit_balance=float(beta_fit_balance),
        )
        # Assets
        if cfg.include_picks or cfg.include_faab:
            picks_by_team, faab_by_team = fetch_team_assets(league_id, int(season), (meta or {}).get('league_info', {}))
        else:
            picks_by_team, faab_by_team = {}, {}

        # Helper for label
        meta_last = meta
        owner_to_team_name = (meta_last or {}).get('owner_to_team_name', {}) or {}
        owner_to_manager = (meta_last or {}).get('owner_to_manager', {}) or {}
        def _team_label2(tid: str) -> str:
            return f"{owner_to_team_name.get(str(tid), str(tid))} ({owner_to_manager.get(str(tid), '')})".strip()

        # Scan all targets
        targets = list(owner_to_team_name.keys()) or list((meta_last or {}).get('players_names_by_team', {}).keys())
        targets = [str(t) for t in targets if str(t) != str(user_team_id)]
        proposals_all: List[pd.DataFrame] = []
        for tid in targets:
            df_prop = propose_trades_enhanced(
                user_team=str(user_team_id),
                target_team=str(tid),
                meta=meta,
                values_df=values_df,
                picks_by_team=picks_by_team,
                faab_by_team=faab_by_team,
                cfg=cfg,
            )
            if not df_prop.empty:
                df_prop.insert(0, 'target_team', _team_label2(tid))
                proposals_all.append(df_prop.head(cfg.max_proposals_per_team))
        if not proposals_all:
            st.info("No suitable packages found across the league under the current settings.")
        else:
            out = pd.concat(proposals_all, ignore_index=True)
            # Rank globally by fit balance, then your gain
            if 'fit_balance' not in out.columns and 'fit_user' in out.columns and 'fit_target' in out.columns:
                out['fit_balance'] = out[['fit_user','fit_target']].min(axis=1)
            # Compute trade_score if not present
            if 'trade_score' not in out.columns and 'fit_balance' in out.columns and 'my_gain' in out.columns:
                out['trade_score'] = float(alpha_my_gain) * out['my_gain'] + float(beta_fit_balance) * out['fit_balance']
            out = out.sort_values(['trade_score','fit_balance','my_gain','mutual_gain','fit_score'], ascending=[False, False, False, False, False]).reset_index(drop=True)
            best = out.iloc[0]
            st.success(f"Best vs {best['target_team']}: send {best['send']} ⇄ receive {best['receive']} | trade_score {best['trade_score']:+.2f} | your Δ {best['my_gain']:+.2f} | fit_balance {best['fit_balance']:+.2f}")
            top_n = st.slider("Show top N results", min_value=5, max_value=100, value=max_results, step=5)
            desired_cols = ['trade_score','target_team','send','receive','you_picks','you_faab','they_picks','they_faab','my_gain','their_gain','mutual_gain','fit_user','fit_target','fit_balance','fit_score','notes']
            for col in desired_cols:
                if col not in out.columns:
                    if col in ('target_team','send','receive','you_picks','they_picks','notes'):
                        out[col] = ''
                    elif col in ('you_faab','they_faab'):
                        out[col] = 0
                    else:
                        out[col] = 0.0
            st.dataframe(out[desired_cols].head(top_n), use_container_width=True, hide_index=True)
            st.markdown("""
            ### Column glossary
            - target_team: The opponent team for this proposal.
            - send: What you would give up (players and any sweeteners).
            - receive: What you would get back (players and any sweeteners).
            - you_picks / they_picks: Draft pick rounds included by each side (e.g., R2 means a 2nd‑round pick).
            - you_faab / they_faab: FAAB dollars included by each side.
            - my_gain: Your net value gain (value of receive minus value of send). Higher is better for you.
            - their_gain: The other team’s net value gain. Positive means they benefit too.
            - mutual_gain: If both sides gain, this is the smaller of the two gains (a quick win‑win signal).
            - fit_user / fit_target: How much the trade improves each team’s weakest positions (more is better).
            - fit_balance: The lower of (fit_user, fit_target). Higher means the trade helps both teams’ needs more evenly.
            - trade_score: Combined ranking score = alpha*my_gain + beta*fit_balance (set by the sliders).
            - fit_score: Overall positional fit = (fit_user + fit_target) scaled by the Fit weight slider.
            - notes: Extra tags like “sweetened” to indicate picks/FAAB were added.

            ### Methodology (short)
            - Player value = VOR + z_boost*max(0, z_ppg) − vol_weight*rel_vol, then adjusted by position/mode and an injury factor.
            - Replacement is lineup‑aware (using your Sleeper roster settings) via nth‑starter or median‑starter.
            - Fit metrics reward adding value to weak positions and penalize taking from strengths.
            - Picks/FAAB are converted to value using your sliders, and proposals are ranked by trade_score.

            ### How to evaluate quickly
            - Use trade_score as the primary sort; it blends “how much you win” and “positional fit balance.”
            - Check my_gain > 0. If it’s small, ensure fit_balance is high so you’re solving the right roster problems.
            - If you’re targeting a specific need, sort or filter by fit_user to prioritize help to your roster.
            - Prefer proposals that don’t require large sweeteners unless the my_gain or fit_balance justify them.
            - Watch injury_factor on the Value board: big discounts can make “name” players look fairly priced or risky.
            - In Superflex, QB replacement is higher—compare QB-for-QB deals where possible.
            """)

with tabs[4]:
    st.subheader("Trade evaluator")
    if not user_team_id or not target_team_id:
        st.info("Select your team and a target team above to evaluate a trade.")
    else:
        # Assets for FAAB limits (optional)
        try:
            _picks_tmp, _faab_tmp = fetch_team_assets(league_id, int(season), (meta or {}).get('league_info', {}))
        except Exception:
            _picks_tmp, _faab_tmp = {}, {}
        pos_map: Dict[str, str] = {row['player']: row['pos'] for _, row in values_df[['player','pos']].iterrows()}
        vals_map: Dict[str, float] = _build_value_lookup(values_df)
        user_roster = _roster_names_for(meta, str(user_team_id)) if user_team_id else []
        tgt_roster = _roster_names_for(meta, str(target_team_id)) if target_team_id else []
        user_opts = [p for p in user_roster if p in vals_map]
        tgt_opts = [p for p in tgt_roster if p in vals_map]
        col1, col2 = st.columns(2)
        with col1:
            you_send_players = st.multiselect("You send (players)", options=sorted(user_opts))
            st.caption("Draft picks you can include (future seasons only)")
            # Build availability for your future picks (season+1..+3)
            try:
                draft_rounds = int(((meta or {}).get('league_info', {}) or {}).get('draft_rounds') or 4)
            except Exception:
                draft_rounds = 4
            future_seasons = [int(season)+1, int(season)+2, int(season)+3]
            you_picks_avail: Dict[Tuple[int,int], int] = {}
            for p in _picks_tmp.get(str(user_team_id), []) or []:
                try:
                    sv = p.get('season'); rv = p.get('round')
                    s = int(sv) if isinstance(sv, (int, str)) and str(sv).isdigit() else None
                    r = int(rv) if isinstance(rv, (int, str)) and str(rv).isdigit() else None
                except Exception:
                    s, r = None, None
                if isinstance(s, int) and isinstance(r, int) and s in future_seasons and 1 <= r <= draft_rounds:
                    you_picks_avail[(s, r)] = you_picks_avail.get((s, r), 0) + 1
            you_picks_sel: Dict[Tuple[int,int], int] = {}
            # Render as grid: one row per season, small counters per round to minimize scrolling
            for s in future_seasons:
                # Label above the inputs for this season
                st.markdown(f"**You: {s}**")
                round_cols = st.columns(draft_rounds)
                for idx, r in enumerate(range(1, draft_rounds+1)):
                    avail = int(you_picks_avail.get((s, r), 0))
                    if avail <= 0:
                        continue
                    key = f"ys_{s}_R{r}"
                    with round_cols[idx]:
                        you_picks_sel[(s, r)] = st.number_input(f"R{r}", min_value=0, max_value=avail, value=0, step=1, key=key)
            # Allow FAAB entry even if league cap isn't exposed: default to a generous max
            _faab_cap_you = _faab_tmp.get(str(user_team_id))
            try:
                faab_cap_you = float(_faab_cap_you) if _faab_cap_you is not None and float(_faab_cap_you) > 0 else 1000.0
            except Exception:
                faab_cap_you = 1000.0
            faab_you = float(st.number_input("You send FAAB ($)", min_value=0.0, max_value=faab_cap_you, value=0.0, step=1.0))
        with col2:
            they_send_players = st.multiselect("They send (players)", options=sorted(tgt_opts))
            st.caption("Draft picks they can include (future seasons only)")
            they_picks_avail: Dict[Tuple[int,int], int] = {}
            for p in _picks_tmp.get(str(target_team_id), []) or []:
                try:
                    sv = p.get('season'); rv = p.get('round')
                    s = int(sv) if isinstance(sv, (int, str)) and str(sv).isdigit() else None
                    r = int(rv) if isinstance(rv, (int, str)) and str(rv).isdigit() else None
                except Exception:
                    s, r = None, None
                if isinstance(s, int) and isinstance(r, int) and s in future_seasons and 1 <= r <= draft_rounds:
                    they_picks_avail[(s, r)] = they_picks_avail.get((s, r), 0) + 1
            they_picks_sel: Dict[Tuple[int,int], int] = {}
            for s in future_seasons:
                # Label above the inputs for this season
                st.markdown(f"**They: {s}**")
                round_cols_t = st.columns(draft_rounds)
                for idx, r in enumerate(range(1, draft_rounds+1)):
                    avail = int(they_picks_avail.get((s, r), 0))
                    if avail <= 0:
                        continue
                    key = f"ts_{s}_R{r}"
                    with round_cols_t[idx]:
                        they_picks_sel[(s, r)] = st.number_input(f"R{r}", min_value=0, max_value=avail, value=0, step=1, key=key)
            _faab_cap_they = _faab_tmp.get(str(target_team_id))
            try:
                faab_cap_they = float(_faab_cap_they) if _faab_cap_they is not None and float(_faab_cap_they) > 0 else 1000.0
            except Exception:
                faab_cap_they = 1000.0
            faab_they = float(st.number_input("They send FAAB ($)", min_value=0.0, max_value=faab_cap_they, value=0.0, step=1.0))
        # Compute values
        pick_values_map = {1: float(st.session_state.get('pv1', 6.0)), 2: float(st.session_state.get('pv2', 3.5)), 3: float(st.session_state.get('pv3', 2.0)), 4: float(st.session_state.get('pv4', 1.2))}
        you_picks_val = 0.0
        they_picks_val = 0.0
        try:
            for (s, r), cnt in (you_picks_sel or {}).items():
                if int(cnt) > 0:
                    you_picks_val += int(cnt) * float(pick_values_map.get(int(r), 0.0))
        except Exception:
            pass
        try:
            for (s, r), cnt in (they_picks_sel or {}).items():
                if int(cnt) > 0:
                    they_picks_val += int(cnt) * float(pick_values_map.get(int(r), 0.0))
        except Exception:
            pass
        send_val = sum(vals_map.get(n, 0.0) for n in you_send_players) + you_picks_val + float(faab_you) * float(st.session_state.get('pv0', faab_value_per_dollar))
        recv_val = sum(vals_map.get(n, 0.0) for n in they_send_players) + they_picks_val + float(faab_they) * float(st.session_state.get('pv0', faab_value_per_dollar))
        my_gain = recv_val - send_val
        their_gain = -my_gain

        needs_z_all = _compute_team_needs_z(meta, values_df)
        u_z = needs_z_all.get(str(user_team_id), {k:0.0 for k in ['QB','RB','WR','TE']})
        t_z = needs_z_all.get(str(target_team_id), {k:0.0 for k in ['QB','RB','WR','TE']})
        def _fit_eval(team_z, recv, send):
            total = 0.0
            for n in recv:
                v = vals_map.get(n, 0.0); z = float(team_z.get(pos_map.get(n,''), 0.0))
                total += v * max(0.0, -z)
            for n in send:
                v = vals_map.get(n, 0.0); z = float(team_z.get(pos_map.get(n,''), 0.0))
                total -= v * max(0.0, +z)
            return total
        fit_user = _fit_eval(u_z, they_send_players, you_send_players)
        fit_target = _fit_eval(t_z, you_send_players, they_send_players)
        mutual_gain = min(my_gain, their_gain) if my_gain>0 and their_gain>0 else 0.0
        fit_score = (fit_user + fit_target) * float(fit_weight)

        st.markdown("#### Evaluation")
        # Summarize selected picks for clarity
        def _picks_summary(sel: Dict[Tuple[int,int], int]) -> str:
            parts = []
            for (s,r), cnt in sorted(sel.items()):
                if int(cnt) > 0:
                    parts.append(f"{s} R{r} x{int(cnt)}")
            return ", ".join(parts) if parts else "(none)"
        st.write({
            'my_gain': round(my_gain,3),
            'their_gain': round(their_gain,3),
            'mutual_gain': round(mutual_gain,3),
            'you_picks': _picks_summary(you_picks_sel),
            'they_picks': _picks_summary(they_picks_sel),
            'fit_user': round(fit_user,3),
            'fit_target': round(fit_target,3),
            'fit_score': round(fit_score,3),
        })

with tabs[5]:
    st.subheader("Assets: picks and FAAB (live from Sleeper)")
    # Pull actual picks and FAAB regardless of proposal toggles
    # Refresh daily at 3 AM Eastern by keying cache to a daily window id
    def _assets_window_key(now_utc: Optional[pd.Timestamp] = None) -> str:
        try:
            # Use zoneinfo if available
            import datetime as _dt
            if now_utc is None:
                now_utc = pd.Timestamp.utcnow()
            dt_utc = _dt.datetime.utcfromtimestamp(now_utc.timestamp())
            if ZoneInfo is not None:
                et = dt_utc.replace(tzinfo=_dt.timezone.utc).astimezone(ZoneInfo("America/New_York"))
                boundary = et.replace(hour=3, minute=0, second=0, microsecond=0)
                if et >= boundary:
                    return f"ET-{et.date()}-post3am"
                else:
                    y = et.date() - _dt.timedelta(days=1)
                    return f"ET-{y}-post3am"
            else:
                # Fallback: approximate ET as UTC-5 (may be off during DST)
                et = dt_utc - _dt.timedelta(hours=5)
                boundary = et.replace(hour=3, minute=0, second=0, microsecond=0)
                if et >= boundary:
                    return f"approxET-{et.date()}-post3am"
                else:
                    y = et.date() - _dt.timedelta(days=1)
                    return f"approxET-{y}-post3am"
        except Exception:
            # Ultimate fallback: daily UTC window at 08:00 (roughly 3am ET standard)
            try:
                import datetime as _dt
                now = _dt.datetime.utcnow()
                boundary = now.replace(hour=8, minute=0, second=0, microsecond=0)
                if now >= boundary:
                    return f"UTC-{now.date()}-08"
                else:
                    y = now.date() - _dt.timedelta(days=1)
                    return f"UTC-{y}-08"
            except Exception:
                return "fallback-window"

    @st.cache_data(show_spinner=False)
    def _cached_fetch_assets(league_id_in: str, season_in: int, league_info_in: dict, window_key: str, cache_version: str):
        # window_key and cache_version are included in the cache key to control refresh cadence
        return fetch_team_assets(league_id_in, int(season_in), league_info_in)
    try:
        league_info = (meta or {}).get('league_info', {}) or {}
        window_key = _assets_window_key()
        picks_by_team, faab_by_team = _cached_fetch_assets(league_id, int(season), league_info, window_key, cache_version="v2-future-only-3yrs")
        st.caption("Assets auto-refresh daily at 3 AM Eastern. Use the sidebar's 'Refresh data cache' to force an immediate update.")
    except Exception as e:
        picks_by_team, faab_by_team = {}, {}
        st.warning(f"Could not fetch assets: {e}")

    owner_to_team_name = (meta or {}).get('owner_to_team_name', {}) or {}
    owner_to_manager = (meta or {}).get('owner_to_manager', {}) or {}
    def _team_label_assets(tid: str) -> str:
        name = owner_to_team_name.get(str(tid), str(tid))
        mgr = owner_to_manager.get(str(tid), '')
        return f"{name} ({mgr})".strip()

    # FAAB table
    try:
        faab_rows = []
        # Pull raw settings for transparency/debug
        try:
            client_dbg = SleeperClient()
            rosters_dbg = client_dbg.get_rosters(league_id) or []
            settings_by_owner = {}
            for rdbg in rosters_dbg:
                oid_dbg = rdbg.get('owner_id'); rid_dbg = rdbg.get('roster_id')
                tid_dbg = str(oid_dbg) if oid_dbg is not None else (str(rid_dbg) if rid_dbg is not None else '')
                settings_by_owner[tid_dbg] = (rdbg.get('settings', {}) or {})
        except Exception:
            settings_by_owner = {}
        for tid, amt in (faab_by_team or {}).items():
            try:
                s = settings_by_owner.get(str(tid), {})
                faab_rows.append({
                    'team': _team_label_assets(tid),
                    'owner_id': str(tid),
                    'faab_remaining': round(float(amt or 0.0), 2),
                    'league_budget_cap': ((meta or {}).get('league_info', {}) or {}).get('settings', {}).get('waiver_budget', (meta or {}).get('league_info', {}).get('waiver_budget', None)),
                    'raw_waiver_budget': s.get('waiver_budget'),
                    'raw_faab': s.get('faab'),
                    'raw_waiver_budget_total': s.get('waiver_budget_total'),
                    'raw_waiver_budget_used': s.get('waiver_budget_used'),
                })
            except Exception:
                continue
        df_faab = pd.DataFrame(faab_rows)
        if not df_faab.empty:
            st.markdown("### FAAB remaining")
            # Hide owner_id, raw_waiver_budget, raw_faab, raw_waiver_budget_total as requested
            show_cols = ['team','faab_remaining','league_budget_cap','raw_waiver_budget_used']
            st.dataframe(df_faab[show_cols].sort_values('faab_remaining', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No FAAB data available.")
    except Exception:
        st.caption("Unable to display FAAB table.")

    # Picks table (counts by season/round)
    try:
        # Display exactly the next three seasons (exclude current), regardless of any historical entries
        all_picks = picks_by_team or {}
        seasons = [int(season)+1, int(season)+2, int(season)+3]
        try:
            default_rounds = int(((meta or {}).get('league_info', {}) or {}).get('draft_rounds') or 4)
        except Exception:
            default_rounds = 4
        rounds = list(range(1, default_rounds+1))

        # Build rows per team
        pick_rows: List[Dict[str, object]] = []
        for tid, plist in all_picks.items():
            row: Dict[str, object] = {'team': _team_label_assets(tid)}
            # initialize counts to 0
            for s in seasons:
                for r in rounds:
                    row[f"{s} R{r}"] = 0
            # count
            for p in plist:
                try:
                    _s = p.get('season'); _r = p.get('round')
                    s = int(_s) if _s is not None and str(_s).isdigit() else None
                    r = int(_r) if _r is not None and str(_r).isdigit() else None
                    # Only count future seasons we display
                    key = f"{s} R{r}"
                    if isinstance(s, int) and isinstance(r, int) and s in seasons and key in row:
                        current_val = row.get(key, 0)
                        current_int: int
                        if isinstance(current_val, (int, float, str)):
                            try:
                                current_int = int(current_val)
                            except Exception:
                                current_int = 0
                        else:
                            current_int = 0
                        row[key] = current_int + 1
                except Exception:
                    continue
            # totals
            try:
                total = 0
                for s in seasons:
                    for r in rounds:
                        v = row.get(f"{s} R{r}", 0)
                        if isinstance(v, (int, float, str)):
                            try:
                                total += int(v)
                            except Exception:
                                total += 0
                        else:
                            total += 0
                row['total_picks'] = int(total)
            except Exception:
                row['total_picks'] = 0
            pick_rows.append(row)
        df_picks = pd.DataFrame(pick_rows)
        if not df_picks.empty:
            # Order columns: team, seasons/rounds grid, total
            grid_cols = [f"{s} R{r}" for s in seasons for r in rounds]
            show_cols = ['team'] + grid_cols + ['total_picks']
            st.markdown("### Draft picks (current owner)")
            st.dataframe(df_picks[show_cols].sort_values(['total_picks','team'], ascending=[False, True]), use_container_width=True, hide_index=True)
            st.caption("Counts reflect current pick ownership by season and round. Ownership updated using Sleeper's traded_picks feed. Rounds inferred from league settings.")
        else:
            st.caption("No draft pick data available.")
    except Exception:
        st.caption("Unable to display picks table.")
