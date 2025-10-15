import os
import json
import pandas as pd
import streamlit as st
import altair as alt
from collections import Counter, defaultdict

# Local imports
from src.cli import build_report, score_breakdown

# Bump this to invalidate Streamlit cache when data shape/mapping changes
CACHE_VERSION = "v4.1-sim-fa-allplayers"

DEFAULT_LEAGUE = os.environ.get("FT_LEAGUE_ID", "1260098143609966592")
DEFAULT_SEASON = int(os.environ.get("FT_SEASON", "2025"))
DEFAULT_WEEK = int(os.environ.get("FT_WEEK", "6"))

st.set_page_config(page_title="Fantasy Scoring Explorer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
# Subtle CSS for spacing and density
st.markdown(
    """
    <style>
    h2, h3, h4, h5 { margin-top: 0.6rem; }
    .stDataFrame table { font-size: 0.92rem; }
    .streamlit-expanderHeader { font-weight: 600; }
    .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Fantasy Scoring Explorer")

with st.sidebar:
    st.header("Inputs")
    st.write("Configure data scope and source. Use Simulator to experiment with scoring.")
    st.divider()
    league_id = st.text_input("Sleeper league id", value=DEFAULT_LEAGUE)
    season = st.number_input("Season", min_value=2010, max_value=2100, value=DEFAULT_SEASON, step=1)
    # Week selector: include Full Season option
    _week_options = ["Full Season"] + list(range(1, 19))
    # Default to Full Season
    _week_choice = st.selectbox("Week", options=_week_options, index=0, help="Pick a single week or Full Season")
    week = None if _week_choice == "Full Season" else int(_week_choice)
    source = st.selectbox("Source", options=["auto","sleeper","pbp"], index=1)
    cache_dir = st.text_input("Cache dir", value=".cache")
    st.divider()
    roster_filter = st.selectbox(
        "Roster filter",
        options=["All", "Starters only", "Bench only"],
        index=0,
        help="Filter the table to only starters or only bench for the selected week"
    )
    if st.button("Refresh data cache"):
        st.cache_data.clear()
    explain_player = st.text_input("Explain player (exact display name)", value="Josh Allen")
    st.caption("Tip: Use the table below to find the exact display name for breakdowns.")
    name_filter = st.text_input("Filter by player name (contains)", value="")
    # Removed compact mode per request: use default Streamlit sizing

# Compact mode removed

@st.cache_data(show_spinner=False)
def load_data(league_id: str, season: int, week: int | None, source: str, cache_dir: str, version: str):
    report, player_points, meta = build_report(
        league_id,
        season,
        week,
        infer=True,
        match_threshold=0.78,
        source=source,
        scoring_path=None,
        assume_buckets='infer',
        cache_dir=cache_dir,
        debug_source=False,
        overrides_path=None,
    )
    return report, player_points, meta

# Determine last completed week by requiring both scheduled matchups and positive weekly points
def detect_last_completed_week() -> int | None:
    last = None
    for w in range(1, 19):
        try:
            _r, _pp, _m = load_data(league_id, int(season), w, source, cache_dir, CACHE_VERSION)
        except Exception:
            continue
        matchups_by_id = _m.get('matchups_by_id', {}) or {}
        try:
            pf_total = sum(float(v) for v in (_pp or {}).values())
        except Exception:
            pf_total = 0.0
        if matchups_by_id and pf_total > 0:
            last = w
    return last

with st.spinner("Loading data..."):
    report, player_points, meta = load_data(league_id, int(season), week, source, cache_dir, CACHE_VERSION)

with st.expander("Context", expanded=False):
    try:
        _last_completed = detect_last_completed_week()
    except Exception:
        _last_completed = None
    _scope_label = f"Week {int(week)}" if week is not None else "Full Season"
    _used_src = meta.get('used_source') or source
    c1, c2, c3, c4, c5 = st.columns([2,1,1,1,2])
    with c1:
        st.caption(f"League: {league_id}")
    with c2:
        st.caption(f"Season: {int(season)}")
    with c3:
        st.caption(f"Scope: {_scope_label}")
    with c4:
        st.caption(f"Last completed wk: {_last_completed if _last_completed is not None else '-'}")
    with c5:
        st.caption(f"Data source: {_used_src}")

page_tabs = st.tabs(["📈 Summary", "🤝 Matchups", "📊 Rankings", "🧪 Simulator"])
def build_df():
    # Full Season: aggregate across all completed weeks
    if week is None:
        last_week = detect_last_completed_week()
        if not last_week:
            return pd.DataFrame(columns=["player","team","pos","points","starter"])

        cum_points: dict[str, float] = defaultdict(float)
        team_counts: dict[str, Counter] = defaultdict(Counter)
        pos_map: dict[str, str | None] = {}

        meta_last = None
        report_last = None
        # Aggregate by roster filter across weeks
        for w in range(1, last_week + 1):
            r_w, pp_w, m_w = load_data(league_id, int(season), w, source, cache_dir, CACHE_VERSION)
            meta_last = m_w
            report_last = r_w
            starters_w = m_w.get('starters_names_by_team', {}) or {}
            players_w = m_w.get('players_names_by_team', {}) or {}
            for team_id, tdata in r_w.items():
                starters_set = set(starters_w.get(team_id, set()))
                all_players_set = set(players_w.get(team_id, set()))
                if roster_filter == "Starters only":
                    roster_players = starters_set
                elif roster_filter == "Bench only":
                    roster_players = all_players_set - starters_set
                else:
                    roster_players = all_players_set
                for pname in roster_players:
                    pts = float(pp_w.get(pname, 0.0))
                    cum_points[pname] += pts
                    team_counts[pname][team_id] += 1
                    if pname not in pos_map:
                        pos_map[pname] = m_w.get('name_pos', {}).get(pname)

        # Choose display team as most frequent team across season
        owner_to_manager = (meta_last or meta).get('owner_to_manager', {})
        owner_to_team_name = (meta_last or meta).get('owner_to_team_name', {})
        # Also try to source manager from report when available
        rows = []
        for pname, pts in cum_points.items():
            # identify most frequent team id
            team_id = None
            if team_counts[pname]:
                team_id = team_counts[pname].most_common(1)[0][0]
            manager = None
            team_name = None
            if team_id is not None:
                tdata = (report_last or report).get(team_id, {}) if isinstance((report_last or report), dict) else {}
                manager = (tdata.get('manager') if isinstance(tdata, dict) else None) or owner_to_manager.get(str(team_id)) or owner_to_manager.get(team_id)
                team_name = (tdata.get('team_name') if isinstance(tdata, dict) else None) or owner_to_team_name.get(str(team_id)) or owner_to_team_name.get(team_id) or str(team_id)
            team_display = f"{team_name} ({manager})" if manager else (team_name or "")
            rows.append({
                "player": pname,
                "team": team_display,
                "pos": pos_map.get(pname),
                "points": round(pts, 2),
                "starter": False,  # ambiguous across season
            })
        df = pd.DataFrame(rows)
        if name_filter:
            df = df[df["player"].str.contains(name_filter, case=False, na=False)]
        return df

    # Single-week view (existing behavior)
    rows = []
    owner_to_manager = meta.get('owner_to_manager', {})
    owner_to_team_name = meta.get('owner_to_team_name', {})
    for team_id, tdata in report.items():
        starters_set = set(meta.get('starters_names_by_team', {}).get(team_id, set()))
        manager = tdata.get('manager') or owner_to_manager.get(str(team_id)) or owner_to_manager.get(team_id) or ""
        team_name = tdata.get('team_name') or owner_to_team_name.get(str(team_id)) or owner_to_team_name.get(team_id) or str(team_id)
        team_display = f"{team_name} ({manager})" if manager else team_name
        for pname, weeks in tdata.get("players", {}).items():
            if roster_filter == "Starters only" and starters_set and pname not in starters_set:
                continue
            if roster_filter == "Bench only" and starters_set and pname in starters_set:
                continue
            pos = meta.get('name_pos', {}).get(pname)
            pts = player_points.get(pname, 0.0)
            rows.append({
                "player": pname,
                "team": team_display,
                "pos": pos,
                "points": round(pts, 2),
                "starter": pname in starters_set,
            })
    df = pd.DataFrame(rows)
    if name_filter:
        df = df[df["player"].str.contains(name_filter, case=False, na=False)]
    return df

with page_tabs[0]:
    st.subheader("Summary")
    # If in Full Season mode, add an "as of Week X" caption to show the last completed week
    if week is None:
        try:
            _last_wk_caption = detect_last_completed_week()
            if _last_wk_caption:
                st.caption(f"As of completed Week {_last_wk_caption}")
        except Exception:
            pass
    tabs = st.tabs(["All Players", "QB", "RB", "WR", "TE", "Breakdown"])

    df_all = build_df()
    # Reorder columns for readability (no weeks/manager columns)
    desired_cols = ["player", "pos", "points", "team", "starter"]
    df_all = df_all.reindex(columns=[c for c in desired_cols if c in df_all.columns] + [c for c in df_all.columns if c not in desired_cols])
    # In Full Season mode, the 'starter' concept is ambiguous across weeks; hide the column
    if week is None and 'starter' in df_all.columns:
        df_all = df_all.drop(columns=['starter'])
    # Hint if team names look numeric (owner_id fallback)
    if 'team' in df_all.columns and df_all['team'].astype(str).str.fullmatch(r'\d+').any():
        st.caption("If team names look numeric, click 'Refresh data cache' in the sidebar to reload.")

    # Top-N positional distribution
    st.markdown("### Top-N positional distribution")
    top_n_choice = st.selectbox(
        "Top N",
        options=[12, 25, 50, 75, 100],
        index=0,
        help="Top-N by points for the current selection (week/full season and roster filter)."
    )
    metric_choice_summary = st.selectbox(
        "Display metric",
        options=["Count", "Percent", "Avg points", "Points %"],
        index=0,
        key="summary_topn_metric",
        help="Choose which metric the chart shows across positions."
    )
    if df_all.empty:
        st.info("No player data available for this selection.")
    else:
        valid_positions = ["QB", "RB", "WR", "TE"]
        df_all_ranked = df_all.copy()
        df_all_ranked["rank"] = df_all_ranked["points"].rank(method="min", ascending=False)
        df_ranked = df_all_ranked[df_all_ranked["pos"].isin(valid_positions)].sort_values("points", ascending=False)
        df_top = df_ranked.head(int(top_n_choice))
        denom = max(1, len(df_top))
        counts = df_top["pos"].value_counts().reindex(valid_positions, fill_value=0).astype(float)
        import numpy as _np
        # Compute all metric series
        perc = (counts / float(denom) * 100.0)
        avg_pts = (df_top.groupby("pos")["points"].mean().reindex(valid_positions)) if not df_top.empty else pd.Series([0,0,0,0], index=valid_positions, dtype=float)
        sum_pts = (df_top.groupby("pos")["points"].sum().reindex(valid_positions, fill_value=0.0)) if not df_top.empty else pd.Series([0,0,0,0], index=valid_positions, dtype=float)
        total_pts = float(df_top["points"].sum()) or 1.0
        pts_pct = (sum_pts / total_pts * 100.0)
        # Pick metric
        if metric_choice_summary == "Count":
            series = counts
            fmt = "{:,.0f}"
        elif metric_choice_summary == "Percent":
            series = perc
            fmt = "{:.1f}%"
        elif metric_choice_summary == "Avg points":
            series = avg_pts
            fmt = "{:.2f}"
        else:
            series = pts_pct
            fmt = "{:.1f}%"
        chart_df = pd.DataFrame({"value": series.reindex(valid_positions).values}, index=valid_positions)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(chart_df, use_container_width=True)
        with c2:
            tbl = pd.DataFrame({
                "pos": valid_positions,
                metric_choice_summary: series.reindex(valid_positions).round(2).values,
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        # Average points, points share, and relative efficiency per position within Top-N (detailed table)
        if not df_top.empty:
            count_share = counts / float(denom)
            points_share = (sum_pts / total_pts).fillna(0.0)
            _cs = _np.asarray(count_share.values, dtype=float)
            _ps = _np.asarray(points_share.values, dtype=float)
            _rel_vals = _np.divide(_ps, _cs, out=_np.zeros_like(_ps), where=_cs != 0)
            _count_share_vals = _np.asarray(count_share.values, dtype=float)
            _points_share_vals = _np.asarray(points_share.values, dtype=float)
            dom_df = pd.DataFrame({
                "pos": valid_positions,
                "count": counts.values.astype(int),
                "count %": (_count_share_vals * 100.0).round(1),
                "avg points": avg_pts.round(2).values,
                "points %": (_points_share_vals * 100.0).round(1),
                "rel efficiency": _np.round(_rel_vals, 2),
            })
            st.dataframe(dom_df, use_container_width=True, hide_index=True)

        # Top-N by team/manager breakdown
        st.markdown("#### Top-N by team/manager")
        # df_all['team'] already includes team name + (manager) when available
        team_counts_rows = []
        for team_label, group in df_top.groupby('team', dropna=False):
            # Count by pos for valid positions
            pos_counts = group['pos'].value_counts().reindex(valid_positions, fill_value=0)
            row = {
                'team': team_label if pd.notna(team_label) else '',
                'total': int(len(group)),
            }
            for p in valid_positions:
                row[p] = int(pos_counts.get(p, 0))
            team_counts_rows.append(row)
        df_team_counts = pd.DataFrame(team_counts_rows)
        if not df_team_counts.empty:
            # Sort by total desc, then by QB/RB/WR/TE counts for tie-breakers
            sort_cols = ['total'] + valid_positions
            df_team_counts = df_team_counts.sort_values(sort_cols, ascending=[False] + [False]*len(valid_positions)).reset_index(drop=True)
            st.dataframe(df_team_counts, use_container_width=True, hide_index=True)
        else:
            st.caption("No teams found in the current Top-N selection.")
    with tabs[0]:
        _df = df_all.sort_values("points", ascending=False).copy()
        _df.insert(0, "rank", range(1, len(_df) + 1))
        # Ensure rank is first
        cols = ["rank"] + [c for c in _df.columns if c != "rank"]
        _df = _df[cols]
        st.dataframe(_df, use_container_width=True, hide_index=True)
    with tabs[1]:
        _df = df_all[df_all["pos"]=="QB"].sort_values("points", ascending=False).copy()
        _df.insert(0, "rank", range(1, len(_df) + 1))
        _df = _df[["rank"] + [c for c in _df.columns if c != "rank"]]
        st.dataframe(_df, use_container_width=True, hide_index=True)
    with tabs[2]:
        _df = df_all[df_all["pos"]=="RB"].sort_values("points", ascending=False).copy()
        _df.insert(0, "rank", range(1, len(_df) + 1))
        _df = _df[["rank"] + [c for c in _df.columns if c != "rank"]]
        st.dataframe(_df, use_container_width=True, hide_index=True)
    with tabs[3]:
        _df = df_all[df_all["pos"]=="WR"].sort_values("points", ascending=False).copy()
        _df.insert(0, "rank", range(1, len(_df) + 1))
        _df = _df[["rank"] + [c for c in _df.columns if c != "rank"]]
        st.dataframe(_df, use_container_width=True, hide_index=True)
    with tabs[4]:
        _df = df_all[df_all["pos"]=="TE"].sort_values("points", ascending=False).copy()
        _df.insert(0, "rank", range(1, len(_df) + 1))
        _df = _df[["rank"] + [c for c in _df.columns if c != "rank"]]
        st.dataframe(_df, use_container_width=True, hide_index=True)
    with tabs[5]:
        st.subheader("Explain breakdown")
        if explain_player:
            # Find first week with data (or chosen week)
            weeks_map = None
            for team_id, tdata in report.items():
                if explain_player in tdata.get("players", {}):
                    weeks_map = tdata["players"][explain_player]
                    break
            if not weeks_map:
                st.info("Player not found in current report.")
            else:
                # If specific week, use that; else show all available weeks
                if (week is not None) and (str(int(week)) in weeks_map):
                    wk_keys = [str(int(week))]
                else:
                    wk_keys = sorted(weeks_map.keys(), key=lambda x: int(x))
                for wk in wk_keys:
                    stats = {k: float(v) for k, v in weeks_map[wk].items() if isinstance(v, (int, float))}
                    # Attach position for TE bonus logic
                    pos = meta.get('name_pos', {}).get(explain_player)
                    if pos:
                        stats['_pos'] = pos
                    scoring_rules = meta.get('scoring_rules', {})
                    breakdown = score_breakdown(stats, scoring_rules)
                    total = sum(breakdown.values())
                    st.markdown(f"#### {explain_player} W{wk}: {total:.2f}")
                    # Build a detailed table: stat | value | points per | points
                    rows = []
                    for stat_key, stat_points in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True):
                        factor = scoring_rules.get(stat_key)
                        value = stats.get(stat_key)
                        if isinstance(factor, (int, float)) and isinstance(value, (int, float)):
                            pp = factor
                            val = value
                        else:
                            pp = None
                            val = None
                        rows.append({
                            'stat': stat_key,
                            'value': val,
                            'points per': pp,
                            'points': stat_points,
                        })
                    import pandas as _pd
                    df_break = _pd.DataFrame(rows)
                    # Keep only rows where value is numeric and non-zero, or where value is NaN but points != 0 (e.g., tier/bonus rows)
                    if not df_break.empty:
                        mask = ((df_break['value'].notna()) & (df_break['value'] != 0)) | ((df_break['value'].isna()) & (df_break['points'] != 0))
                        df_break = df_break[mask]
                    # Order columns and format
                    df_break = df_break.reindex(columns=['stat', 'value', 'points per', 'points'])
                    st.dataframe(df_break, use_container_width=True, hide_index=True)

        with st.expander("Roster lookup (who has this player this week?)"):
            lookup_name = st.text_input("Find player on rosters", value="Aaron Rodgers")
            if lookup_name:
                owners = []
                starters_map = meta.get('starters_names_by_team', {})
                players_map = meta.get('players_names_by_team', {})
                owner_to_manager = meta.get('owner_to_manager', {})
                owner_to_team_name = meta.get('owner_to_team_name', {})
                for team_id, names in players_map.items():
                    if any(lookup_name.lower() == n.lower() for n in names):
                        owners.append({
                            'team': owner_to_team_name.get(str(team_id)) or str(team_id),
                            'manager': (report.get(team_id, {}).get('manager') if isinstance(report.get(team_id), dict) else None) or owner_to_manager.get(str(team_id)),
                            'starter': any(lookup_name.lower() == n.lower() for n in starters_map.get(team_id, set())),
                        })
                if owners:
                    st.write(owners)
                else:
                    st.info("Not found on any Week-roster. They may be a free agent for the selected week, or on a different week.")


with page_tabs[1]:
    st.subheader("Weekly matchups")
    if week:
        matchups_by_id = meta.get('matchups_by_id', {}) or {}
        starters_map = meta.get('starters_names_by_team', {}) or {}
        players_map = meta.get('players_names_by_team', {}) or {}
        owner_to_manager = meta.get('owner_to_manager', {})
        owner_to_team_name = meta.get('owner_to_team_name', {})
        scoring_rules = meta.get('scoring_rules', {})

        # Helper to compute team totals and build roster tables
        def team_display(team_id: str) -> str:
            manager = report.get(team_id, {}).get('manager') or owner_to_manager.get(str(team_id)) or ''
            team_name = report.get(team_id, {}).get('team_name') or owner_to_team_name.get(str(team_id)) or str(team_id)
            return f"{team_name} ({manager})" if manager else team_name

        def _player_week_points(pname: str, weeks_map: dict) -> float:
            # Sum points for available weeks (weekly mode usually 1)
            pos = meta.get('name_pos', {}).get(pname)
            total = 0.0
            for _wk, wkstats in (weeks_map or {}).items():
                if not wkstats:
                    continue
                stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                if pos:
                    stats_numeric['_pos'] = pos
                bd = score_breakdown(stats_numeric, meta.get('scoring_rules', {}))
                total += sum(bd.values())
            return round(total, 2)

        def build_roster_df(team_id: str, only_starters: bool | None = None) -> pd.DataFrame:
            rows = []
            starters = set(starters_map.get(team_id, set()))
            all_players = set(players_map.get(team_id, set()))
            roster_players = starters if only_starters is True else (all_players - starters if only_starters is False else all_players)
            for pname in sorted(roster_players):
                # compute points from this team's report entry to avoid cross-team name collisions
                weeks_map = (report.get(team_id, {}).get('players') or {}).get(pname, {})
                pts = _player_week_points(pname, weeks_map)
                pos = meta.get('name_pos', {}).get(pname)
                rows.append({
                    'player': pname,
                    'pos': pos,
                    'starter': pname in starters,
                    'points': round(pts, 2),
                })
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values(['starter','points'], ascending=[False, False])
                df.insert(0, 'rank', range(1, len(df)+1))
            return df

        matchup_ids_sorted = sorted(matchups_by_id.keys(), key=lambda x: (x is None, str(x)))
        view = st.radio("Roster group", options=["All", "Starters", "Bench"], horizontal=True, index=1)

        # Determine which players count for totals given the selected view
        def team_total(team_id: str, only_starters: bool | None) -> float:
            starters = set(starters_map.get(team_id, set()))
            all_players = set(players_map.get(team_id, set()))
            roster_players = starters if only_starters is True else (all_players - starters if only_starters is False else all_players)
            total = 0.0
            team_players = (report.get(team_id, {}) or {}).get('players', {})
            for pname in roster_players:
                weeks_map = team_players.get(pname, {})
                total += _player_week_points(pname, weeks_map)
            return round(total, 2)

        only_starters = True if view == "Starters" else (False if view == "Bench" else None)
        # Compute totals and median across all teams this week for icon display
        teams_this_week = set()
        for tids in matchups_by_id.values():
            teams_this_week.update(tids)
        pf_by_team = {tid: team_total(tid, only_starters) for tid in teams_this_week}
        try:
            import statistics as _stats
            median_pf = _stats.median(list(pf_by_team.values())) if pf_by_team else 0.0
        except Exception:
            median_pf = 0.0

        # Legend for icons
        st.caption("Icons: 🏆 winner • 🟢 above weekly median")

        for mid in matchup_ids_sorted:
            team_ids = matchups_by_id.get(mid, [])
            if not team_ids:
                continue
            # Some matchup groups might have more than 2 (e.g., byes or incomplete data). We render in pairs.
            for i in range(0, len(team_ids), 2):
                pair = team_ids[i:i+2]
                cols = st.columns(2)
                # Determine winner for the pair (if exactly 2 teams and non-tie)
                winner_idx = None
                if len(pair) == 2:
                    t0 = pf_by_team.get(pair[0], 0.0)
                    t1 = pf_by_team.get(pair[1], 0.0)
                    if t0 > t1:
                        winner_idx = 0
                    elif t1 > t0:
                        winner_idx = 1
                for idx, team_id in enumerate(pair):
                    with cols[idx]:
                        # Icons: winner trophy and median marker
                        icons = ""
                        if winner_idx is not None and idx == winner_idx:
                            icons += " 🏆"
                        if pf_by_team.get(team_id, 0.0) > median_pf:
                            icons += " 🟢"
                        st.markdown(f"##### {team_display(team_id)}{icons}")
                        df_team = build_roster_df(team_id, only_starters=only_starters)
                        total_points = pf_by_team.get(team_id, 0.0)
                        st.markdown(f"Total: {total_points:.2f}")
                        if not df_team.empty:
                            st.dataframe(df_team[['rank','player','pos','starter','points']], use_container_width=True, hide_index=True)
                        else:
                            st.caption("No players found for this selection.")

# Rankings tab: rank by Win/Loss (starters totals) then Points For
with page_tabs[2]:
    st.subheader("Rankings")
    if week is None:
        include_median = st.checkbox("Include median bonus", value=True, help="Above the weekly median = +1 win; below = +1 loss. Equal to median = no change.")
        owner_to_manager = meta.get('owner_to_manager', {})
        owner_to_team_name = meta.get('owner_to_team_name', {})

        def team_display(team_id: str) -> str:
            manager = owner_to_manager.get(str(team_id)) or ''
            team_name = owner_to_team_name.get(str(team_id)) or str(team_id)
            return f"{team_name} ({manager})" if manager else team_name

        standings = {}
        def ensure_team(tid: str):
            if tid not in standings:
                standings[tid] = {
                    'team': team_display(tid),
                    'wins': 0,
                    'losses': 0,
                    'points_for': 0.0,
                    'points_against': 0.0,
                }

        # detect last completed week
        last_week = detect_last_completed_week()
        if not last_week:
            st.info("No completed weeks found.")
        else:
            for w in range(1, last_week + 1):
                r_w, pp_w, m_w = load_data(league_id, int(season), int(w), source, cache_dir, CACHE_VERSION)
                matchups_by_id_w = m_w.get('matchups_by_id', {}) or {}
                starters_map_w = m_w.get('starters_names_by_team', {}) or {}
                # compute starters-only PF for that week
                pf_week = {}
                for tid in set([t for lst in matchups_by_id_w.values() for t in lst]):
                    total = 0.0
                    for pname in starters_map_w.get(tid, set()):
                        total += float(pp_w.get(pname, 0.0))
                    pf_week[tid] = round(total, 2)
                # head-to-head
                for _, tids in matchups_by_id_w.items():
                    if len(tids) != 2:
                        continue
                    a, b = tids[0], tids[1]
                    ensure_team(a); ensure_team(b)
                    pa = pf_week.get(a, 0.0); pb = pf_week.get(b, 0.0)
                    standings[a]['points_for'] += pa; standings[b]['points_for'] += pb
                    standings[a]['points_against'] += pb; standings[b]['points_against'] += pa
                    if pa > pb:
                        standings[a]['wins'] += 1; standings[b]['losses'] += 1
                    elif pb > pa:
                        standings[b]['wins'] += 1; standings[a]['losses'] += 1
                # median bonus
                if include_median and pf_week:
                    import statistics as _st
                    med = _st.median(list(pf_week.values()))
                    for tid in set(pf_week.keys()):
                        ensure_team(tid)
                        if pf_week.get(tid, 0.0) > med:
                            standings[tid]['wins'] += 1
                        elif pf_week.get(tid, 0.0) < med:
                            standings[tid]['losses'] += 1

            # display standings
            rows = []
            for tid, rec in standings.items():
                rows.append({
                    'team': rec['team'],
                    'wins': rec['wins'],
                    'losses': rec['losses'],
                    'points for': round(rec['points_for'], 2),
                    'points against': round(rec['points_against'], 2),
                })
            df_rank = pd.DataFrame(rows)
            if df_rank.empty:
                st.info("No matchup data available for rankings.")
            else:
                df_rank = df_rank.sort_values(['wins', 'points for'], ascending=[False, False]).reset_index(drop=True)
                df_rank.insert(0, 'rank', range(1, len(df_rank)+1))
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
            st.caption(f"As of completed Week {last_week}")
    elif week:
        include_median = st.checkbox("Include median bonus", value=True, help="Above the weekly median = +1 win; below = +1 loss. Equal to median = no change.")
        owner_to_manager = meta.get('owner_to_manager', {})
        owner_to_team_name = meta.get('owner_to_team_name', {})

        # Display helpers use global meta (stable across weeks)
        def team_display(team_id: str) -> str:
            manager = owner_to_manager.get(str(team_id)) or ''
            team_name = owner_to_team_name.get(str(team_id)) or str(team_id)
            return f"{team_name} ({manager})" if manager else team_name

        # Initialize standings accumulator
        standings = {}
        def ensure_team(tid: str):
            if tid not in standings:
                standings[tid] = {
                    'team': team_display(tid),
                    'wins': 0,
                    'losses': 0,
                    'points_for': 0.0,
                    'points_against': 0.0,
                }

        # Accumulate from week 1 through the selected week (inclusive)
        for w in range(1, int(week) + 1):
            r_w, pp_w, m_w = load_data(league_id, int(season), int(w), source, cache_dir, CACHE_VERSION)
            matchups_by_id_w = m_w.get('matchups_by_id', {}) or {}
            starters_map_w = m_w.get('starters_names_by_team', {}) or {}
            # Per-week starters-only points for a team
            def starters_total_week(team_id: str) -> float:
                names = starters_map_w.get(team_id, set())
                total = 0.0
                for pname in names:
                    total += float(pp_w.get(pname, 0.0))
                return round(total, 2)

            for mid, tids in matchups_by_id_w.items():
                if len(tids) != 2:
                    continue
                a, b = tids[0], tids[1]
                ensure_team(a); ensure_team(b)
                pf_a = starters_total_week(a)
                pf_b = starters_total_week(b)
                standings[a]['points_for'] += pf_a
                standings[b]['points_for'] += pf_b
                standings[a]['points_against'] += pf_b
                standings[b]['points_against'] += pf_a
                if pf_a > pf_b:
                    standings[a]['wins'] += 1
                    standings[b]['losses'] += 1
                elif pf_b > pf_a:
                    standings[b]['wins'] += 1
                    standings[a]['losses'] += 1
                else:
                    # tie: no instruction provided; leave wins/losses unchanged
                    pass

            # Median bonus for the week
            if include_median:
                try:
                    # Build per-team starters PF for all teams that appeared in any matchup this week
                    teams_this_week = set()
                    for tids in matchups_by_id_w.values():
                        teams_this_week.update(tids)
                    if teams_this_week:
                        pf_week_list = []
                        pf_by_team = {}
                        for tid in teams_this_week:
                            ensure_team(tid)
                            pf = starters_total_week(tid)
                            pf_by_team[tid] = pf
                            pf_week_list.append(pf)
                        import statistics as _stats
                        med = _stats.median(pf_week_list)
                        for tid, pf in pf_by_team.items():
                            if pf > med:
                                standings[tid]['wins'] += 1
                            elif pf < med:
                                standings[tid]['losses'] += 1
                            else:
                                # exactly median: no change per league description
                                pass
                except Exception:
                    # If anything goes wrong, skip median bonus silently
                    pass

        # Build dataframe
        rows = []
        for tid, rec in standings.items():
            rows.append({
                'team': rec['team'],
                'wins': rec['wins'],
                'losses': rec['losses'],
                'points for': round(rec['points_for'], 2),
                'points against': round(rec['points_against'], 2),
            })
        df_rank = pd.DataFrame(rows)
        if df_rank.empty:
            st.info("No matchup data available for rankings.")
        else:
            df_rank = df_rank.sort_values(['wins', 'points for'], ascending=[False, False]).reset_index(drop=True)
            df_rank.insert(0, 'rank', range(1, len(df_rank)+1))
            st.dataframe(df_rank, use_container_width=True, hide_index=True)
    else:
        st.info("Select a specific week to view rankings.")

# Global footer caption
_wk_label = week if week is not None else "Full Season"
st.caption(f"Data source used: {meta.get('used_source') or 'unknown'} | League: {league_id} | Season: {season} | Week: {_wk_label}")

# Simulator tab: adjust scoring and see effects on players, matchups, and rankings
with page_tabs[3]:
    st.subheader("Simulator")
    st.caption("Adjust scoring settings and compare original vs simulated results for the selected week (and cumulative rankings up to this week).")
    if week is None:
        st.info("Select a specific week to use the Simulator. Full-season mode is not supported for simulation.")
        st.stop()

    base_rules = meta.get('scoring_rules', {}) or {}
    EXCLUDED_POS_SIM = {"K","DEF","DST","D","LB","DB","DL","CB","S","DE","DT","IDP","PK"}
    # Heuristic: treat a player's week as offensive if common offensive stat keys are present
    def _has_offensive_keys(stats: dict) -> bool:
        if not stats:
            return False
        for k in stats.keys():
            if k.startswith('pass_') or k.startswith('rush_') or k.startswith('rec') or k.startswith('two_pt'):
                return True
        return False

    # Shared mapping between UI sim_* keys and rule keys
    UI_KEY_MAP = {
        'sim_pass_yds':'pass_yds','sim_rush_yds':'rush_yds','sim_rec_yds':'rec_yds','sim_pass_tds':'pass_tds',
        'sim_rush_tds':'rush_tds','sim_rec_tds':'rec_tds','sim_rec':'rec','sim_rush_att':'rush_att','sim_int':'int',
        'sim_bonus_rec_te':'bonus_rec_te','sim_fum_lost':'fum_lost','sim_pass_att':'pass_att','sim_pass_2pt':'pass_2pt',
        'sim_rec_2pt':'rec_2pt','sim_rush_2pt':'rush_2pt','sim_rec_0_4':'rec_0_4','sim_rec_5_9':'rec_5_9',
        'sim_rec_10_19':'rec_10_19','sim_rec_20_29':'rec_20_29','sim_rec_30_39':'rec_30_39','sim_rec_40p':'rec_40p',
        'sim_pass_fd':'pass_fd','sim_rush_fd':'rush_fd','sim_rec_fd':'rec_fd','sim_sack_taken':'sack_taken',
        'sim_bonus_rush_100_199':'bonus_rush_100_199','sim_bonus_rush_200p':'bonus_rush_200p',
        'sim_bonus_rec_100_199':'bonus_rec_100_199','sim_bonus_rec_200p':'bonus_rec_200p',
        'sim_bonus_pass_300_399':'bonus_pass_300_399','sim_bonus_pass_400p':'bonus_pass_400p',
        'sim_bonus_comb_100_199':'bonus_comb_100_199','sim_bonus_comb_200p':'bonus_comb_200p',
        'sim_bonus_pass_cmp_25p':'bonus_pass_cmp_25p','sim_bonus_carries_20p':'bonus_carries_20p',
    }
    UI_KEY_MAP_REV = {v: k for k, v in UI_KEY_MAP.items()}

    # Handle applying presets or reset before any widgets are instantiated (prevents mutation errors)
    if st.session_state.get('do_apply_preset', False) and isinstance(st.session_state.get('pending_preset_values'), dict):
        preset_vals = st.session_state.get('pending_preset_values') or {}
        # consume flags
        st.session_state.pop('do_apply_preset', None)
        st.session_state.pop('pending_preset_values', None)
        # apply provided sim_* values
        for sk, sv in preset_vals.items():
            try:
                st.session_state[sk] = float(sv)
            except Exception:
                pass
        try:
            st.rerun()
        except Exception:
            pass

    # Handle reset before any widgets are instantiated (prevents Streamlit session_state mutation errors)
    if st.session_state.get('do_reset_sim', False):
        # consume flag
        st.session_state.pop('do_reset_sim', None)
        # Map UI keys to base rule keys and defaults
        for _k, _rk in UI_KEY_MAP.items():
            try:
                st.session_state[_k] = float(base_rules.get(_rk, 0.0))
            except Exception:
                st.session_state[_k] = 0.0
        # Now rerun so widgets pick up new state as their current values
        try:
            st.rerun()
        except Exception:
            pass

    # Presets UI (after pre-widget handlers, before controls)
    with st.expander("Presets", expanded=False):
        preset_names = [
            "Balanced half-PPR",
            "No-PPR, first-down focus",
            "QB-Focused",
            "RB-Focused",
            "WR-Focused",
            "TE-Focused",
            "Maximum parity",
        ]
        chosen = st.selectbox("Choose preset", preset_names, index=0)
        if st.button("Apply preset", type="primary"):
            def apply_preset_rules(rule_overrides: dict[str, float]):
                # Translate rule keys to sim_* keys
                pending: dict[str, float] = {}
                for rk, val in rule_overrides.items():
                    sk = UI_KEY_MAP_REV.get(rk)
                    if sk is not None:
                        pending[sk] = float(val)
                st.session_state['pending_preset_values'] = pending
                st.session_state['do_apply_preset'] = True
                try:
                    st.rerun()
                except Exception:
                    pass
            if chosen == "Balanced half-PPR":
                apply_preset_rules({
                    # QB
                    'pass_yds': 0.04, 'pass_tds': 4, 'int': -1.5, 'sack_taken': -0.5, 'pass_att': 0.0, 'bonus_pass_cmp_25p': 1.0, 'pass_fd': 0.0,
                    # RB
                    'rush_yds': 0.1, 'rush_tds': 6, 'rush_att': 0.15, 'bonus_rush_100_199': 2.0, 'bonus_rush_200p': 3.0, 'bonus_carries_20p': 1.0, 'rush_fd': 0.2,
                    # WR/TE
                    'rec_yds': 0.1, 'rec_tds': 6, 'rec': 0.5, 'bonus_rec_te': 0.5, 'rec_fd': 0.15,
                    # Thresholds
                    'bonus_comb_100_199': 1.0, 'bonus_comb_200p': 2.0,
                    # Buckets disabled by default
                    'rec_0_4': 0.0, 'rec_5_9': 0.0, 'rec_10_19': 0.0, 'rec_20_29': 0.0, 'rec_30_39': 0.0, 'rec_40p': 0.0,
                })
            elif chosen == "No-PPR, first-down focus":
                apply_preset_rules({
                    # QB
                    'pass_yds': 0.04, 'pass_tds': 4, 'int': -2.0, 'sack_taken': -0.5, 'pass_att': 0.0, 'bonus_pass_cmp_25p': 1.0, 'pass_fd': 0.0,
                    # RB
                    'rush_yds': 0.1, 'rush_tds': 6, 'rush_att': 0.1, 'rush_fd': 0.25, 'bonus_rush_100_199': 2.0, 'bonus_rush_200p': 3.0, 'bonus_carries_20p': 1.0,
                    # WR/TE
                    'rec_yds': 0.1, 'rec_tds': 6, 'rec': 0.0, 'bonus_rec_te': 0.5, 'rec_fd': 0.25,
                    # Thresholds
                    'bonus_comb_100_199': 1.0, 'bonus_comb_200p': 2.0,
                    # Buckets disabled
                    'rec_0_4': 0.0, 'rec_5_9': 0.0, 'rec_10_19': 0.0, 'rec_20_29': 0.0, 'rec_30_39': 0.0, 'rec_40p': 0.0,
                })
            elif chosen == "QB-Focused":
                apply_preset_rules({
                    # Boost QB efficiency and dampen penalties slightly
                    'pass_yds': 0.05, 'pass_tds': 5.0, 'int': -1.0, 'sack_taken': -0.2, 'bonus_pass_cmp_25p': 2.0,
                    # Light QB first downs, no pay for attempts
                    'pass_fd': 0.1, 'pass_att': 0.0,
                    # Keep RB/WR/TE moderate
                    'rush_yds': 0.1, 'rush_tds': 6.0, 'rush_att': 0.05, 'rush_fd': 0.1,
                    'rec_yds': 0.1, 'rec_tds': 6.0, 'rec': 0.25, 'rec_fd': 0.1, 'bonus_rec_te': 0.4,
                    # Thresholds conservative
                    'bonus_rush_100_199': 1.0, 'bonus_rush_200p': 2.0,
                    'bonus_pass_300_399': 1.0, 'bonus_pass_400p': 2.0,
                    'bonus_comb_100_199': 1.0, 'bonus_comb_200p': 2.0,
                    'bonus_carries_20p': 0.5,
                    # Buckets off
                    'rec_0_4': 0.0, 'rec_5_9': 0.0, 'rec_10_19': 0.0, 'rec_20_29': 0.0, 'rec_30_39': 0.0, 'rec_40p': 0.0,
                })
            elif chosen == "RB-Focused":
                apply_preset_rules({
                    # Slightly nerf QB and boost ground game
                    'pass_yds': 0.04, 'pass_tds': 4.0, 'int': -2.0, 'sack_taken': -0.5, 'bonus_pass_cmp_25p': 0.5,
                    'rush_yds': 0.1, 'rush_tds': 6.0, 'rush_att': 0.25, 'rush_fd': 0.3,
                    'bonus_rush_100_199': 3.0, 'bonus_rush_200p': 4.0, 'bonus_carries_20p': 2.0,
                    # Receiving modest to keep WRs in check; TE subdued
                    'rec_yds': 0.1, 'rec_tds': 6.0, 'rec': 0.25, 'rec_fd': 0.1, 'bonus_rec_te': 0.3,
                    # Combined yards still matter
                    'bonus_comb_100_199': 2.0, 'bonus_comb_200p': 3.0,
                    # Buckets off
                    'rec_0_4': 0.0, 'rec_5_9': 0.0, 'rec_10_19': 0.0, 'rec_20_29': 0.0, 'rec_30_39': 0.0, 'rec_40p': 0.0,
                })
            elif chosen == "WR-Focused":
                apply_preset_rules({
                    # Keep QB steady; emphasize receptions and receiving FDs
                    'pass_yds': 0.04, 'pass_tds': 4.0, 'int': -1.5, 'sack_taken': -0.3, 'bonus_pass_cmp_25p': 1.0,
                    'rush_yds': 0.1, 'rush_tds': 6.0, 'rush_att': 0.05, 'rush_fd': 0.1,
                    'rec_yds': 0.1, 'rec_tds': 6.0, 'rec': 1.0, 'rec_fd': 0.3, 'bonus_rec_te': 0.4,
                    # Gentle distance buckets to reward route depth
                    'rec_0_4': 0.00, 'rec_5_9': 0.02, 'rec_10_19': 0.04, 'rec_20_29': 0.06, 'rec_30_39': 0.08, 'rec_40p': 0.10,
                    # Thresholds moderate
                    'bonus_comb_100_199': 1.0, 'bonus_comb_200p': 2.0,
                })
            elif chosen == "TE-Focused":
                apply_preset_rules({
                    # Keep QB/RB/WR near standard; give TEs real help
                    'pass_yds': 0.04, 'pass_tds': 4.0, 'int': -1.5, 'sack_taken': -0.3,
                    'rush_yds': 0.1, 'rush_tds': 6.0, 'rush_att': 0.1, 'rush_fd': 0.15,
                    'rec_yds': 0.1, 'rec_tds': 6.0, 'rec': 0.5, 'rec_fd': 0.25, 'bonus_rec_te': 0.75,
                    # Buckets off
                    'rec_0_4': 0.0, 'rec_5_9': 0.0, 'rec_10_19': 0.0, 'rec_20_29': 0.0, 'rec_30_39': 0.0, 'rec_40p': 0.0,
                    # Thresholds modest
                    'bonus_comb_100_199': 1.0, 'bonus_comb_200p': 2.0,
                })
            elif chosen == "Maximum parity":
                apply_preset_rules({
                    # Compress variance: penalize mistakes, reward sustained chains, trim spikes
                    'pass_yds': 0.04, 'pass_tds': 4.0, 'int': -2.0, 'sack_taken': -0.5, 'bonus_pass_cmp_25p': 0.5,
                    'rush_yds': 0.1, 'rush_tds': 6.0, 'rush_att': 0.1,
                    'rec_yds': 0.1, 'rec_tds': 6.0, 'rec': 0.3, 'bonus_rec_te': 0.4,
                    # Reward first downs across the board slightly
                    'pass_fd': 0.05, 'rush_fd': 0.1, 'rec_fd': 0.1,
                    # Lower threshold spikes
                    'bonus_pass_300_399': 1.0, 'bonus_pass_400p': 1.0,
                    'bonus_rush_100_199': 1.0, 'bonus_rush_200p': 1.0,
                    'bonus_rec_100_199': 1.0, 'bonus_rec_200p': 1.0,
                    'bonus_comb_100_199': 1.0, 'bonus_comb_200p': 1.0,
                    'bonus_carries_20p': 0.5,
                    # Buckets off
                    'rec_0_4': 0.0, 'rec_5_9': 0.0, 'rec_10_19': 0.0, 'rec_20_29': 0.0, 'rec_30_39': 0.0, 'rec_40p': 0.0,
                })
    # Helper: render a number_input with base value shown and an icon if adjusted
    def sim_number(label: str, base_key: str, key: str, default: float = 0.0, step: float = 0.1, fmt: str | None = None) -> float:
        import math as _math
        base_raw = base_rules.get(base_key, default)
        try:
            base_val = float(base_raw)
        except Exception:
            base_val = float(default)
        # If user has already adjusted this setting in session_state, show an icon in the label
        changed_before = key in st.session_state and not _math.isclose(float(st.session_state.get(key, base_val)), base_val, rel_tol=1e-9, abs_tol=1e-9)
        # Show actual base value without rounding in the label
        now_txt = str(base_raw)
        full_label = f"{label} (Now {now_txt})" + (" ✏️" if changed_before else "")
        val = st.number_input(full_label, value=base_val, step=step, format=fmt, key=key)
        return float(val)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        s_pass_yds = sim_number("Pass yds", 'pass_yds', "sim_pass_yds", default=0.04, step=0.01, fmt="%.3f")
        s_rush_yds = sim_number("Rush yds", 'rush_yds', "sim_rush_yds", default=0.1, step=0.01, fmt="%.3f")
    with c2:
        s_pass_tds = sim_number("Pass TD", 'pass_tds', "sim_pass_tds", default=4, step=0.1)
        s_rush_tds = sim_number("Rush TD", 'rush_tds', "sim_rush_tds", default=6, step=0.1)
    with c3:
        s_rec_yds = sim_number("Rec yds", 'rec_yds', "sim_rec_yds", default=0.1, step=0.01, fmt="%.3f")
        s_rec_tds = sim_number("Rec TD", 'rec_tds', "sim_rec_tds", default=6, step=0.1)
    with c4:
        s_rec = sim_number("Receptions", 'rec', "sim_rec", default=0.0, step=0.1)
        s_rush_att = sim_number("Rush att", 'rush_att', "sim_rush_att", default=0.0, step=0.1)
    with c5:
        s_int = sim_number("QB INT", 'int', "sim_int", default=-1, step=0.1)
        s_te_bonus = sim_number("TE Bonus", 'bonus_rec_te', "sim_bonus_rec_te", default=0.0, step=0.1)
    if st.button("Reset to league settings", type="secondary"):
        # Defer resetting until before widgets are created on next run
        st.session_state['do_reset_sim'] = True
        try:
            st.rerun()
        except Exception:
            pass
    with st.expander("More scoring options"):
        cA, cB, cC = st.columns(3)
        with cA:
            s_fum_lost = sim_number("Fumbles lost", 'fum_lost', "sim_fum_lost", default=0.0, step=0.1)
            s_pass_att = sim_number("Pass attempts", 'pass_att', "sim_pass_att", default=0.0, step=0.1)
            s_pass_2pt = sim_number("Pass 2PT", 'pass_2pt', "sim_pass_2pt", default=base_rules.get('two_pt', 0.0), step=0.1)
        with cB:
            st.caption("Reception buckets (points per catch in distance bucket)")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                s_rec_0_4 = sim_number("0-4 Yd Rec", 'rec_0_4', "sim_rec_0_4", default=0.0, step=0.1)
                s_rec_10_19 = sim_number("10-19 Yd Rec", 'rec_10_19', "sim_rec_10_19", default=0.0, step=0.1)
                s_rec_30_39 = sim_number("30-39 Yd Rec", 'rec_30_39', "sim_rec_30_39", default=0.0, step=0.1)
            with bcol2:
                s_rec_5_9 = sim_number("5-9 Yd Rec", 'rec_5_9', "sim_rec_5_9", default=0.0, step=0.1)
                s_rec_20_29 = sim_number("20-29 Yd Rec", 'rec_20_29', "sim_rec_20_29", default=0.0, step=0.1)
                s_rec_40p = sim_number("40+ Yd Rec", 'rec_40p', "sim_rec_40p", default=0.0, step=0.1)
        with cC:
            s_rec_2pt = sim_number("Reception 2PT", 'rec_2pt', "sim_rec_2pt", default=base_rules.get('two_pt', 0.0), step=0.1)
            s_rush_2pt = sim_number("Rush 2PT", 'rush_2pt', "sim_rush_2pt", default=base_rules.get('two_pt', 0.0), step=0.1)

        # Advanced bonuses and stats (applied when stats are present)
        with st.expander("Advanced bonuses and first downs"):
            cL, cM, cR = st.columns(3)
            with cL:
                st.caption("First downs (per first down)")
                s_pass_fd = sim_number("1st Down Pass", 'pass_fd', "sim_pass_fd", default=0.0, step=0.1)
                s_rush_fd = sim_number("1st Down Rush", 'rush_fd', "sim_rush_fd", default=0.0, step=0.1)
                s_rec_fd = sim_number("1st Down Rec", 'rec_fd', "sim_rec_fd", default=0.0, step=0.1)
                s_sack_taken = sim_number("QB Sacked", 'sack_taken', "sim_sack_taken", default=0.0, step=0.1)
            with cM:
                st.caption("Threshold bonuses (points when condition met)")
                b_rush_100_199 = sim_number("rush 100-199 yds", 'bonus_rush_100_199', "sim_bonus_rush_100_199", default=0.0, step=0.1)
                b_rush_200p = sim_number("rush 200+ yds", 'bonus_rush_200p', "sim_bonus_rush_200p", default=0.0, step=0.1)
                b_rec_100_199 = sim_number("rec 100-199 yds", 'bonus_rec_100_199', "sim_bonus_rec_100_199", default=0.0, step=0.1)
                b_rec_200p = sim_number("rec 200+ yds", 'bonus_rec_200p', "sim_bonus_rec_200p", default=0.0, step=0.1)
            with cR:
                b_pass_300_399 = sim_number("pass 300-399 yds", 'bonus_pass_300_399', "sim_bonus_pass_300_399", default=0.0, step=0.1)
                b_pass_400p = sim_number("pass 400+ yds", 'bonus_pass_400p', "sim_bonus_pass_400p", default=0.0, step=0.1)
                b_comb_100_199 = sim_number("combined 100-199 (rush+rec)", 'bonus_comb_100_199', "sim_bonus_comb_100_199", default=0.0, step=0.1)
                b_comb_200p = sim_number("combined 200+ (rush+rec)", 'bonus_comb_200p', "sim_bonus_comb_200p", default=0.0, step=0.1)
                b_pass_cmp_25p = sim_number("25+ pass completions", 'bonus_pass_cmp_25p', "sim_bonus_pass_cmp_25p", default=0.0, step=0.1)
                b_carries_20p = sim_number("20+ carries", 'bonus_carries_20p', "sim_bonus_carries_20p", default=0.0, step=0.1)

        # Removed the dynamic 'All league offensive keys' expander and associated settings (including pass_int),
        # as they were redundant or confusing. The simulator already uses 'Interception (thrown)' via the 'int' key.


    # Build simulated scoring rules by overriding base values and keeping other keys
    sim_rules = dict(base_rules)
    sim_rules.update({
        'pass_yds': s_pass_yds,
        'rush_yds': s_rush_yds,
        'rec_yds': s_rec_yds,
        'pass_tds': s_pass_tds,
        'rush_tds': s_rush_tds,
        'rec_tds': s_rec_tds,
        'rec': s_rec,
        'rush_att': s_rush_att,
        'int': s_int,
        'bonus_rec_te': s_te_bonus,
    })
    # Add expanded options
    sim_rules.update({
        'fum_lost': s_fum_lost,
        'pass_att': s_pass_att,
        'pass_2pt': s_pass_2pt,
        'rec_2pt': s_rec_2pt,
        'rush_2pt': s_rush_2pt,
        'rec_0_4': s_rec_0_4,
        'rec_5_9': s_rec_5_9,
        'rec_10_19': s_rec_10_19,
        'rec_20_29': s_rec_20_29,
        'rec_30_39': s_rec_30_39,
        'rec_40p': s_rec_40p,
    })
    # Advanced bonuses
    sim_rules.update({
        'pass_fd': s_pass_fd,
        'rush_fd': s_rush_fd,
        'rec_fd': s_rec_fd,
        'sack_taken': s_sack_taken,
        'bonus_rush_100_199': b_rush_100_199,
        'bonus_rush_200p': b_rush_200p,
        'bonus_rec_100_199': b_rec_100_199,
        'bonus_rec_200p': b_rec_200p,
        'bonus_pass_300_399': b_pass_300_399,
        'bonus_pass_400p': b_pass_400p,
        'bonus_comb_100_199': b_comb_100_199,
        'bonus_comb_200p': b_comb_200p,
        'bonus_pass_cmp_25p': b_pass_cmp_25p,
        'bonus_carries_20p': b_carries_20p,
    })
    # No dynamic other_updates; sim_rules contains only the explicit options above.
    # Remove pass_int from sim rules; engine uses 'int' for offensive interceptions
    if 'pass_int' in sim_rules:
        sim_rules.pop('pass_int', None)
    # Tier editing/keep toggle removed: we retain any league-provided tiers implicitly

    # --- Export Summary (PDF) ---
    def _format_num(v: float | int | str | None) -> str:
        try:
            if v is None:
                return "-"
            if isinstance(v, (int, float)):
                s = ("{:.6f}".format(float(v))).rstrip('0').rstrip('.')
                return s if s != "-0" else "0"
            return str(v)
        except Exception:
            return str(v)

    # Human-readable labels for keys we expose in the simulator
    _LABELS = {
        'pass_yds': 'Pass yards', 'rush_yds': 'Rush yards', 'rec_yds': 'Rec yards',
        'pass_tds': 'Pass TD', 'rush_tds': 'Rush TD', 'rec_tds': 'Rec TD',
        'rec': 'Receptions', 'rush_att': 'Rush attempts', 'int': 'Interceptions (thrown)',
        'bonus_rec_te': 'TE bonus (per rec)', 'fum_lost': 'Fumbles lost', 'pass_att': 'Pass attempts',
        'pass_2pt': 'Pass 2PT', 'rec_2pt': 'Rec 2PT', 'rush_2pt': 'Rush 2PT',
        'rec_0_4': 'Rec 0–4 yds', 'rec_5_9': 'Rec 5–9 yds', 'rec_10_19': 'Rec 10–19 yds',
        'rec_20_29': 'Rec 20–29 yds', 'rec_30_39': 'Rec 30–39 yds', 'rec_40p': 'Rec 40+ yds',
        'pass_fd': '1st down pass', 'rush_fd': '1st down rush', 'rec_fd': '1st down rec', 'sack_taken': 'QB sacked',
        'bonus_rush_100_199': 'Bonus rush 100–199', 'bonus_rush_200p': 'Bonus rush 200+',
        'bonus_rec_100_199': 'Bonus rec 100–199', 'bonus_rec_200p': 'Bonus rec 200+',
        'bonus_pass_300_399': 'Bonus pass 300–399', 'bonus_pass_400p': 'Bonus pass 400+',
        'bonus_comb_100_199': 'Bonus combined 100–199', 'bonus_comb_200p': 'Bonus combined 200+',
        'bonus_pass_cmp_25p': 'Bonus 25+ pass comp', 'bonus_carries_20p': 'Bonus 20+ carries',
    }
    _ORDER = [
        'pass_yds','rush_yds','rec_yds','pass_tds','rush_tds','rec_tds',
        'rec','rush_att','int','fum_lost','sack_taken',
        'pass_att','pass_2pt','rec_2pt','rush_2pt',
        'pass_fd','rush_fd','rec_fd','bonus_rec_te',
        'bonus_pass_300_399','bonus_pass_400p',
        'bonus_rush_100_199','bonus_rush_200p',
        'bonus_rec_100_199','bonus_rec_200p',
        'bonus_comb_100_199','bonus_comb_200p',
        'bonus_pass_cmp_25p','bonus_carries_20p',
        'rec_0_4','rec_5_9','rec_10_19','rec_20_29','rec_30_39','rec_40p',
    ]

    def _build_summary_rows() -> list[tuple[str, str, str]]:
        rows = []
        seen = set()
        # only include keys that appear in either base_rules or sim_rules
        all_keys = [k for k in _ORDER if (k in base_rules or k in sim_rules)]
        for k in all_keys:
            label = _LABELS.get(k, k)
            before = _format_num(base_rules.get(k, 0.0))
            after = _format_num(sim_rules.get(k, base_rules.get(k, 0.0)))
            rows.append((label, before, after))
            seen.add(k)
        # Include any extra keys present in sim_rules but not in ORDER (rare)
        for k in sim_rules.keys():
            if k in seen:
                continue
            label = _LABELS.get(k, k)
            before = _format_num(base_rules.get(k))
            after = _format_num(sim_rules.get(k))
            rows.append((label, before, after))
        return rows

    def _make_pdf_reportlab(title: str, subtitle: str, rows: list[tuple[str,str,str]]) -> bytes | None:
        # Dynamically import reportlab to avoid static import errors when it's not installed
        try:
            import importlib
            from io import BytesIO
            rl_pagesizes = importlib.import_module('reportlab.lib.pagesizes')
            rl_lib = importlib.import_module('reportlab.lib')
            rl_styles = importlib.import_module('reportlab.lib.styles')
            rl_platypus = importlib.import_module('reportlab.platypus')
        except Exception:
            return None
        buf = BytesIO()
        SimpleDocTemplate = getattr(rl_platypus, 'SimpleDocTemplate')
        Table = getattr(rl_platypus, 'Table')
        TableStyle = getattr(rl_platypus, 'TableStyle')
        Paragraph = getattr(rl_platypus, 'Paragraph')
        Spacer = getattr(rl_platypus, 'Spacer')
        letter = getattr(rl_pagesizes, 'letter')
        colors = getattr(rl_lib, 'colors')
        getSampleStyleSheet = getattr(rl_styles, 'getSampleStyleSheet')
        doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph(title, styles['Title']))
        if subtitle:
            story.append(Paragraph(subtitle, styles['Normal']))
        story.append(Spacer(1, 12))
        data = [["Setting", "Before", "After"]] + [[l,b,a] for (l,b,a) in rows]
        # Wider first column for labels; numbers get compact columns
        table = Table(data, colWidths=[280, 120, 120], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eeeeee')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
            ('ALIGN', (0,1), (0,-1), 'LEFT'),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
            # Zebra striping for readability
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fbfbfb')]),
        ]))
        story.append(table)
        doc.build(story)
        return buf.getvalue()

    def _escape_pdf_text(s: str) -> str:
        return s.replace('\\', r'\\').replace('(', r'\(').replace(')', r'\)')

    def _make_pdf_minimal(title: str, subtitle: str, rows: list[tuple[str,str,str]]) -> bytes:
        # Build a very simple one-page PDF with Helvetica and absolute-positioned text via Tm
        from io import BytesIO
        buf = BytesIO()
        lines = []
        lines.append("BT")
        # Title
        lines.append("/F1 14 Tf")
        lines.append("1 0 0 1 50 760 Tm")
        lines.append(f"({_escape_pdf_text(title)}) Tj")
        y = 740
        # Subtitle
        if subtitle:
            lines.append("/F1 10 Tf")
            lines.append(f"1 0 0 1 50 {y} Tm")
            lines.append(f"({_escape_pdf_text(subtitle)}) Tj")
            y -= 20
        # Header row
        lines.append("/F1 11 Tf")
        lines.append(f"1 0 0 1 50 {y} Tm (Setting) Tj")
        lines.append(f"1 0 0 1 250 {y} Tm (Before) Tj")
        lines.append(f"1 0 0 1 350 {y} Tm (After) Tj")
        y -= 16
        # Data rows
        lines.append("/F1 10 Tf")
        for (label, before, after) in rows:
            if y < 60:
                break  # keep to one page for this fallback
            lines.append(f"1 0 0 1 50 {y} Tm ({_escape_pdf_text(str(label))}) Tj")
            lines.append(f"1 0 0 1 250 {y} Tm ({_escape_pdf_text(str(before))}) Tj")
            lines.append(f"1 0 0 1 350 {y} Tm ({_escape_pdf_text(str(after))}) Tj")
            y -= 14
        lines.append("ET")
        stream = "\n".join(lines).encode('latin-1', errors='ignore')

        # Build PDF objects
        objects = []
        def w(x: bytes):
            buf.write(x)
        xref = []
        def add_obj(obj_bytes: bytes) -> int:
            xref.append(buf.tell())
            w(f"{len(xref)} 0 obj\n".encode('ascii'))
            w(obj_bytes)
            w(b"\nendobj\n")
            return len(xref)

        w(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
        # 1: Catalog
        cat_id = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
        # 2: Pages
        pages_id = add_obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        # 3: Page
        page_obj = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        page_id = add_obj(page_obj)
        # 4: Contents
        contents_stream = b"<< /Length " + str(len(stream)).encode('ascii') + b" >>\nstream\n" + stream + b"\nendstream"
        contents_id = add_obj(contents_stream)
        # 5: Font
        font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        # xref
        xref_start = buf.tell()
        w(b"xref\n")
        w(f"0 {len(xref)+1}\n".encode('ascii'))
        w(b"0000000000 65535 f \n")
        for off in xref:
            w(f"{off:010d} 00000 n \n".encode('ascii'))
        # trailer
        w(b"trailer\n")
        trailer = f"<< /Size {len(xref)+1} /Root 1 0 R >>\n".encode('ascii')
        w(trailer)
        w(b"startxref\n")
        w(f"{xref_start}\n".encode('ascii'))
        w(b"%%EOF")
        return buf.getvalue()

    rows_summary = _build_summary_rows()
    title = "Scoring settings summary"
    scope = f"League {league_id} • Season {season} • Week {week}"
    subtitle = f"{scope} | Source: {meta.get('used_source') or source}"

    # Try to use reportlab first, fall back to a minimal PDF if unavailable
    if not rows_summary:
        st.warning("No scoring settings to export right now. Try adjusting settings or reset to league settings, then export again.")
    else:
        pdf_bytes = _make_pdf_reportlab(title, subtitle, rows_summary)
        if pdf_bytes is None:
            pdf_bytes = _make_pdf_minimal(title, subtitle, rows_summary)
        st.download_button(
            label="Export scoring summary (PDF)",
            data=pdf_bytes,
            file_name=f"scoring_summary_{season}_wk{week}.pdf",
            mime="application/pdf",
            help="Downloads a concise PDF showing Before vs After scoring values. Uses reportlab if installed, else a basic PDF fallback.",
        )

    st.divider()
    # Flip Maximizer — place just above Players section
    with st.expander("Flip maximizer (experimental)", expanded=False):
        st.caption("Search random scoring presets to maximize flipped matchups for the chosen scope.")
        # Roster group for optimization (independent of other controls)
        fm_view = st.radio("Roster group (optimizer)", options=["All","Starters","Bench"], horizontal=True, index=1, key="fm_roster_group")
        fm_only_starters = True if fm_view == "Starters" else (False if fm_view == "Bench" else None)
        n_trials = st.slider("Trials", min_value=10, max_value=200, value=60, step=10, help="More trials = longer search")
        apply_best = st.checkbox("Apply best preset automatically", value=True, key="fm_apply_best")
        # Define a compact search space (reasonable ranges)
        search_space = {
            'pass_yds': (0.02, 0.08), 'pass_tds': (3.0, 6.0), 'int': (-4.0, 0.0), 'sack_taken': (-1.0, 0.0),
            'rush_yds': (0.05, 0.15), 'rush_tds': (4.0, 8.0), 'rush_att': (0.0, 0.3),
            'rec_yds': (0.05, 0.15), 'rec': (0.0, 1.5), 'rec_fd': (0.0, 0.5),
            'rush_fd': (0.0, 0.5), 'pass_fd': (0.0, 0.3), 'bonus_rec_te': (0.0, 1.0),
            'bonus_rush_100_199': (0.0, 3.0), 'bonus_rush_200p': (0.0, 4.0),
            'bonus_rec_100_199': (0.0, 3.0), 'bonus_rec_200p': (0.0, 4.0),
            'bonus_pass_300_399': (0.0, 3.0), 'bonus_pass_400p': (0.0, 4.0),
        }
        vary_receiving = st.checkbox("Vary receiving weights", value=True, key="fm_vary_rec")
        vary_rushing = st.checkbox("Vary rushing weights", value=True, key="fm_vary_rush")
        vary_qb = st.checkbox("Vary QB weights", value=True, key="fm_vary_qb")
        close_thresh = st.number_input("Close game threshold (pts)", min_value=0.0, value=10.0, step=0.5,
                                       help="Only count or weight flips when the official margin is within this many points (0 = disabled)", key="fm_close_thresh")
        use_weighting = st.checkbox("Weight flips by closeness", value=False,
                                    help="If enabled: weight = max(0, 1 - margin/threshold). If disabled: only count flips with margin <= threshold when threshold > 0.", key="fm_use_weighting")
        # Regularization: penalize large deviations from base
        reg_strength = st.slider("Stay near base (penalty)", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                                 help="Higher = prefer rule values closer to league base. 0 = ignore distance from base.", key="fm_reg")
        rng_seed = st.text_input("Random seed (optional)", value="", help="Set for reproducible search. Leave blank for random.", key="fm_seed")
        # Filter search space
        chosen_keys = []
        for k in search_space.keys():
            if k in ('rec','rec_yds','rec_fd','bonus_rec_te','bonus_rec_100_199','bonus_rec_200p') and not vary_receiving:
                continue
            if k in ('rush_yds','rush_tds','rush_att','rush_fd','bonus_rush_100_199','bonus_rush_200p') and not vary_rushing:
                continue
            if k in ('pass_yds','pass_tds','int','sack_taken','pass_fd','bonus_pass_300_399','bonus_pass_400p') and not vary_qb:
                continue
            chosen_keys.append(k)

        # Scope selection
        try:
            wk_int = int(week) if isinstance(week, int) else int(week or 1)
        except Exception:
            wk_int = 1
        scope_options = ["This week only", f"Season to date (1..{wk_int})"]
        try:
            _lastwk = detect_last_completed_week()
            if _lastwk and _lastwk != wk_int:
                scope_options.append(f"Full season (1..{_lastwk})")
        except Exception:
            pass
        scope_choice = st.radio("Scope", options=scope_options, index=0, key="fm_scope")

        run_search = st.button("Search for flip-max preset", type="primary", key="fm_run_search")
        if run_search:
            import random as _rnd
            progress = st.progress(0)
            best_flips = -1.0
            best_rules_overrides = None
            # Seed RNG if provided
            try:
                if rng_seed:
                    _ = int(rng_seed)
                    _rnd.seed(_)
            except Exception:
                pass

            # Local helper: recompute points for a given weekly report with candidate rules
            def _recompute_points_for_report(_report: dict, _rules: dict, _name_pos: dict) -> dict:
                pts = {}
                for team_id, tdata in (_report or {}).items():
                    for pname, weeks in (tdata.get('players') or {}).items():
                        total = 0.0
                        pos = _name_pos.get(pname)
                        if EXCLUDED_POS_SIM and pos in EXCLUDED_POS_SIM:
                            any_off = any(_has_offensive_keys(wk) for wk in weeks.values() if wk)
                            if not any_off:
                                pts[pname] = 0.0
                                continue
                        for wkstats in (weeks or {}).values():
                            if not wkstats:
                                continue
                            stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                            if pos:
                                stats_numeric['_pos'] = pos
                            bd = score_breakdown(stats_numeric, _rules)
                            total += sum(bd.values())
                        pts[pname] = total
                return pts

            def _eval_weeks():
                if scope_choice.startswith("Season to date"):
                    return list(range(1, int(wk_int) + 1))
                elif scope_choice.startswith("Full season"):
                    try:
                        lw = detect_last_completed_week()
                    except Exception:
                        lw = int(wk_int)
                    lw = lw or int(wk_int)
                    return list(range(1, int(lw) + 1))
                else:
                    return [int(wk_int)]

            weeks_to_eval = _eval_weeks()
            if not weeks_to_eval:
                st.info("No weeks to evaluate.")
            else:
                weekly_data = []
                for w in weeks_to_eval:
                    try:
                        r_w, pp_w, m_w = load_data(league_id, int(season), int(w), source, cache_dir, CACHE_VERSION)
                        matchups_by_id_w = m_w.get('matchups_by_id', {}) or {}
                        if not matchups_by_id_w:
                            continue
                        starters_w = m_w.get('starters_names_by_team', {}) or {}
                        players_w = m_w.get('players_names_by_team', {}) or {}
                        teams_w = set([tid for tids in matchups_by_id_w.values() for tid in tids])
                        roster_map_w = {}
                        for tid in teams_w:
                            st_set = set(starters_w.get(tid, set()))
                            all_set = set(players_w.get(tid, set()))
                            roster_map_w[tid] = (st_set if fm_only_starters is True else (all_set - st_set if fm_only_starters is False else all_set))
                        pf_before_w = {}
                        for tid in teams_w:
                            total = 0.0
                            for pname in roster_map_w.get(tid, set()):
                                total += float(pp_w.get(pname, 0.0))
                            pf_before_w[tid] = round(total, 2)
                        weekly_data.append({'report': r_w, 'name_pos': m_w.get('name_pos', {}), 'roster_map': roster_map_w, 'matchups': matchups_by_id_w, 'teams': list(teams_w), 'pf_before': pf_before_w})
                    except Exception:
                        continue

                if not weekly_data:
                    st.info("No matchup data found for the selected scope.")
                else:
                    to_vary = chosen_keys if chosen_keys else list(search_space.keys())
                    for i in range(1, n_trials + 1):
                        cand = dict(base_rules)
                        for key in to_vary:
                            lo, hi = search_space[key]
                            r = _rnd.random()
                            if r < 0.2:
                                val = lo
                            elif r > 0.8:
                                val = hi
                            else:
                                val = lo + (hi - lo) * _rnd.random()
                            cand[key] = float(val)

                        flips_total = 0.0
                        for wd in weekly_data:
                            cand_points_w = _recompute_points_for_report(wd['report'], cand, wd['name_pos'])
                            pf_after_w = {}
                            for tid in wd['teams']:
                                total = 0.0
                                for pname in wd['roster_map'].get(tid, set()):
                                    total += float(cand_points_w.get(pname, 0.0))
                                pf_after_w[tid] = round(total, 2)
                            for mid, tids in wd['matchups'].items():
                                if len(tids) < 2:
                                    continue
                                for j in range(0, len(tids), 2):
                                    pair = tids[j:j+2]
                                    if len(pair) != 2:
                                        continue
                                    a, b = pair[0], pair[1]
                                    before_a = wd['pf_before'].get(a,0.0); before_b = wd['pf_before'].get(b,0.0)
                                    w_before = 0 if before_a > before_b else (1 if before_b > before_a else None)
                                    w_after = 0 if pf_after_w.get(a,0.0) > pf_after_w.get(b,0.0) else (1 if pf_after_w.get(b,0.0) > pf_after_w.get(a,0.0) else None)
                                    if w_before is not None and w_after is not None and w_before != w_after:
                                        margin = abs(before_a - before_b)
                                        if use_weighting and close_thresh > 0:
                                            weight = max(0.0, 1.0 - (margin / float(close_thresh)))
                                            flips_total += weight
                                        elif close_thresh > 0:
                                            if margin <= close_thresh:
                                                flips_total += 1.0
                                        else:
                                            flips_total += 1.0
                        # Apply regularization penalty: subtract weighted normalized distance from base
                        if reg_strength > 0:
                            dist = 0.0
                            for k in to_vary:
                                lo, hi = search_space.get(k, (0.0, 1.0))
                                span = max(1e-9, hi - lo)
                                base_val = float(base_rules.get(k, lo))
                                v = float(cand.get(k, base_val))
                                # Normalize distance within [0,1]
                                d = abs(v - base_val) / span
                                dist += d
                            # Average distance across varied keys
                            dist = dist / max(1, len(to_vary))
                            objective = float(flips_total) - float(reg_strength) * float(dist)
                        else:
                            objective = float(flips_total)
                        if objective > best_flips:
                            best_flips = objective
                            best_rules_overrides = {k: cand[k] for k in to_vary}
                        progress.progress(i / n_trials)

                    # Report objective; note: if regularized, the score includes penalty so it's not pure flips
                    if reg_strength > 0:
                        st.markdown(f"Best objective (flips minus penalty): **{best_flips:.3f}** over {n_trials} trials across {len(weekly_data)} week(s)")
                    else:
                        if use_weighting:
                            st.markdown(f"Best weighted flips found: **{best_flips:.2f}** over {n_trials} trials across {len(weekly_data)} week(s)")
                        else:
                            st.markdown(f"Best flips found: **{int(best_flips)}** over {n_trials} trials across {len(weekly_data)} week(s)")
                    if best_rules_overrides:
                        st.json(best_rules_overrides)
                        if apply_best:
                            pending: dict[str, float] = {}
                            for rk, val in best_rules_overrides.items():
                                sk = UI_KEY_MAP_REV.get(rk)
                                if sk is not None:
                                    pending[sk] = float(val)
                            st.session_state['pending_preset_values'] = pending
                            st.session_state['do_apply_preset'] = True
                            st.success("Applying best preset…")
                            try:
                                st.rerun()
                            except Exception:
                                pass
                        else:
                            if st.button("Apply best preset", type="primary", key="fm_apply_btn"):
                                pending: dict[str, float] = {}
                                for rk, val in best_rules_overrides.items():
                                    sk = UI_KEY_MAP_REV.get(rk)
                                    if sk is not None:
                                        pending[sk] = float(val)
                                st.session_state['pending_preset_values'] = pending
                                st.session_state['do_apply_preset'] = True
                                try:
                                    st.rerun()
                                except Exception:
                                    pass

    st.markdown("### Players — before vs after (this week)")
    include_fas = st.checkbox("Include free agents (view-only)", value=False, help="Show players not rostered this week. Does not affect team totals/matchups.")

    # Recompute player points with simulated rules
    def recompute_player_points(_report, _rules, _name_pos, exclude_positions: set[str] | None = None):
        pts = {}
        for team_id, tdata in _report.items():
            for pname, weeks in tdata.get('players', {}).items():
                total = 0.0
                pos = _name_pos.get(pname)
                if exclude_positions and pos in exclude_positions:
                    # Do not exclude if weeks show offensive stat keys (name collision safety)
                    any_off = any(_has_offensive_keys(wkstats) for wkstats in weeks.values() if wkstats)
                    if not any_off:
                        pts[pname] = 0.0
                        continue
                for wk, wkstats in weeks.items():
                    if not wkstats:
                        continue
                    stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                    if pos:
                        stats_numeric['_pos'] = pos
                    bd = score_breakdown(stats_numeric, _rules)
                    total += sum(bd.values())
                pts[pname] = total
        return pts

    sim_player_points = recompute_player_points(report, sim_rules, meta.get('name_pos', {}), exclude_positions=EXCLUDED_POS_SIM)

    # Build player table
    def build_player_df(points_map, extra_points_map: dict | None = None):
        rows = []
        owner_to_manager = meta.get('owner_to_manager', {})
        owner_to_team_name = meta.get('owner_to_team_name', {})
        for team_id, tdata in report.items():
            starters_set = set(meta.get('starters_names_by_team', {}).get(team_id, set()))
            manager = tdata.get('manager') or owner_to_manager.get(str(team_id)) or owner_to_manager.get(team_id) or ""
            team_name = tdata.get('team_name') or owner_to_team_name.get(str(team_id)) or owner_to_team_name.get(team_id) or str(team_id)
            team_display = f"{team_name} ({manager})" if manager else team_name
            for pname, weeks in tdata.get("players", {}).items():
                pos = meta.get('name_pos', {}).get(pname)
                rows.append({
                    "player": pname,
                    "team": team_display,
                    "pos": pos,
                    "points": round(points_map.get(pname, 0.0), 2),
                })
        # Append free agents
        if extra_points_map:
            for pname, pts in extra_points_map.items():
                rows.append({
                    "player": pname,
                    "team": "Free Agent",
                    "pos": meta.get('name_pos', {}).get(pname),
                    "points": round(pts, 2),
                })
        df = pd.DataFrame(rows)
        return df

    # Recompute baseline (before) with base rules and excluded positions for fair comparison
    baseline_player_points = recompute_player_points(report, base_rules, meta.get('name_pos', {}), exclude_positions=EXCLUDED_POS_SIM)
    # Optionally extend with free agents (view-only)
    fa_points_before = {}
    fa_points_after = {}
    if include_fas:
        # Build set of names already in report
        report_names = set()
        for tdata in report.values():
            report_names.update((tdata.get('players') or {}).keys())
        week_all_stats = meta.get('week_all_player_stats', {}) or {}
        for pname, statmap in week_all_stats.items():
            # Exclude pseudo-team placeholders (e.g., TEAM_KC, TEAM_NYG)
            if isinstance(pname, str) and pname.upper().startswith("TEAM_"):
                continue
            if pname in report_names:
                continue
            # statmap is a single-week stats dict; apply exclusions cautiously (don't exclude if offensive keys present)
            pos = meta.get('name_pos', {}).get(pname)
            # Exclude irrelevant FA positions per request
            FA_EXCLUDE_POS = {"0","C","CB","DB","DE","DL","DT","K","P","LB","NT","OL","SS","T"}
            if pos in FA_EXCLUDE_POS:
                continue
            has_off = any(k.startswith('pass_') or k.startswith('rush_') or k.startswith('rec') or k.startswith('two_pt') for k in statmap.keys())
            if pos in EXCLUDED_POS_SIM and not has_off:
                continue
            # score for before
            bd_b = score_breakdown(statmap, base_rules)
            fa_points_before[pname] = sum(bd_b.values())
            # score for after
            bd_a = score_breakdown(statmap, sim_rules)
            fa_points_after[pname] = sum(bd_a.values())

    df_before = build_player_df(baseline_player_points, extra_points_map=(fa_points_before if include_fas else None))
    df_after = build_player_df(sim_player_points, extra_points_map=(fa_points_after if include_fas else None))
    df_players = df_before.merge(df_after, on=["player","team","pos"], how="outer", suffixes=("_before","_after")).fillna(0)
    # Points deltas and ranks
    df_players["delta"] = df_players["points_after"] - df_players["points_before"]
    df_players["rank_before"] = df_players["points_before"].rank(method="min", ascending=False).astype(int)
    df_players["rank_after"] = df_players["points_after"].rank(method="min", ascending=False).astype(int)
    df_players["rank_change"] = df_players["rank_before"] - df_players["rank_after"]  # + means moved up
    # Order by new rank
    df_players = df_players.sort_values(["rank_after","points_after"], ascending=[True, False]).reset_index(drop=True)
    # Display formatting with Styler so dtype stays numeric for proper sorting
    cols_players = ["rank_after","rank_change","player","pos","team","points_before","points_after","delta"]
    df_display = df_players[cols_players].copy()
    styler_players = df_display.style.format({
        "rank_change": "{:+d}",
        "points_before": "{:.2f}",
        "points_after": "{:.2f}",
        "delta": "{:+.2f}",
    })
    # Add position-filtered tabs for Players table
    tabs_players = st.tabs(["All Players", "QB", "RB", "WR", "TE"])
    with tabs_players[0]:
        st.dataframe(styler_players, use_container_width=True, hide_index=True)
    def _show_pos_tab(pos: str, tab_idx: int):
        with tabs_players[tab_idx]:
            _df = df_players[df_players['pos'] == pos][cols_players].copy()
            # keep formatting consistent
            _sty = _df.style.format({
                "rank_change": "{:+d}",
                "points_before": "{:.2f}",
                "points_after": "{:.2f}",
                "delta": "{:+.2f}",
            })
            st.dataframe(_sty, use_container_width=True, hide_index=True)
    _show_pos_tab("QB", 1)
    _show_pos_tab("RB", 2)
    _show_pos_tab("WR", 3)
    _show_pos_tab("TE", 4)

    # Top-N sections moved below Matchups

    st.divider()
    st.markdown("### Matchups — before vs after (this week)")
    starters_map = meta.get('starters_names_by_team', {}) or {}
    players_map = meta.get('players_names_by_team', {}) or {}
    owner_to_manager = meta.get('owner_to_manager', {})
    owner_to_team_name = meta.get('owner_to_team_name', {})
    matchups_by_id = meta.get('matchups_by_id', {}) or {}
    view_sim = st.radio("Roster group (sim)", options=["All","Starters","Bench"], horizontal=True, index=1)
    only_starters_sim = True if view_sim == "Starters" else (False if view_sim == "Bench" else None)

    def _player_week_points_with_rules(pname: str, weeks_map: dict, rules: dict, exclude_positions: set[str] | None = None) -> float:
        pos = meta.get('name_pos', {}).get(pname)
        if exclude_positions and pos in exclude_positions:
            any_off = any(_has_offensive_keys(wkstats) for wkstats in (weeks_map or {}).values() if wkstats)
            if not any_off:
                return 0.0
        total = 0.0
        for _wk, wkstats in (weeks_map or {}).items():
            if not wkstats:
                continue
            stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
            if pos:
                stats_numeric['_pos'] = pos
            bd = score_breakdown(stats_numeric, rules)
            total += sum(bd.values())
        return round(total, 2)

    def team_total_with(points_map, team_id: str, only_starters: bool | None, rules_override: dict | None = None) -> float:
        starters = set(starters_map.get(team_id, set()))
        all_players = set(players_map.get(team_id, set()))
        roster_players = starters if only_starters is True else (all_players - starters if only_starters is False else all_players)
        total = 0.0
        team_players = (report.get(team_id, {}) or {}).get('players', {})
        for pname in roster_players:
            weeks_map = team_players.get(pname, {})
            if rules_override is None:
                total += _player_week_points_with_rules(pname, weeks_map, meta.get('scoring_rules', {}), exclude_positions=None)
            else:
                total += _player_week_points_with_rules(pname, weeks_map, rules_override, exclude_positions=EXCLUDED_POS_SIM)
        return round(total, 2)

    def team_display(team_id: str) -> str:
        manager = report.get(team_id, {}).get('manager') or owner_to_manager.get(str(team_id)) or ''
        team_name = report.get(team_id, {}).get('team_name') or owner_to_team_name.get(str(team_id)) or str(team_id)
        return f"{team_name} ({manager})" if manager else team_name

    import statistics as _stats
    matchup_ids_sorted = sorted(matchups_by_id.keys(), key=lambda x: (x is None, str(x)))
    # Precompute medians and totals
    teams_this_week = set([tid for tids in matchups_by_id.values() for tid in tids])
    pf_before = {tid: team_total_with(baseline_player_points, tid, only_starters_sim, base_rules) for tid in teams_this_week}
    pf_after = {tid: team_total_with(sim_player_points, tid, only_starters_sim, sim_rules) for tid in teams_this_week}
    med_before = _stats.median(list(pf_before.values())) if pf_before else 0.0
    med_after = _stats.median(list(pf_after.values())) if pf_after else 0.0

    # Preset impact summary: flipped matchups, avg delta by position, top risers/fallers
    try:
        flips_total = 0
        for mid in matchup_ids_sorted:
            tids = matchups_by_id.get(mid, [])
            if not tids:
                continue
            for i in range(0, len(tids), 2):
                pair = tids[i:i+2]
                if len(pair) != 2:
                    continue
                w_before = w_after = None
                if pf_before.get(pair[0],0.0) > pf_before.get(pair[1],0.0):
                    w_before = 0
                elif pf_before.get(pair[1],0.0) > pf_before.get(pair[0],0.0):
                    w_before = 1
                if pf_after.get(pair[0],0.0) > pf_after.get(pair[1],0.0):
                    w_after = 0
                elif pf_after.get(pair[1],0.0) > pf_after.get(pair[0],0.0):
                    w_after = 1
                if w_before is not None and w_after is not None and w_before != w_after:
                    flips_total += 1
        with st.expander("Preset impact summary", expanded=False):
            st.markdown(f"Matchups flipped under current settings: **{flips_total}**")
            include_fas_impact = st.checkbox("Include free agents in impact lists", value=False, key="sim_include_fas_impact")
            _dfp = df_players.copy()
            if not include_fas_impact:
                _dfp = _dfp[_dfp['team'] != 'Free Agent']
            # Average delta by position
            try:
                pos_avg = _dfp.groupby('pos', dropna=False)['delta'].mean().reset_index().rename(columns={'delta':'avg_delta'})
                pos_avg = pos_avg[pos_avg['pos'].isin(['QB','RB','WR','TE'])]
                st.dataframe(pos_avg.sort_values('pos'), use_container_width=True, hide_index=True)
            except Exception:
                pass
            # Top risers/fallers by position
            tab_risers = st.tabs(["QB", "RB", "WR", "TE"])
            def _top_lists_for(pos: str, idx: int):
                with tab_risers[idx]:
                    sdf = _dfp[_dfp['pos']==pos].sort_values('delta', ascending=False)
                    rises = sdf.head(5)[['player','team','delta','rank_change']]
                    falls = sdf.tail(5).sort_values('delta', ascending=True)[['player','team','delta','rank_change']]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("Top 5 risers (Δ pts)")
                        st.dataframe(rises.style.format({'delta': '{:+.2f}', 'rank_change': '{:+d}'}), use_container_width=True, hide_index=True)
                    with c2:
                        st.markdown("Top 5 fallers (Δ pts)")
                        st.dataframe(falls.style.format({'delta': '{:+.2f}', 'rank_change': '{:+d}'}), use_container_width=True, hide_index=True)
            _top_lists_for('QB', 0)
            _top_lists_for('RB', 1)
            _top_lists_for('WR', 2)
            _top_lists_for('TE', 3)
    except Exception:
        pass


    for mid in matchup_ids_sorted:
        tids = matchups_by_id.get(mid, [])
        if not tids:
            continue
        for i in range(0, len(tids), 2):
            pair = tids[i:i+2]
            cols = st.columns(2)
            # Determine winners
            winner_before = winner_after = None
            if len(pair) == 2:
                if pf_before.get(pair[0],0.0) > pf_before.get(pair[1],0.0):
                    winner_before = 0
                elif pf_before.get(pair[1],0.0) > pf_before.get(pair[0],0.0):
                    winner_before = 1
                if pf_after.get(pair[0],0.0) > pf_after.get(pair[1],0.0):
                    winner_after = 0
                elif pf_after.get(pair[1],0.0) > pf_after.get(pair[0],0.0):
                    winner_after = 1
            # Pair-level indicator if outcome flipped under simulated scoring
            if len(pair) == 2:
                flipped = (winner_before is not None and winner_after is not None and winner_before != winner_after)
                if flipped:
                    st.markdown("**🚨 FLIPPED! 🚨**")
            for idx, tid in enumerate(pair):
                with cols[idx]:
                    icons = ""
                    if winner_after is not None and idx == winner_after:
                        icons += " 🏆"
                    if pf_after.get(tid,0.0) > med_after:
                        icons += " 🟢"
                    st.markdown(f"##### {team_display(tid)}{icons}")
                    before_val = pf_before.get(tid, 0.0)
                    after_val = pf_after.get(tid, 0.0)
                    delta_val = after_val - before_val
                    st.markdown(f"Total (after): {after_val:.2f}  •  Δ: {delta_val:+.2f}")

    # (Top-N positional distribution graphs moved to bottom of Simulator)

    # (Top-N by team/manager moved below the positional distribution charts)

    with st.expander("Explain a team total (per-player)"):
        # Pick a single team to inspect per-player contributions
        all_team_ids = sorted({tid for tids in matchups_by_id.values() for tid in tids})
        options = [(tid, team_display(tid)) for tid in all_team_ids]
        if options:
            selected_label = st.selectbox("Team", options=[lbl for _, lbl in options])
            # Map label back to id
            selected_tid = next((tid for tid, lbl in options if lbl == selected_label), None)
            if selected_tid is not None:
                starters = set(starters_map.get(selected_tid, set()))
                all_players_set = set(players_map.get(selected_tid, set()))
                roster_players = starters if only_starters_sim is True else (all_players_set - starters if only_starters_sim is False else all_players_set)
                team_players = (report.get(selected_tid, {}) or {}).get('players', {})
                rows_dbg = []
                for pname in sorted(roster_players):
                    weeks_map = team_players.get(pname, {})
                    pos = meta.get('name_pos', {}).get(pname)
                    # Official (matchups-style): no excluded positions
                    official = _player_week_points_with_rules(pname, weeks_map, base_rules, exclude_positions=None)
                    # Simulator after: use sim_rules and same exclusion policy as main sim views
                    sim_after = _player_week_points_with_rules(pname, weeks_map, sim_rules, exclude_positions=EXCLUDED_POS_SIM)
                    rows_dbg.append({
                        'player': pname,
                        'pos': pos,
                        'official': official,
                        'sim_after': sim_after,
                        'delta': round(sim_after - official, 2),
                    })
                import pandas as _pd
                df_dbg = _pd.DataFrame(rows_dbg)
                if not df_dbg.empty:
                    df_dbg = df_dbg.sort_values(['official','sim_after'], ascending=[False, False]).reset_index(drop=True)
                    # Signed delta formatting
                    df_dbg['delta'] = df_dbg['delta'].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(df_dbg[['player','pos','official','sim_after','delta']], use_container_width=True, hide_index=True)
                    st.caption(f"Official total: {df_dbg['official'].astype(float).sum():.2f} • Simulator after total: {df_dbg['sim_after'].astype(float).sum():.2f}")

    st.divider()
    st.markdown("### Rankings — before vs after (cumulative through selected week)")
    include_median = st.checkbox("Include median bonus (sim)", value=True)

    # Build cumulative standings for before and after
    def build_standings_cumulative(points_func):
        standings = {}
        def ensure(tid, name_provider):
            if tid not in standings:
                standings[tid] = {'team': name_provider(tid), 'wins':0, 'losses':0, 'pf':0.0, 'pa':0.0}
        def name_provider(tid):
            return team_display(tid)
        # If no specific week is selected, nothing to accumulate
        if week is None:
            return {}
        for w in range(1, int(week)+1):
            r_w, pp_w, m_w = load_data(league_id, int(season), int(w), source, cache_dir, CACHE_VERSION)
            starters_map_w = m_w.get('starters_names_by_team', {}) or {}
            matchups_by_id_w = m_w.get('matchups_by_id', {}) or {}
            # compute per-team PF for this week using provided points_func
            team_list = set([t for lst in matchups_by_id_w.values() for t in lst])
            pf_week = {}
            for tid in team_list:
                names = starters_map_w.get(tid, set())
                pf = 0.0
                for pname in names:
                    pf += float(points_func(r_w, m_w, pname, pp_w))
                pf_week[tid] = round(pf,2)
            # head-to-head
            for _, tids in matchups_by_id_w.items():
                if len(tids) != 2:
                    continue
                a, b = tids[0], tids[1]
                ensure(a, name_provider); ensure(b, name_provider)
                pa = pf_week.get(a,0.0); pb = pf_week.get(b,0.0)
                standings[a]['pf'] += pa; standings[b]['pf'] += pb
                standings[a]['pa'] += pb; standings[b]['pa'] += pa
                if pa > pb:
                    standings[a]['wins'] += 1; standings[b]['losses'] += 1
                elif pb > pa:
                    standings[b]['wins'] += 1; standings[a]['losses'] += 1
            # median bonus
            if include_median and team_list:
                import statistics as _st
                med = _st.median(list(pf_week.values())) if pf_week else 0.0
                for tid in team_list:
                    ensure(tid, name_provider)
                    if pf_week.get(tid,0.0) > med:
                        standings[tid]['wins'] += 1
                    elif pf_week.get(tid,0.0) < med:
                        standings[tid]['losses'] += 1
        return standings

    # points_func for before uses cached per-player points (pp_w)
    def points_before(_report, _meta_w, pname, pp_w):
        # recompute with base rules and excluded positions (but don't exclude if offensive stats present)
        pos = meta.get('name_pos', {}).get(pname)
        for tdata in _report.values():
            if pname in tdata.get('players', {}):
                weeks_map = tdata['players'][pname]
                if pos in EXCLUDED_POS_SIM:
                    any_off = any(_has_offensive_keys(wkstats) for wkstats in weeks_map.values() if wkstats)
                    if not any_off:
                        return 0.0
                for wkstats in weeks_map.values():
                    stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                    if pos:
                        stats_numeric['_pos'] = pos
                    bd = score_breakdown(stats_numeric, base_rules)
                    return sum(bd.values())
        return 0.0

    # points_func for after recomputes with sim_rules
    def points_after(_report, _meta_w, pname, pp_w):
        # find the player's weekly stats in _report
        # build a tiny lookup cache per invocation for speed could be added later
        for tdata in _report.values():
            if pname in tdata.get('players', {}):
                weeks_map = tdata['players'][pname]
                # use any week's stats (only one week per _report)
                for wkstats in weeks_map.values():
                    pos = meta.get('name_pos', {}).get(pname)
                    if pos in EXCLUDED_POS_SIM:
                        any_off = any(_has_offensive_keys(wkstats2) for wkstats2 in weeks_map.values() if wkstats2)
                        if not any_off:
                            return 0.0
                    stats_numeric = {k: float(v) for k, v in wkstats.items() if isinstance(v, (int, float))}
                    pos = meta.get('name_pos', {}).get(pname)
                    if pos:
                        stats_numeric['_pos'] = pos
                    bd = score_breakdown(stats_numeric, sim_rules)
                    return sum(bd.values())
        return 0.0

    standings_before = build_standings_cumulative(points_before)
    standings_after = build_standings_cumulative(points_after)

    # Merge into a comparison table
    all_tids = sorted(set(standings_before.keys()) | set(standings_after.keys()))
    rows = []
    for tid in all_tids:
        b = standings_before.get(tid, {'team': team_display(tid), 'wins':0,'losses':0,'pf':0.0,'pa':0.0})
        a = standings_after.get(tid, {'team': team_display(tid), 'wins':0,'losses':0,'pf':0.0,'pa':0.0})
        rows.append({
            'team': a['team'] or b['team'],
            'wins_before': b['wins'], 'losses_before': b['losses'], 'pf_before': round(b['pf'],2), 'pa_before': round(b['pa'],2),
            'wins_after': a['wins'], 'losses_after': a['losses'], 'pf_after': round(a['pf'],2), 'pa_after': round(a['pa'],2),
            'wins_delta': a['wins'] - b['wins'], 'pf_delta': round(a['pf'] - b['pf'], 2),
        })
    df_stand = pd.DataFrame(rows)
    if not df_stand.empty:
        df_stand = df_stand.sort_values(['wins_after','pf_after'], ascending=[False, False]).reset_index(drop=True)
        df_stand.insert(0, 'rank_after', range(1, len(df_stand)+1))
        styler_stand = df_stand.style.format({
            'wins_before': '{:d}',
            'losses_before': '{:d}',
            'pf_before': '{:.2f}',
            'pa_before': '{:.2f}',
            'wins_after': '{:d}',
            'losses_after': '{:d}',
            'pf_after': '{:.2f}',
            'pa_after': '{:.2f}',
            'wins_delta': '{:+d}',
            'pf_delta': '{:+.2f}',
        })
        st.dataframe(styler_stand, use_container_width=True, hide_index=True)
    else:
        st.info("No standings available.")

    # --- Bottom: Top-N positional distribution (simulated) with side-by-side charts ---
    st.divider()
    st.markdown("### Top-N positional distribution (simulated)")
    sim_top_n = st.selectbox(
        "Top N (sim)",
        options=[12, 25, 50, 75, 100],
        index=0,
        key="sim_top_n_choice",
        help="Top-N by points for After (simulated) and Before (official), each ranked independently."
    )
    metric_choice = st.selectbox(
        "Display metric",
        options=["Count", "Percent", "Avg points", "Points %"],
        index=0,
        key="sim_topn_metric",
        help="Choose what both charts display."
    )
    valid_positions = ["QB", "RB", "WR", "TE"]

    def _compute_topn_metrics(df_src: pd.DataFrame, points_col: str) -> dict:
        df_src = df_src.copy()
        df_src = df_src[df_src["pos"].isin(valid_positions)]
        df_ranked = df_src.sort_values(points_col, ascending=False)
        df_top = df_ranked.head(int(sim_top_n))
        if df_top.empty:
            return {
                "denom": 0,
                "count": pd.Series([0,0,0,0], index=valid_positions, dtype=float),
                "percent": pd.Series([0,0,0,0], index=valid_positions, dtype=float),
                "avg_points": pd.Series([0,0,0,0], index=valid_positions, dtype=float),
                "points_percent": pd.Series([0,0,0,0], index=valid_positions, dtype=float),
            }
        denom = max(1, len(df_top))
        counts = df_top["pos"].value_counts().reindex(valid_positions, fill_value=0).astype(float)
        perc = (counts / float(denom) * 100.0)
        avg_pts = df_top.groupby("pos")[points_col].mean().reindex(valid_positions)
        sum_pts = df_top.groupby("pos")[points_col].sum().reindex(valid_positions, fill_value=0.0)
        total_pts = float(df_top[points_col].sum()) or 1.0
        pts_pct = (sum_pts / total_pts * 100.0)
        return {
            "denom": denom,
            "count": counts,
            "percent": perc,
            "avg_points": avg_pts,
            "points_percent": pts_pct,
        }

    m_after = _compute_topn_metrics(df_players.rename(columns={"points_after": "points"}), "points")
    m_before = _compute_topn_metrics(df_players.rename(columns={"points_before": "points"}), "points")

    if metric_choice == "Count":
        series_after = m_after["count"].reindex(valid_positions)
        series_before = m_before["count"].reindex(valid_positions)
    elif metric_choice == "Percent":
        series_after = m_after["percent"].reindex(valid_positions)
        series_before = m_before["percent"].reindex(valid_positions)
    elif metric_choice == "Avg points":
        series_after = m_after["avg_points"].reindex(valid_positions)
        series_before = m_before["avg_points"].reindex(valid_positions)
    else:
        series_after = m_after["points_percent"].reindex(valid_positions)
        series_before = m_before["points_percent"].reindex(valid_positions)

    # Clustered (grouped) bar chart: positions on x-axis, two columns for Before/After using Altair
    chart_df = pd.DataFrame({
        "pos": valid_positions,
        "Before (Actual)": series_before.values,
        "After Sim": series_after.values,
    })
    # Melt to long form for Altair
    chart_long = chart_df.melt(id_vars=["pos"], var_name="series_label", value_name="value")
    # Axis title varies by metric
    if metric_choice == "Percent" or metric_choice == "Points %":
        y_title = "Percent"
        y_axis = alt.Axis(title=y_title, grid=True)
    elif metric_choice == "Avg points":
        y_title = "Avg points"
        y_axis = alt.Axis(title=y_title, grid=True)
    else:
        y_title = "Players"
        y_axis = alt.Axis(title=y_title, grid=True)
    after_color = "#1f77b4"  # blue/teal
    before_color = "#ff7f0e"  # orange
    color_scale = alt.Scale(domain=["After Sim", "Before (Actual)"], range=[after_color, before_color])
    clustered = (
        alt.Chart(chart_long)
        .mark_bar()
        .encode(
            x=alt.X("pos:N", title=None),
            y=alt.Y("value:Q", axis=y_axis),
            color=alt.Color("series_label:N", scale=color_scale, legend=alt.Legend(title=None)),
            xOffset="series_label:N",
            tooltip=["pos:N", "series_label:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=320)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    st.altair_chart(clustered, use_container_width=True)
    # Exact values table beneath (with Change = After - Before)
    _before_vals = series_before.round(2).values
    _after_vals = series_after.round(2).values
    _change_vals = (series_after - series_before).round(2).values
    tbl = pd.DataFrame({
        "pos": valid_positions,
        "After (Sim)": _after_vals,
        "Before (Actual)": _before_vals,
        "Change": _change_vals,
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption(f"Each series uses its own Top-{int(sim_top_n)} ranking.")

    # Now place Top-N by team/manager (simulated) below as a single combined table with Change columns
    st.markdown("#### Top-N by team/manager (simulated)")
    def _topn_team_counts(df_src: pd.DataFrame, points_col: str, top_n_val: int) -> pd.DataFrame:
        valid_positions = ["QB", "RB", "WR", "TE"]
        df_ranked = df_src[df_src["pos"].isin(valid_positions)].sort_values(points_col, ascending=False)
        df_top = df_ranked.head(top_n_val)
        rows = []
        for team_label, group in df_top.groupby('team', dropna=False):
            pos_counts = group['pos'].value_counts().reindex(valid_positions, fill_value=0)
            row = {
                'team': team_label if pd.notna(team_label) else '',
                'total': int(len(group)),
            }
            for p in valid_positions:
                row[p] = int(pos_counts.get(p, 0))
            rows.append(row)
        return pd.DataFrame(rows)

    _topn_val = int(st.session_state.get('sim_top_n_choice', int(sim_top_n)))
    df_after_src = df_players.rename(columns={"points_after": "points"})
    df_before_src = df_players.rename(columns={"points_before": "points"})
    df_after_counts = _topn_team_counts(df_after_src, "points", _topn_val)
    df_before_counts = _topn_team_counts(df_before_src, "points", _topn_val)

    # Merge and compute changes
    combined = df_after_counts.merge(df_before_counts, on='team', how='outer', suffixes=('_after','_before')).fillna(0)
    # Ensure numeric types
    for col in combined.columns:
        if col != 'team':
            try:
                combined[col] = combined[col].astype(int)
            except Exception:
                combined[col] = combined[col]
    # Add change columns for total and each position
    for base in ['total','QB','RB','WR','TE']:
        a = f"{base}_after"; b = f"{base}_before"; c = f"{base}_change"
        combined[c] = combined.get(a, 0) - combined.get(b, 0)
    # Column order: team, total_after, total_before, total_change, then per-pos triplets
    ordered_cols = ['team','total_after','total_before','total_change']
    for p in ['QB','RB','WR','TE']:
        ordered_cols += [f"{p}_after", f"{p}_before", f"{p}_change"]
    for col in combined.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    combined = combined.reindex(columns=ordered_cols)
    # Sort by total_after desc, then WR/RB/QB/TE counts (after)
    sort_cols = ['total_after','WR_after','RB_after','QB_after','TE_after']
    combined = combined.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
    st.dataframe(combined, use_container_width=True, hide_index=True)
