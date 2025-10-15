# Fantasy Scoring Explorer (Streamlit)

A simple Streamlit dashboard to browse league scoring results and drill into per-stat breakdowns.

## Quick start

1. Install dependencies (in your active Python env):

```powershell
pip install -r requirements.txt
```

2. Run the app:

```powershell
streamlit run .\streamlit_app.py
```

3. In the sidebar, set:
- League ID (defaults to your league)
- Season and Week
- Source (auto/sleeper/pbp)
- Cache dir (optional)

4. Use the "Explain player" input to see a stat-by-stat breakdown for a specific player. The exact display name matches the main table.

## Notes
- The app calls the same programmatic API as the CLI (`build_report`) and uses `score_breakdown` for the per-stat contributions.
- It respects your league's scoring settings when available. You can still override via CLI if needed.
- Cached downloads are stored under the directory you specify.
