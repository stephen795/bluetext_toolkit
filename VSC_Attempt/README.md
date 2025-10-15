Fantasy Football Scoring Toolkit
================================

Simulate how alternative scoring settings impact player points, weekly matchups, and (eventually) season outcomes. Supports importing real Sleeper league rosters & scoring, historical play-by-play (nflfastR style) or current-season weekly Sleeper stats with automatic fallback, plus flexible bucket / tier bonuses.

Key Features
------------
* Sleeper league import (rosters + scoring settings)
* Data sources:
  * Full play-by-play (historical seasons) with reception yard buckets & rush attempts extracted
  * Sleeper weekly stats fallback (for current or missing seasons/weeks)
* Automatic fallback: if PBP fetch fails (or `--source sleeper` chosen) weekly stats are used; optional bucket inference
* Bucket-based reception bonuses (rec_0_4 ... rec_40p) replacing per-reception PPR if desired
* Non-stacking yardage tiers (e.g., 100+ / 200+ bonuses)
* TE-specific reception bonus (`bonus_rec_te`)
* Rush attempt scoring (`rush_att`)
* Fuzzy name matching with configurable threshold & manual overrides
* File-based caching for PBP and weekly stats
* Debug raw source snippet output
* Summary table + JSON structured dump for downstream processing/tests

Installation
------------
Install dependencies (example):

```powershell
pip install -r requirements.txt
```

Sample Simulation (Synthetic Data)
----------------------------------
```powershell
python -m src.simulate --data data/sample_season.json --scoring data/sample_scoring.json
```

Sleeper League Import
---------------------
```powershell
python -m src.cli import-sleeper --league <LEAGUE_ID> --season 2024 --week 1 \
  --match-threshold 0.78 --assume-buckets infer --cache-dir .cache
```

Common Flags
------------
* `--league <id>`: Sleeper league id (required)
* `--season <year>`: Season year
* `--week <n>`: Specific week (omit for all weeks found in source/PBP)
* `--source {auto|pbp|sleeper}`:
  * `pbp` = force play-by-play
  * `sleeper` = force weekly stats
  * `auto` = try PBP then fallback to weekly
* `--scoring file.json`: Override league scoring with custom JSON
* `--assume-buckets {infer|none}`: When weekly stats lack bucket counts, infer (distribute receptions into yard buckets) or skip
* `--match-threshold <0-1>`: Fuzzy match threshold (higher => stricter)
* `--overrides overrides.json`: Manual name mapping (canonical -> Sleeper full_name)
* `--cache-dir path`: Store cached API responses and PBP downloads
* `--debug-source`: Emit truncated raw weekly stats block (guards at 600 chars)
* `--no-infer`: Disable bucket inference entirely

Overrides File Format
---------------------
JSON mapping of canonical display name to Sleeper's full_name string:
```json
{
  "Josh Allen": "J AllenQ",
  "Kenneth Walker III": "K Walker"
}
```
During output, the canonical (left side) appears in tables; the right side is used for matching.

Custom Scoring JSON
-------------------
You can fully define scoring (overrides Sleeper league settings when provided):
```json
{
  "pass_yds": 0.04,
  "pass_tds": 4,
  "int": -2,
  "rush_yds": 0.1,
  "rush_tds": 6,
  "rec": 0,
  "rec_yds": 0.1,
  "rec_tds": 6,
  "rush_att": 0.25,
  "rec_0_4": 0.5,
  "rec_5_9": 0.75,
  "rec_10_19": 1.0,
  "rec_20_29": 1.0,
  "rec_30_39": 1.25,
  "rec_40p": 1.5,
  "bonus_rec_te": 0.75,
  "tier_pass_yds": [ [300, 3], [400, 5] ],
  "tier_rush_yds": [ [100, 3] ]
}
```

Caching
-------
Provide `--cache-dir` to reuse previously fetched PBP or weekly stats. Cache keys are SHA256 hashes of request parameters; delete the directory to invalidate.

Output Structure
----------------
Two sentinels delimit machine-readable sections:
* `SUMMARY_TABLE_START` ... summary table lines
* `REPORT_JSON_START` ... pretty-printed JSON of the internal report `{ team_id -> { players -> { player_name -> { week -> stats }}}}`

Testing
-------
```powershell
python -m unittest discover -s tests -p test_*.py -v
```

Selected Tests
--------------
* `test_custom_league_scoring.py`: Verifies exact scoring for provided players
* `test_weekly_stats_fallback.py`: Exercises PBP failure -> weekly stats fallback path
* `test_caching.py`: Ensures second run hits cache (no refetch)
* `test_cli_overrides.py`: Validates canonical override display
* `test_name_matcher.py`: Fuzzy matching + overrides logic

Development Ideas / Next Steps
------------------------------
* Persist season-long standings aggregation
* Add lineup optimization vs. bench
* Enrich with opponent defensive adjustments
* Export CSV/Parquet
* Add type checking & lint (mypy / ruff)

License
-------
Prototype / personal use; add a LICENSE file if distributing.