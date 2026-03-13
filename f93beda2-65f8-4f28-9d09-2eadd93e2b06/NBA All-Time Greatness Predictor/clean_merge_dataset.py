
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Step 5: Clean, Normalize & Merge All Data
# Produces:
#   1. career_df   — one row per player (career totals + advanced + honors)
#   2. seasons_df  — per-season longitudinal table (for age curve modeling)
# ─────────────────────────────────────────────────────────────────

# ── Build career DataFrame ────────────────────────────────────────
career_df = pd.DataFrame(all_career_records)
print(f"Career records loaded: {len(career_df)} players")
print(f"  Active: {(career_df['player_status']=='active').sum()}")
print(f"  Historical: {(career_df['player_status']=='historical').sum()}")

# ── Add per-game stats (derived from career totals) ──────────────
career_df["career_ppg"]  = career_df["career_pts"]  / career_df["career_g"]
career_df["career_rpg"]  = career_df["career_trb"]  / career_df["career_g"]
career_df["career_apg"]  = career_df["career_ast"]  / career_df["career_g"]
career_df["career_spg"]  = career_df["career_stl"]  / career_df["career_g"]
career_df["career_bpg"]  = career_df["career_blk"]  / career_df["career_g"]
career_df["career_topg"] = career_df["career_tov"]  / career_df["career_g"]
career_df["career_mpg"]  = career_df["career_mp"]   / career_df["career_g"]

# ── Merge honors data ─────────────────────────────────────────────
honors_cols = ["player_name","mvp","dpoy","finals_mvp","all_nba_1st","all_nba_2nd",
               "all_nba_3rd","championships","allstar","honors_index"]
career_df = career_df.merge(
    honors_df[honors_cols],
    on="player_name",
    how="left"
)

# ── Fill missing honors with 0 (players not in honors DB) ────────
honor_fill_cols = ["mvp","dpoy","finals_mvp","all_nba_1st","all_nba_2nd",
                   "all_nba_3rd","championships","allstar","honors_index"]
career_df[honor_fill_cols] = career_df[honor_fill_cols].fillna(0)

# ── Build per-season longitudinal DataFrame ───────────────────────
if all_season_records:
    seasons_df = pd.concat(all_season_records, ignore_index=True)
    print(f"\nPer-season records: {len(seasons_df)} rows")
    print(f"Columns: {list(seasons_df.columns[:15])}…")
else:
    seasons_df = pd.DataFrame()
    print("\n⚠️  No season records available")

# ── Clean season DataFrame ────────────────────────────────────────
def _safe_num(series):
    return pd.to_numeric(series, errors='coerce')

if not seasons_df.empty:
    # Standardize critical numeric columns
    numeric_season_cols = {
        "season_age": "age",
        "season_g": "games",
        "season_mp": "mp",
        "season_pts": "pts",
        "season_trb": "trb",
        "season_ast": "ast",
        "season_stl": "stl",
        "season_blk": "blk",
        "season_per": "per",
        "season_ts_pct": "ts_pct",
        "season_ws": "ws",
        "season_ws_per48": "ws_per_48",
        "season_bpm": "bpm",
        "season_vorp": "vorp",
        "season_obpm": "obpm",
        "season_dbpm": "dbpm",
        "season_fg_pct": "fg_pct",
        "season_efg_pct": "efg_pct",
    }

    for new_col, src_col in numeric_season_cols.items():
        if src_col in seasons_df.columns:
            seasons_df[new_col] = _safe_num(seasons_df[src_col])
        else:
            seasons_df[new_col] = np.nan

    # Per-game season stats
    seasons_df["season_ppg"]  = seasons_df["season_pts"] / seasons_df["season_g"].replace(0, np.nan)
    seasons_df["season_rpg"]  = seasons_df["season_trb"] / seasons_df["season_g"].replace(0, np.nan)
    seasons_df["season_apg"]  = seasons_df["season_ast"] / seasons_df["season_g"].replace(0, np.nan)

    # Parse birth year / season year for age curve modeling
    # season_year format is e.g. "2023-24" → keep as string but extract start year
    if "season_year" in seasons_df.columns:
        seasons_df["season_start_year"] = _safe_num(
            seasons_df["season_year"].fillna("").str.extract(r'^(\d{4})')[0]
        )

    # Keep only columns we actually need
    keep_cols = [
        "player_name", "slug", "player_status", "season_year", "season_start_year",
        "season_age", "team_name_abbr", "pos",
        "season_g", "season_mp",
        "season_pts", "season_trb", "season_ast", "season_stl", "season_blk",
        "season_ppg", "season_rpg", "season_apg",
        "season_per", "season_ts_pct", "season_ws", "season_ws_per48",
        "season_bpm", "season_vorp", "season_obpm", "season_dbpm",
        "season_fg_pct", "season_efg_pct",
    ]
    available_cols = [c for c in keep_cols if c in seasons_df.columns]
    seasons_df = seasons_df[available_cols].copy()

    # Drop rows with no age data (can't use for age curves)
    seasons_df = seasons_df.dropna(subset=["season_age"]).reset_index(drop=True)
    print(f"Clean season rows (with age): {len(seasons_df)}")

# ── Critical column null audit ────────────────────────────────────
print("\n── Career DataFrame Null Audit ──")
critical_cols = ["career_g", "career_pts", "career_per", "career_ws",
                 "career_bpm", "career_vorp", "player_status"]
null_counts = career_df[critical_cols].isna().sum()
print(null_counts.to_string())

# Players with NaN in key advanced stats (usually pre-1974 era or ABA era)
nan_adv = career_df[career_df["career_per"].isna()]["player_name"].tolist()
print(f"\nPlayers with missing advanced stats (pre-BPM era): {nan_adv}")

# ── Final counts ──────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"✓ CAREER TABLE:  {len(career_df)} players × {len(career_df.columns)} columns")
print(f"✓ SEASONS TABLE: {len(seasons_df)} season-rows × {len(seasons_df.columns)} columns")
print(f"\n── Active benchmark players in career_df ──")
print(career_df[career_df["player_status"]=="active"][
    ["player_name","career_g","career_ppg","career_per","career_ws","career_vorp","career_bpm"]
].round(1).to_string(index=False))
