
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Step 6: Final Validation + Save
# Validates data quality, confirms the 5 active benchmarks,
# tags the final dataset, and saves to CSV files.
# ─────────────────────────────────────────────────────────────────

# ── Active benchmark 5 must ALL be present ───────────────────────
BENCHMARK_5 = ["Victor Wembanyama", "Shai Gilgeous-Alexander",
                "Nikola Jokic", "LeBron James", "Stephen Curry"]

missing_benchmarks = [p for p in BENCHMARK_5 if p not in career_df["player_name"].values]
assert len(missing_benchmarks) == 0, f"Missing benchmarks: {missing_benchmarks}"
print("✓ All 5 active benchmark players present in career_df")

# ── Tag benchmark players ─────────────────────────────────────────
career_df["is_benchmark"] = career_df["player_name"].isin(BENCHMARK_5)

# ── Status breakdown ──────────────────────────────────────────────
print(f"\nPlayer breakdown:")
print(f"  Active (current season): {(career_df['player_status']=='active').sum()}")
print(f"  Historical (retired):    {(career_df['player_status']=='historical').sum()}")
print(f"  Benchmark 5:             {career_df['is_benchmark'].sum()}")
print(f"  Total:                   {len(career_df)}")

# ── Validate no nulls in critical columns for historical players ──
# Pre-BPM era players (Wilt, Allen Iverson, etc.) get median imputation
_pre_era = career_df[career_df["career_bpm"].isna()]["player_name"].tolist()

if _pre_era:
    print(f"\n⚠️  {len(_pre_era)} players with missing BPM/VORP (pre-BPM era):")
    for _p in _pre_era:
        print(f"   – {_p}")

    # Impute with median of historical players from same approximate era
    # PER and WS are available for most so use those to estimate
    _hist_median_bpm  = career_df.loc[career_df["career_bpm"].notna(), "career_bpm"].median()
    _hist_median_vorp = career_df.loc[career_df["career_vorp"].notna(), "career_vorp"].median()

    career_df["career_bpm_imputed"]  = career_df["career_bpm"].isna()
    career_df["career_vorp_imputed"] = career_df["career_vorp"].isna()

    # For players with PER > 20, use a PER-adjusted estimate:
    # BPM ≈ (PER - 15) * 0.4  (rough historical calibration)
    def _estimate_bpm(row):
        if pd.notna(row["career_bpm"]):
            return row["career_bpm"]
        if pd.notna(row["career_per"]):
            return (row["career_per"] - 15.0) * 0.4
        return _hist_median_bpm

    def _estimate_vorp(row):
        if pd.notna(row["career_vorp"]):
            return row["career_vorp"]
        if pd.notna(row["career_ws"]):
            return row["career_ws"] * 0.03  # rough WS→VORP proxy
        return _hist_median_vorp

    career_df["career_bpm"]  = career_df.apply(_estimate_bpm, axis=1)
    career_df["career_vorp"] = career_df.apply(_estimate_vorp, axis=1)

    # Update OBPM/DBPM similarly
    career_df["career_obpm"] = career_df["career_obpm"].fillna(career_df["career_bpm"] * 0.7)
    career_df["career_dbpm"] = career_df["career_dbpm"].fillna(career_df["career_bpm"] * 0.3)

    print(f"  → Estimated BPM/VORP from PER/WS for {len(_pre_era)} players")

# ── Final null count on critical columns ─────────────────────────
print("\n── Post-imputation null counts ──")
_crit = ["career_g","career_pts","career_per","career_ws","career_bpm",
         "career_vorp","career_ppg","career_rpg","career_apg","player_status"]
_nulls = career_df[_crit].isna().sum()
print(_nulls[_nulls > 0].to_string() if (_nulls > 0).any() else "  ✓ No nulls in critical columns")

# ── Save to CSV ───────────────────────────────────────────────────
career_df.to_csv("nba_career_stats.csv", index=False)
seasons_df.to_csv("nba_seasons_longitudinal.csv", index=False)
print(f"\n✓ Saved nba_career_stats.csv ({len(career_df)} rows × {len(career_df.columns)} cols)")
print(f"✓ Saved nba_seasons_longitudinal.csv ({len(seasons_df)} rows × {len(seasons_df.columns)} cols)")

# ── Summary of final dataset ──────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"\nCareer DataFrame columns ({len(career_df.columns)} total):")
print("  Counting: career_g, career_pts, career_trb, career_ast, career_stl, career_blk")
print("  Per-game: career_ppg, career_rpg, career_apg, career_spg, career_bpg")
print("  Advanced: career_per, career_ts_pct, career_efg_pct, career_ws, career_ws_per48")
print("            career_bpm, career_vorp, career_obpm, career_dbpm")
print("  Honors:   mvp, dpoy, finals_mvp, all_nba_1st/2nd/3rd, championships, allstar, honors_index")
print("  Tags:     player_status (active/historical), is_benchmark")

print(f"\nSeason DataFrame columns ({len(seasons_df.columns)} total):")
print("  Identifiers: player_name, slug, season_year, season_start_year, season_age")
print("  Stats:       season_g, season_ppg, season_rpg, season_apg")
print("  Advanced:    season_per, season_ts_pct, season_ws, season_bpm, season_vorp")

print(f"\n── Top 15 Historical Greats (by Career WS) ──")
print(career_df[career_df["player_status"]=="historical"].nlargest(15, "career_ws")[
    ["player_name","career_ppg","career_rpg","career_ws","career_per","career_bpm","championships"]
].round(1).to_string(index=False))

print(f"\n── 5 Active Benchmarks ──")
print(career_df[career_df["is_benchmark"]][
    ["player_name","career_g","career_ppg","career_per","career_ws","career_bpm",
     "mvp","championships","allstar"]
].round(1).to_string(index=False))
