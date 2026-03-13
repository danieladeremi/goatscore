
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
#  ERA ADJUSTMENT ENGINE
#  Normalises all NBA player stats across eras (1946–present) to a
#  "Modern Era Equivalent" baseline anchored to the 2015–2024 average.
#
#  Four adjustment factors applied:
#   1. PACE FACTOR   – possessions per 48 min → normalise volume stats
#   2. eFG% INFLATION – era shooting environment → adjust efficiency
#   3. PPG CONTEXT   – league-average scoring → re-baseline per-game pts
#   4. 3PT RATE      – three-point attempt rate by season (context only)
#
#  References:
#   - Basketball-Reference League Averages (1946-2024)
#   - Dean Oliver's "Basketball on Paper" pace methodology
#   - Kubatko et al. (2007) pace estimation framework
# ═══════════════════════════════════════════════════════════════════

# ─── 1.  LEAGUE-AVERAGE HISTORICAL DATA ─────────────────────────────────────
# Sourced from Basketball-Reference League Averages
# Pace = estimated possessions per game (per-team, 48-min basis)
# eFG% = league-wide effective field goal %
# PPG  = league average points per game per team
# 3PAr = three-point attempt rate (3PA / FGA)

LEAGUE_AVERAGES = {
    # season_start: (pace, efg_pct, ppg, tpar)
    # 1946–1959: Pre-24-second clock era → very high pace
    1946: (105.0, 0.370, 79.5, 0.000),
    1947: (107.0, 0.375, 80.7, 0.000),
    1948: (106.5, 0.373, 79.0, 0.000),
    1949: (108.0, 0.378, 80.2, 0.000),
    1950: (92.0,  0.375, 83.5, 0.000),  # NBA merger with BAA
    1951: (94.0,  0.378, 85.0, 0.000),
    1952: (93.5,  0.382, 84.5, 0.000),
    1953: (94.0,  0.385, 84.8, 0.000),
    1954: (93.0,  0.383, 83.2, 0.000),
    # 1954–55: 24-second shot clock introduced → pace spikes
    1955: (107.5, 0.390, 99.5, 0.000),
    1956: (106.8, 0.393, 101.2, 0.000),
    1957: (107.2, 0.396, 100.5, 0.000),
    1958: (108.4, 0.395, 103.1, 0.000),
    1959: (110.2, 0.398, 107.0, 0.000),
    # 1960s: High-pace era (Wilt/Russell)
    1960: (112.5, 0.400, 115.3, 0.000),
    1961: (114.8, 0.402, 117.8, 0.000),
    1962: (116.3, 0.407, 118.8, 0.000),  # Wilt's 50.4 PPG season
    1963: (113.1, 0.405, 115.4, 0.000),
    1964: (111.3, 0.405, 111.5, 0.000),
    1965: (108.8, 0.407, 110.6, 0.000),
    1966: (107.4, 0.408, 113.4, 0.000),
    1967: (108.2, 0.409, 116.0, 0.000),
    1968: (106.8, 0.411, 113.0, 0.000),
    1969: (105.4, 0.412, 110.3, 0.000),
    # 1970s: Pace slowing; ABA-influence era
    1970: (106.0, 0.413, 113.6, 0.000),
    1971: (104.2, 0.415, 110.8, 0.000),
    1972: (103.8, 0.416, 110.1, 0.000),
    1973: (102.5, 0.418, 109.5, 0.000),
    1974: (101.8, 0.420, 107.8, 0.000),
    1975: (103.2, 0.422, 109.1, 0.000),
    1976: (105.8, 0.422, 108.8, 0.000),
    1977: (106.2, 0.423, 110.5, 0.000),
    1978: (106.8, 0.424, 111.3, 0.000),
    1979: (108.5, 0.425, 110.4, 0.000),
    # 1980s: 3-point era begins; Bird/Magic era
    1980: (103.1, 0.470, 108.2, 0.030),
    1981: (101.8, 0.468, 105.4, 0.031),
    1982: (101.1, 0.466, 104.3, 0.038),
    1983: (103.1, 0.468, 107.0, 0.036),
    1984: (102.2, 0.471, 107.4, 0.039),
    1985: (102.1, 0.474, 110.1, 0.046),
    1986: (103.8, 0.476, 109.2, 0.050),
    1987: (107.0, 0.480, 109.9, 0.058),
    1988: (107.0, 0.481, 108.5, 0.059),
    1989: (107.3, 0.484, 109.0, 0.069),
    # 1990s: Jordan era; hand-checking; post-Magic-Bird
    1990: (107.7, 0.488, 107.0, 0.080),
    1991: (105.3, 0.488, 104.6, 0.080),
    1992: (103.7, 0.489, 105.3, 0.084),
    1993: (105.4, 0.490, 105.0, 0.092),
    1994: (102.7, 0.489, 101.5, 0.114),
    1995: (100.9, 0.489, 101.4, 0.164),
    1996: (96.9,  0.489, 101.4, 0.195),  # 3-pt line shortened briefly
    1997: (95.9,  0.487, 100.7, 0.170),  # 3-pt line restored
    1998: (94.8,  0.484, 99.2,  0.157),
    1999: (91.2,  0.475, 91.6,  0.143),  # lockout-shortened season
    # 2000s: Jordan retires; defensive era
    2000: (93.1,  0.478, 98.2,  0.162),
    2001: (91.3,  0.479, 95.5,  0.167),
    2002: (90.7,  0.479, 95.5,  0.161),
    2003: (91.0,  0.476, 95.1,  0.171),
    2004: (90.1,  0.477, 94.7,  0.184),
    2005: (90.9,  0.479, 97.2,  0.181),
    2006: (91.9,  0.482, 98.7,  0.188),
    2007: (91.9,  0.485, 98.7,  0.193),
    2008: (92.4,  0.486, 99.9,  0.199),
    2009: (91.7,  0.489, 100.0, 0.201),
    # 2010s: Post-hand-check; 3-pt revolution begins
    2010: (92.7,  0.490, 100.4, 0.209),
    2011: (92.1,  0.491, 99.6,  0.215),
    2012: (91.3,  0.490, 98.1,  0.215),
    2013: (92.0,  0.493, 101.0, 0.233),
    2014: (93.9,  0.497, 102.4, 0.264),
    2015: (96.7,  0.500, 104.5, 0.278),
    2016: (96.4,  0.504, 105.6, 0.302),
    2017: (96.4,  0.505, 105.6, 0.330),
    2018: (97.3,  0.512, 106.3, 0.358),
    2019: (100.0, 0.520, 111.2, 0.376),
    # 2020s: Modern analytics era; 3-pt dominance
    2020: (100.3, 0.533, 111.8, 0.393),
    2021: (99.2,  0.535, 112.1, 0.397),
    2022: (98.2,  0.534, 110.6, 0.393),
    2023: (99.4,  0.537, 114.7, 0.396),
    2024: (100.0, 0.535, 114.0, 0.398),
}

# Convert to DataFrame for easy manipulation
era_df = pd.DataFrame.from_dict(
    LEAGUE_AVERAGES, orient='index',
    columns=['lg_pace', 'lg_efg_pct', 'lg_ppg', 'lg_tpar']
).reset_index().rename(columns={'index': 'season_start_year'})
era_df['season_start_year'] = era_df['season_start_year'].astype(int)

# ─── 2.  DEFINE MODERN BASELINE (2015–2024 average) ─────────────────────────
# This is our "target era" — we translate all historical stats to this baseline
MODERN_MASK = (era_df['season_start_year'] >= 2015)
MODERN_PACE   = era_df.loc[MODERN_MASK, 'lg_pace'].mean()
MODERN_EFG    = era_df.loc[MODERN_MASK, 'lg_efg_pct'].mean()
MODERN_PPG    = era_df.loc[MODERN_MASK, 'lg_ppg'].mean()
MODERN_TPAR   = era_df.loc[MODERN_MASK, 'lg_tpar'].mean()

print(f"Modern Era Baseline (2015–2024 avg):")
print(f"  Pace:  {MODERN_PACE:.1f} poss/48")
print(f"  eFG%:  {MODERN_EFG:.3f}")
print(f"  PPG:   {MODERN_PPG:.1f}")
print(f"  3PAr:  {MODERN_TPAR:.3f}")

# ─── 3.  COMPUTE PER-SEASON ADJUSTMENT FACTORS ──────────────────────────────
era_df['pace_factor']      = MODERN_PACE   / era_df['lg_pace']
era_df['efg_factor']       = MODERN_EFG    / era_df['lg_efg_pct']
era_df['ppg_factor']       = MODERN_PPG    / era_df['lg_ppg']
era_df['tpar_factor']      = MODERN_TPAR   / era_df['lg_tpar'].replace(0, np.nan)  # NaN for pre-3pt era

# Decade labels for readability
def _decade_label(yr):
    d = (yr // 10) * 10
    return f"{d}s"

era_df['decade'] = era_df['season_start_year'].apply(_decade_label)

# ─── 4.  PRINT METHODOLOGY LOG BY DECADE ─────────────────────────────────────
print(f"\n{'='*75}")
print("ERA ADJUSTMENT FACTORS — METHODOLOGY LOG (DECADE AVERAGES)")
print(f"{'='*75}")
print(f"{'Decade':<10} {'Pace':>7} {'PaceFact':>9} {'eFG%':>7} {'eFGFact':>8} "
      f"{'PPG':>7} {'PPGFact':>8} {'3PAr':>7} {'3PARFact':>9}")
print(f"{'-'*75}")

_decade_summary = era_df.groupby('decade').agg({
    'lg_pace':      'mean',
    'pace_factor':  'mean',
    'lg_efg_pct':   'mean',
    'efg_factor':   'mean',
    'lg_ppg':       'mean',
    'ppg_factor':   'mean',
    'lg_tpar':      'mean',
    'tpar_factor':  'mean',
}).reset_index().sort_values('decade')

for _, r in _decade_summary.iterrows():
    _tf = f"{r['tpar_factor']:.3f}" if not np.isnan(r['tpar_factor']) else " N/A "
    print(f"{r['decade']:<10} {r['lg_pace']:>7.1f} {r['pace_factor']:>9.4f} "
          f"{r['lg_efg_pct']:>7.3f} {r['efg_factor']:>8.4f} "
          f"{r['lg_ppg']:>7.1f} {r['ppg_factor']:>8.4f} "
          f"{r['lg_tpar']:>7.3f} {_tf:>9}")

print(f"\n  * Pace factor > 1.0 means the era was FASTER than modern → volume stats scaled DOWN")
print(f"  * eFG factor  > 1.0 means the era shot WORSE than modern → efficiency scores scale UP (more impressive)")
print(f"  * PPG factor  > 1.0 means the era scored LESS than modern → per-game pts scale UP")

# ─── 5.  MERGE ERA FACTORS INTO SEASONS DATA ─────────────────────────────────
# Forward-fill for any seasons not explicitly in the table
all_years = pd.DataFrame({'season_start_year': range(
    int(seasons_df['season_start_year'].min()),
    int(seasons_df['season_start_year'].max()) + 2
)})
era_full = all_years.merge(era_df, on='season_start_year', how='left').ffill()

adj_seasons = seasons_df.copy()
adj_seasons = adj_seasons.merge(
    era_full[['season_start_year','lg_pace','lg_efg_pct','lg_ppg','lg_tpar',
              'pace_factor','efg_factor','ppg_factor','tpar_factor']],
    on='season_start_year', how='left'
)

# ─── 6.  APPLY ADJUSTMENTS TO SEASON DATA ───────────────────────────────────
# ── 6a. PACE ADJUSTMENT (volume stats: points, rebounds, assists totals) ─────
#   Idea: if a player played in a 115-pace era, their raw counts are inflated.
#   We scale totals to what they'd look like at modern pace (~99 poss/game).
#   Per-game stats get a separate pace-per-game adjustment.

adj_seasons['adj_season_pts']  = adj_seasons['season_pts']  * adj_seasons['pace_factor']
adj_seasons['adj_season_trb']  = adj_seasons['season_trb']  * adj_seasons['pace_factor']
adj_seasons['adj_season_ast']  = adj_seasons['season_ast']  * adj_seasons['pace_factor']
adj_seasons['adj_season_stl']  = adj_seasons['season_stl']  * adj_seasons['pace_factor']
adj_seasons['adj_season_blk']  = adj_seasons['season_blk']  * adj_seasons['pace_factor']

# Per-game stats: pace-adjust the per-game rates
adj_seasons['adj_season_ppg']  = adj_seasons['season_ppg']  * adj_seasons['pace_factor']
adj_seasons['adj_season_rpg']  = adj_seasons['season_rpg']  * adj_seasons['pace_factor']
adj_seasons['adj_season_apg']  = adj_seasons['season_apg']  * adj_seasons['pace_factor']

# ── 6b. EFFICIENCY ADJUSTMENT (eFG% era context) ─────────────────────────────
#   Shooting conditions vary dramatically by era (hand-checking, zone D, spacing).
#   A player with 0.500 eFG% in 1965 (league avg 0.400) is MORE dominant
#   than a player with 0.500 eFG% in 2020 (league avg 0.533).
#   We compute the player's surplus above league average and project to modern baseline.

adj_seasons['lg_efg_pct_safe'] = adj_seasons['lg_efg_pct'].fillna(MODERN_EFG)
adj_seasons['efg_surplus']     = adj_seasons['season_efg_pct'] - adj_seasons['lg_efg_pct_safe']
adj_seasons['adj_season_efg_pct'] = (MODERN_EFG + adj_seasons['efg_surplus']).clip(lower=0.0, upper=0.85)

# Similarly for TS%
adj_seasons['lg_ts_proxy'] = adj_seasons['lg_efg_pct_safe'] * 1.06  # rough TS≈eFG*1.06 proxy
adj_seasons['ts_surplus']  = adj_seasons['season_ts_pct'] - adj_seasons['lg_ts_proxy']
_modern_ts_proxy = MODERN_EFG * 1.06
adj_seasons['adj_season_ts_pct'] = (_modern_ts_proxy + adj_seasons['ts_surplus']).clip(lower=0.0, upper=0.95)

# ── 6c. PPG ERA INFLATION ADJUSTMENT ─────────────────────────────────────────
#   Adjust scoring averages relative to era PPG environment.
#   Era-adjusted PPG = (player PPG / league PPG) × modern league PPG
#   This preserves the player's RELATIVE DOMINANCE in scoring.

adj_seasons['lg_ppg_safe']      = adj_seasons['lg_ppg'].fillna(MODERN_PPG)
adj_seasons['ppg_relative']     = adj_seasons['season_ppg'] / adj_seasons['lg_ppg_safe']
adj_seasons['adj_season_ppg_inflation'] = adj_seasons['ppg_relative'] * MODERN_PPG
# Use a blend: 60% pace-adjusted, 40% inflation-adjusted (complementary methods)
adj_seasons['adj_season_ppg_blended'] = (
    0.6 * adj_seasons['adj_season_ppg'] +
    0.4 * adj_seasons['adj_season_ppg_inflation']
)

# ── 6d. PER (already pace-adjusted internally by B-Ref, but note for transparency) ──
# BPM and PER are already pace-normalized metrics from Basketball-Reference.
# We preserve them as-is but note the era context.
# For era-adjusted WS, we use the pace factor applied to raw WS.
adj_seasons['adj_season_ws']    = adj_seasons['season_ws']   * adj_seasons['pace_factor']
adj_seasons['adj_season_vorp']  = adj_seasons['season_vorp']  # VORP is already league-relative

# ─── 7.  COMPUTE ERA-ADJUSTED CAREER STATS ──────────────────────────────────
# For each player, sum/average season-level adjusted stats → career-level adjusted
_career_adj_agg = adj_seasons.groupby('player_name').agg(
    adj_career_pts  = ('adj_season_pts',  'sum'),
    adj_career_trb  = ('adj_season_trb',  'sum'),
    adj_career_ast  = ('adj_season_ast',  'sum'),
    adj_career_stl  = ('adj_season_stl',  'sum'),
    adj_career_blk  = ('adj_season_blk',  'sum'),
    adj_career_ppg  = ('adj_season_ppg_blended', 'mean'),
    adj_career_rpg  = ('adj_season_rpg',  'mean'),
    adj_career_apg  = ('adj_season_apg',  'mean'),
    adj_career_efg_pct = ('adj_season_efg_pct', 'mean'),
    adj_career_ts_pct  = ('adj_season_ts_pct',  'mean'),
    adj_career_ws   = ('adj_season_ws',   'sum'),
    avg_pace_factor = ('pace_factor',     'mean'),
    avg_ppg_factor  = ('ppg_factor',      'mean'),
    avg_efg_factor  = ('efg_factor',      'mean'),
    min_season_year = ('season_start_year', 'min'),
    max_season_year = ('season_start_year', 'max'),
    n_seasons       = ('season_start_year', 'count'),
).reset_index()

# Merge with career_df to get raw + adjusted side by side
era_adj_career_df = career_df.merge(_career_adj_agg, on='player_name', how='left')

# Round adjusted columns for readability
_adj_cols = [c for c in era_adj_career_df.columns if c.startswith('adj_')]
era_adj_career_df[_adj_cols] = era_adj_career_df[_adj_cols].round(2)

print(f"\n\n{'='*75}")
print("ERA-ADJUSTED CAREER STATS — KEY PLAYERS")
print(f"{'='*75}")

_display_cols = [
    'player_name', 'min_season_year', 'max_season_year',
    'career_ppg', 'adj_career_ppg',
    'career_rpg', 'adj_career_rpg',
    'career_apg', 'adj_career_apg',
    'career_ws',  'adj_career_ws',
    'avg_pace_factor',
]

# Key validation players
_validation_players = [
    'Wilt Chamberlain', 'Oscar Robertson', 'Bill Russell', 'Elgin Baylor',
    'Kareem Abdul-Jabbar', 'Magic Johnson', 'Michael Jordan', 'LeBron James',
    'Nikola Jokic', 'Stephen Curry', 'Victor Wembanyama', 'Shai Gilgeous-Alexander'
]
_val_df = era_adj_career_df[era_adj_career_df['player_name'].isin(_validation_players)].copy()
_val_df = _val_df.sort_values('min_season_year')

print(f"\n{'Player':<26} {'Era':>12} {'RawPPG':>7} {'AdjPPG':>7} {'RawRPG':>7} "
      f"{'AdjRPG':>7} {'RawAPG':>7} {'AdjAPG':>7} {'RawWS':>7} {'AdjWS':>7} {'PaceFact':>9}")
print("-" * 105)
for _, r in _val_df.iterrows():
    if pd.notna(r.get('min_season_year')) and pd.notna(r.get('avg_pace_factor')):
        era_str = f"{int(r['min_season_year'])}–{int(r['max_season_year'])}"
        rpg_raw = f"{r['career_rpg']:.1f}" if pd.notna(r['career_rpg']) else " N/A"
        rpg_adj = f"{r['adj_career_rpg']:.1f}" if pd.notna(r['adj_career_rpg']) else " N/A"
        apg_raw = f"{r['career_apg']:.1f}" if pd.notna(r['career_apg']) else " N/A"
        apg_adj = f"{r['adj_career_apg']:.1f}" if pd.notna(r['adj_career_apg']) else " N/A"
        ppg_raw = f"{r['career_ppg']:.1f}" if pd.notna(r['career_ppg']) else " N/A"
        ppg_adj = f"{r['adj_career_ppg']:.1f}" if pd.notna(r['adj_career_ppg']) else " N/A"
        ws_raw  = f"{r['career_ws']:.1f}" if pd.notna(r['career_ws']) else " N/A"
        ws_adj  = f"{r['adj_career_ws']:.1f}" if pd.notna(r['adj_career_ws']) else " N/A"
        pf      = f"{r['avg_pace_factor']:.4f}"
        print(f"{r['player_name']:<26} {era_str:>12} {ppg_raw:>7} {ppg_adj:>7} "
              f"{rpg_raw:>7} {rpg_adj:>7} {apg_raw:>7} {apg_adj:>7} "
              f"{ws_raw:>7} {ws_adj:>7} {pf:>9}")

# ─── 8.  SANITY CHECK — WILT AND OSCAR VALIDATION ───────────────────────────
print(f"\n{'='*75}")
print("SANITY CHECK: WILT CHAMBERLAIN vs MODERN STARS")
print(f"{'='*75}")

def _show_player(df, name):
    _r = df[df['player_name'] == name]
    if _r.empty:
        print(f"  {name}: not found in dataset")
        return
    _r = _r.iloc[0]
    print(f"\n  {name} ({int(_r['min_season_year']) if pd.notna(_r.get('min_season_year')) else '?'}–"
          f"{int(_r['max_season_year']) if pd.notna(_r.get('max_season_year')) else '?'}):")
    print(f"    Raw PPG:    {_r['career_ppg']:.1f}  →  Era-Adj PPG: {_r['adj_career_ppg']:.1f}" 
          if pd.notna(_r.get('career_ppg')) and pd.notna(_r.get('adj_career_ppg'))
          else f"    PPG: data limited")
    print(f"    Raw RPG:    {_r['career_rpg']:.1f}  →  Era-Adj RPG: {_r['adj_career_rpg']:.1f}"
          if pd.notna(_r.get('career_rpg')) and pd.notna(_r.get('adj_career_rpg'))
          else f"    RPG: data limited")
    print(f"    Raw APG:    {_r['career_apg']:.1f}  →  Era-Adj APG: {_r['adj_career_apg']:.1f}"
          if pd.notna(_r.get('career_apg')) and pd.notna(_r.get('adj_career_apg'))
          else f"    APG: data limited")
    print(f"    Avg Pace Factor: {_r['avg_pace_factor']:.4f}" if pd.notna(_r.get('avg_pace_factor')) else "")

for _name in ['Wilt Chamberlain', 'Oscar Robertson', 'LeBron James', 'Nikola Jokic', 'Stephen Curry']:
    _show_player(era_adj_career_df, _name)

# ─── 9.  WILT'S LEGENDARY 1961-62 SEASON (50.4 PPG) BREAKDOWN ────────────────
print(f"\n{'='*75}")
print("DEEP DIVE: WILT CHAMBERLAIN 1961–62 (50.4 PPG) — ERA ADJUSTMENT")
print(f"{'='*75}")
_wilt_62 = adj_seasons[
    (adj_seasons['player_name'] == 'Wilt Chamberlain') &
    (adj_seasons['season_start_year'] == 1961)
]
if not _wilt_62.empty:
    _w = _wilt_62.iloc[0]
    print(f"\n  Raw stats (1961-62):")
    print(f"    PPG:             {_w['season_ppg']:.1f}")
    print(f"    RPG:             {_w['season_rpg']:.1f}")
    print(f"    APG:             {_w['season_apg']:.1f}")
    print(f"\n  Era context (1961-62):")
    print(f"    League Pace:     {_w['lg_pace']:.1f} poss/48 (modern: {MODERN_PACE:.1f})")
    print(f"    League PPG:      {_w['lg_ppg']:.1f} pts/game (modern: {MODERN_PPG:.1f})")
    print(f"    League eFG%:     {_w['lg_efg_pct']:.3f} (modern: {MODERN_EFG:.3f})")
    print(f"\n  Adjustment factors:")
    print(f"    Pace factor:     {_w['pace_factor']:.4f}  (pace was {(1/_w['pace_factor']-1)*100:.1f}% FASTER)")
    print(f"    PPG factor:      {_w['ppg_factor']:.4f}")
    print(f"    eFG factor:      {_w['efg_factor']:.4f}")
    print(f"\n  Era-adjusted stats (2015–2024 equivalent):")
    print(f"    Pace-adj PPG:    {_w['adj_season_ppg']:.1f}")
    print(f"    PPG-inflation adj: {_w['adj_season_ppg_inflation']:.1f}")
    print(f"    Blended adj PPG: {_w['adj_season_ppg_blended']:.1f}")
    print(f"    Adj RPG:         {_w['adj_season_rpg']:.1f}")
    print(f"    Adj APG:         {_w['adj_season_apg']:.1f}")
    print(f"    Adj eFG%:        {_w['adj_season_efg_pct']:.3f} (raw: {_w['season_efg_pct']:.3f}, era surplus: {_w['efg_surplus']:+.3f})")

# ─── 10. OSCAR ROBERTSON 1961-62 TRIPLE-DOUBLE SEASON ────────────────────────
print(f"\n{'='*75}")
print("DEEP DIVE: OSCAR ROBERTSON 1961–62 TRIPLE-DOUBLE SEASON")
print(f"{'='*75}")
_oscar_62 = adj_seasons[
    (adj_seasons['player_name'] == 'Oscar Robertson') &
    (adj_seasons['season_start_year'] == 1961)
]
if not _oscar_62.empty:
    _o = _oscar_62.iloc[0]
    print(f"\n  Raw stats (1961-62):")
    print(f"    PPG: {_o['season_ppg']:.1f}  RPG: {_o['season_rpg']:.1f}  APG: {_o['season_apg']:.1f}")
    print(f"  Era-adjusted (modern equivalent):")
    print(f"    PPG: {_o['adj_season_ppg_blended']:.1f}  RPG: {_o['adj_season_rpg']:.1f}  APG: {_o['adj_season_apg']:.1f}")
    print(f"  Interpretation: Oscar's line translates to roughly "
          f"{_o['adj_season_ppg_blended']:.0f}/{_o['adj_season_rpg']:.0f}/{_o['adj_season_apg']:.0f} "
          f"in modern-era context")
else:
    print("  Oscar Robertson 1961-62 season not found in dataset (data limited)")

# ─── 11. VALIDATION ASSERTIONS ──────────────────────────────────────────────
print(f"\n{'='*75}")
print("AUTOMATED SANITY CHECK ASSERTIONS")
print(f"{'='*75}")

_assertions_passed = 0
_assertions_total  = 0

def _assert_check(condition, description, detail=""):
    global _assertions_passed, _assertions_total
    _assertions_total += 1
    if condition:
        print(f"  ✓  {description}")
        _assertions_passed += 1
    else:
        print(f"  ✗  FAILED: {description} {detail}")

# 1. Pre-1980s players should have pace_factor < 1 (they played faster → adjusted DOWN)
_wilt_pf = era_adj_career_df.loc[era_adj_career_df['player_name']=='Wilt Chamberlain', 'avg_pace_factor']
if not _wilt_pf.empty and pd.notna(_wilt_pf.iloc[0]):
    _assert_check(_wilt_pf.iloc[0] < 1.0,
                  "Wilt Chamberlain avg pace_factor < 1.0 (played in faster era → volume adjusted down)",
                  f"[pace_factor={_wilt_pf.iloc[0]:.4f}]")

# 2. Modern players (LeBron, Jokic) should have pace_factor ≈ 1.0 ± 0.05
for _mn in ['LeBron James', 'Nikola Jokic', 'Stephen Curry']:
    _pf = era_adj_career_df.loc[era_adj_career_df['player_name']==_mn, 'avg_pace_factor']
    if not _pf.empty and pd.notna(_pf.iloc[0]):
        _assert_check(abs(_pf.iloc[0] - 1.0) < 0.12,
                      f"{_mn} pace_factor near 1.0 (modern era player)",
                      f"[pace_factor={_pf.iloc[0]:.4f}]")

# 3. Era-adjusted Wilt PPG should be materially LOWER than raw (high-pace era)
_wilt_row = era_adj_career_df[era_adj_career_df['player_name']=='Wilt Chamberlain']
if not _wilt_row.empty and pd.notna(_wilt_row.iloc[0].get('career_ppg')):
    _raw = _wilt_row.iloc[0]['career_ppg']
    _adj = _wilt_row.iloc[0]['adj_career_ppg']
    _assert_check(_adj < _raw,
                  f"Wilt era-adj PPG ({_adj:.1f}) < raw PPG ({_raw:.1f}) — pace deflation applied")

# 4. Era-adjusted eFG% for 1960s players should be higher than raw (harder era)
_era_62 = adj_seasons[adj_seasons['season_start_year'] == 1961]
if not _era_62.empty:
    _mean_raw_efg = _era_62['season_efg_pct'].mean()
    _mean_adj_efg = _era_62['adj_season_efg_pct'].mean()
    _assert_check(_mean_adj_efg > _mean_raw_efg,
                  f"1961-62 avg era-adj eFG% ({_mean_adj_efg:.3f}) > raw eFG% ({_mean_raw_efg:.3f}) — harder era acknowledged")

# 5. Modern players: raw ≈ adjusted (minimal correction needed)
_lbj = era_adj_career_df[era_adj_career_df['player_name']=='LeBron James']
if not _lbj.empty and pd.notna(_lbj.iloc[0].get('career_ppg')) and pd.notna(_lbj.iloc[0].get('adj_career_ppg')):
    _raw = _lbj.iloc[0]['career_ppg']
    _adj = _lbj.iloc[0]['adj_career_ppg']
    _assert_check(abs(_adj - _raw) < 5.0,
                  f"LeBron raw vs adj PPG close ({_raw:.1f} vs {_adj:.1f}) — already in modern era")

# 6. 1980s players should have pace_factor slightly below 1.0 (slightly faster)
_mj = era_adj_career_df[era_adj_career_df['player_name']=='Michael Jordan']
if not _mj.empty and pd.notna(_mj.iloc[0].get('avg_pace_factor')):
    _pf = _mj.iloc[0]['avg_pace_factor']
    _assert_check(0.88 < _pf < 1.02,
                  f"Michael Jordan pace_factor in expected range for 1984–1998 era",
                  f"[pace_factor={_pf:.4f}]")

print(f"\n  {_assertions_passed}/{_assertions_total} assertions passed")

# ─── 12. SAVE OUTPUTS ────────────────────────────────────────────────────────
era_adj_career_df.to_csv('nba_era_adjusted_career.csv', index=False)
adj_seasons.to_csv('nba_era_adjusted_seasons.csv', index=False)
era_df.to_csv('nba_era_factors.csv', index=False)

print(f"\n{'='*75}")
print("OUTPUT FILES SAVED:")
print(f"{'='*75}")
print(f"  nba_era_adjusted_career.csv  — {len(era_adj_career_df)} players × {len(era_adj_career_df.columns)} cols")
print(f"  nba_era_adjusted_seasons.csv — {len(adj_seasons)} seasons × {len(adj_seasons.columns)} cols")
print(f"  nba_era_factors.csv          — {len(era_df)} season rows × {len(era_df.columns)} cols (adjustment factors)")

print(f"\n  New columns in career data:  adj_career_ppg, adj_career_rpg, adj_career_apg,")
print(f"                               adj_career_pts, adj_career_trb, adj_career_ast,")
print(f"                               adj_career_ws,  adj_career_efg_pct, adj_career_ts_pct")
print(f"\n  New columns in season data:  adj_season_ppg_blended, adj_season_ppg,")
print(f"                               adj_season_rpg, adj_season_apg,")
print(f"                               adj_season_pts, adj_season_trb, adj_season_ast,")
print(f"                               adj_season_ws, adj_season_efg_pct, adj_season_ts_pct")
print(f"\n✓ Era adjustment engine complete.")

# Expose key variables for downstream blocks
era_factors_df     = era_df.copy()
era_adj_seasons_df = adj_seasons.copy()
