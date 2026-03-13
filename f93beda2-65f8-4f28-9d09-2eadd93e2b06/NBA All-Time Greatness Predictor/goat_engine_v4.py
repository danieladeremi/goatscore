
"""
GOAT SCORE ENGINE v4 — Honor Weight Rebalance + Physical Dominance Removal + Curry Fix

Changes from v3:
  (A) HONOR WEIGHTS REBALANCED:
      - MVP: 0.35 (increased, MVP >> DPOY ≈ First Team All-NBA)
      - All-NBA weighted: 0.28 (1st=8pts, 2nd=4pts, 3rd=1pt → tiered gap)
      - Championships: 0.18 (reduced)
      - All-Star: 0.10
      - All-Defense weighted: 1st=3pts, 2nd=1.5pts (First Team All-Defense ≈ Second Team All-NBA)
      - DPOY: 0.05 (DPOY ≈ First Team All-NBA per selection)
      - Finals MVP: 0.04
      → All-Def 1st ≈ All-NBA 2nd; All-Def 2nd ≈ All-NBA 3rd (ticket spec)
  (B) PHYSICAL DOMINATION STAT REMOVED ENTIRELY from peak pillar
      → Peak pillar = pure PER/TS%/BPM peak with era discount
  (C) STEPHEN CURRY FIX:
      - Career gate v3 uses "<2 champs AND <3 All-NBA" → Curry has 4 chips + 10 All-NBA = passes gate
      - But supporting_cast_df has 97 rows (missing 2 players from longitudinal data)
      - Curry IS present in data (rank 33 in v3 with score 42.8 — hurt by:
        (i)  5-pillar equal weight where defense pillar scores him 0 (he's 0,0 All-Def)
        (ii) peak_era_mult = 0.80 (modern era -20% discount) crushing his peak
      - Fix: Move from 5-pillar to 4-pillar (drop defense as separate pillar, keep it in honors only)
        AND adjust peak era discount: 2015-era peak should only get -10% (not -20%) 
        since Curry peak was 2015-16 (scoring explosion era, not modern volume-stat era)
      → Curry should rank top 10-12 given 2x MVP (including unanimous), 4 chips, 10 All-NBA

Outputs: goat_scores_v4, pillar_scores_v4, active_projections_v4, goat_engine_v4_leaderboard_chart
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
_ACTIVE_5_V4 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
                'Nikola Jokic', 'LeBron James', 'Stephen Curry']

_CURRENT_AGES_V4 = {
    'Victor Wembanyama':       21,
    'Shai Gilgeous-Alexander': 26,
    'Nikola Jokic':            29,
    'LeBron James':            40,
    'Stephen Curry':           36,
}

_GAMES_PER_SEASON_V4 = 82
_REPL_WS_V4 = 2.0

# 4-pillar weights (defense moved into honors, no longer a separate pillar)
_DEFAULT_WEIGHTS_V4 = {
    'volume_longevity':   0.25,
    'peak_dominance':     0.25,
    'context_value':      0.25,
    'honors_recognition': 0.25,
}

def _normalize_weights_v4(w):
    total = sum(max(0, v) for v in w.values())
    if total == 0:
        n = len(w)
        return {k: 1/n for k in w}
    return {k: max(0, v) / total for k, v in w.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (B): PEAK ERA DISCOUNT — refined
# Original: ≥2010 → 0.80 (too harsh, hurt Curry badly)
# New: ≥2015 → 0.90; 2010-2014 → 0.93; 1990-2009 → 1.00; pre-1990 → 1.08
# ═══════════════════════════════════════════════════════════════════════════════
def _peak_era_discount_v4(peak_year):
    """Returns multiplier for peak pillar based on peak season era."""
    if peak_year >= 2015:
        return 0.90   # -10% for late modern era (not -20%)
    elif peak_year >= 2010:
        return 0.93   # -7% for early modern era
    elif peak_year >= 1990:
        return 1.00   # baseline
    else:
        return 1.08   # +8% for pre-1990 undercount

# ═══════════════════════════════════════════════════════════════════════════════
# ALL-DEFENSIVE DATABASE (same as v3)
# ═══════════════════════════════════════════════════════════════════════════════
_ALL_DEF_DB_V4 = {
    'Michael Jordan':            (8, 1),
    'LeBron James':              (5, 1),
    'Kareem Abdul-Jabbar':       (5, 6),
    'Bill Russell':              (0, 0),
    'Wilt Chamberlain':          (0, 0),
    'Magic Johnson':             (0, 0),
    'Larry Bird':                (0, 3),
    'Tim Duncan':                (8, 7),
    'Hakeem Olajuwon':           (5, 4),
    "Shaquille O'Neal":          (3, 0),
    'Kobe Bryant':               (9, 3),
    'Kevin Garnett':             (9, 3),
    'Gary Payton':               (9, 2),
    'Dennis Rodman':             (7, 1),
    'Ben Wallace':               (4, 2),
    'Rudy Gobert':               (4, 2),
    'Dikembe Mutombo':           (2, 6),
    'David Robinson':            (4, 4),
    'Scottie Pippen':            (8, 2),
    'Karl Malone':               (4, 1),
    'John Stockton':             (2, 3),
    'Charles Barkley':           (0, 5),
    'Patrick Ewing':             (1, 3),
    'Kevin Durant':              (1, 0),
    'Giannis Antetokounmpo':     (3, 2),
    'Kawhi Leonard':             (5, 4),
    'Nikola Jokic':              (0, 0),
    'Stephen Curry':             (0, 0),
    'Shai Gilgeous-Alexander':   (1, 0),
    'Victor Wembanyama':         (1, 0),
    'Chris Paul':                (1, 5),
    'Dwyane Wade':               (3, 2),
    'Jason Kidd':                (4, 5),
    'Dirk Nowitzki':             (0, 1),
    'James Harden':              (0, 0),
    'Russell Westbrook':         (0, 0),
    'Paul Pierce':               (0, 1),
    'Ray Allen':                 (0, 0),
    'Isiah Thomas':              (0, 1),
    'Clyde Drexler':             (0, 2),
    'Tony Parker':               (0, 0),
    'Manu Ginobili':             (0, 0),
    'Julius Erving':             (0, 2),
    'Oscar Robertson':           (0, 0),
    'Jerry West':                (4, 2),
    'Elgin Baylor':              (0, 0),
    'Bob Cousy':                 (0, 0),
    'Elvin Hayes':               (0, 0),
    'Bob Pettit':                (0, 0),
    'Pete Maravich':             (0, 0),
    'Moses Malone':              (1, 1),
    'Rick Barry':                (0, 2),
    'John Havlicek':             (5, 3),
    'Sam Jones':                 (0, 0),
    'Bob McAdoo':                (0, 0),
    'Dave Cowens':               (1, 0),
    'Billy Cunningham':          (0, 1),
    'Spencer Haywood':           (0, 0),
    'Bill Walton':               (1, 1),
    'Artis Gilmore':             (0, 0),
    'Nate Archibald':            (0, 0),
    'Walt Frazier':              (6, 1),
    'Willis Reed':               (0, 0),
    'Alonzo Mourning':           (2, 2),
    'Dominique Wilkins':         (0, 0),
    'Alex English':              (0, 0),
    'Bernard King':              (0, 0),
}

# Superteam discount (same as v3)
_SUPERTEAM_RING_YEARS_V4 = {
    'Kevin Durant':  [2017, 2018],
    'LeBron James':  [2012, 2013],
    'Dwyane Wade':   [2012, 2013],
    'Ray Allen':     [2008],
    'Paul Pierce':   [2008],
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
_career_v4 = era_adj_career_df.copy()
_seasons_v4 = era_adj_seasons_df.copy()

print(f"✓ Loaded {len(_career_v4)} career records, {len(_seasons_v4)} season records")

# Verify Curry is in career data
_curry_row = _career_v4[_career_v4['player_name'] == 'Stephen Curry']
print(f"  Curry in career_df: {not _curry_row.empty}, "
      f"seasons: {_career_v4[_career_v4['player_name']=='Stephen Curry']['n_seasons'].values}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTING CAST
# ═══════════════════════════════════════════════════════════════════════════════
_career_v4 = _career_v4.merge(
    supporting_cast_df[['player_name', 'individual_brilliance_score',
                         'prime_team_win_pct', 'avg_team_win_pct',
                         'sc_mult', 'prime_avg_apg']],
    on='player_name', how='left'
)
_career_v4['individual_brilliance_score'] = _career_v4['individual_brilliance_score'].fillna(50)
_career_v4['prime_team_win_pct']          = _career_v4['prime_team_win_pct'].fillna(0.500)
_career_v4['avg_team_win_pct']            = _career_v4['avg_team_win_pct'].fillna(0.500)
_career_v4['sc_mult']                     = _career_v4['sc_mult'].fillna(1.0)
_career_v4['prime_avg_apg']               = _career_v4['prime_avg_apg'].fillna(0)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1: VOLUME / LONGEVITY (same as v3)
# ═══════════════════════════════════════════════════════════════════════════════
_vol_cols_v4 = {
    'adj_career_pts': 'vp4_pts',
    'adj_career_trb': 'vp4_trb',
    'adj_career_ast': 'vp4_ast',
    'career_g':       'vp4_g',
    'n_seasons':      'vp4_seas',
}
for _src, _dst in _vol_cols_v4.items():
    _col = _career_v4[_src].fillna(0)
    _mx = _col.max()
    _career_v4[_dst] = (_col / _mx * 100) if _mx > 0 else 0.0

_career_v4['pillar4_volume_raw'] = _career_v4[list(_vol_cols_v4.values())].mean(axis=1)

# Stockton system-dependency discount
_ast_sorted_v4 = _career_v4.nlargest(5, 'career_ast')['player_name'].tolist()
def _stockton_discount_v4(row):
    if (row['player_name'] in _ast_sorted_v4 and
            row.get('prime_team_win_pct', 0.5) > 0.60 and
            row.get('championships', 0) == 0):
        return 0.90
    return 1.0

_career_v4['stockton_mult_v4'] = _career_v4.apply(_stockton_discount_v4, axis=1)
_career_v4['pillar4_volume'] = (
    _career_v4['pillar4_volume_raw'] *
    _career_v4['sc_mult'].clip(0.88, 1.05) *
    _career_v4['stockton_mult_v4']
).clip(0, 100)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2: PEAK DOMINANCE
# FIX (B): NO physical dominance stat — pure PER/TS%/BPM with refined era discount
# ═══════════════════════════════════════════════════════════════════════════════
_PEAK_METRICS_V4 = ['season_per', 'season_ts_pct', 'season_bpm']

def _best3avg_v4(grp, col):
    _vals = grp.sort_values('season_start_year')[col].dropna().values
    if len(_vals) == 0:
        return np.nan, None
    if len(_vals) < 3:
        _idx = np.argmax(_vals)
        _years = grp.sort_values('season_start_year')['season_start_year'].values
        _peak_yr = int(_years[_idx]) if len(_years) > _idx else 2000
        return float(np.mean(_vals)), _peak_yr
    _rolled = np.convolve(_vals, np.ones(3) / 3, mode='valid')
    _best_idx = np.argmax(_rolled)
    _years = grp.sort_values('season_start_year')['season_start_year'].values
    _peak_yr = int(_years[_best_idx + 1]) if len(_years) > _best_idx + 1 else int(_years[-1])
    return float(np.max(_rolled)), _peak_yr

_pk4_rows = []
for _pname, _grp in _seasons_v4.groupby('player_name'):
    _pk = {'player_name': _pname}
    _peak_years_m = []
    for _m in _PEAK_METRICS_V4:
        _val, _yr = _best3avg_v4(_grp, _m)
        if _m == 'season_bpm':
            _val = min(_val, 8.0) if _val is not None and not np.isnan(_val) else _val
        _pk[f'pk4_{_m}'] = _val
        if _yr is not None:
            _peak_years_m.append(_yr)
    _pk['peak_year_v4'] = int(np.median(_peak_years_m)) if _peak_years_m else 2000
    _pk4_rows.append(_pk)

_peak_df_v4 = pd.DataFrame(_pk4_rows)

# Normalize peak metrics
for _m in _PEAK_METRICS_V4:
    _col = _peak_df_v4[f'pk4_{_m}'].fillna(0)
    _mn_p, _mx_p = _col.min(), _col.max()
    _peak_df_v4[f'pk4_{_m}_n'] = ((_col - _mn_p) / (_mx_p - _mn_p) * 100) if _mx_p > _mn_p else 50.0

# Pure peak = equal-weighted PER/TS%/BPM, no physical dominance
_peak_df_v4['pillar4_peak_raw'] = _peak_df_v4[[f'pk4_{m}_n' for m in _PEAK_METRICS_V4]].mean(axis=1)

# Merge peak data
_career_v4 = _career_v4.merge(
    _peak_df_v4[['player_name', 'pillar4_peak_raw', 'peak_year_v4'] +
                 [f'pk4_{m}_n' for m in _PEAK_METRICS_V4]],
    on='player_name', how='left'
)
_career_v4['pillar4_peak_raw'] = _career_v4['pillar4_peak_raw'].fillna(
    _career_v4['pillar4_peak_raw'].median()
)
_career_v4['peak_year_v4'] = _career_v4['peak_year_v4'].fillna(2000).astype(int)

# Apply refined peak era discount
_career_v4['peak_era_mult_v4'] = _career_v4['peak_year_v4'].apply(_peak_era_discount_v4)
_career_v4['pillar4_peak'] = (
    _career_v4['pillar4_peak_raw'] * _career_v4['peak_era_mult_v4']
).clip(0, 100)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 3: CONTEXT-ADJUSTED VALUE (same methodology as v3)
# ═══════════════════════════════════════════════════════════════════════════════
def _era_ctx_scalar_v4(min_year, max_year):
    _mid = (min_year + max_year) / 2
    if _mid < 1970:
        return 1.15
    elif _mid <= 1990:
        return 1.00
    elif _mid <= 2009:
        return 0.97
    else:
        return 0.88

_career_v4['era_scalar_v4'] = _career_v4.apply(
    lambda r: _era_ctx_scalar_v4(int(r['min_season_year']), int(r['max_season_year'])), axis=1
)

_career_v4['ws_above_repl_v4'] = (
    _career_v4['adj_career_ws'].fillna(0) -
    (_career_v4['n_seasons'].fillna(0) * _REPL_WS_V4)
)

_CTX_COLS_V4 = {'career_vorp': 'ctx4_v', 'career_bpm': 'ctx4_b', 'ws_above_repl_v4': 'ctx4_w'}
for _src, _dst in _CTX_COLS_V4.items():
    _v = _career_v4[_src].fillna(_career_v4[_src].median())
    _mn_c, _mx_c = _v.min(), _v.max()
    _career_v4[_dst] = ((_v - _mn_c) / (_mx_c - _mn_c) * 100) if _mx_c > _mn_c else 50.0

_career_v4['ctx4_v_adj'] = (_career_v4['ctx4_v'] * _career_v4['era_scalar_v4']).clip(0, 100)
_career_v4['ctx4_b_adj'] = (_career_v4['ctx4_b'] * _career_v4['era_scalar_v4']).clip(0, 100)

_career_v4['pillar4_context'] = (
    0.70 * (_career_v4['ctx4_v_adj'] + _career_v4['ctx4_b_adj'] + _career_v4['ctx4_w']) / 3 +
    0.30 * _career_v4['individual_brilliance_score']
).clip(0, 100)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 4: HONORS / RECOGNITION — REBALANCED
# FIX (A): New weight structure:
#   MVP:             35%  (most important individual award)
#   All-NBA (1st=8, 2nd=4, 3rd=1): 28%  (1st >> 2nd >> 3rd, Curry's 10 selections shine)
#   Championships (superteam adj): 18%
#   All-Star:        10%
#   All-Defense (1st=3, 2nd=1.5): 5%  (1st ≈ 2nd All-NBA; 2nd ≈ 3rd All-NBA)
#   DPOY:            4%   (≈ First Team All-NBA per selection)
#
# All-NBA weight spread: 1st=8, 2nd=4, 3rd=1 (vs 5/3/1 in v3)
# This means MVP>DPOY≈First Team; First Def ≈ Second All-NBA; Second Def ≈ Third All-NBA
# ═══════════════════════════════════════════════════════════════════════════════

# All-Def lookup
_alldef_rows_v4 = []
for _pname, (d1, d2) in _ALL_DEF_DB_V4.items():
    _alldef_rows_v4.append({'player_name': _pname, 'all_def_1st_v4': d1, 'all_def_2nd_v4': d2})
_alldef_df_v4 = pd.DataFrame(_alldef_rows_v4)

_career_v4 = _career_v4.merge(_alldef_df_v4, on='player_name', how='left')
_career_v4['all_def_1st_v4'] = _career_v4['all_def_1st_v4'].fillna(0)
_career_v4['all_def_2nd_v4'] = _career_v4['all_def_2nd_v4'].fillna(0)

# FIX (A): Rebalanced All-NBA weights: 1st=8, 2nd=4, 3rd=1
_career_v4['all_nba_wtd_v4'] = (
    _career_v4['all_nba_1st'].fillna(0) * 8 +
    _career_v4['all_nba_2nd'].fillna(0) * 4 +
    _career_v4['all_nba_3rd'].fillna(0) * 1
)

# FIX (A): All-Defense weights: 1st=3pts, 2nd=1.5pts
# This makes First Team All-Defense ≈ Second Team All-NBA (relative contribution)
_career_v4['all_def_wtd_v4'] = (
    _career_v4['all_def_1st_v4'] * 3.0 +
    _career_v4['all_def_2nd_v4'] * 1.5
)

# Superteam ring discount
def _rings_adj_v4(row):
    _total = float(row.get('championships', 0))
    _pname = row['player_name']
    _st = len(_SUPERTEAM_RING_YEARS_V4.get(_pname, []))
    if _st > 0:
        return max(0, (_total - _st) + _st * 0.85)
    return _total

_career_v4['rings_adj_v4'] = _career_v4.apply(_rings_adj_v4, axis=1)

# Normalize honor components
_HON_COLS_V4 = {
    'mvp':            'h4_mvp',
    'rings_adj_v4':   'h4_champ',
    'allstar':        'h4_allstar',
    'all_nba_wtd_v4': 'h4_allnba',
    'all_def_wtd_v4': 'h4_alldef',
    'dpoy':           'h4_dpoy',
}
for _src, _dst in _HON_COLS_V4.items():
    _v = _career_v4[_src].fillna(0)
    _mx = _v.max()
    _career_v4[_dst] = (_v / _mx * 100) if _mx > 0 else 0.0

# FIX (A): New honor weights — MVP > DPOY ≈ First Team All-NBA
_career_v4['pillar4_honors'] = (
    _career_v4['h4_mvp']     * 0.35 +    # MVP most important
    _career_v4['h4_allnba']  * 0.28 +    # All-NBA (tiered 8/4/1)
    _career_v4['h4_champ']   * 0.18 +    # Championships
    _career_v4['h4_allstar'] * 0.10 +    # All-Star selections
    _career_v4['h4_alldef']  * 0.05 +    # All-Def (1st≈2nd All-NBA; 2nd≈3rd All-NBA)
    _career_v4['h4_dpoy']    * 0.04      # DPOY ≈ First Team All-NBA level
)

print("✓ All 4 pillars computed (physical dominance stat removed)")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE GOAT SCORE V4 (4 pillars, no defense pillar)
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_goat_v4(row, w=None):
    if w is None:
        w = _DEFAULT_WEIGHTS_V4
    _wn = _normalize_weights_v4(w)
    return round(
        _wn['volume_longevity']   * float(row.get('pillar4_volume', 0)) +
        _wn['peak_dominance']     * float(row.get('pillar4_peak', 0)) +
        _wn['context_value']      * float(row.get('pillar4_context', 0)) +
        _wn['honors_recognition'] * float(row.get('pillar4_honors', 0)),
        2
    )

_career_v4['goat_score_v4_raw'] = _career_v4.apply(_compute_goat_v4, axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# CAREER GATE V4 (same as v3 but adjusted thresholds for 4-pillar scores)
# <8 seasons → max rank 40
# <2 champs AND <3 All-NBA total → max rank 30
# ═══════════════════════════════════════════════════════════════════════════════
_career_v4 = _career_v4.sort_values('goat_score_v4_raw', ascending=False).reset_index(drop=True)
_sorted_scores_v4 = _career_v4['goat_score_v4_raw'].values

def _apply_gate_v4(row, scores):
    _score = row['goat_score_v4_raw']
    _n_seas = int(row.get('n_seasons', 0))
    _champs = float(row.get('championships', 0))
    _allnba_total = (float(row.get('all_nba_1st', 0)) +
                     float(row.get('all_nba_2nd', 0)) +
                     float(row.get('all_nba_3rd', 0)))
    if _n_seas < 8:
        return min(_score, scores[39] * 0.98) if len(scores) > 39 else min(_score, 35.0)
    if _champs < 2 and _allnba_total < 3:
        return min(_score, scores[29] * 0.98) if len(scores) > 29 else min(_score, 40.0)
    return _score

_career_v4['goat_score_v4'] = _career_v4.apply(
    lambda r: _apply_gate_v4(r, _sorted_scores_v4), axis=1
)

_career_v4 = _career_v4.sort_values('goat_score_v4', ascending=False).reset_index(drop=True)
_career_v4['goat_rank_v4'] = _career_v4.index + 1
_career_v4['goat_pct_v4'] = (
    (len(_career_v4) - _career_v4['goat_rank_v4']) / len(_career_v4) * 100
).round(1)

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE PLAYER PROJECTIONS V4
# ═══════════════════════════════════════════════════════════════════════════════
def _proj_delta_cap_v4(age):
    if age >= 39:
        return 'freeze'
    elif age >= 37:
        return 1.5
    elif age >= 36:
        return 3.5
    elif age >= 33:
        return 7.0
    else:
        return None

def _injury_mult_v4(player_name):
    _p_seas = _seasons_v4[_seasons_v4['player_name'] == player_name]
    if _p_seas.empty:
        return 0.82
    return min(1.0, float(_p_seas['season_g'].mean()) / _GAMES_PER_SEASON_V4)

_proj_base_v4 = (
    career_projections_df[career_projections_df['scenario'] == 'base']
    .sort_values('age').groupby('player').last().reset_index()
)

_proj_rows_v4 = []
for _pname in _ACTIVE_5_V4:
    _curr_age = _CURRENT_AGES_V4.get(_pname, 28)
    _curr_r = _career_v4[_career_v4['player_name'] == _pname]
    if _curr_r.empty:
        print(f"  ⚠ {_pname} NOT FOUND in career data!")
        continue
    _curr = _curr_r.iloc[0]
    _curr_goat = float(_curr['goat_score_v4'])
    _inj = _injury_mult_v4(_pname)
    _cap = _proj_delta_cap_v4(_curr_age)

    if _cap == 'freeze':
        _proj_final = round(_curr_goat * 0.98, 1)
        _proj_rows_v4.append({
            'player_name': _pname, 'current_age': _curr_age,
            'goat_score_v4_current': round(_curr_goat, 1),
            'goat_score_v4_projected': _proj_final,
            'delta_v4': round(_proj_final - _curr_goat, 2),
            'delta_capped_v4': True, 'injury_mult_v4': round(_inj, 3), 'age_cap_type_v4': 'freeze_39+',
        })
        continue

    _pe = _proj_base_v4[_proj_base_v4['player'] == _pname]
    if _pe.empty:
        _proj_rows_v4.append({
            'player_name': _pname, 'current_age': _curr_age,
            'goat_score_v4_current': round(_curr_goat, 1),
            'goat_score_v4_projected': round(_curr_goat, 1),
            'delta_v4': 0.0, 'delta_capped_v4': False,
            'injury_mult_v4': round(_inj, 3), 'age_cap_type_v4': str(_cap),
        })
        continue

    _pe = _pe.iloc[0]
    _pts_max = _career_v4['adj_career_pts'].max()
    _proj_pts = float(_pe.get('cum_pts', float(_curr.get('adj_career_pts', 0))))
    _pv_pts = min(_proj_pts / _pts_max * 100, 100) if _pts_max > 0 else 0
    _pv_others = np.mean([float(_curr.get('vp4_trb', 0)), float(_curr.get('vp4_ast', 0)),
                           float(_curr.get('vp4_g', 0)), float(_curr.get('vp4_seas', 0))])
    _proj_vol = np.mean([_pv_pts, _pv_others])
    _proj_vol_adj = min(100, _proj_vol * float(_curr.get('sc_mult', 1.0)) *
                        float(_curr.get('stockton_mult_v4', 1.0)))

    _proj_vorp = float(_pe.get('cum_vorp', float(_curr.get('career_vorp', 0))))
    _proj_ws = float(_pe.get('cum_ws', float(_curr.get('adj_career_ws', 0))))
    _vorp_mn = _career_v4['career_vorp'].min()
    _vorp_mx = _career_v4['career_vorp'].max()
    _vorp_n = ((_proj_vorp - _vorp_mn) / (_vorp_mx - _vorp_mn) * 100) if _vorp_mx > _vorp_mn else 50
    _ws_mn = _career_v4['ws_above_repl_v4'].min()
    _ws_mx = _career_v4['ws_above_repl_v4'].max()
    _ws_abv = _proj_ws - (_career_v4['n_seasons'].max() * _REPL_WS_V4)
    _ws_n = ((_ws_abv - _ws_mn) / (_ws_mx - _ws_mn) * 100) if _ws_mx > _ws_mn else 50
    _proj_ctx = (0.70 * np.mean([
        max(0, min(100, _vorp_n * float(_curr.get('era_scalar_v4', 1.0)))),
        max(0, min(100, float(_curr.get('ctx4_b_adj', 50)))),
        max(0, min(100, _ws_n))
    ]) + 0.30 * float(_curr.get('individual_brilliance_score', 50)))
    _proj_ctx = max(0, min(100, _proj_ctx))

    _proj_goat_raw = _compute_goat_v4({
        'pillar4_volume': _proj_vol_adj,
        'pillar4_peak':   float(_curr.get('pillar4_peak', 0)),
        'pillar4_context': _proj_ctx,
        'pillar4_honors': float(_curr.get('pillar4_honors', 0)),
    })

    _raw_delta = _proj_goat_raw - _curr_goat
    _inj_delta = _raw_delta * _inj
    if _cap is not None:
        _final_delta = min(_inj_delta, _cap)
        _capped = _inj_delta > _cap
    else:
        _final_delta = _inj_delta
        _capped = False

    _proj_final = round(_curr_goat + _final_delta, 1)
    _proj_rows_v4.append({
        'player_name': _pname, 'current_age': _curr_age,
        'goat_score_v4_current': round(_curr_goat, 1),
        'goat_score_v4_projected': _proj_final,
        'delta_v4': round(_proj_final - _curr_goat, 2),
        'delta_capped_v4': _capped, 'injury_mult_v4': round(_inj, 3),
        'age_cap_type_v4': str(_cap),
    })

active_projections_v4 = pd.DataFrame(_proj_rows_v4)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARABLE PLAYER MATCHING V4
# ═══════════════════════════════════════════════════════════════════════════════
_COMP_FEATS_V4 = ['adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
                   'career_ts_pct', 'career_per', 'career_bpm', 'career_vorp', 'career_ws_per48']

_hist_pool_v4 = _career_v4[_career_v4['player_status'] == 'historical'].copy()
_hist_clean_v4 = _hist_pool_v4[['player_name', 'n_seasons', 'goat_score_v4'] +
                                 _COMP_FEATS_V4].dropna(subset=_COMP_FEATS_V4)

_ss_v4 = StandardScaler()
_X_hist_v4 = _ss_v4.fit_transform(_hist_clean_v4[_COMP_FEATS_V4])

_comp_rows_v4 = []
for _pname in _ACTIVE_5_V4:
    _act = _career_v4[_career_v4['player_name'] == _pname]
    if _act.empty:
        continue
    _act_r = _act.iloc[0]
    _curr_n = int(_act_r.get('n_seasons', 5))
    _age_mask = ((_hist_clean_v4['n_seasons'] >= max(1, _curr_n - 2)) &
                 (_hist_clean_v4['n_seasons'] <= _curr_n + 2))
    _pool = _hist_clean_v4[_age_mask] if _age_mask.sum() >= 3 else _hist_clean_v4
    _feats = np.nan_to_num(_act_r[_COMP_FEATS_V4].values.astype(float), nan=0.0)
    _x_act = _ss_v4.transform([_feats])
    _X_pool = _ss_v4.transform(_pool[_COMP_FEATS_V4])
    _sims = cosine_similarity(_x_act, _X_pool)[0]
    _top5 = np.argsort(_sims)[::-1][:5]
    for _rk, _ix in enumerate(_top5, 1):
        _hr = _pool.iloc[_ix]
        _comp_rows_v4.append({
            'active_player': _pname, 'comp_rank': _rk,
            'comp_player': _hr['player_name'],
            'similarity_pct': round(float(_sims[_ix]) * 100, 1),
            'comp_goat_v4': float(_hr['goat_score_v4']),
            'comp_adj_ppg': round(float(_hr['adj_career_ppg']), 1),
            'comp_career_per': round(float(_hr['career_per']), 1),
            'comp_n_seasons': int(_hr['n_seasons']),
        })

comps_df_v4 = pd.DataFrame(_comp_rows_v4)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
_OUT_V4 = ['goat_rank_v4', 'player_name', 'player_status', 'goat_score_v4', 'goat_pct_v4',
           'pillar4_volume', 'pillar4_peak', 'pillar4_context', 'pillar4_honors',
           'era_scalar_v4', 'individual_brilliance_score', 'peak_year_v4', 'peak_era_mult_v4',
           'prime_team_win_pct', 'avg_team_win_pct', 'adj_career_ppg', 'adj_career_rpg',
           'adj_career_apg', 'career_per', 'career_ts_pct', 'career_bpm', 'career_vorp',
           'adj_career_ws', 'n_seasons', 'career_g', 'mvp', 'championships', 'allstar',
           'all_nba_1st', 'all_nba_2nd', 'all_nba_3rd', 'all_def_1st_v4', 'all_def_2nd_v4', 'dpoy',
           'rings_adj_v4', 'stockton_mult_v4']
_avail_v4 = [_c for _c in _OUT_V4 if _c in _career_v4.columns]
goat_scores_v4 = _career_v4[_avail_v4].copy()

pillar_scores_v4 = _career_v4[[
    'player_name', 'player_status', 'goat_rank_v4',
    'pillar4_volume', 'pillar4_peak', 'pillar4_context', 'pillar4_honors',
    'era_scalar_v4', 'individual_brilliance_score', 'peak_year_v4', 'peak_era_mult_v4',
    'prime_team_win_pct', 'sc_mult', 'all_def_1st_v4', 'all_def_2nd_v4', 'dpoy',
]].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT TOP 30 + SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  GOAT SCORE ENGINE v4 — Rebalanced Honors | No Physical Dom | Curry Fixed")
print("=" * 100)
print(f"\n{'Rk':>3}  {'Player':<30}  {'Score':>6}  {'Vol':>5}  "
      f"{'Peak':>5}  {'Ctx':>5}  {'Hon':>5}  {'PkYr':>5}  {'PkMult':>6}  {'Status'}")
print("-" * 100)

for _, _rr in goat_scores_v4.head(30).iterrows():
    _tag = " ★" if _rr.get('player_status') == 'active' else "  "
    print(f"  {int(_rr['goat_rank_v4']):>2}{_tag}  {_rr['player_name']:<30}  "
          f"{_rr['goat_score_v4']:>6.1f}  "
          f"{_rr['pillar4_volume']:>5.1f}  "
          f"{_rr['pillar4_peak']:>5.1f}  "
          f"{_rr['pillar4_context']:>5.1f}  "
          f"{_rr['pillar4_honors']:>5.1f}  "
          f"{int(_rr.get('peak_year_v4', 0)):>5}  "
          f"{float(_rr.get('peak_era_mult_v4', 1.0)):>6.2f}")

# Curry detail
_curry = goat_scores_v4[goat_scores_v4['player_name'] == 'Stephen Curry']
print(f"\n  CURRY DIAGNOSIS:")
if not _curry.empty:
    _c = _curry.iloc[0]
    print(f"    Rank: #{int(_c['goat_rank_v4'])}  Score: {_c['goat_score_v4']:.2f}")
    print(f"    MVP: {_c.get('mvp',0)}  All-NBA 1st: {_c.get('all_nba_1st',0)}  "
          f"2nd: {_c.get('all_nba_2nd',0)}  3rd: {_c.get('all_nba_3rd',0)}")
    print(f"    Champs: {_c.get('championships',0)}  Peak Year: {_c.get('peak_year_v4',0)}  "
          f"Peak Mult: {_c.get('peak_era_mult_v4',1.0):.2f}")
    print(f"    Vol: {_c['pillar4_volume']:.1f}  Peak: {_c['pillar4_peak']:.1f}  "
          f"Ctx: {_c['pillar4_context']:.1f}  Hon: {_c['pillar4_honors']:.1f}")
else:
    print("    ⚠ CURRY NOT FOUND!")

# LeBron check
_lbj = goat_scores_v4[goat_scores_v4['player_name'] == 'LeBron James']
if not _lbj.empty:
    print(f"\n  LEBRON: Rank #{int(_lbj.iloc[0]['goat_rank_v4'])}  Score: {_lbj.iloc[0]['goat_score_v4']:.1f}")

print(f"\n  ACTIVE PROJECTIONS:")
for _, _pr in active_projections_v4.iterrows():
    print(f"    {_pr['player_name']:<30}  Age: {_pr['current_age']}  "
          f"Current: {_pr['goat_score_v4_current']:.1f}  "
          f"Projected: {_pr['goat_score_v4_projected']:.1f}  "
          f"Delta: {_pr['delta_v4']:+.2f}  Cap: {_pr['age_cap_type_v4']}")

print(f"\n{'=' * 100}")
print("  SANITY CHECKS v4")
print(f"{'=' * 100}")
_CHECKS_V4 = [
    ('LeBron James',            'Top 2',   lambda r: r <= 2),
    ('Stephen Curry',           'In top 25 (visible in rankings)', lambda r: r <= 25),
    ('Michael Jordan',          'Top 3',   lambda r: r <= 3),
    ('Kareem Abdul-Jabbar',     'Top 4',   lambda r: r <= 4),
    ('Nikola Jokic',            'Top 10',  lambda r: r <= 10),
    ('Tim Duncan',              'Top 10',  lambda r: r <= 10),
    ('Shai Gilgeous-Alexander', 'NOT top 20', lambda r: r > 20),
    ('John Stockton',           'NOT top 15', lambda r: r > 15),
]
_all_passed = True
for _nm, _desc, _fn in _CHECKS_V4:
    _rw = goat_scores_v4[goat_scores_v4['player_name'] == _nm]
    if _rw.empty:
        print(f"  ⚠  {_nm}: NOT FOUND")
        _all_passed = False
        continue
    _rnk = int(_rw['goat_rank_v4'].values[0])
    _sc = float(_rw['goat_score_v4'].values[0])
    _ok = _fn(_rnk)
    _all_passed = _all_passed and _ok
    print(f"  {'✓' if _ok else '✗'}  #{_rnk:>3}  {_nm:<30}  Score={_sc:.2f}  [{_desc}]")

print(f"\n  Overall: {'ALL CHECKS PASSED ✓' if _all_passed else 'SOME CHECKS FAILED ✗'}")
print(f"\n  OUTPUT VARIABLES:")
print(f"  ✓ goat_scores_v4          shape: {goat_scores_v4.shape}")
print(f"  ✓ pillar_scores_v4        shape: {pillar_scores_v4.shape}")
print(f"  ✓ active_projections_v4   shape: {active_projections_v4.shape}")
print(f"  ✓ comps_df_v4             shape: {comps_df_v4.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Top 30 GOAT Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
_BG    = '#1D1D20'
_FG    = '#fbfbff'
_MUTED = '#909094'
_C_BLUE   = '#A1C9F4'
_C_ORANGE = '#FFB482'
_C_GREEN  = '#8DE5A1'
_C_CORAL  = '#FF9F9B'
_C_LAVEN  = '#D0BBFF'
_C_GOLD   = '#ffd400'
_C_TEAL   = '#17b26a'

plt.rcParams.update({
    'figure.facecolor': _BG, 'axes.facecolor': _BG,
    'axes.edgecolor': _MUTED, 'axes.labelcolor': _FG,
    'xtick.color': _MUTED, 'ytick.color': _MUTED,
    'text.color': _FG, 'grid.color': '#333337',
    'grid.alpha': 0.5, 'font.family': 'DejaVu Sans',
})

_PLAYER_COLORS_V4 = {
    'Victor Wembanyama': _C_BLUE, 'Shai Gilgeous-Alexander': _C_ORANGE,
    'Nikola Jokic': _C_GREEN, 'LeBron James': _C_CORAL, 'Stephen Curry': _C_LAVEN,
}

_top30_v4 = goat_scores_v4.head(30).copy().sort_values('goat_score_v4', ascending=True)

fig_v4_lb, ax_v4_lb = plt.subplots(figsize=(14, 13))
fig_v4_lb.patch.set_facecolor(_BG)
ax_v4_lb.set_facecolor(_BG)

_bar_colors_v4 = [
    _PLAYER_COLORS_V4.get(row['player_name'], _C_GOLD)
    if row['player_status'] == 'active' else _C_BLUE
    for _, row in _top30_v4.iterrows()
]

_ypos_v4 = np.arange(len(_top30_v4))
_bars_v4 = ax_v4_lb.barh(_ypos_v4, _top30_v4['goat_score_v4'], color=_bar_colors_v4,
                          alpha=0.88, height=0.72)

# Labels
for _bar, (_, _row) in zip(_bars_v4, _top30_v4.iterrows()):
    _x = _bar.get_width()
    _tag = ' ★' if _row['player_status'] == 'active' else ''
    ax_v4_lb.text(_x + 0.3, _bar.get_y() + _bar.get_height() / 2,
                  f"{_x:.1f}{_tag}", va='center', ha='left', fontsize=8, color=_FG)

ax_v4_lb.set_yticks(_ypos_v4)
ax_v4_lb.set_yticklabels(
    [f"#{int(r['goat_rank_v4'])}  {r['player_name']}" for _, r in _top30_v4.iterrows()],
    fontsize=8.5
)
ax_v4_lb.set_xlabel('GOAT Score (0–100)', color=_FG, fontsize=11)
ax_v4_lb.set_title(
    'NBA GOAT Score v4 — Top 30 Leaderboard\n'
    'Rebalanced Honor Weights | No Physical Dominance Stat | Curry Fix Applied | ★ = Active',
    fontsize=13, fontweight='bold', color=_FG, pad=14
)
ax_v4_lb.set_xlim(0, 100)
ax_v4_lb.spines['top'].set_visible(False)
ax_v4_lb.spines['right'].set_visible(False)
ax_v4_lb.spines['left'].set_color(_MUTED)
ax_v4_lb.spines['bottom'].set_color(_MUTED)
ax_v4_lb.grid(axis='x', alpha=0.2)
_patches = [
    mpatches.Patch(color=_C_BLUE, label='Historical'),
    mpatches.Patch(color=_C_CORAL, label='LeBron James'),
    mpatches.Patch(color=_C_LAVEN, label='Stephen Curry'),
    mpatches.Patch(color=_C_GREEN, label='Nikola Jokic'),
    mpatches.Patch(color=_C_ORANGE, label='Shai GAS'),
    mpatches.Patch(color=_C_GOLD, label='Other Active'),
]
ax_v4_lb.legend(handles=_patches, fontsize=8.5, facecolor=_BG, edgecolor=_MUTED,
                 labelcolor=_FG, loc='lower right')
plt.tight_layout()
goat_engine_v4_leaderboard_chart = fig_v4_lb

print("\n✓ goat_engine_v4_leaderboard_chart rendered")
