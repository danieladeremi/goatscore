
"""
GOAT SCORE ENGINE v3 — Complete Rebuild with All 8 Bias Fixes

(1) PEAK ERA DISCOUNT: -20% for 2010-2024 peak seasons, baseline 1990-2009, +10% pre-1990
    Also cap BPM contribution in peak pillar at 8.0
(2) DEFENSE PILLAR: 5th pillar from All-Def selections, DPOY, era-adj BPG/SPG, reputation flag
(3) ALL-DEFENSIVE IN HONORS: All-Def 1st=4pts, 2nd=2pts added to honors pillar
(4) STOCKTON FIX: -10% volume discount if top-5 all-time assists + prime win% > 60% + 0 rings
(5) KD SUPERTEAM DISCOUNT: 15% discount on rings contribution in honors if superteam champion
(6) LEBRON PROJECTION HARD CAP: age 39+ → projected = current * 0.98
(7) SHAQ PHYSICAL DOMINANCE: (BPG*2.5 + RPG*0.8) era-adjusted → 20% of peak pillar
(8) CAREER GATE V3: <8 seasons → max rank 40; <2 champs AND <3 All-NBA → max rank 25

Outputs: goat_df_v3, pillar_scores_v3, active_projections_v3, comps_df_v3
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
_ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
             'Nikola Jokic', 'LeBron James', 'Stephen Curry']

_CURRENT_AGES = {
    'Victor Wembanyama':       21,
    'Shai Gilgeous-Alexander': 26,
    'Nikola Jokic':            29,
    'LeBron James':            40,
    'Stephen Curry':           36,
}

_GAMES_PER_SEASON = 82
_REPL_WS = 2.0  # replacement-level WS per season

# 5-pillar weights (equal default)
_DEFAULT_WEIGHTS_V3 = {
    'volume_longevity':   0.20,
    'peak_dominance':     0.20,
    'context_value':      0.20,
    'honors_recognition': 0.20,
    'defensive_legacy':   0.20,
}

def _normalize_weights(w):
    total = sum(max(0, v) for v in w.values())
    if total == 0:
        n = len(w)
        return {k: 1/n for k in w}
    return {k: max(0, v) / total for k, v in w.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (1): PEAK ERA DISCOUNT SCALARS
# Peak season era: most-productive 3-season window peak year
# 2010-2024: -20% discount; 1990-2009: baseline; pre-1990: +10% bonus
# ═══════════════════════════════════════════════════════════════════════════════
def _peak_era_discount(peak_year):
    """Returns multiplier applied to peak pillar score based on peak season era."""
    if peak_year >= 2010:
        return 0.80   # -20% for modern era metric inflation
    elif peak_year >= 1990:
        return 1.00   # baseline
    else:
        return 1.10   # +10% for pre-1990 undercount

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (2): DEFENSE PILLAR DATABASE
# All-Defensive Team selections (manual, era-verified)
# ═══════════════════════════════════════════════════════════════════════════════
_ALL_DEF_DB = {
    # player_name: (all_def_1st, all_def_2nd)
    'Michael Jordan':            (8, 1),
    'LeBron James':              (5, 1),
    'Kareem Abdul-Jabbar':       (5, 6),
    'Bill Russell':              (0, 0),   # pre-All-Def era, handled via rep flag
    'Wilt Chamberlain':          (0, 0),   # pre-All-Def era, handled via rep flag
    'Magic Johnson':             (0, 0),
    'Larry Bird':                (0, 3),
    'Tim Duncan':                (8, 7),
    'Hakeem Olajuwon':           (5, 4),
    'Shaquille O\'Neal':         (3, 0),
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
    'Patrick Ewing':             (1, 3),
    'Dominique Wilkins':         (0, 0),
    'Alex English':              (0, 0),
    'Bernard King':              (0, 0),
}

# Manual defensive reputation flags (historically elite defenders)
# These players get a bonus 15-point rep bonus in defense pillar
_DEF_REP_PLAYERS = {
    'Hakeem Olajuwon':   20,  # most versatile defender + 3.7 BPG career
    'Dennis Rodman':     25,  # greatest defensive rebounder + 2x DPOY + 7x All-Def 1st
    'Bill Russell':      25,  # greatest winner/team defender, pre-All-Def era
    'Wilt Chamberlain':  10,  # dominant interior, but not lockdown
    'Ben Wallace':       20,  # 4x DPOY, franchise-altering defense
    'Rudy Gobert':       15,  # 4x DPOY but limited on perimeter
    'Gary Payton':       20,  # only PG DPOY, 9x All-Def 1st
    'Scottie Pippen':    15,  # elite 2-way, 8x All-Def 1st
    'Kevin Garnett':     15,  # vocal leader, 9x All-Def 1st, transformed defense
    'Kobe Bryant':       15,  # 9x All-Def 1st, locked up opposing stars
    'Michael Jordan':    15,  # DPOY + 9x All-Def 1st
    'Kawhi Leonard':     15,  # 5x All-Def, 2x DPOY
    'Walt Frazier':      10,  # 6x All-Def 1st, elite perimeter stopper
    'David Robinson':    10,  # 4x All-Def, elite shot blocker
    'Dikembe Mutombo':   15,  # 4x DPOY, premier rim protector
    'Tim Duncan':        10,  # 8x All-Def 1st + 7x 2nd
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (5): SUPERTEAM DISCOUNT — players who won rings on 3+ All-Star teams
# ═══════════════════════════════════════════════════════════════════════════════
# Championship seasons where team had 3+ All-Stars simultaneously
_SUPERTEAM_RING_YEARS = {
    # player: list of championship years on superteams
    'Kevin Durant':      [2017, 2018],           # GSW with Curry/KT/KD/Klay/Dray
    'LeBron James':      [2012, 2013],            # Heat Big 3 + Wade/Bosh
    'Dwyane Wade':       [2012, 2013],            # Heat Big 3
    'Chris Bosh':        [2012, 2013],            # Heat Big 3
    'Ray Allen':         [2008],                  # Boston Big 3 + KG
    'Paul Pierce':       [2008],                  # Boston Big 3
    'Shaquille O\'Neal': [],                      # LAL was built around him
    'Kobe Bryant':       [],                      # LAL was built around him/Shaq
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA from upstream
# ═══════════════════════════════════════════════════════════════════════════════
_career_v3 = era_adj_career_df.copy()
_seasons_v3 = era_adj_seasons_df.copy()

print(f"✓ Loaded {len(_career_v3)} players, {len(_seasons_v3)} season records")

# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTING CAST — reuse from goat_engine_v2
# ═══════════════════════════════════════════════════════════════════════════════
_career_v3 = _career_v3.merge(
    supporting_cast_df[['player_name', 'individual_brilliance_score',
                         'prime_team_win_pct', 'avg_team_win_pct',
                         'sc_mult', 'prime_weighted_bpm_surplus',
                         'prime_avg_apg']],
    on='player_name', how='left'
)
_career_v3['individual_brilliance_score'] = _career_v3['individual_brilliance_score'].fillna(50)
_career_v3['prime_team_win_pct'] = _career_v3['prime_team_win_pct'].fillna(0.500)
_career_v3['avg_team_win_pct'] = _career_v3['avg_team_win_pct'].fillna(0.500)
_career_v3['sc_mult'] = _career_v3['sc_mult'].fillna(1.0)
_career_v3['prime_avg_apg'] = _career_v3['prime_avg_apg'].fillna(0)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1: VOLUME / LONGEVITY
# ═══════════════════════════════════════════════════════════════════════════════
_vol_cols = {
    'adj_career_pts': 'vp3_pts',
    'adj_career_trb': 'vp3_trb',
    'adj_career_ast': 'vp3_ast',
    'career_g':       'vp3_g',
    'n_seasons':      'vp3_seas',
}
for _src, _dst in _vol_cols.items():
    _col = _career_v3[_src].fillna(0)
    _mx = _col.max()
    _career_v3[_dst] = (_col / _mx * 100) if _mx > 0 else 0.0

_career_v3['pillar3_volume_raw'] = _career_v3[list(_vol_cols.values())].mean(axis=1)

# FIX (4): STOCKTON SYSTEM-DEPENDENCY CHECK
# Top-5 all-time assists AND prime win% > 60% AND 0 championships → -10% on volume
_ast_sorted = _career_v3.nlargest(5, 'career_ast')['player_name'].tolist()
def _stockton_discount(row):
    """Apply -10% volume discount for assist-dependent system players."""
    if (row['player_name'] in _ast_sorted and
        row.get('prime_team_win_pct', 0.5) > 0.60 and
        row.get('championships', 0) == 0):
        return 0.90
    return 1.0

_career_v3['stockton_mult'] = _career_v3.apply(_stockton_discount, axis=1)
_career_v3['pillar3_volume_adj'] = (
    _career_v3['pillar3_volume_raw'] *
    _career_v3['sc_mult'].clip(0.88, 1.05) *
    _career_v3['stockton_mult']
).clip(0, 100)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2: PEAK DOMINANCE (with Era Discount + BPM Cap + Physical Dominance)
# ═══════════════════════════════════════════════════════════════════════════════
_PEAK_METRICS = ['season_per', 'season_ts_pct', 'season_bpm']

def _best3avg_v3(grp, col):
    _vals = grp.sort_values('season_start_year')[col].dropna().values
    if len(_vals) == 0:
        return np.nan, None
    if len(_vals) < 3:
        # Find year of max val
        _idx = np.argmax(_vals)
        _years = grp.sort_values('season_start_year')['season_start_year'].values
        _peak_yr = int(_years[_idx]) if len(_years) > _idx else 2000
        return float(np.mean(_vals)), _peak_yr
    # Rolling 3-season average
    _rolled = np.convolve(_vals, np.ones(3)/3, mode='valid')
    _best_idx = np.argmax(_rolled)
    # Get the middle year of best 3-season window
    _years = grp.sort_values('season_start_year')['season_start_year'].values
    _peak_yr = int(_years[_best_idx + 1]) if len(_years) > _best_idx + 1 else int(_years[-1])
    return float(np.max(_rolled)), _peak_yr

# FIX (7): PHYSICAL DOMINANCE sub-score for big men
# (BPG * 2.5 + RPG * 0.8) era-adjusted, contributing 20% of peak pillar
def _era_adj_physical(row):
    """Era-adjust blocks and rebounds per game for physical dominance."""
    # Pace-adjust: multiply by modern_pace/their_era_pace
    _era_factor = float(row.get('avg_pace_factor', 1.0))
    _bpg = float(row.get('career_bpg', 0) or 0)
    _rpg = float(row.get('career_rpg', 0) or 0)
    # Era-adjust by pace factor
    _bpg_adj = _bpg * min(1.2, max(0.9, _era_factor))
    _rpg_adj = _rpg * min(1.1, max(0.9, _era_factor))
    return (_bpg_adj * 2.5) + (_rpg_adj * 0.8)

_pk3_rows = []
for _pname, _grp in _seasons_v3.groupby('player_name'):
    _pk = {'player_name': _pname}
    _peak_years_m = []
    for _m in _PEAK_METRICS:
        _val, _yr = _best3avg_v3(_grp, _m)
        if _m == 'season_bpm':
            # FIX (1): Cap BPM at 8.0 (league leader level) to prevent modern outlier inflation
            _val = min(_val, 8.0) if _val is not None and not np.isnan(_val) else _val
        _pk[f'pk3_{_m}'] = _val
        if _yr is not None:
            _peak_years_m.append(_yr)
    # Peak year = average of peak years across metrics
    _pk['peak_year'] = int(np.median(_peak_years_m)) if _peak_years_m else 2000
    _pk3_rows.append(_pk)

_peak_df_v3 = pd.DataFrame(_pk3_rows)

# Normalize peak metrics
for _m in _PEAK_METRICS:
    _col = _peak_df_v3[f'pk3_{_m}'].fillna(0)
    _mn_p, _mx_p = _col.min(), _col.max()
    _peak_df_v3[f'pk3_{_m}_n'] = ((_col - _mn_p) / (_mx_p - _mn_p) * 100) if _mx_p > _mn_p else 50.0

_peak_df_v3['pillar3_peak_raw'] = _peak_df_v3[[f'pk3_{m}_n' for m in _PEAK_METRICS]].mean(axis=1)

# Merge peak data back to career
_career_v3 = _career_v3.merge(
    _peak_df_v3[['player_name', 'pillar3_peak_raw', 'peak_year'] + [f'pk3_{m}_n' for m in _PEAK_METRICS]],
    on='player_name', how='left'
)
_career_v3['pillar3_peak_raw'] = _career_v3['pillar3_peak_raw'].fillna(
    _career_v3['pillar3_peak_raw'].median()
)
_career_v3['peak_year'] = _career_v3['peak_year'].fillna(2000).astype(int)

# FIX (7): Physical dominance sub-score (20% of peak pillar)
_career_v3['phys_dom_raw'] = _career_v3.apply(_era_adj_physical, axis=1)
_phys_mx = _career_v3['phys_dom_raw'].max()
_career_v3['phys_dom_norm'] = (_career_v3['phys_dom_raw'] / _phys_mx * 100).clip(0, 100)

# Blend: 80% PER/TS/BPM peak + 20% physical dominance
_career_v3['pillar3_peak_blended'] = (
    0.80 * _career_v3['pillar3_peak_raw'] +
    0.20 * _career_v3['phys_dom_norm']
)

# FIX (1): Apply peak era discount
_career_v3['peak_era_mult'] = _career_v3['peak_year'].apply(_peak_era_discount)
_career_v3['pillar3_peak'] = (
    _career_v3['pillar3_peak_blended'] * _career_v3['peak_era_mult']
).clip(0, 100)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 3: CONTEXT-ADJUSTED VALUE
# (era scalar + individual brilliance)
# ═══════════════════════════════════════════════════════════════════════════════
def _era_ctx_scalar(min_year, max_year):
    _mid = (min_year + max_year) / 2
    if _mid < 1970:
        return 1.15
    elif _mid <= 1990:
        return 1.00
    elif _mid <= 2009:
        return 0.97
    else:
        return 0.88  # Modern BPM/VORP inflation discount

_career_v3['era_scalar_v3'] = _career_v3.apply(
    lambda r: _era_ctx_scalar(int(r['min_season_year']), int(r['max_season_year'])), axis=1
)

_career_v3['ws_above_repl_v3'] = (
    _career_v3['adj_career_ws'].fillna(0) -
    (_career_v3['n_seasons'].fillna(0) * _REPL_WS)
)

_CTX_COLS = {'career_vorp': 'ctx3_v', 'career_bpm': 'ctx3_b', 'ws_above_repl_v3': 'ctx3_w'}
for _src, _dst in _CTX_COLS.items():
    _v = _career_v3[_src].fillna(_career_v3[_src].median())
    _mn_c, _mx_c = _v.min(), _v.max()
    _career_v3[_dst] = ((_v - _mn_c) / (_mx_c - _mn_c) * 100) if _mx_c > _mn_c else 50.0

_career_v3['ctx3_v_adj'] = (_career_v3['ctx3_v'] * _career_v3['era_scalar_v3']).clip(0, 100)
_career_v3['ctx3_b_adj'] = (_career_v3['ctx3_b'] * _career_v3['era_scalar_v3']).clip(0, 100)

_career_v3['pillar3_context'] = (
    0.70 * (_career_v3['ctx3_v_adj'] + _career_v3['ctx3_b_adj'] + _career_v3['ctx3_w']) / 3 +
    0.30 * _career_v3['individual_brilliance_score']
).clip(0, 100)

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 4: HONORS / RECOGNITION
# FIX (3): Add All-Defensive selections (1st=4pts, 2nd=2pts)
# FIX (5): KD superteam ring discount (15% off ring contribution)
# ═══════════════════════════════════════════════════════════════════════════════

# Build All-Def lookup DataFrame
_alldef_rows = []
for _pname, (d1, d2) in _ALL_DEF_DB.items():
    _alldef_rows.append({'player_name': _pname, 'all_def_1st': d1, 'all_def_2nd': d2})
_alldef_df = pd.DataFrame(_alldef_rows)

_career_v3 = _career_v3.merge(_alldef_df, on='player_name', how='left')
_career_v3['all_def_1st'] = _career_v3['all_def_1st'].fillna(0)
_career_v3['all_def_2nd'] = _career_v3['all_def_2nd'].fillna(0)

# All-NBA weighted (existing)
_career_v3['all_nba_weighted_v3'] = (
    _career_v3['all_nba_1st'].fillna(0) * 5 +
    _career_v3['all_nba_2nd'].fillna(0) * 3 +
    _career_v3['all_nba_3rd'].fillna(0) * 1
)

# FIX (3): All-Defensive weighted score
_career_v3['all_def_weighted'] = (
    _career_v3['all_def_1st'] * 4 +
    _career_v3['all_def_2nd'] * 2
)

# FIX (5): Superteam ring discount
def _rings_after_superteam_discount(row):
    """Apply 15% discount to rings won on superteams."""
    _total_rings = float(row.get('championships', 0))
    _pname = row['player_name']
    _st_rings = len(_SUPERTEAM_RING_YEARS.get(_pname, []))
    if _st_rings > 0:
        _discounted = _st_rings * 0.85  # 15% discount on superteam rings
        _clean_rings = _total_rings - _st_rings
        return max(0, _clean_rings + _discounted)
    return _total_rings

_career_v3['rings_adj'] = _career_v3.apply(_rings_after_superteam_discount, axis=1)

# Normalize honor components
_HON_COLS = {
    'mvp':               'h3_mvp',
    'rings_adj':         'h3_champ',
    'allstar':           'h3_allstar',
    'all_nba_weighted_v3': 'h3_allnba',
    'all_def_weighted':  'h3_alldef',  # FIX (3)
    'dpoy':              'h3_dpoy',    # DPOY also in honors
    'finals_mvp':        'h3_fmvp',
}
for _src, _dst in _HON_COLS.items():
    _v = _career_v3[_src].fillna(0)
    _mx = _v.max()
    _career_v3[_dst] = (_v / _mx * 100) if _mx > 0 else 0.0

_career_v3['pillar3_honors'] = (
    _career_v3['h3_mvp']     * 0.25 +
    _career_v3['h3_allnba']  * 0.22 +
    _career_v3['h3_champ']   * 0.20 +
    _career_v3['h3_allstar'] * 0.12 +
    _career_v3['h3_alldef']  * 0.10 +  # FIX (3): All-Def in honors
    _career_v3['h3_dpoy']    * 0.06 +
    _career_v3['h3_fmvp']    * 0.05
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (2): PILLAR 5 — DEFENSIVE LEGACY
# Components: All-Def selections, DPOY, era-adj BPG, era-adj SPG, rep flag
# ═══════════════════════════════════════════════════════════════════════════════

# Era-adjusted defensive stats (BPG, SPG)
# Blocks and steals are era-dependent - more blocks in 1970s/80s, fewer in modern
_MODERN_BPG_BASELINE = 0.35  # modern avg BPG for position
_MODERN_SPG_BASELINE = 0.65  # modern avg SPG

_career_v3['era_adj_bpg'] = (
    _career_v3['career_bpg'].fillna(0) *
    _career_v3['avg_pace_factor'].fillna(1.0).clip(0.85, 1.15)
)
_career_v3['era_adj_spg'] = (
    _career_v3['career_spg'].fillna(0) *
    _career_v3['avg_pace_factor'].fillna(1.0).clip(0.85, 1.15)
)

# Defensive rep bonus
_career_v3['def_rep_bonus'] = _career_v3['player_name'].map(_DEF_REP_PLAYERS).fillna(0)

# Compute raw defense score (not yet normalized to 0-100)
_career_v3['def_alldef_pts'] = (
    _career_v3['all_def_1st'] * 5 +
    _career_v3['all_def_2nd'] * 3
)
_career_v3['def_dpoy_pts']   = _career_v3['dpoy'].fillna(0) * 10
_career_v3['def_bpg_pts']    = (_career_v3['era_adj_bpg'] * 15).clip(0, 50)
_career_v3['def_spg_pts']    = (_career_v3['era_adj_spg'] * 10).clip(0, 30)

_career_v3['def_raw_score'] = (
    _career_v3['def_alldef_pts'] +
    _career_v3['def_dpoy_pts'] +
    _career_v3['def_bpg_pts'] +
    _career_v3['def_spg_pts'] +
    _career_v3['def_rep_bonus']
)

# Normalize to 0-100
_def_mx = _career_v3['def_raw_score'].max()
_career_v3['pillar3_defense'] = (
    (_career_v3['def_raw_score'] / _def_mx * 100)
).clip(0, 100)

print("✓ All 5 pillars computed")

# Verify key defenders score 80+ on defense pillar
_def_checks = ['Hakeem Olajuwon', 'Dennis Rodman', 'Bill Russell']
for _dc in _def_checks:
    _r = _career_v3[_career_v3['player_name'] == _dc]
    if not _r.empty:
        _ds = float(_r['pillar3_defense'].values[0])
        _icon = '✓' if _ds >= 80 else '✗'
        print(f"  {_icon} {_dc} Defense Pillar: {_ds:.1f} (target: 80+)")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE GOAT SCORE V3 (5 pillars)
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_goat_v3(row, w=None):
    if w is None:
        w = _DEFAULT_WEIGHTS_V3
    _wn = _normalize_weights(w)
    return round(
        _wn['volume_longevity']   * float(row.get('pillar3_volume_adj', 0)) +
        _wn['peak_dominance']     * float(row.get('pillar3_peak', 0)) +
        _wn['context_value']      * float(row.get('pillar3_context', 0)) +
        _wn['honors_recognition'] * float(row.get('pillar3_honors', 0)) +
        _wn['defensive_legacy']   * float(row.get('pillar3_defense', 0)),
        2
    )

_career_v3['goat_score_v3_raw'] = _career_v3.apply(_compute_goat_v3, axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (8): CAREER GATE V3
# <8 seasons → max rank 40
# <2 champs AND <3 All-NBA total → max rank 25
# ═══════════════════════════════════════════════════════════════════════════════
_career_v3 = _career_v3.sort_values('goat_score_v3_raw', ascending=False).reset_index(drop=True)
_career_v3['tentative_rank'] = _career_v3.index + 1

def _apply_career_gate_v3(row, all_scores):
    _score = row['goat_score_v3_raw']
    _n_seas = int(row.get('n_seasons', 0))
    _champs = float(row.get('championships', 0))
    _allnba_total = (float(row.get('all_nba_1st', 0)) +
                     float(row.get('all_nba_2nd', 0)) +
                     float(row.get('all_nba_3rd', 0)))

    # Gate 1: <8 seasons → max rank 40
    if _n_seas < 8:
        # Cap score to be at rank 41 (score of player at rank 40)
        _rank40_score = all_scores[39] if len(all_scores) > 39 else 35.0
        return min(_score, _rank40_score * 0.98)

    # Gate 2: <2 champs AND <3 All-NBA → max rank 25
    if _champs < 2 and _allnba_total < 3:
        _rank25_score = all_scores[24] if len(all_scores) > 24 else 42.0
        return min(_score, _rank25_score * 0.98)

    return _score

_sorted_scores = _career_v3['goat_score_v3_raw'].sort_values(ascending=False).values
_career_v3['goat_score_v3'] = _career_v3.apply(
    lambda r: _apply_career_gate_v3(r, _sorted_scores), axis=1
)

# Re-sort and rank
_career_v3 = _career_v3.sort_values('goat_score_v3', ascending=False).reset_index(drop=True)
_career_v3['goat_rank_v3'] = _career_v3.index + 1
_career_v3['goat_percentile_v3'] = (
    (len(_career_v3) - _career_v3['goat_rank_v3']) / len(_career_v3) * 100
).round(1)

print(f"✓ GOAT v3 scores computed for {len(_career_v3)} players")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX (6): LEBRON PROJECTION HARD CAP + ACTIVE PLAYER PROJECTIONS V3
# age 39+: projected = current * 0.98 (flat or very slight decline)
# age 37-38: projected = current + min(delta, 1.5)
# ═══════════════════════════════════════════════════════════════════════════════
def _proj_delta_cap_v3(age):
    if age >= 39:
        return 'freeze'  # current * 0.98
    elif age >= 37:
        return 1.5
    elif age >= 36:
        return 3.5
    elif age >= 33:
        return 7.0
    else:
        return None  # Full projection

def _injury_mult_v3(player_name):
    _p_seas = _seasons_v3[_seasons_v3['player_name'] == player_name]
    if _p_seas.empty:
        return 0.82
    return min(1.0, float(_p_seas['season_g'].mean()) / _GAMES_PER_SEASON)

_proj_base_v3 = (
    career_projections_df[career_projections_df['scenario'] == 'base']
    .sort_values('age')
    .groupby('player')
    .last()
    .reset_index()
)

_proj_rows_v3 = []
for _pname in _ACTIVE_5:
    _curr_age = _CURRENT_AGES.get(_pname, 28)
    _curr_r = _career_v3[_career_v3['player_name'] == _pname]
    if _curr_r.empty:
        continue
    _curr = _curr_r.iloc[0]
    _curr_goat = float(_curr['goat_score_v3'])
    _inj_mult = _injury_mult_v3(_pname)
    _cap = _proj_delta_cap_v3(_curr_age)

    # FIX (6): LeBron age 39+ hard cap
    if _cap == 'freeze':
        _proj_final = round(_curr_goat * 0.98, 1)
        _proj_rows_v3.append({
            'player_name':         _pname,
            'current_age':         _curr_age,
            'goat_score_v3_current': round(_curr_goat, 1),
            'goat_score_v3_projected': _proj_final,
            'delta_v3':            round(_proj_final - _curr_goat, 2),
            'delta_capped_v3':     True,
            'injury_multiplier_v3': round(_inj_mult, 3),
            'age_cap_type':        'freeze_39+',
            'proj_pillar3_volume':  round(float(_curr.get('pillar3_volume_adj', 0)), 1),
            'proj_pillar3_peak':    round(float(_curr.get('pillar3_peak', 0)), 1),
            'proj_pillar3_context': round(float(_curr.get('pillar3_context', 0)), 1),
            'proj_pillar3_honors':  round(float(_curr.get('pillar3_honors', 0)), 1),
            'proj_pillar3_defense': round(float(_curr.get('pillar3_defense', 0)), 1),
        })
        continue

    # Get upstream projection end-state
    _pe = _proj_base_v3[_proj_base_v3['player'] == _pname]
    if _pe.empty:
        _proj_final = round(_curr_goat, 1)
    else:
        _pe = _pe.iloc[0]
        # Estimate projected volume improvement
        _pts_max = _career_v3['adj_career_pts'].max()
        _proj_pts = float(_pe.get('cum_pts', float(_curr.get('adj_career_pts', 0))))
        _pv_pts = min(_proj_pts / _pts_max * 100, 100) if _pts_max > 0 else 0
        _pv_others = np.mean([float(_curr.get('vp3_trb', 0)),
                               float(_curr.get('vp3_ast', 0)),
                               float(_curr.get('vp3_g', 0)),
                               float(_curr.get('vp3_seas', 0))])
        _proj_vol = np.mean([_pv_pts, _pv_others])
        _proj_vol_adj = min(100, _proj_vol * float(_curr.get('sc_mult', 1.0)) *
                            float(_curr.get('stockton_mult', 1.0)))

        # Context improvement from projected VORP/WS
        _proj_vorp = float(_pe.get('cum_vorp', float(_curr.get('career_vorp', 0))))
        _proj_ws   = float(_pe.get('cum_ws', float(_curr.get('adj_career_ws', 0))))
        _vorp_mn = _career_v3['career_vorp'].min()
        _vorp_mx = _career_v3['career_vorp'].max()
        _vorp_n = ((_proj_vorp - _vorp_mn) / (_vorp_mx - _vorp_mn) * 100) if _vorp_mx > _vorp_mn else 50
        _ws_abv = _proj_ws - (_career_v3['n_seasons'].max() * _REPL_WS)
        _ws_mn = _career_v3['ws_above_repl_v3'].min()
        _ws_mx = _career_v3['ws_above_repl_v3'].max()
        _ws_n = ((_ws_abv - _ws_mn) / (_ws_mx - _ws_mn) * 100) if _ws_mx > _ws_mn else 50
        _proj_ctx = (0.70 * np.mean([
            max(0, min(100, _vorp_n * float(_curr.get('era_scalar_v3', 1.0)))),
            max(0, min(100, float(_curr.get('ctx3_b_adj', 50)))),
            max(0, min(100, _ws_n))
        ]) + 0.30 * float(_curr.get('individual_brilliance_score', 50)))
        _proj_ctx = max(0, min(100, _proj_ctx))

        _proj_goat_raw = _compute_goat_v3({
            'pillar3_volume_adj': _proj_vol_adj,
            'pillar3_peak':       float(_curr.get('pillar3_peak', 0)),
            'pillar3_context':    _proj_ctx,
            'pillar3_honors':     float(_curr.get('pillar3_honors', 0)),
            'pillar3_defense':    float(_curr.get('pillar3_defense', 0)),
        })

        _raw_delta = _proj_goat_raw - _curr_goat
        _inj_adj_delta = _raw_delta * _inj_mult

        if _cap is not None:
            _final_delta = min(_inj_adj_delta, _cap)
            _capped = _inj_adj_delta > _cap
        else:
            _final_delta = _inj_adj_delta
            _capped = False

        _proj_final = round(_curr_goat + _final_delta, 1)

    _proj_rows_v3.append({
        'player_name':           _pname,
        'current_age':           _curr_age,
        'goat_score_v3_current': round(_curr_goat, 1),
        'goat_score_v3_projected': _proj_final,
        'delta_v3':              round(_proj_final - _curr_goat, 2),
        'delta_capped_v3':       _capped if '_capped' in dir() else False,
        'injury_multiplier_v3':  round(_inj_mult, 3),
        'age_cap_type':          str(_cap),
        'proj_pillar3_volume':   round(float(_curr.get('pillar3_volume_adj', 0)), 1),
        'proj_pillar3_peak':     round(float(_curr.get('pillar3_peak', 0)), 1),
        'proj_pillar3_context':  round(float(_curr.get('pillar3_context', 0)), 1),
        'proj_pillar3_honors':   round(float(_curr.get('pillar3_honors', 0)), 1),
        'proj_pillar3_defense':  round(float(_curr.get('pillar3_defense', 0)), 1),
    })

active_projections_v3 = pd.DataFrame(_proj_rows_v3)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARABLE PLAYER MATCHING V3
# ═══════════════════════════════════════════════════════════════════════════════
_COMP_FEATS_V3 = ['adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
                   'career_ts_pct', 'career_per', 'career_bpm',
                   'career_vorp', 'career_ws_per48']

_hist_pool_v3 = _career_v3[_career_v3['player_status'] == 'historical'].copy()
_hist_clean_v3 = _hist_pool_v3[
    ['player_name', 'n_seasons', 'goat_score_v3'] + _COMP_FEATS_V3
].dropna(subset=_COMP_FEATS_V3)

_ss_v3 = StandardScaler()
_X_hist_v3 = _ss_v3.fit_transform(_hist_clean_v3[_COMP_FEATS_V3])

_comp_rows_v3 = []
for _pname in _ACTIVE_5:
    _act = _career_v3[_career_v3['player_name'] == _pname]
    if _act.empty:
        continue
    _act_r = _act.iloc[0]
    _curr_n = int(_act_r.get('n_seasons', 5))
    _age_mask = ((_hist_clean_v3['n_seasons'] >= max(1, _curr_n - 2)) &
                 (_hist_clean_v3['n_seasons'] <= _curr_n + 2))
    _pool_v3 = _hist_clean_v3[_age_mask]
    if len(_pool_v3) < 3:
        _pool_v3 = _hist_clean_v3

    _feats_v3 = _act_r[_COMP_FEATS_V3].values
    _feats_v3 = np.nan_to_num(_feats_v3.astype(float), nan=0.0)
    _x_act_v3 = _ss_v3.transform([_feats_v3])
    _X_pool_v3 = _ss_v3.transform(_pool_v3[_COMP_FEATS_V3])
    _sims_v3 = cosine_similarity(_x_act_v3, _X_pool_v3)[0]
    _top5_v3 = np.argsort(_sims_v3)[::-1][:5]

    for _rk, _ix in enumerate(_top5_v3, 1):
        _hr = _pool_v3.iloc[_ix]
        _comp_rows_v3.append({
            'active_player':    _pname,
            'active_age':       _CURRENT_AGES.get(_pname, 28),
            'comp_rank':        _rk,
            'comp_player':      _hr['player_name'],
            'similarity_pct':   round(float(_sims_v3[_ix]) * 100, 1),
            'comp_goat_v3':     float(_hr['goat_score_v3']),
            'comp_adj_ppg':     round(float(_hr['adj_career_ppg']), 1),
            'comp_career_per':  round(float(_hr['career_per']), 1),
            'comp_career_bpm':  round(float(_hr['career_bpm']), 2),
            'comp_n_seasons':   int(_hr['n_seasons']),
        })

comps_df_v3 = pd.DataFrame(_comp_rows_v3)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT TABLES
# ═══════════════════════════════════════════════════════════════════════════════
_OUT_V3 = ['goat_rank_v3', 'player_name', 'player_status', 'goat_score_v3',
           'goat_percentile_v3',
           'pillar3_volume_adj', 'pillar3_peak', 'pillar3_context',
           'pillar3_honors', 'pillar3_defense',
           'era_scalar_v3', 'individual_brilliance_score', 'peak_year',
           'prime_team_win_pct', 'avg_team_win_pct',
           'adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
           'career_per', 'career_ts_pct', 'career_bpm', 'career_vorp',
           'adj_career_ws', 'n_seasons', 'career_g',
           'mvp', 'championships', 'allstar', 'all_nba_1st', 'all_nba_2nd',
           'all_nba_3rd', 'all_def_1st', 'all_def_2nd', 'dpoy',
           'career_bpg', 'career_spg', 'phys_dom_norm', 'peak_era_mult',
           'stockton_mult', 'rings_adj']
_avail_v3 = [_c for _c in _OUT_V3 if _c in _career_v3.columns]
goat_df_v3 = _career_v3[_avail_v3].copy()

pillar_scores_v3 = _career_v3[[
    'player_name', 'player_status', 'goat_rank_v3',
    'pillar3_volume_adj', 'pillar3_peak', 'pillar3_context',
    'pillar3_honors', 'pillar3_defense',
    'era_scalar_v3', 'individual_brilliance_score',
    'peak_year', 'peak_era_mult', 'phys_dom_norm',
    'prime_team_win_pct', 'sc_mult', 'stockton_mult',
    'all_def_1st', 'all_def_2nd', 'dpoy', 'def_rep_bonus',
]].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT FULL TOP 25 + SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("  GOAT SCORE ENGINE v3 — 8 Bias Fixes | 5 Pillars")
print("=" * 110)
print(f"\n{'Rk':>3}  {'Player':<28}  {'Score':>5}  "
      f"{'Vol':>5}  {'Peak':>5}  {'Ctx':>5}  {'Hon':>5}  {'Def':>5}  "
      f"{'PkYr':>5}  {'PkMult':>6}  {'PhysDom':>7}")
print("-" * 110)

for _, _rr in goat_df_v3.head(25).iterrows():
    _active_tag = " ★" if _rr.get('player_status') == 'active' else "  "
    print(f"  {int(_rr['goat_rank_v3']):>2}{_active_tag}  {_rr['player_name']:<28}  "
          f"{_rr['goat_score_v3']:>5.1f}  "
          f"{_rr['pillar3_volume_adj']:>5.1f}  "
          f"{_rr['pillar3_peak']:>5.1f}  "
          f"{_rr['pillar3_context']:>5.1f}  "
          f"{_rr['pillar3_honors']:>5.1f}  "
          f"{_rr['pillar3_defense']:>5.1f}  "
          f"{int(_rr.get('peak_year', 0)):>5}  "
          f"{float(_rr.get('peak_era_mult', 1.0)):>6.2f}  "
          f"{float(_rr.get('phys_dom_norm', 0)):>7.1f}")

print(f"\n{'─' * 90}")
print("  ACTIVE PLAYER PROJECTIONS v3 (Age Gate + Hard LeBron Cap)")
print(f"{'─' * 90}")
print(f"  {'Player':<30}  {'Age':>4}  {'Current':>8}  {'Projected':>10}  {'Delta':>7}  {'Cap Type'}")
print("  " + "-" * 80)
for _, _pr in active_projections_v3.iterrows():
    print(f"  {_pr['player_name']:<30}  {_pr['current_age']:>4}  "
          f"{_pr['goat_score_v3_current']:>8.1f}  "
          f"{_pr['goat_score_v3_projected']:>10.1f}  "
          f"{_pr['delta_v3']:>+7.2f}  "
          f"{_pr['age_cap_type']}")

print(f"\n{'═' * 110}")
print("  SANITY CHECKS v3")
print(f"{'═' * 110}")
_CHECKS_V3 = [
    ('Michael Jordan',          'Top 3',      lambda r: r <= 3),
    ('LeBron James',            'Top 3',      lambda r: r <= 3),
    ('Kareem Abdul-Jabbar',     'Top 3',      lambda r: r <= 3),
    ('Shaquille O\'Neal',       'Top 8',      lambda r: r <= 8),
    ('Hakeem Olajuwon',         'Top 12',     lambda r: r <= 12),
    ('Nikola Jokic',            'Top 10 max', lambda r: r <= 10),
    ('Kevin Durant',            'NOT top 5',  lambda r: r > 5),
    ('John Stockton',           'NOT top 15', lambda r: r > 15),
    ('Shai Gilgeous-Alexander', 'NOT top 30', lambda r: r > 30),
]
for _nm, _desc, _fn in _CHECKS_V3:
    _rw = goat_df_v3[goat_df_v3['player_name'] == _nm]
    if _rw.empty:
        print(f"  ⚠  {_nm}: NOT FOUND")
        continue
    _rnk = int(_rw['goat_rank_v3'].values[0])
    _sc  = float(_rw['goat_score_v3'].values[0])
    _ok  = _fn(_rnk)
    print(f"  {'✓' if _ok else '✗'}  #{_rnk:>3}  {_nm:<30}  Score={_sc:>5.1f}  [{_desc}]")

# LeBron delta check
_lbj = active_projections_v3[active_projections_v3['player_name'] == 'LeBron James']
if not _lbj.empty:
    _lbj_delta = float(_lbj['delta_v3'].values[0])
    _lbj_ok = abs(_lbj_delta) < 1.0
    print(f"\n  {'✓' if _lbj_ok else '✗'}  LeBron delta={_lbj_delta:+.2f}  [|delta| < 1.0]")

# Defense pillar checks
print(f"\n  DEFENSE PILLAR CHECKS (target: 80+):")
for _dc in ['Hakeem Olajuwon', 'Dennis Rodman', 'Bill Russell']:
    _rw = goat_df_v3[goat_df_v3['player_name'] == _dc]
    if not _rw.empty:
        _ds = float(_rw['pillar3_defense'].values[0])
        print(f"  {'✓' if _ds >= 80 else '✗'}  {_dc:<28}  Defense={_ds:.1f}")

print(f"\n  OUTPUT VARIABLES:")
print(f"  ✓ goat_df_v3             shape: {goat_df_v3.shape}")
print(f"  ✓ pillar_scores_v3       shape: {pillar_scores_v3.shape}")
print(f"  ✓ active_projections_v3  shape: {active_projections_v3.shape}")
print(f"  ✓ comps_df_v3            shape: {comps_df_v3.shape}")
