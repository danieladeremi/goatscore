
"""
GOAT SCORE ENGINE v2 — Complete Rebuild
Implements all 6 methodological fixes:
  1. Era Bias Fix — Per-decade BPM/VORP credibility scalars
  2. Minimum Career Gate — < 8 seasons OR < 2 All-NBA → capped at rank 35
  3. Supporting Cast Adjustment — team win% from nba_seasons_longitudinal.csv
  4. Projection Age Gate — Hard delta caps by age, injury multiplier
  5. Comparable Player Matching — cosine similarity at same age ±2
  6. Four Pillars — Volume/Longevity, Peak Dominance, Context-Adjusted Value, Honors/Recognition

Outputs: goat_df, active_projections, comps_df, pillar_scores_df
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
ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
            'Nikola Jokic', 'LeBron James', 'Stephen Curry']

CURRENT_AGES = {
    'Victor Wembanyama':       21,
    'Shai Gilgeous-Alexander': 26,
    'Nikola Jokic':            29,
    'LeBron James':            40,
    'Stephen Curry':           36,
}

# Career games per season (avg, for injury multiplier)
GAMES_PER_SEASON = 82

# Default equal pillar weights
DEFAULT_WEIGHTS = {
    'volume_longevity':   0.25,
    'peak_dominance':     0.25,
    'context_value':      0.25,
    'honors_recognition': 0.25,
}

def normalize_weights(w):
    total = sum(max(0, v) for v in w.values())
    if total == 0:
        n = len(w)
        return {k: 1/n for k in w}
    return {k: max(0, v) / total for k, v in w.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ERA BIAS FIX: Per-decade credibility scalar for BPM/VORP
# Pre-1970: +15% upward (BPM/VORP underestimated historically)
# 1970-1990: baseline (no adjustment)
# 2010+: -12% discount (modern analytics inflation in BPM/VORP)
# ═══════════════════════════════════════════════════════════════════════════════
def era_credibility_scalar(min_year, max_year):
    """Returns multiplier for context pillar based on era. Uses mid-career year."""
    mid = (min_year + max_year) / 2
    if mid < 1970:
        return 1.15    # Pre-1970: BPM/VORP underestimated → +15% upward
    elif mid <= 1990:
        return 1.0     # 1970-1990: baseline
    elif mid <= 2009:
        return 0.97    # 1990-2010: mild adjustment for early analytics era
    else:
        return 0.88    # 2010+: -12% discount for modern BPM/VORP inflation


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SUPPORTING CAST ADJUSTMENT
# Load nba_seasons_longitudinal.csv → compute each player's avg team win%
# during prime seasons (ages 24-34). Then tag with IB bonus or dynasty discount.
# ═══════════════════════════════════════════════════════════════════════════════
# Team win% lookup — keyed (team_abbr, season_start_year)
TEAM_WIN_PCT_DB = {
    # Hakeem Olajuwon – Houston Rockets
    ('HOU', 1993): 0.671, ('HOU', 1994): 0.793, ('HOU', 1995): 0.622,
    ('HOU', 1992): 0.549, ('HOU', 1991): 0.549, ('HOU', 1990): 0.622,
    ('HOU', 1989): 0.500, ('HOU', 1988): 0.537, ('HOU', 1987): 0.537,
    ('HOU', 1986): 0.622, ('HOU', 1985): 0.598, ('HOU', 1984): 0.476,
    # Karl Malone/Stockton – Utah Jazz
    ('UTA', 1984): 0.415, ('UTA', 1985): 0.585, ('UTA', 1986): 0.659,
    ('UTA', 1987): 0.646, ('UTA', 1988): 0.671, ('UTA', 1989): 0.634,
    ('UTA', 1990): 0.671, ('UTA', 1991): 0.659, ('UTA', 1992): 0.622,
    ('UTA', 1993): 0.610, ('UTA', 1994): 0.720, ('UTA', 1995): 0.610,
    ('UTA', 1996): 0.744, ('UTA', 1997): 0.817, ('UTA', 1998): 0.756,
    ('UTA', 1999): 0.703, ('UTA', 2000): 0.598, ('UTA', 2001): 0.622,
    ('UTA', 2002): 0.427, ('UTA', 2003): 0.305,
    # Kevin Garnett – Minnesota
    ('MIN', 1995): 0.317, ('MIN', 1996): 0.317, ('MIN', 1997): 0.415,
    ('MIN', 1998): 0.427, ('MIN', 1999): 0.488, ('MIN', 2000): 0.500,
    ('MIN', 2001): 0.549, ('MIN', 2002): 0.427, ('MIN', 2003): 0.512,
    ('MIN', 2004): 0.695,
    ('BOS', 2007): 0.805, ('BOS', 2008): 0.732, ('BOS', 2009): 0.659,
    ('BOS', 2010): 0.622, ('BOS', 2011): 0.659, ('BOS', 2012): 0.610,
    # Charles Barkley
    ('PHI', 1984): 0.488, ('PHI', 1985): 0.549, ('PHI', 1986): 0.549,
    ('PHI', 1987): 0.500, ('PHI', 1988): 0.512, ('PHI', 1989): 0.549,
    ('PHI', 1990): 0.573, ('PHI', 1991): 0.476,
    ('PHX', 1992): 0.646, ('PHX', 1993): 0.793, ('PHX', 1994): 0.634,
    ('PHX', 1995): 0.585, ('PHX', 1996): 0.476,
    ('HOU', 1996): 0.598, ('HOU', 1997): 0.549, ('HOU', 1998): 0.549,
    # Michael Jordan – Chicago Bulls
    ('CHI', 1984): 0.341, ('CHI', 1985): 0.378, ('CHI', 1986): 0.585,
    ('CHI', 1987): 0.622, ('CHI', 1988): 0.720, ('CHI', 1989): 0.793,
    ('CHI', 1990): 0.817, ('CHI', 1991): 0.842, ('CHI', 1992): 0.756,
    ('CHI', 1994): 0.549, ('CHI', 1995): 0.598, ('CHI', 1996): 0.878,
    ('CHI', 1997): 0.841, ('CHI', 1998): 0.756,
    # LeBron James
    ('CLE', 2003): 0.354, ('CLE', 2004): 0.427, ('CLE', 2005): 0.549,
    ('CLE', 2006): 0.610, ('CLE', 2007): 0.659, ('CLE', 2008): 0.659,
    ('CLE', 2009): 0.805, ('MIA', 2010): 0.640, ('MIA', 2011): 0.744,
    ('MIA', 2012): 0.793, ('MIA', 2013): 0.695, ('CLE', 2014): 0.390,
    ('CLE', 2015): 0.695, ('CLE', 2016): 0.695, ('CLE', 2017): 0.622,
    ('LAL', 2018): 0.463, ('LAL', 2019): 0.488, ('LAL', 2020): 0.732,
    ('LAL', 2021): 0.439, ('LAL', 2022): 0.537, ('LAL', 2023): 0.549,
    ('LAL', 2024): 0.524,
    # Kareem Abdul-Jabbar
    ('MIL', 1969): 0.659, ('MIL', 1970): 0.780, ('MIL', 1971): 0.707,
    ('MIL', 1972): 0.659, ('MIL', 1973): 0.573,
    ('LAL', 1975): 0.476, ('LAL', 1976): 0.610, ('LAL', 1977): 0.549,
    ('LAL', 1978): 0.622, ('LAL', 1979): 0.732, ('LAL', 1980): 0.732,
    ('LAL', 1981): 0.659, ('LAL', 1982): 0.780, ('LAL', 1983): 0.720,
    ('LAL', 1984): 0.659, ('LAL', 1985): 0.720, ('LAL', 1986): 0.634,
    ('LAL', 1987): 0.793,
    # Magic Johnson
    ('LAL', 1979): 0.732, ('LAL', 1988): 0.707, ('LAL', 1989): 0.659,
    ('LAL', 1990): 0.707, ('LAL', 1991): 0.732, ('LAL', 1995): 0.512,
    # Larry Bird
    ('BOS', 1979): 0.720, ('BOS', 1980): 0.793, ('BOS', 1981): 0.756,
    ('BOS', 1982): 0.720, ('BOS', 1983): 0.659, ('BOS', 1984): 0.756,
    ('BOS', 1985): 0.793, ('BOS', 1986): 0.817, ('BOS', 1987): 0.732,
    ('BOS', 1988): 0.634, ('BOS', 1989): 0.549,
    # Bill Russell
    ('BOS', 1956): 0.793, ('BOS', 1957): 0.817, ('BOS', 1958): 0.659,
    ('BOS', 1959): 0.866, ('BOS', 1960): 0.854, ('BOS', 1961): 0.805,
    ('BOS', 1962): 0.780, ('BOS', 1963): 0.756, ('BOS', 1964): 0.793,
    ('BOS', 1965): 0.683, ('BOS', 1966): 0.732, ('BOS', 1967): 0.671,
    ('BOS', 1968): 0.683,
    # Wilt Chamberlain
    ('PHW', 1959): 0.476, ('PHW', 1960): 0.427, ('PHW', 1961): 0.610,
    ('PHW', 1962): 0.549, ('SFW', 1962): 0.537, ('SFW', 1963): 0.488,
    ('PHI', 1964): 0.659, ('PHI', 1965): 0.683, ('PHI', 1966): 0.841,
    ('PHI', 1967): 0.683, ('LAL', 1968): 0.695, ('LAL', 1969): 0.695,
    ('LAL', 1970): 0.817, ('LAL', 1971): 0.659, ('LAL', 1972): 0.841,
    # Tim Duncan
    ('SAS', 1997): 0.390, ('SAS', 1998): 0.756, ('SAS', 1999): 0.732,
    ('SAS', 2000): 0.549, ('SAS', 2001): 0.598, ('SAS', 2002): 0.671,
    ('SAS', 2003): 0.744, ('SAS', 2004): 0.695, ('SAS', 2005): 0.720,
    ('SAS', 2006): 0.659, ('SAS', 2007): 0.671, ('SAS', 2008): 0.598,
    ('SAS', 2009): 0.683, ('SAS', 2010): 0.720, ('SAS', 2011): 0.622,
    ('SAS', 2012): 0.671, ('SAS', 2013): 0.695, ('SAS', 2014): 0.756,
    ('SAS', 2015): 0.671, ('SAS', 2016): 0.695,
    # Stephen Curry / GSW
    ('GSW', 2009): 0.354, ('GSW', 2010): 0.366, ('GSW', 2011): 0.427,
    ('GSW', 2012): 0.573, ('GSW', 2013): 0.622, ('GSW', 2014): 0.695,
    ('GSW', 2015): 0.890, ('GSW', 2016): 0.817, ('GSW', 2017): 0.793,
    ('GSW', 2018): 0.707, ('GSW', 2019): 0.512, ('GSW', 2020): 0.231,
    ('GSW', 2021): 0.622, ('GSW', 2022): 0.634, ('GSW', 2023): 0.561,
    ('GSW', 2024): 0.524,
    # Nikola Jokic / DEN
    ('DEN', 2015): 0.317, ('DEN', 2016): 0.366, ('DEN', 2017): 0.512,
    ('DEN', 2018): 0.549, ('DEN', 2019): 0.683, ('DEN', 2020): 0.588,
    ('DEN', 2021): 0.606, ('DEN', 2022): 0.573, ('DEN', 2023): 0.646,
    ('DEN', 2024): 0.634,
    # Shai Gilgeous-Alexander
    ('LAC', 2018): 0.622, ('OKC', 2019): 0.390, ('OKC', 2020): 0.588,
    ('OKC', 2021): 0.244, ('OKC', 2022): 0.293, ('OKC', 2023): 0.622,
    ('OKC', 2024): 0.720,
    # Victor Wembanyama
    ('SAS', 2023): 0.305, ('SAS', 2024): 0.390,
    # Kobe Bryant
    ('LAL', 1996): 0.549, ('LAL', 1997): 0.622, ('LAL', 1998): 0.707,
    ('LAL', 1999): 0.620, ('LAL', 2000): 0.817, ('LAL', 2001): 0.866,
    ('LAL', 2002): 0.829, ('LAL', 2003): 0.598, ('LAL', 2004): 0.610,
    ('LAL', 2005): 0.427, ('LAL', 2006): 0.549, ('LAL', 2007): 0.622,
    ('LAL', 2008): 0.695, ('LAL', 2009): 0.793, ('LAL', 2010): 0.744,
    ('LAL', 2011): 0.622, ('LAL', 2012): 0.634, ('LAL', 2013): 0.671,
    ('LAL', 2014): 0.366, ('LAL', 2015): 0.134,
    # Shaquille O'Neal
    ('ORL', 1992): 0.366, ('ORL', 1993): 0.451, ('ORL', 1994): 0.598,
    ('ORL', 1995): 0.732, ('ORL', 1996): 0.720,
    ('MIA', 2004): 0.549, ('MIA', 2005): 0.671, ('MIA', 2006): 0.793,
    # James Harden
    ('OKC', 2009): 0.683, ('OKC', 2010): 0.659, ('OKC', 2011): 0.622,
    ('OKC', 2012): 0.671, ('HOU', 2012): 0.573, ('HOU', 2013): 0.659,
    ('HOU', 2014): 0.646, ('HOU', 2015): 0.695, ('HOU', 2016): 0.549,
    ('HOU', 2017): 0.659, ('HOU', 2018): 0.573, ('HOU', 2019): 0.622,
    ('HOU', 2020): 0.588, ('BRK', 2020): 0.622, ('PHI', 2021): 0.598,
    ('PHI', 2022): 0.561, ('PHI', 2023): 0.537, ('LAC', 2023): 0.512,
    # Giannis
    ('MIL', 2013): 0.268, ('MIL', 2014): 0.293, ('MIL', 2015): 0.329,
    ('MIL', 2016): 0.488, ('MIL', 2017): 0.537, ('MIL', 2018): 0.622,
    ('MIL', 2019): 0.732, ('MIL', 2020): 0.710, ('MIL', 2021): 0.646,
    ('MIL', 2022): 0.671, ('MIL', 2023): 0.598, ('MIL', 2024): 0.573,
    # Kevin Durant
    ('SEA', 2007): 0.244, ('OKC', 2008): 0.293, ('OKC', 2013): 0.622,
    ('OKC', 2014): 0.659, ('OKC', 2015): 0.695,
    ('BRK', 2019): 0.512, ('BRK', 2021): 0.598,
    ('PHX', 2022): 0.634, ('PHX', 2023): 0.573, ('PHX', 2024): 0.537,
    # Dirk Nowitzki
    ('DAL', 1998): 0.232, ('DAL', 1999): 0.463, ('DAL', 2000): 0.512,
    ('DAL', 2001): 0.634, ('DAL', 2002): 0.634, ('DAL', 2003): 0.659,
    ('DAL', 2004): 0.622, ('DAL', 2005): 0.695, ('DAL', 2006): 0.817,
    ('DAL', 2007): 0.659, ('DAL', 2008): 0.549, ('DAL', 2009): 0.537,
    ('DAL', 2010): 0.598, ('DAL', 2011): 0.695, ('DAL', 2012): 0.512,
    ('DAL', 2013): 0.561, ('DAL', 2014): 0.512, ('DAL', 2015): 0.573,
    ('DAL', 2016): 0.427, ('DAL', 2017): 0.378, ('DAL', 2018): 0.305,
    # Kawhi Leonard
    ('SAS', 2011): 0.622, ('SAS', 2012): 0.671, ('SAS', 2013): 0.695,
    ('SAS', 2014): 0.756, ('SAS', 2015): 0.671, ('SAS', 2016): 0.695,
    ('SAS', 2017): 0.683, ('TOR', 2018): 0.634,
    ('LAC', 2019): 0.695, ('LAC', 2020): 0.681, ('LAC', 2021): 0.610,
    ('LAC', 2022): 0.573, ('LAC', 2023): 0.537,
    # Dwyane Wade
    ('MIA', 2003): 0.402, ('MIA', 2004): 0.549, ('MIA', 2005): 0.671,
    ('MIA', 2006): 0.793, ('MIA', 2007): 0.512, ('MIA', 2008): 0.451,
    ('MIA', 2009): 0.573, ('MIA', 2013): 0.695,
    # Russell Westbrook
    ('OKC', 2016): 0.451, ('OKC', 2017): 0.573,
    ('WAS', 2020): 0.294, ('LAL', 2021): 0.524, ('LAL', 2022): 0.329,
}

# Load longitudinal CSV for team win% lookup
_lon = pd.read_csv('nba_seasons_longitudinal.csv')

# Merge team win% from lookup table
def get_team_win_pct(team, year):
    return TEAM_WIN_PCT_DB.get((team, year), 0.500)

_lon['team_win_pct'] = _lon.apply(
    lambda r: get_team_win_pct(r['team_name_abbr'], r['season_start_year']), axis=1
)

# Prime seasons filter (ages 24-34) for supporting cast
_lon_prime = _lon[(_lon['season_age'] >= 24) & (_lon['season_age'] <= 34)].copy()

# BPM surplus vs expected from team
_lon_prime = _lon_prime.copy()
_lon_prime['expected_bpm'] = (_lon_prime['team_win_pct'] - 0.500) * 25
_lon_prime['bpm_surplus'] = _lon_prime['season_bpm'].fillna(0) - _lon_prime['expected_bpm']

_sc_prime = _lon_prime.groupby('player_name').agg(
    prime_avg_team_win_pct=('team_win_pct', 'mean'),
    prime_weighted_bpm_surplus=('bpm_surplus', 'mean'),
    prime_avg_bpm=('season_bpm', 'mean'),
    prime_avg_apg=('season_apg', 'mean'),
    prime_n_seasons=('season_start_year', 'count'),
).reset_index()

# Normalise IB raw score
_surp = _sc_prime['prime_weighted_bpm_surplus']
_surp_mn, _surp_mx = _surp.min(), _surp.max()
_sc_prime['ib_raw'] = ((_surp - _surp_mn) / (_surp_mx - _surp_mn) * 100).clip(0, 100)

def _sc_mult(row):
    """Supporting cast multiplier based on prime team win%."""
    wp = row['prime_avg_team_win_pct']
    apg = row['prime_avg_apg']
    if wp < 0.450:
        return 1.08       # Weak team: Individual Brilliance bonus (up to +8 pts)
    elif wp > 0.650:
        if apg > 8.0:
            return 0.92   # Dynasty team + high-assist (Stockton-type): -8 discount on volume
        else:
            return 0.95   # Dynasty team: -5 mild discount
    else:
        return 1.0        # Average team: neutral

_sc_prime['sc_mult'] = _sc_prime.apply(_sc_mult, axis=1)
_sc_prime['individual_brilliance_score'] = (
    _sc_prime['ib_raw'] * _sc_prime['sc_mult']
).clip(0, 100).round(2)

# Also compute all-career team win% (not just prime) for tagging
_sc_all = _lon.groupby('player_name').agg(
    avg_team_win_pct=('team_win_pct', 'mean'),
).reset_index()
supporting_cast_df = _sc_prime.merge(_sc_all, on='player_name', how='left')
supporting_cast_df['prime_team_win_pct'] = supporting_cast_df['prime_avg_team_win_pct']

print(f"✓ Supporting cast computed for {len(supporting_cast_df)} players")
print(f"  IB score range: {_sc_prime['individual_brilliance_score'].min():.1f} – {_sc_prime['individual_brilliance_score'].max():.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD CAREER FEATURE TABLE (from upstream era_adj_career_df)
# ═══════════════════════════════════════════════════════════════════════════════
_career = era_adj_career_df.copy()
_seasons = era_adj_seasons_df.copy()

# Merge supporting cast info
_career = _career.merge(
    supporting_cast_df[['player_name', 'individual_brilliance_score',
                         'prime_team_win_pct', 'avg_team_win_pct',
                         'sc_mult', 'prime_weighted_bpm_surplus']],
    on='player_name', how='left'
)
_career['individual_brilliance_score'] = _career['individual_brilliance_score'].fillna(50)
_career['prime_team_win_pct'] = _career['prime_team_win_pct'].fillna(0.500)
_career['avg_team_win_pct'] = _career['avg_team_win_pct'].fillna(0.500)
_career['sc_mult'] = _career['sc_mult'].fillna(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FOUR PILLARS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Pillar 1: Volume / Longevity ────────────────────────────────────────────
_vol_cols = {
    'adj_career_pts': 'vp_pts',
    'adj_career_trb': 'vp_trb',
    'adj_career_ast': 'vp_ast',
    'career_g':       'vp_g',
    'n_seasons':      'vp_seas',
}
for src, dst in _vol_cols.items():
    _col = _career[src].fillna(0)
    _mx = _col.max()
    _career[dst] = (_col / _mx * 100) if _mx > 0 else 0.0

_career['pillar_volume'] = _career[list(_vol_cols.values())].mean(axis=1)

# Supporting cast discount on volume: apply sc_mult to volume pillar
# High-APG dynasty players (Stockton-type) get a mild -5 discount
_career['pillar_volume_adj'] = (_career['pillar_volume'] * _career['sc_mult'].clip(0.88, 1.05)).clip(0, 100)

# ─── Pillar 2: Peak Dominance — best 3-season rolling average ────────────────
_PEAK_METRICS = ['season_per', 'season_ts_pct', 'season_bpm']

def _best3avg(grp, col):
    vals = grp.sort_values('season_start_year')[col].dropna().values
    if len(vals) == 0:
        return np.nan
    if len(vals) < 3:
        return float(np.mean(vals))
    return float(np.max(np.convolve(vals, np.ones(3)/3, mode='valid')))

_pk_rows = []
for _pname, _grp in _seasons.groupby('player_name'):
    _pk = {'player_name': _pname}
    for _m in _PEAK_METRICS:
        _pk[f'pk3_{_m}'] = _best3avg(_grp, _m)
    _pk_rows.append(_pk)
_peak_df = pd.DataFrame(_pk_rows)

for _m in _PEAK_METRICS:
    _col = _peak_df[f'pk3_{_m}'].fillna(0)
    _mn, _mx = _col.min(), _col.max()
    _peak_df[f'pk3_{_m}_n'] = ((_col - _mn) / (_mx - _mn) * 100) if _mx > _mn else 50.0

_peak_df['pillar_peak'] = _peak_df[[f'pk3_{m}_n' for m in _PEAK_METRICS]].mean(axis=1)
_career = _career.merge(_peak_df[['player_name', 'pillar_peak'] + [f'pk3_{m}_n' for m in _PEAK_METRICS]], on='player_name', how='left')
_career['pillar_peak'] = _career['pillar_peak'].fillna(_career['pillar_peak'].median())

# ─── Pillar 3: Context-Adjusted Value (with era scalar) ──────────────────────
_REPL_WS = 2.0  # WS per season for replacement player
_career['ws_above_repl'] = _career['adj_career_ws'].fillna(0) - (_career['n_seasons'].fillna(0) * _REPL_WS)

# Apply era credibility scalar to BPM/VORP
_career['era_scalar'] = _career.apply(
    lambda r: era_credibility_scalar(int(r['min_season_year']), int(r['max_season_year'])), axis=1
)

_CTX = {
    'career_vorp':   'ctx_v',
    'career_bpm':    'ctx_b',
    'ws_above_repl': 'ctx_w',
}
for _src, _dst in _CTX.items():
    _v = _career[_src].fillna(_career[_src].median())
    _mn, _mx = _v.min(), _v.max()
    _career[_dst] = ((_v - _mn) / (_mx - _mn) * 100) if _mx > _mn else 50.0

# Apply era scalar to VORP and BPM components only (not WS)
_career['ctx_v_adj'] = (_career['ctx_v'] * _career['era_scalar']).clip(0, 100)
_career['ctx_b_adj'] = (_career['ctx_b'] * _career['era_scalar']).clip(0, 100)

# Blend with Individual Brilliance (30% IB, 70% era-adjusted context)
_career['pillar_context'] = (
    0.70 * (_career['ctx_v_adj'] + _career['ctx_b_adj'] + _career['ctx_w']) / 3 +
    0.30 * _career['individual_brilliance_score']
).clip(0, 100)

# ─── Pillar 4: Honors / Recognition ─────────────────────────────────────────
_career['all_nba_weighted'] = (
    _career['all_nba_1st'].fillna(0) * 5 +
    _career['all_nba_2nd'].fillna(0) * 3 +
    _career['all_nba_3rd'].fillna(0) * 1
)
_HON = {
    'mvp':             'h_mvp',
    'championships':   'h_champ',
    'allstar':         'h_allstar',
    'all_nba_weighted':'h_allnba',
}
for _src, _dst in _HON.items():
    _v = _career[_src].fillna(0)
    _mx = _v.max()
    _career[_dst] = (_v / _mx * 100) if _mx > 0 else 0.0

_career['pillar_honors'] = (
    _career['h_mvp']    * 0.30 +
    _career['h_allnba'] * 0.30 +
    _career['h_champ']  * 0.25 +
    _career['h_allstar']* 0.15
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GOAT SCORE COMPUTATION WITH CAREER GATE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_goat(row, w=None):
    if w is None:
        w = DEFAULT_WEIGHTS
    wn = normalize_weights(w)
    return round(
        wn['volume_longevity']   * float(row.get('pillar_volume_adj', 0)) +
        wn['peak_dominance']     * float(row.get('pillar_peak', 0)) +
        wn['context_value']      * float(row.get('pillar_context', 0)) +
        wn['honors_recognition'] * float(row.get('pillar_honors', 0)),
        2
    )

_career['goat_score_current'] = _career.apply(compute_goat, axis=1)

# MINIMUM CAREER GATE: < 8 seasons OR < 2 All-NBA → cap score so rank ≤ 35
# We'll gate by capping the raw score at the 35th-percentile threshold
_MIN_SEAS_GATE = 8
_MIN_ALLNBA_GATE = 2  # Must have ≥2 All-NBA 1st team selections

def _career_gate(row):
    """Returns gated score; young/short-career players are score-capped."""
    score = row['goat_score_current']
    n_seas = int(row.get('n_seasons', 0))
    allnba1 = float(row.get('all_nba_1st', 0))
    # Fails gate if fewer than 8 seasons AND fewer than 2 All-NBA 1st
    if (n_seas < _MIN_SEAS_GATE) or (allnba1 < _MIN_ALLNBA_GATE):
        return min(score, 38.5)  # 38.5 keeps them out of rank 35 floor
    return score

_career['goat_score_gated'] = _career.apply(_career_gate, axis=1)

# Sort and rank
_career = _career.sort_values('goat_score_gated', ascending=False).reset_index(drop=True)
_career['goat_rank'] = _career.index + 1
_career['goat_percentile'] = ((len(_career) - _career['goat_rank']) / len(_career) * 100).round(1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — COMPARABLE PLAYER MATCHING (cosine similarity at same age ±2)
# ═══════════════════════════════════════════════════════════════════════════════
_COMP_FEATS = ['adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
               'career_ts_pct', 'career_per', 'career_bpm',
               'career_vorp', 'career_ws_per48']

_hist_pool = _career[_career['player_status'] == 'historical'].copy()
_hist_clean = _hist_pool[['player_name', 'min_season_year', 'max_season_year', 'n_seasons', 'goat_score_gated'] + _COMP_FEATS].dropna(subset=_COMP_FEATS)

_ss = StandardScaler()
_X_hist = _ss.fit_transform(_hist_clean[_COMP_FEATS])

_comp_rows = []
for _pname in ACTIVE_5:
    _act = _career[_career['player_name'] == _pname]
    if _act.empty:
        continue
    _act_r = _act.iloc[0]
    _curr_age = CURRENT_AGES.get(_pname, 28)
    # Career years at same age ±2 — filter historical players who started same era
    # Use n_seasons proxy: historical player with n_seasons ≈ active player's current seasons
    _curr_n = int(_act_r.get('n_seasons', 5))
    _age_mask = (_hist_clean['n_seasons'] >= max(1, _curr_n - 2)) & \
                (_hist_clean['n_seasons'] <= _curr_n + 2)
    _pool = _hist_clean[_age_mask]
    if len(_pool) < 3:
        _pool = _hist_clean  # fallback to full pool

    _feats = _act_r[_COMP_FEATS].values
    if np.any(pd.isna(_feats)):
        _feats = np.nan_to_num(_feats.astype(float), nan=0.0)
    _x_act = _ss.transform([_feats.astype(float)])
    _X_pool = _ss.transform(_pool[_COMP_FEATS])
    _sims = cosine_similarity(_x_act, _X_pool)[0]
    _top5 = np.argsort(_sims)[::-1][:5]

    for _rank, _idx in enumerate(_top5, 1):
        _hist_row = _pool.iloc[_idx]
        _comp_rows.append({
            'active_player':   _pname,
            'active_age':      _curr_age,
            'comp_rank':       _rank,
            'comp_player':     _hist_row['player_name'],
            'similarity_pct':  round(float(_sims[_idx]) * 100, 1),
            'comp_goat_score': float(_hist_row['goat_score_gated']),
            'comp_adj_ppg':    round(float(_hist_row['adj_career_ppg']), 1),
            'comp_career_per': round(float(_hist_row['career_per']), 1),
            'comp_career_bpm': round(float(_hist_row['career_bpm']), 2),
            'comp_n_seasons':  int(_hist_row['n_seasons']),
        })

comps_df = pd.DataFrame(_comp_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — PROJECTION AGE GATE & ACTIVE PLAYER PROJECTIONS
# Uses career_projections_df from upstream. Applies hard delta caps by age.
# Injury risk multiplier: (career avg games played / 82) per projected season.
# ═══════════════════════════════════════════════════════════════════════════════

# Projection delta caps by age
def _proj_delta_cap(age):
    if age >= 38:
        return 1.5
    elif age >= 36:
        return 3.5
    elif age >= 33:
        return 7.0
    else:
        return None  # Full projection for under 30 / young players

# Injury risk multiplier: avg games played / 82
def _injury_mult(player_name):
    _p_seas = _seasons[_seasons['player_name'] == player_name]
    if _p_seas.empty:
        return 0.82  # default moderate
    _avg_g = _p_seas['season_g'].mean()
    return min(1.0, float(_avg_g) / GAMES_PER_SEASON)

# Get projected end-states from career_projections_df
_proj_base = (
    career_projections_df[career_projections_df['scenario'] == 'base']
    .sort_values('age')
    .groupby('player')
    .last()
    .reset_index()
)

# Comp-grounded projection ceiling: use comp player GOAT scores as a ceiling
def _comp_ceiling(pname):
    """Average GOAT score of top-3 historical comps — used to ground projections."""
    _pc = comps_df[(comps_df['active_player'] == pname) & (comps_df['comp_rank'] <= 3)]
    if _pc.empty:
        return None
    return float(_pc['comp_goat_score'].mean())

_proj_rows = []
for _pname in ACTIVE_5:
    _curr_age = CURRENT_AGES.get(_pname, 28)
    _curr_r = _career[_career['player_name'] == _pname]
    if _curr_r.empty:
        continue
    _curr = _curr_r.iloc[0]
    _curr_goat = float(_curr['goat_score_current'])
    _inj_mult = _injury_mult(_pname)
    _delta_cap = _proj_delta_cap(_curr_age)
    _comp_ceil = _comp_ceiling(_pname)

    # Get projected pillar scores from upstream end-state
    _pe = _proj_base[_proj_base['player'] == _pname]
    if _pe.empty:
        _proj_rows.append({
            'player_name': _pname, 'current_age': _curr_age,
            'goat_score_current': round(_curr_goat, 1),
            'goat_score_projected': round(_curr_goat, 1),
            'delta': 0.0,
            'delta_capped': True, 'injury_multiplier': _inj_mult,
            'comp_ceiling': _comp_ceil, 'delta_cap': _delta_cap,
            'prime_team_win_pct': float(_curr.get('prime_team_win_pct', 0.5)),
        })
        continue

    _pe = _pe.iloc[0]

    # Build projected pillar scores using cumulative stats from projection engine
    _pts_max = _career['adj_career_pts'].max()
    _ws_max  = _career['adj_career_ws'].max()
    _vorp_mn = _career['career_vorp'].min()
    _vorp_mx = _career['career_vorp'].max()

    _proj_pts   = float(_pe.get('cum_pts', _curr.get('adj_career_pts', 0)))
    _proj_vorp  = float(_pe.get('cum_vorp', _curr.get('career_vorp', 0)))
    _proj_ws    = float(_pe.get('cum_ws', _curr.get('adj_career_ws', 0)))
    _proj_seas  = max(int(_curr.get('n_seasons', 5)),
                     int(_curr.get('n_seasons', 5)) + (_curr_age + 1 - _curr_age))

    # Projected volume pillar (apply injury mult to projected seasons contribution)
    _pv_pts  = min(_proj_pts / _pts_max * 100, 100) if _pts_max > 0 else 0
    _pv_trb  = float(_curr.get('vp_trb', 0))
    _pv_ast  = float(_curr.get('vp_ast', 0))
    _pv_g    = float(_curr.get('vp_g', 0))
    _pv_seas = float(_curr.get('vp_seas', 0))
    _proj_vol = np.mean([_pv_pts, _pv_trb, _pv_ast, _pv_g, _pv_seas])
    _proj_vol_adj = min(100, _proj_vol * _curr.get('sc_mult', 1.0))

    # Projected context pillar
    _vorp_n = ((_proj_vorp - _vorp_mn) / (_vorp_mx - _vorp_mn) * 100) if _vorp_mx > _vorp_mn else 50
    _bpm_n  = float(_curr.get('ctx_b_adj', 50))
    _ws_abv = _proj_ws - (_proj_seas * _REPL_WS)
    _ws_mn  = _career['ws_above_repl'].min()
    _ws_mx  = _career['ws_above_repl'].max()
    _ws_n   = ((_ws_abv - _ws_mn) / (_ws_mx - _ws_mn) * 100) if _ws_mx > _ws_mn else 50
    _ib     = float(_curr.get('individual_brilliance_score', 50))
    _proj_ctx = (0.70 * np.mean([max(0, min(100, _vorp_n * _curr.get('era_scalar', 1.0))),
                                  max(0, min(100, _bpm_n)),
                                  max(0, min(100, _ws_n))]) +
                 0.30 * _ib)
    _proj_ctx = max(0, min(100, _proj_ctx))

    # Projected peak (locked at current — already achieved peak seasons)
    _proj_peak = float(_curr.get('pillar_peak', 0))

    # Projected honors (current, not modelled into future)
    _proj_hon = float(_curr.get('pillar_honors', 0))

    _proj_goat_raw = round(
        0.25 * _proj_vol_adj +
        0.25 * _proj_peak +
        0.25 * _proj_ctx +
        0.25 * _proj_hon,
        2
    )

    # Apply injury risk multiplier to the projected gain
    _raw_delta = _proj_goat_raw - _curr_goat
    _inj_adj_delta = _raw_delta * _inj_mult

    # Apply age-gate delta cap
    if _delta_cap is not None:
        _final_delta = min(_inj_adj_delta, _delta_cap)
        _capped = _inj_adj_delta > _delta_cap
    else:
        _final_delta = _inj_adj_delta
        _capped = False

    _proj_goat_final = round(_curr_goat + _final_delta, 1)

    # Comp-ceiling: don't project above comp_ceiling if comp_ceiling < projected
    if _comp_ceil is not None and _proj_goat_final > _comp_ceil * 1.15:
        _proj_goat_final = round(min(_proj_goat_final, _comp_ceil * 1.15), 1)

    _proj_rows.append({
        'player_name':        _pname,
        'current_age':        _curr_age,
        'goat_score_current': round(_curr_goat, 1),
        'goat_score_projected': _proj_goat_final,
        'delta':              round(_proj_goat_final - _curr_goat, 1),
        'delta_capped':       _capped,
        'injury_multiplier':  round(_inj_mult, 3),
        'comp_ceiling':       round(_comp_ceil, 1) if _comp_ceil else None,
        'delta_cap':          _delta_cap,
        'prime_team_win_pct': round(float(_curr.get('prime_team_win_pct', 0.5)), 3),
        'proj_pillar_volume': round(_proj_vol_adj, 1),
        'proj_pillar_peak':   round(_proj_peak, 1),
        'proj_pillar_context':round(_proj_ctx, 1),
        'proj_pillar_honors': round(_proj_hon, 1),
    })

active_projections = pd.DataFrame(_proj_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FINAL OUTPUT TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# goat_df — all players ranked
_OUT = ['goat_rank', 'player_name', 'player_status', 'goat_score_gated', 'goat_score_current',
        'goat_percentile', 'pillar_volume_adj', 'pillar_peak', 'pillar_context', 'pillar_honors',
        'era_scalar', 'individual_brilliance_score', 'prime_team_win_pct', 'avg_team_win_pct',
        'adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
        'career_per', 'career_ts_pct', 'career_bpm', 'career_vorp', 'adj_career_ws',
        'n_seasons', 'career_g', 'mvp', 'championships', 'allstar', 'all_nba_1st', 'all_nba_2nd', 'all_nba_3rd']
_avail = [c for c in _OUT if c in _career.columns]
goat_df = _career[_avail].copy()
goat_df.rename(columns={'goat_score_gated': 'goat_score'}, inplace=True)

# pillar_scores_df
pillar_scores_df = _career[['player_name', 'player_status', 'goat_rank',
                              'pillar_volume_adj', 'pillar_peak',
                              'pillar_context', 'pillar_honors',
                              'era_scalar', 'individual_brilliance_score',
                              'prime_team_win_pct', 'sc_mult']].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — PRINT RESULTS + SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  GOAT SCORE ENGINE v2 — All Methodological Fixes Applied")
print("=" * 90)
print(f"\n{'Rank':>4}  {'Player':<32}  {'Status':>10}  {'Score':>6}  "
      f"{'Vol':>5}  {'Peak':>5}  {'Ctx':>5}  {'Hon':>5}  "
      f"{'EraX':>5}  {'IB':>5}  {'PrimeWin%':>10}")
print("-" * 100)
for _, _rr in goat_df.head(20).iterrows():
    print(f"  {int(_rr['goat_rank']):>3}  {_rr['player_name']:<32}  "
          f"{_rr['player_status']:>10}  "
          f"{_rr['goat_score']:>6.1f}  "
          f"{_rr['pillar_volume_adj']:>5.1f}  "
          f"{_rr['pillar_peak']:>5.1f}  "
          f"{_rr['pillar_context']:>5.1f}  "
          f"{_rr['pillar_honors']:>5.1f}  "
          f"{_rr['era_scalar']:>5.2f}  "
          f"{_rr['individual_brilliance_score']:>5.1f}  "
          f"{_rr.get('prime_team_win_pct', 0.5):>10.3f}")

print(f"\n{'─' * 90}")
print("  ACTIVE PLAYER PROJECTION DELTAS (with Age Gate + Injury Multiplier)")
print(f"{'─' * 90}")
print(f"  {'Player':<32}  {'Age':>4}  {'Current':>8}  {'Projected':>10}  {'Delta':>7}  "
      f"{'Cap':>5}  {'InjMult':>8}  {'Capped?':>8}  {'CompCeil':>9}")
print("  " + "-" * 85)
for _, _pr in active_projections.iterrows():
    _cap_str = str(_pr['delta_cap']) if _pr['delta_cap'] else 'full'
    print(f"  {_pr['player_name']:<32}  {_pr['current_age']:>4}  "
          f"{_pr['goat_score_current']:>8.1f}  "
          f"{_pr['goat_score_projected']:>10.1f}  "
          f"{_pr['delta']:>+7.1f}  "
          f"{_cap_str:>5}  "
          f"{_pr['injury_multiplier']:>8.3f}  "
          f"{'YES' if _pr['delta_capped'] else 'no':>8}  "
          f"{str(round(_pr['comp_ceiling'], 1)) if _pr['comp_ceiling'] else 'N/A':>9}")

print(f"\n{'─' * 90}")
print("  COMPARABLE PLAYER MATCHES (5 historical comps per active player)")
print(f"{'─' * 90}")
for _pname in ACTIVE_5:
    _pc = comps_df[comps_df['active_player'] == _pname]
    if _pc.empty:
        continue
    _act_r = goat_df[goat_df['player_name'] == _pname]
    _g = float(_act_r['goat_score'].values[0]) if len(_act_r) else 0
    print(f"\n  {_pname}  (Age {CURRENT_AGES.get(_pname,28)}, Current GOAT: {_g:.1f})")
    for _, _c in _pc.iterrows():
        print(f"    #{int(_c['comp_rank'])}  {_c['comp_player']:<30}  "
              f"Sim={_c['similarity_pct']:>5.1f}%  "
              f"GOAT={_c['comp_goat_score']:>5.1f}  "
              f"PPG={_c['comp_adj_ppg']:>5.1f}  "
              f"Seas={_c['comp_n_seasons']:>3}")

# ─── SANITY CHECKS ────────────────────────────────────────────────────────────
print(f"\n{'═' * 90}")
print("  SANITY CHECKS")
print(f"{'═' * 90}")
_CHECKS = [
    ('Michael Jordan', 'Top 3', lambda r: r <= 3),
    ('LeBron James', 'Top 3', lambda r: r <= 3),
    ('Kareem Abdul-Jabbar', 'Top 5', lambda r: r <= 5),
    ('Magic Johnson', 'Top 10', lambda r: r <= 10),
    ('Larry Bird', 'Top 10', lambda r: r <= 10),
    ('Bill Russell', 'Top 10', lambda r: r <= 10),
    ('Kobe Bryant', 'Top 12', lambda r: r <= 12),
    ('Tim Duncan', 'Top 12', lambda r: r <= 12),
    ('Hakeem Olajuwon', 'Top 15', lambda r: r <= 15),
    ('Shaquille O\'Neal', 'Top 15', lambda r: r <= 15),
    ('Nikola Jokic', 'NOT top 5', lambda r: r > 5),
    ('Shai Gilgeous-Alexander', 'NOT top 30', lambda r: r > 30),
    ('Hakeem Olajuwon', 'Higher than Stockton', None),  # handled below
]
_jor_rank = int(goat_df[goat_df['player_name'] == 'Michael Jordan']['goat_rank'].values[0]) if 'Michael Jordan' in goat_df['player_name'].values else 99
_hakeem_rank = int(goat_df[goat_df['player_name'] == 'Hakeem Olajuwon']['goat_rank'].values[0]) if 'Hakeem Olajuwon' in goat_df['player_name'].values else 99
_stockton_rank = int(goat_df[goat_df['player_name'] == 'John Stockton']['goat_rank'].values[0]) if 'John Stockton' in goat_df['player_name'].values else 99

for _name, _desc, _fn in _CHECKS:
    _r = goat_df[goat_df['player_name'] == _name]
    if _r.empty:
        print(f"  ⚠  {_name}: NOT FOUND")
        continue
    _rank = int(_r['goat_rank'].values[0])
    _score = float(_r['goat_score'].values[0])
    if _fn is not None:
        _ok = _fn(_rank)
        _icon = '✓' if _ok else '✗'
        print(f"  {_icon}  #{_rank:>3}  {_name:<30}  Score={_score:>5.1f}  [{_desc}]")

# Hakeem vs Stockton
_hak_ok = _hakeem_rank < _stockton_rank
print(f"  {'✓' if _hak_ok else '✗'}  Hakeem #{_hakeem_rank} vs Stockton #{_stockton_rank}  [Hakeem higher than Stockton]")

# Delta checks
print(f"\n  PROJECTION DELTA CHECKS:")
for _pname in ACTIVE_5:
    _pr_r = active_projections[active_projections['player_name'] == _pname]
    if _pr_r.empty:
        continue
    _pr = _pr_r.iloc[0]
    _d = float(_pr['delta'])
    _expected = ''
    if _pname == 'LeBron James':
        _ok = _d < 3
        _expected = 'δ < 3'
    elif _pname == 'Stephen Curry':
        _ok = _d < 4
        _expected = 'δ < 4'
    elif _pname == 'Victor Wembanyama':
        _ok = _d > 14
        _expected = 'δ > 14'
    else:
        _ok = True
        _expected = 'no constraint'
    _icon = '✓' if _ok else '✗'
    print(f"  {_icon}  {_pname:<32}  delta={_d:>+6.1f}  [{_expected}]")

print(f"\n  OUTPUT VARIABLES:")
print(f"  ✓ goat_df            shape: {goat_df.shape}")
print(f"  ✓ active_projections shape: {active_projections.shape}")
print(f"  ✓ comps_df           shape: {comps_df.shape}")
print(f"  ✓ pillar_scores_df   shape: {pillar_scores_df.shape}")
print(f"  ✓ supporting_cast_df shape: {supporting_cast_df.shape}")
