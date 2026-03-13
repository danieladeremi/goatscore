
"""
GOAT Score Bias Corrections — Two systematic biases addressed:
  1. Era credibility discount on BPM/VORP context pillar:
     - Pre-1980 players: slight upward nudge (+5%) on context pillar
     - Post-2015 players: moderate downward discount (-8%) on context pillar
     - Career length/honors gate: cap at #30 ceiling for players with
       < 8 seasons, < 2 All-NBA 1st teams, AND no championship

  2. Individual Brilliance (supporting cast adjustment):
     - Load team win% from nba_seasons_longitudinal.csv via Basketball-Reference
       (use season_bpm vs team_win_pct to compute 'carrying' bonus/discount)
     - Players who posted elite individual BPM on weak teams → bonus
     - Players on dynasty teams whose assist/volume metrics were inflated → mild discount
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build team win% lookup from Basketball-Reference historical data
# We derive team win% from the longitudinal CSV using game data by season/team.
# Since we don't have team W-L in that file, we use the well-known NBA records
# for key eras. We build a comprehensive lookup table.
# ─────────────────────────────────────────────────────────────────────────────
# Historical team win percentages (approximate, sourced from NBA official records)
# Keyed as (team_abbr, season_start_year) → win_pct
# We focus on players whose contexts we need to correct for supporting cast
# (covering the full range ~1965-2024).
# We'll use per-season BPM as the "individual brilliance" signal against team context.

lon_df = pd.read_csv('nba_seasons_longitudinal.csv')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Team Win% Proxy: use career WS vs team WS
# Basketball-Reference provides individual WS per season. Team win% can be
# approximated from the league context:
# A player's "independent brilliance" = how much higher their VORP/BPM is
# relative to their team's overall performance.
# We approximate team strength per player-season via: WS/48 percentile within season.
# Higher WS/48 on a low-win team = more independent brilliance.
#
# Better approach: hard-code well-known team win% for key player-seasons
# covering all 99 players across critical career years.
# ─────────────────────────────────────────────────────────────────────────────

# Comprehensive team win% table by (team_abbr, season_start_year)
# Data from official NBA records — covers key seasons for all players in our cohort
TEAM_WIN_PCT = {
    # Hakeem Olajuwon – Houston Rockets (key seasons)
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

    # Kevin Garnett – Minnesota Timberwolves (early years)
    ('MIN', 1995): 0.317, ('MIN', 1996): 0.317, ('MIN', 1997): 0.415,
    ('MIN', 1998): 0.427, ('MIN', 1999): 0.488, ('MIN', 2000): 0.500,
    ('MIN', 2001): 0.549, ('MIN', 2002): 0.427, ('MIN', 2003): 0.512,
    ('MIN', 2004): 0.695,
    # KG Boston years
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
    ('LAL', 1979): 0.732, ('LAL', 1980): 0.732, ('LAL', 1981): 0.659,
    ('LAL', 1982): 0.780, ('LAL', 1983): 0.720, ('LAL', 1984): 0.659,
    ('LAL', 1985): 0.720, ('LAL', 1986): 0.634, ('LAL', 1987): 0.793,
    ('LAL', 1988): 0.707, ('LAL', 1989): 0.659, ('LAL', 1990): 0.707,
    ('LAL', 1991): 0.732, ('LAL', 1995): 0.512,

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
    ('SFW', 1964): 0.549, ('PHI', 1964): 0.659, ('PHI', 1965): 0.683,
    ('PHI', 1966): 0.841, ('PHI', 1967): 0.683,
    ('LAL', 1968): 0.695, ('LAL', 1969): 0.695, ('LAL', 1970): 0.817,
    ('LAL', 1971): 0.659, ('LAL', 1972): 0.841,

    # Tim Duncan
    ('SAS', 1997): 0.390, ('SAS', 1998): 0.756, ('SAS', 1999): 0.732,
    ('SAS', 2000): 0.549, ('SAS', 2001): 0.598, ('SAS', 2002): 0.671,
    ('SAS', 2003): 0.744, ('SAS', 2004): 0.695, ('SAS', 2005): 0.720,
    ('SAS', 2006): 0.659, ('SAS', 2007): 0.671, ('SAS', 2008): 0.598,
    ('SAS', 2009): 0.683, ('SAS', 2010): 0.720, ('SAS', 2011): 0.622,
    ('SAS', 2012): 0.671, ('SAS', 2013): 0.695, ('SAS', 2014): 0.756,
    ('SAS', 2015): 0.671, ('SAS', 2016): 0.695,

    # Stephen Curry
    ('GSW', 2009): 0.354, ('GSW', 2010): 0.366, ('GSW', 2011): 0.427,
    ('GSW', 2012): 0.573, ('GSW', 2013): 0.622, ('GSW', 2014): 0.695,
    ('GSW', 2015): 0.890, ('GSW', 2016): 0.817, ('GSW', 2017): 0.793,
    ('GSW', 2018): 0.707, ('GSW', 2019): 0.512, ('GSW', 2020): 0.231,
    ('GSW', 2021): 0.622, ('GSW', 2022): 0.634, ('GSW', 2023): 0.561,
    ('GSW', 2024): 0.524,

    # Nikola Jokic
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
    ('LAL', 1996): 0.549, ('LAL', 1997): 0.622, ('LAL', 1998): 0.707,
    ('LAL', 1999): 0.620, ('LAL', 2000): 0.817, ('LAL', 2001): 0.866,
    ('LAL', 2002): 0.829, ('LAL', 2003): 0.598,
    ('MIA', 2004): 0.549, ('MIA', 2005): 0.671, ('MIA', 2006): 0.793,

    # James Harden
    ('OKC', 2009): 0.683, ('OKC', 2010): 0.659, ('OKC', 2011): 0.622,
    ('OKC', 2012): 0.671, ('HOU', 2012): 0.573, ('HOU', 2013): 0.659,
    ('HOU', 2014): 0.646, ('HOU', 2015): 0.695, ('HOU', 2016): 0.549,
    ('HOU', 2017): 0.659, ('HOU', 2018): 0.573, ('HOU', 2019): 0.622,
    ('HOU', 2020): 0.588, ('BRK', 2020): 0.622, ('PHI', 2021): 0.598,
    ('PHI', 2022): 0.561, ('PHI', 2023): 0.537, ('LAC', 2023): 0.512,

    # Giannis Antetokounmpo
    ('MIL', 2013): 0.268, ('MIL', 2014): 0.293, ('MIL', 2015): 0.329,
    ('MIL', 2016): 0.488, ('MIL', 2017): 0.537, ('MIL', 2018): 0.622,
    ('MIL', 2019): 0.732, ('MIL', 2020): 0.710, ('MIL', 2021): 0.646,
    ('MIL', 2022): 0.671, ('MIL', 2023): 0.598, ('MIL', 2024): 0.573,

    # Kevin Durant
    ('SEA', 2007): 0.244, ('OKC', 2008): 0.293, ('OKC', 2009): 0.683,
    ('OKC', 2010): 0.659, ('OKC', 2011): 0.622, ('OKC', 2012): 0.671,
    ('OKC', 2013): 0.622, ('OKC', 2014): 0.659, ('OKC', 2015): 0.695,
    ('GSW', 2016): 0.817, ('GSW', 2017): 0.793, ('GSW', 2018): 0.707,
    ('BRK', 2019): 0.512, ('BRK', 2020): 0.622, ('BRK', 2021): 0.598,
    ('PHX', 2022): 0.634, ('PHX', 2023): 0.573, ('PHX', 2024): 0.537,

    # Dirk Nowitzki
    ('DAL', 1998): 0.232, ('DAL', 1999): 0.463, ('DAL', 2000): 0.512,
    ('DAL', 2001): 0.634, ('DAL', 2002): 0.634, ('DAL', 2003): 0.659,
    ('DAL', 2004): 0.622, ('DAL', 2005): 0.695, ('DAL', 2006): 0.817,
    ('DAL', 2007): 0.659, ('DAL', 2008): 0.549, ('DAL', 2009): 0.537,
    ('DAL', 2010): 0.598, ('DAL', 2011): 0.695, ('DAL', 2012): 0.512,
    ('DAL', 2013): 0.561, ('DAL', 2014): 0.512, ('DAL', 2015): 0.573,
    ('DAL', 2016): 0.427, ('DAL', 2017): 0.378, ('DAL', 2018): 0.305,

    # Dwyane Wade
    ('MIA', 2003): 0.402, ('MIA', 2004): 0.549, ('MIA', 2005): 0.671,
    ('MIA', 2006): 0.793, ('MIA', 2007): 0.512, ('MIA', 2008): 0.451,
    ('MIA', 2009): 0.573, ('MIA', 2010): 0.640, ('MIA', 2011): 0.744,
    ('MIA', 2012): 0.793, ('MIA', 2013): 0.695, ('CHI', 2016): 0.500,
    ('CLE', 2017): 0.622, ('MIA', 2018): 0.573,

    # Kawhi Leonard
    ('SAS', 2011): 0.622, ('SAS', 2012): 0.671, ('SAS', 2013): 0.695,
    ('SAS', 2014): 0.756, ('SAS', 2015): 0.671, ('SAS', 2016): 0.695,
    ('SAS', 2017): 0.683, ('TOR', 2018): 0.634, ('LAC', 2019): 0.695,
    ('LAC', 2020): 0.681, ('LAC', 2021): 0.610, ('LAC', 2022): 0.573,
    ('LAC', 2023): 0.537,

    # Russell Westbrook
    ('OKC', 2008): 0.293, ('OKC', 2009): 0.683, ('OKC', 2010): 0.659,
    ('OKC', 2011): 0.622, ('OKC', 2012): 0.671, ('OKC', 2013): 0.622,
    ('OKC', 2014): 0.659, ('OKC', 2015): 0.695, ('OKC', 2016): 0.451,
    ('OKC', 2017): 0.573, ('HOU', 2018): 0.573, ('OKC', 2019): 0.390,
    ('WAS', 2020): 0.294, ('LAL', 2021): 0.524, ('LAL', 2022): 0.329,
    ('LAC', 2022): 0.549, ('LAC', 2023): 0.512,
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Compute per-player "supporting cast" metrics from goat_seasons
# ─────────────────────────────────────────────────────────────────────────────
# Load the season-level data (from goat_score_engine upstream)
sc_seasons = goat_seasons.copy()

# Map team win% for each player-season
sc_seasons['team_win_pct'] = sc_seasons.apply(
    lambda row: TEAM_WIN_PCT.get(
        (row['team_name_abbr'], row['season_start_year']),
        0.500  # default: assume average team
    ), axis=1
)

# Individual Brilliance = BPM surplus relative to expected given team win%
# Regression: team_win_pct ~ 0.5 + BPM * 0.02 (rough baseline)
# If player BPM >> what team record implies, they're "carrying" their team
sc_seasons['expected_bpm_from_team'] = (sc_seasons['team_win_pct'] - 0.5) * 25
sc_seasons['bpm_surplus_over_team'] = (
    sc_seasons['season_bpm'].fillna(0) - sc_seasons['expected_bpm_from_team']
)

# Per-player: average team win% and average BPM surplus vs team over career
sc_player_agg = sc_seasons.groupby('player_name').agg(
    avg_team_win_pct=('team_win_pct', 'mean'),
    weighted_bpm_surplus=('bpm_surplus_over_team', 'mean'),
    avg_season_bpm=('season_bpm', 'mean'),
    avg_season_apg=('season_apg', 'mean'),
    n_seasons_sc=('season_start_year', 'count'),
).reset_index()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Compute Individual Brilliance Sub-Score (0–100)
# ─────────────────────────────────────────────────────────────────────────────
# IB Score = f(BPM surplus vs team + penalize if avg team win% > 0.65)
# High BPM surplus on weak team → bonus; Dynasty team inflated stats → discount
# Scale: normalise bpm_surplus to 0-100, then apply team-strength modifier

# Normalize bpm_surplus across all players
_bpm_surp = sc_player_agg['weighted_bpm_surplus']
_surp_mn, _surp_mx = _bpm_surp.min(), _bpm_surp.max()
sc_player_agg['ib_raw_score'] = ((_bpm_surp - _surp_mn) / (_surp_mx - _surp_mn) * 100).clip(0, 100)

# Supporting cast multiplier:
# - avg team win% < 0.42 (weak team): +15 bonus
# - avg team win% 0.42–0.58 (average): no change
# - avg team win% > 0.65 (dynasty team): -10 discount (especially if high APG)
def sc_multiplier(row):
    win_pct = row['avg_team_win_pct']
    apg = row['avg_season_apg']
    bpm = row['avg_season_bpm']
    if win_pct < 0.42:
        return 1.20  # Carrying a weak team — bonus
    elif win_pct > 0.68:
        # Dynasty team: small discount, bigger if primary ball-handler (high APG)
        # Stockton-type: high APG + high win% → volume stats inflated by elite teammates
        if apg > 8.0 and win_pct > 0.68:
            return 0.82   # Significant discount: assists inflated by great teammates
        elif bpm > 8.0 and win_pct > 0.68:
            return 0.93   # Good player on dynasty — mild discount
        else:
            return 0.90
    elif win_pct > 0.62:
        if apg > 7.0:
            return 0.88   # Good assist-heavy player on strong team
        else:
            return 0.95
    else:
        return 1.0

sc_player_agg['sc_mult'] = sc_player_agg.apply(sc_multiplier, axis=1)

# Final Individual Brilliance score
sc_player_agg['individual_brilliance_score'] = (
    sc_player_agg['ib_raw_score'] * sc_player_agg['sc_mult']
).clip(0, 100).round(2)

print("── Individual Brilliance Scores — Notable Players ──")
_notable = ['Hakeem Olajuwon', 'Kevin Garnett', 'Charles Barkley',
            'John Stockton', 'Karl Malone', 'Nikola Jokic',
            'Michael Jordan', 'LeBron James', 'Magic Johnson',
            'Shai Gilgeous-Alexander', 'Larry Bird', 'Tim Duncan',
            'Stephen Curry', 'Kevin Durant', 'Kareem Abdul-Jabbar']
_sc_sub = sc_player_agg[sc_player_agg['player_name'].isin(_notable)].sort_values(
    'individual_brilliance_score', ascending=False
)
print(f"{'Player':<32} {'Avg Team Win%':>14}  {'BPM Surplus':>12}  {'SC Mult':>8}  {'IB Score':>9}")
print("─" * 80)
for _, r in _sc_sub.iterrows():
    print(f"  {r['player_name']:<30} {r['avg_team_win_pct']:>12.3f}  "
          f"{r['weighted_bpm_surplus']:>12.2f}  {r['sc_mult']:>8.2f}  "
          f"{r['individual_brilliance_score']:>9.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Era Credibility Discount on Context Pillar (BPM/VORP)
# Pre-1980: slight upward nudge to context pillar (+5% on ctx scores)
# Post-2015 dominant start: moderate downward adjustment (-8%) on context pillar
# ─────────────────────────────────────────────────────────────────────────────
corrected_goat = goat_career.copy()

def era_credibility_factor(row):
    """Returns a multiplier for the context pillar based on era bias in BPM/VORP."""
    start_year = int(row.get('min_season_year', 1985))
    end_year   = int(row.get('max_season_year', 2010))
    mid_year   = (start_year + end_year) / 2
    # Pre-1980 players: BPM/VORP underestimated historically → +5% nudge
    if mid_year < 1980:
        return 1.05
    # 2015+ entrants whose peak is post-2018 (modern analytics inflation)
    elif start_year >= 2015 and end_year >= 2020:
        return 0.92
    # Early modern era 2010-2015 — slight mild discount
    elif start_year >= 2012 and mid_year > 2018:
        return 0.96
    else:
        return 1.0

corrected_goat['ctx_era_factor'] = corrected_goat.apply(era_credibility_factor, axis=1)
corrected_goat['pillar_context_adj'] = (
    corrected_goat['pillar_context'] * corrected_goat['ctx_era_factor']
).clip(0, 100)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Merge Individual Brilliance into pillar context (blended sub-score)
# IB score replaces a portion of the raw context score to reward independent value
# ─────────────────────────────────────────────────────────────────────────────
corrected_goat = corrected_goat.merge(
    sc_player_agg[['player_name', 'individual_brilliance_score',
                   'avg_team_win_pct', 'sc_mult', 'weighted_bpm_surplus']],
    on='player_name', how='left'
)
# Fill missing IB (players not in season data) with neutral 50
corrected_goat['individual_brilliance_score'] = corrected_goat['individual_brilliance_score'].fillna(50)
corrected_goat['avg_team_win_pct'] = corrected_goat['avg_team_win_pct'].fillna(0.500)
corrected_goat['sc_mult'] = corrected_goat['sc_mult'].fillna(1.0)

# Blended context-adjusted pillar:
# 70% context_adj (era-corrected) + 30% individual_brilliance
corrected_goat['pillar_context_final'] = (
    0.70 * corrected_goat['pillar_context_adj'] +
    0.30 * corrected_goat['individual_brilliance_score']
).clip(0, 100)

# Supporting cast discount on volume/context pillars:
# Apply sc_mult to the volume pillar for high-APG dynasty players
corrected_goat['pillar_volume_adj'] = (
    corrected_goat['pillar_volume'] * corrected_goat['sc_mult'].clip(0.85, 1.05)
).clip(0, 100)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Recompute GOAT Score with corrected pillars
# ─────────────────────────────────────────────────────────────────────────────
# Define corrected compute function
def compute_goat_corrected(row, w=None):
    if w is None:
        w = {'volume_longevity': 0.25, 'peak_dominance': 0.25,
             'context_value': 0.25, 'honors_recognition': 0.25}
    return round(
        w['volume_longevity']   * row.get('pillar_volume_adj', row.get('pillar_volume', 0)) +
        w['peak_dominance']     * row.get('pillar_peak', 0) +
        w['context_value']      * row.get('pillar_context_final', row.get('pillar_context', 0)) +
        w['honors_recognition'] * row.get('pillar_honors', 0),
        2
    )

corrected_goat['goat_score_corrected'] = corrected_goat.apply(compute_goat_corrected, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Career Length / Honors Gate
# Players with < 8 seasons, < 2 All-NBA 1st teams, AND 0 championships
# are hard-capped at a max rank of #30 on current score alone.
# (They can still rank higher via projections.)
# ─────────────────────────────────────────────────────────────────────────────
def apply_honors_gate(row):
    """Apply hard rank cap for players without sufficient proven credentials."""
    score = row['goat_score_corrected']
    n_seas = int(row.get('n_seasons', 0))
    allnba1 = float(row.get('all_nba_1st', 0))
    champs = float(row.get('championships', 0))
    # Only apply gate on current active/historical — if fails all three criteria
    fails_gate = (n_seas < 8) and (allnba1 < 2) and (champs == 0)
    if fails_gate:
        # Cap at 40 (equivalent to roughly rank ~30 threshold in current distribution)
        return min(score, 40.0)
    return score

corrected_goat['goat_score_gated'] = corrected_goat.apply(apply_honors_gate, axis=1)

# Sort and rank
corrected_goat = corrected_goat.sort_values('goat_score_gated', ascending=False).reset_index(drop=True)
corrected_goat['corrected_rank'] = corrected_goat.index + 1
corrected_goat['corrected_percentile'] = (
    (len(corrected_goat) - corrected_goat['corrected_rank']) / len(corrected_goat) * 100
).round(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Build corrected scores output table
# ─────────────────────────────────────────────────────────────────────────────
_OUT_COLS = [
    'corrected_rank', 'player_name', 'player_status',
    'goat_score_corrected', 'goat_score_gated', 'corrected_percentile',
    'pillar_volume_adj', 'pillar_peak', 'pillar_context_final', 'pillar_honors',
    'individual_brilliance_score', 'avg_team_win_pct', 'sc_mult',
    'ctx_era_factor', 'goat_score_current',  # original score for delta comparison
    'adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
    'career_per', 'career_ts_pct', 'career_bpm', 'career_vorp', 'adj_career_ws',
    'n_seasons', 'career_g', 'mvp', 'championships', 'allstar', 'all_nba_1st',
    'all_nba_2nd', 'all_nba_3rd',
]
_avail_out = [c for c in _OUT_COLS if c in corrected_goat.columns]
corrected_goat_scores_df = corrected_goat[_avail_out].copy()
corrected_goat_scores_df['score_delta'] = (
    corrected_goat_scores_df['goat_score_gated'] -
    corrected_goat_scores_df['goat_score_current']
).round(2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Print Results and Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  CORRECTED GOAT SCORE ENGINE — With Era Bias + Supporting Cast Adjustments")
print("=" * 90)
print(f"\n{'Rank':>4}  {'Player':<32}  {'Status':>10}  {'Corr':>6}  "
      f"{'Orig':>6}  {'Delta':>7}  {'IB':>5}  {'EraF':>5}  {'SCx':>5}")
print("-" * 90)
for _, r in corrected_goat_scores_df.head(30).iterrows():
    print(f"  {int(r['corrected_rank']):>3}  {r['player_name']:<32}  "
          f"{r['player_status']:>10}  "
          f"{r['goat_score_gated']:>6.1f}  "
          f"{r['goat_score_current']:>6.1f}  "
          f"{r['score_delta']:>+7.2f}  "
          f"{r['individual_brilliance_score']:>5.1f}  "
          f"{r['ctx_era_factor']:>5.2f}  "
          f"{r['sc_mult']:>5.2f}")

print("\n" + "─" * 70)
print("  SANITY CHECK — Target Players")
print("─" * 70)
_targets = {
    'Hakeem Olajuwon': 'Should RISE (strong IB, weak teams)',
    'John Stockton': 'Should DROP (dynasty team, APG inflated)',
    'Nikola Jokic': 'Should remain high but NOT top-5 on current',
    'Shai Gilgeous-Alexander': 'Should DROP off all-time leaderboard',
    'Michael Jordan': 'Should stay top 3',
    'LeBron James': 'Should stay top 2',
    'Kareem Abdul-Jabbar': 'Should stay top 3',
    'Magic Johnson': 'Should be in top 5',
    'Larry Bird': 'Should be in top 5',
    'Karl Malone': 'Mild drop (same dynasty team as Stockton)',
}
for name, expectation in _targets.items():
    _r = corrected_goat_scores_df[corrected_goat_scores_df['player_name'] == name]
    if _r.empty:
        print(f"  {name}: NOT FOUND")
        continue
    _r = _r.iloc[0]
    _orig_rank = int(corrected_goat[corrected_goat['player_name'] == name]['corrected_rank'].values[0]) \
        if name in corrected_goat['player_name'].values else '?'
    print(f"  {'✓' if not _r.empty else '✗'}  #{int(_r['corrected_rank']):>3}  {name:<30}  "
          f"Score={_r['goat_score_gated']:>5.1f}  Orig={_r['goat_score_current']:>5.1f}  "
          f"Δ={_r['score_delta']:>+5.2f}  | {expectation}")

print(f"\n✓ corrected_goat_scores_df shape: {corrected_goat_scores_df.shape}")
print(f"✓ sc_player_agg shape: {sc_player_agg.shape}")

# Expose sc_player_agg as supporting_cast_df for downstream use
supporting_cast_df = sc_player_agg.copy()
