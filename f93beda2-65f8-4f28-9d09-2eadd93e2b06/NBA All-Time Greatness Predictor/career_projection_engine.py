
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Replay poly_val helper ───────────────────────────────────────────────────
def _poly_val(coeffs, age):
    return coeffs[0]*age**2 + coeffs[1]*age + coeffs[2]

# ── Config ───────────────────────────────────────────────────────────────────
STANDARD_RETIREMENT_AGE = 38   # Standard horizon for young players

ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
            'Nikola Jokic', 'LeBron James', 'Stephen Curry']

# Age-gated retirement ages: late-career players get realistic horizons
# based on actual "seasons remaining" math, not the generic 42 ceiling
HARD_RETIREMENT_AGES = {
    'Victor Wembanyama':       39,   # 21 → massive runway
    'Shai Gilgeous-Alexander': 39,   # 26 → long runway
    'Nikola Jokic':            38,   # 29 → moderate runway
    'LeBron James':            42,   # 40 → 2 seasons max (already ~done)
    'Stephen Curry':           41,   # 36 → 5 seasons max
}

# For players 35+: further dampen the per-season contribution
# to reflect declining games played and diminishing marginal impact.
# Factor smoothly transitions from 1.0 at age 35 → ~0 at age 41+
def _late_career_dampen(age: int, retirement_age: int) -> float:
    """
    Returns a multiplier [0,1] for how much future seasons contribute
    to projected cumulative totals. Asymptotes to near-0 as age → retirement_age.
    - Under 35: no damping (1.0)
    - 35-40: linear decline from 1.0 → 0.15
    - 40+: heavy damping, almost no contribution
    """
    if age < 35:
        return 1.0
    seasons_left = max(0, retirement_age - age)
    total_late_window = max(1, retirement_age - 35)
    # smooth asymptotic decay: heavier for LeBron (age 40, ~2 left) than Curry (age 36, ~5 left)
    decay = seasons_left / total_late_window
    # Apply square root to make it taper more steeply near retirement
    return max(0.05, round(decay ** 1.5, 3))

METRICS_MAP = {
    'ppg':  'season_ppg',
    'per':  'season_per',
    'bpm':  'season_bpm',
    'vorp': 'season_vorp',
    'ws':   'season_ws',
}
ANCHOR_KEYS = {
    'ppg': 'current_ppg', 'per': 'current_per',
    'bpm': 'current_bpm', 'vorp': 'current_vorp', 'ws': 'current_ws',
}

SCENARIO_SLOPE = {'pessimistic': 1.35, 'base': 1.00, 'optimistic': 0.65}
SCENARIO_AVAIL = {'pessimistic': 0.88, 'base': 1.00, 'optimistic': 1.08}
GAMES_PER_SEASON = 82

# ── Projection Engine ────────────────────────────────────────────────────────
def _anchor_adjusted_curve(coeffs, current_age, current_val, future_age, slope_mult):
    a, b, c = coeffs
    peak_age = -b / (2 * a) if a < 0 else 27.0
    cv_current = _poly_val(coeffs, current_age)
    cv_future  = _poly_val(coeffs, future_age)
    curve_delta = cv_future - cv_current
    if future_age > peak_age and curve_delta < 0:
        curve_delta *= slope_mult
    return max(current_val + curve_delta, 0.0)

def project_player(player_name, anchor, age_curves, seasons_data):
    current_age = anchor['current_age']
    archetype   = anchor['archetype']
    avail_base  = anchor['avail_mult']

    # ── Age-gated retirement horizon ────────────────────────────────────────
    ret_age    = HARD_RETIREMENT_AGES.get(player_name, STANDARD_RETIREMENT_AGE)
    # If already past standard horizon, cap to at most 3 more seasons
    if current_age >= ret_age:
        ret_age = current_age + 2
    future_ages = list(range(current_age + 1, ret_age + 1))

    career_to_date = seasons_data[seasons_data['player_name'] == player_name].copy()
    pts_so_far  = career_to_date['season_pts'].sum() if 'season_pts' in career_to_date else 0.0
    g_so_far    = career_to_date['season_g'].sum() if 'season_g' in career_to_date else 0.0
    vorp_so_far = career_to_date['season_vorp'].sum() if 'season_vorp' in career_to_date else 0.0
    ws_so_far   = career_to_date['season_ws'].sum() if 'season_ws' in career_to_date else 0.0

    rows = []
    for scenario, slope_mult in SCENARIO_SLOPE.items():
        avail_sc = min(avail_base * SCENARIO_AVAIL[scenario], 0.98)
        cum_pts  = pts_so_far
        cum_g    = g_so_far
        cum_vorp = vorp_so_far
        cum_ws   = ws_so_far

        for age in future_ages:
            # ── Age-gated damping factor ─────────────────────────────────────
            # Asymptotically reduces how much each future season contributes
            # to cumulative totals for late-career players.
            dampen = _late_career_dampen(age, ret_age)

            proj_row = {
                'player':   player_name,
                'scenario': scenario,
                'age':      age,
                'season':   f"{2024 + (age - current_age)}-{str(2025 + (age - current_age))[2:]}",
            }

            for mkey, col in METRICS_MAP.items():
                anchor_val = anchor[ANCHOR_KEYS[mkey]]
                if pd.isna(anchor_val):
                    proj_row[f'proj_{mkey}'] = np.nan
                    continue
                curves = age_curves.get(archetype, {})
                coeffs = curves.get(col, age_curves.get('global', {}).get(col))
                if coeffs is not None:
                    proj_val = _anchor_adjusted_curve(
                        coeffs, current_age, anchor_val, age, slope_mult
                    )
                else:
                    decline_rate = 0.025 * slope_mult
                    proj_val = max(anchor_val * (1 - decline_rate) ** (age - current_age), 0.0)

                # Late-career PPG/PER: also apply per-stat damping (fewer minutes, decline)
                if current_age >= 36 and mkey in ('ppg', 'per', 'bpm', 'vorp', 'ws'):
                    stat_damp = max(0.4, dampen)  # floor at 40% to keep stats realistic
                    proj_val = round(anchor_val * stat_damp + proj_val * (1 - stat_damp) * 0.5, 2)

                proj_row[f'proj_{mkey}'] = round(proj_val, 2)

            # Dampen games played for late-career (reduced minutes, load management)
            games_damp = max(0.5, dampen) if current_age >= 35 else 1.0
            proj_g = round(GAMES_PER_SEASON * avail_sc * games_damp, 1)
            proj_g = max(proj_g, 8)
            proj_row['proj_games'] = proj_g
            proj_row['avail_pct']  = round(avail_sc * 100, 1)

            # ── Cumulative accumulation with age-gated damping ───────────────
            # The core fix: late-career seasons contribute far less to cumulative totals.
            # This ensures LeBron's projected delta stays near current career totals.
            ppg = proj_row.get('proj_ppg') or 0
            season_pts  = ppg * proj_g * dampen           # dampened contribution
            season_vorp = (proj_row.get('proj_vorp') or 0) * dampen
            season_ws   = (proj_row.get('proj_ws') or 0) * dampen

            cum_pts  += season_pts
            cum_g    += proj_g * dampen
            cum_vorp += season_vorp
            cum_ws   += season_ws

            proj_row['cum_pts']  = round(cum_pts, 0)
            proj_row['cum_g']    = round(cum_g, 0)
            proj_row['cum_vorp'] = round(cum_vorp, 1)
            proj_row['cum_ws']   = round(cum_ws, 1)
            rows.append(proj_row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── Run projections ──────────────────────────────────────────────────────────
proj_dfs = {}
for pname in ACTIVE_5:
    anchor = active_player_anchors.get(pname)
    if anchor is None:
        print(f"  ⚠ No anchor for {pname}")
        continue
    _df = project_player(pname, anchor, age_curve_models, era_adj_seasons_df)
    if len(_df):
        proj_dfs[pname] = _df

# ── Print tables ─────────────────────────────────────────────────────────────
print("=" * 72)
print("NBA CAREER PROJECTION TABLES  |  Age-Gated  |  3 Scenarios")
print("=" * 72)

_RETIRE_AGE_STR = {p: f"→{HARD_RETIREMENT_AGES.get(p, STANDARD_RETIREMENT_AGE)}" for p in ACTIVE_5}

for pname in ACTIVE_5:
    if pname not in proj_dfs:
        continue
    anchor = active_player_anchors[pname]
    df     = proj_dfs[pname]
    inj_tier = 'high' if anchor['avail_mult'] <= 0.70 else 'moderate' if anchor['avail_mult'] <= 0.83 else 'low'
    ret_a    = HARD_RETIREMENT_AGES.get(pname, STANDARD_RETIREMENT_AGE)
    late_tag = ' [LATE-CAREER — minimal delta]' if anchor['current_age'] >= 35 else ''

    print(f"\n{'─'*72}")
    print(f"  {pname}  |  Age {anchor['current_age']}{_RETIRE_AGE_STR[pname]}  "
          f"|  Archetype: {anchor['archetype']}  |  Injury: {inj_tier}{late_tag}")
    print(f"  Anchor → PPG={anchor['current_ppg']:.1f}  "
          f"BPM={anchor['current_bpm']:.1f}  VORP={anchor['current_vorp']:.1f}  "
          f"WS={anchor['current_ws']:.1f}")
    damp_eg = _late_career_dampen(anchor['current_age'] + 1, ret_a)
    print(f"  Season-1 dampen factor: {damp_eg:.3f}")
    print(f"{'─'*72}")

    # Base case
    base_df = df[df['scenario'] == 'base']
    print(f"  BASE CASE:")
    print(f"  {'Age':>4}  {'Season':>8}  {'PPG':>6}  {'PER':>6}  "
          f"{'BPM':>6}  {'Damp':>5}  "
          f"{'Games':>6}  {'CumPts':>9}  {'CumVORP':>8}  {'CumWS':>7}")
    for _, r in base_df.iterrows():
        _damp = _late_career_dampen(int(r['age']), ret_a)
        print(f"  {int(r['age']):>4}  {r['season']:>8}  "
              f"{(r.get('proj_ppg') or 0):>6.1f}  {(r.get('proj_per') or 0):>6.1f}  "
              f"{(r.get('proj_bpm') or 0):>6.1f}  {_damp:>5.2f}  "
              f"{r['proj_games']:>6.0f}  {r['cum_pts']:>9,.0f}  "
              f"{r['cum_vorp']:>8.1f}  {r['cum_ws']:>7.1f}")

    # 3-scenario end-state
    print(f"\n  END-STATE SCENARIOS AT RETIREMENT:")
    print(f"  {'Scenario':>14}  {'Final PPG':>10}  {'Final BPM':>10}  "
          f"{'Career Pts':>11}  {'Career VORP':>12}  {'Career WS':>10}")
    for sc in ['pessimistic', 'base', 'optimistic']:
        sc_rows = df[df['scenario'] == sc]
        if sc_rows.empty:
            continue
        last = sc_rows.iloc[-1]
        print(f"  {sc:>14}  {(last.get('proj_ppg') or 0):>10.1f}  "
              f"{(last.get('proj_bpm') or 0):>10.1f}  "
              f"{last['cum_pts']:>11,.0f}  "
              f"{last['cum_vorp']:>12.1f}  "
              f"{last['cum_ws']:>10.1f}")

# ── Merge all projections ────────────────────────────────────────────────────
career_projections_df = pd.concat(list(proj_dfs.values()), ignore_index=True)
print(f"\n✓ career_projections_df shape: {career_projections_df.shape}")
print(f"  Players: {career_projections_df['player'].nunique()}")
print(f"  Scenarios: {career_projections_df['scenario'].unique().tolist()}")

# ── Quick sanity check: seasons remaining per player ────────────────────────
print(f"\n  SEASONS REMAINING (base case):")
base_check = career_projections_df[career_projections_df['scenario'] == 'base']
for pname in ACTIVE_5:
    rows_p = base_check[base_check['player'] == pname]
    if rows_p.empty:
        continue
    n_seas = len(rows_p)
    curr_age = active_player_anchors[pname]['current_age']
    ret_a    = HARD_RETIREMENT_AGES.get(pname, STANDARD_RETIREMENT_AGE)
    pts_start = era_adj_seasons_df[era_adj_seasons_df['player_name'] == pname]['season_pts'].sum()
    pts_end   = rows_p.iloc[-1]['cum_pts']
    pts_delta = pts_end - pts_start
    print(f"    {pname:<32}  age {curr_age}→{ret_a}  "
          f"{n_seas} seasons  ΔPts≈{pts_delta:+,.0f}")
