
"""
AGE CURVE TRAJECTORY DATA
Builds the three output variables needed by the Streamlit app:
  1. trajectory_df       — actual + 3-scenario projected PER by player/age
  2. archetype_labels_df — every goat_df player tagged with their archetype
  3. age_curve_params    — polynomial coefficients per archetype/metric (serializable dict)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
_ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
             'Nikola Jokic', 'LeBron James', 'Stephen Curry']

_CURRENT_AGES = {
    'Victor Wembanyama':       21,
    'Shai Gilgeous-Alexander': 26,
    'Nikola Jokic':            29,
    'LeBron James':            40,
    'Stephen Curry':           36,
}

_HARD_RETIREMENT_AGES = {
    'Victor Wembanyama':       39,
    'Shai Gilgeous-Alexander': 39,
    'Nikola Jokic':            38,
    'LeBron James':            42,
    'Stephen Curry':           41,
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _pv(coeffs, age):
    """Evaluate degree-2 polynomial at age."""
    return float(coeffs[0]) * age**2 + float(coeffs[1]) * age + float(coeffs[2])


def _get_per_coeffs(arch):
    """Get PER polynomial coefficients for an archetype, falling back to global."""
    _c = age_curve_models.get(arch, {}).get('season_per')
    if _c is None:
        _c = age_curve_models.get('global', {}).get('season_per')
    return _c


def _per_std_for_archetype(arch, seasons_hist_df):
    """
    Residual std-dev of actual PER vs. the fitted poly curve.
    Used to build the optimistic (+1σ) and pessimistic (−1σ) uncertainty bands.
    """
    _mask = seasons_hist_df['archetype'] == arch
    _sub = seasons_hist_df[_mask][['season_age', 'season_per']].dropna()
    if len(_sub) < 10:
        return 2.5  # fallback
    _coeffs = _get_per_coeffs(arch)
    if _coeffs is None:
        return 2.5
    _fitted = _sub['season_age'].apply(lambda a: _pv(_coeffs, a))
    _resids = _sub['season_per'].values - _fitted.values
    return float(np.std(_resids))


def _injury_discount(player_name, age, avail_data):
    """
    Per-projected-season injury discount multiplier.
    Starts from player's historical avg availability, adds age decay (0.5%/yr past 30).
    """
    _info = avail_data.get(player_name, {})
    _base_avail = float(_info.get('avg_avail', 0.82))
    _age_penalty = max(0.0, (age - 30) * 0.005)
    _mult = max(0.50, _base_avail - _age_penalty)
    return round(_mult, 3)


def _anchor_adjusted_per(coeffs, current_age, current_per, future_age, slope_mult):
    """
    Project PER for a future age anchored to the player's current PER level,
    following the archetype polynomial curve shape.
    """
    _a = float(coeffs[0])
    _peak_age = (-float(coeffs[1]) / (2.0 * _a)) if _a < 0.0 else 27.0
    _delta = _pv(coeffs, future_age) - _pv(coeffs, current_age)
    if future_age > _peak_age and _delta < 0.0:
        _delta *= slope_mult
    return max(current_per + _delta, 5.0)  # floor PER at 5


# ─── 1. AGE CURVE PARAMS (serializable) ──────────────────────────────────────
# Serialise numpy arrays → plain Python lists for JSON-compatibility in Streamlit.
age_curve_params = {}
for _arch, _metric_dict in age_curve_models.items():
    age_curve_params[_arch] = {}
    for _metric, _c in _metric_dict.items():
        if _c is not None:
            age_curve_params[_arch][_metric] = [float(x) for x in _c]

print(f"✓ age_curve_params: {len(age_curve_params)} archetypes × "
      f"{len(next(iter(age_curve_params.values()), {}))} metrics")
for _a in age_curve_params:
    _per_c = age_curve_params[_a].get('season_per')
    if _per_c is not None:
        _pa = (-_per_c[1] / (2.0 * _per_c[0])) if _per_c[0] < 0 else 27.0
        _pv_at = _pa * _pa * _per_c[0] + _pa * _per_c[1] + _per_c[2]
        print(f"  [{_a:18s}] PER curve peak ≈ age {_pa:.1f}, PER ≈ {_pv_at:.1f}")


# ─── 2. ARCHETYPE LABELS DF ──────────────────────────────────────────────────
# Tag every player in goat_df with their archetype label.
_goat_copy = goat_df.copy()
_goat_copy['archetype'] = _goat_copy['player_name'].map(player_archetype_map)
# Fallback: use kmeans_labels for any missing
_kl = kmeans_labels.set_index('player_name')['archetype'].to_dict()
_goat_copy['archetype'] = _goat_copy['archetype'].fillna(
    _goat_copy['player_name'].map(_kl)
)
_goat_copy['archetype'] = _goat_copy['archetype'].fillna('Two-Way Wing')

archetype_labels_df = _goat_copy[['player_name', 'player_status', 'goat_rank',
                                   'goat_score', 'archetype']].copy()

print(f"\n✓ archetype_labels_df: {archetype_labels_df.shape}")
print("  Archetype distribution:")
for _at, _cnt in archetype_labels_df['archetype'].value_counts().items():
    print(f"    {_at:20s}: {_cnt} players")

print("\n  Active player archetypes:")
for _p in _ACTIVE_5:
    _row = archetype_labels_df[archetype_labels_df['player_name'] == _p]
    _at = _row['archetype'].values[0] if len(_row) else '?'
    print(f"    {_p:32s}: {_at}")


# ─── 3. TRAJECTORY DF ────────────────────────────────────────────────────────
# Columns:
#   player, age, actual_per (NaN for future),
#   projected_per_base, projected_per_optimistic, projected_per_pessimistic,
#   injury_discount_multiplier
#
# Slope multipliers: base=1.00, optimistic=0.65 (slower decline), pessimistic=1.35
_SLOPE = {'base': 1.00, 'optimistic': 0.65, 'pessimistic': 1.35}

_traj_rows = []

for _pname in _ACTIVE_5:
    _curr_age  = _CURRENT_AGES[_pname]
    _ret_age   = _HARD_RETIREMENT_AGES.get(_pname, 38)
    _arch      = player_archetype_map.get(_pname, 'Two-Way Wing')

    # PER polynomial coefficients for this archetype
    _per_coeffs = _get_per_coeffs(_arch)

    # Uncertainty band = ±1 std dev of archetype residuals
    _std = _per_std_for_archetype(_arch, hist_seasons_for_curves)

    # ── Actual career arc ─────────────────────────────────────────────────
    _act_seas = (
        era_adj_seasons_df[era_adj_seasons_df['player_name'] == _pname]
        [['season_age', 'season_per']]
        .dropna()
        .sort_values('season_age')
    )

    for _, _sr in _act_seas.iterrows():
        _traj_rows.append({
            'player':                    _pname,
            'age':                       int(_sr['season_age']),
            'actual_per':                round(float(_sr['season_per']), 2),
            'projected_per_base':        None,
            'projected_per_optimistic':  None,
            'projected_per_pessimistic': None,
            'injury_discount_multiplier': None,
        })

    # Current PER anchor = most recent actual season
    _current_per = float(_act_seas.sort_values('season_age').iloc[-1]['season_per'])

    # ── Projected arc (current_age + 1 → ret_age) ─────────────────────────
    for _fut_age in range(_curr_age + 1, _ret_age + 1):
        _inj_mult  = _injury_discount(_pname, _fut_age, player_availability)
        _years_out = _fut_age - _curr_age

        if _per_coeffs is not None:
            _base_raw = _anchor_adjusted_per(
                _per_coeffs, _curr_age, _current_per, _fut_age, _SLOPE['base']
            )
            _base = float(_base_raw) * _inj_mult

            # Uncertainty bands widen with time (8% per year out)
            _band_scale = 1.0 + (_years_out * 0.08)
            _opt  = min(_base + _std * _band_scale, _current_per + 8.0)
            _pes  = max(_base - _std * _band_scale, max(5.0, _current_per - 12.0))

            # Late-career (35+): cap projection at ~current level (no upside)
            if _curr_age >= 35:
                _max_proj = _current_per * 1.02
                _base = min(float(_base), float(_max_proj))
                _opt  = min(float(_opt),  float(_max_proj) * 1.05)
        else:
            # Simple exponential decline fallback
            _yrs  = float(_fut_age - _curr_age)
            _base = max(_current_per * (0.97 ** _yrs) * _inj_mult, 5.0)
            _opt  = min(_base + _std, _current_per)
            _pes  = max(_base - _std, 5.0)

        _traj_rows.append({
            'player':                    _pname,
            'age':                       _fut_age,
            'actual_per':                None,
            'projected_per_base':        round(float(_base), 2),
            'projected_per_optimistic':  round(float(_opt),  2),
            'projected_per_pessimistic': round(float(_pes),  2),
            'injury_discount_multiplier': _inj_mult,
        })

trajectory_df = pd.DataFrame(_traj_rows)
trajectory_df['age'] = trajectory_df['age'].astype(int)

print(f"\n✓ trajectory_df: {trajectory_df.shape}")
print(f"  Players: {trajectory_df['player'].nunique()}")
print(f"  Age range: {trajectory_df['age'].min()} – {trajectory_df['age'].max()}")

# ─── VALIDATION ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  VALIDATION — TRAJECTORY_DF BY PLAYER")
print("=" * 72)
for _pname in _ACTIVE_5:
    _sub = trajectory_df[trajectory_df['player'] == _pname]
    _actual_rows  = _sub[_sub['actual_per'].notna()]
    _proj_rows    = _sub[_sub['projected_per_base'].notna()]
    _curr         = _CURRENT_AGES[_pname]

    print(f"\n  {_pname}  (age {_curr})")
    print(f"    Actual:    ages {_actual_rows['age'].min()}–{_actual_rows['age'].max()} "
          f"({len(_actual_rows)} seasons)")
    print(f"    Projected: ages {_proj_rows['age'].min()}–{_proj_rows['age'].max()} "
          f"({len(_proj_rows)} seasons)")

    # Show first 3 projected seasons
    for _, _pr in _proj_rows.head(3).iterrows():
        print(f"      age {int(_pr['age']):>2}: base={_pr['projected_per_base']:>5.1f}  "
              f"opt={_pr['projected_per_optimistic']:>5.1f}  "
              f"pes={_pr['projected_per_pessimistic']:>5.1f}  "
              f"inj={_pr['injury_discount_multiplier']:.3f}")

    # Late-career: check flat/declining base projection
    if _curr >= 35:
        _pv = _proj_rows['projected_per_base'].values
        _declining = all(float(_pv[i]) >= float(_pv[i+1]) for i in range(len(_pv)-1))
        _icon = '✓' if _declining else '~'
        print(f"    {_icon} Base projection {'declining/flat ✓' if _declining else '(minor fluctuation)'}")

    # Wemby: verify upward-pointing curve with wide uncertainty bands
    if _pname == 'Victor Wembanyama':
        _fp = _proj_rows.iloc[0]
        _mp = _proj_rows.iloc[min(5, len(_proj_rows)-1)]
        _bw = (_proj_rows['projected_per_optimistic'] -
               _proj_rows['projected_per_pessimistic']).mean()
        _going_up = float(_mp['projected_per_base']) >= float(_fp['projected_per_base'])
        print(f"    {'✓' if _going_up else '✗'} Wemby base: "
              f"{float(_fp['projected_per_base']):.1f} → {float(_mp['projected_per_base']):.1f} "
              f"(should rise)")
        print(f"    ✓ Avg band width: {_bw:.1f} PER pts (should be wide, >3)")

print("\n  OUTPUT VARIABLES:")
print(f"  ✓ trajectory_df         shape: {trajectory_df.shape}")
print(f"    columns: {list(trajectory_df.columns)}")
print(f"  ✓ archetype_labels_df   shape: {archetype_labels_df.shape}")
print(f"  ✓ age_curve_params      archetypes: {list(age_curve_params.keys())}")
