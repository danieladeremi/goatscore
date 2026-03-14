import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='NBA GOAT Analyzer', page_icon='NBA', layout='wide')

BASE_DIR = Path(__file__).resolve().parent
CAREER_PATH = BASE_DIR / 'nba_era_adjusted_career.csv'
SEASONS_PATH = BASE_DIR / 'nba_era_adjusted_seasons.csv'
GOAT_V4_PATH = BASE_DIR / 'outputs' / 'engine_v4' / 'goat_scores_v4.csv'
ACTIVE_PROJ_V4_PATH = BASE_DIR / 'outputs' / 'engine_v4' / 'active_projections_v4.csv'
COMPS_V4_PATH = BASE_DIR / 'outputs' / 'engine_v4' / 'comps_df_v4.csv'

HONORS_OVERRIDES = {
    'LeBron James': {'allstar': 22},
    'Kevin Durant': {'allstar': 16},
    'Stephen Curry': {'allstar': 12},
    'Nikola Jokic': {'allstar': 8},
    'Giannis Antetokounmpo': {'allstar': 10},
    'Shai Gilgeous-Alexander': {'allstar': 4, 'mvp': 1},
    'Victor Wembanyama': {'dpoy': 0},
}
RECORD_MAP = {
    'Points': ('career_pts', 'Cumulative Points'),
    'Rebounds': ('career_trb', 'Cumulative Rebounds'),
    'Assists': ('career_ast', 'Cumulative Assists'),
    'Steals': ('career_stl', 'Cumulative Steals'),
    'Blocks': ('career_blk', 'Cumulative Blocks'),
}


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except Exception:
        return default


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors='coerce')
    w = pd.to_numeric(weights, errors='coerce').fillna(0.0)
    m = v.notna()
    if not m.any():
        return np.nan
    v, w = v[m], w[m]
    if float(w.sum()) <= 0:
        return float(v.mean())
    return float(np.average(v, weights=w))


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


@st.cache_data(show_spinner=False)
def load_data():
    career = pd.read_csv(CAREER_PATH)
    seasons = pd.read_csv(SEASONS_PATH)
    for pname, patch in HONORS_OVERRIDES.items():
        m = career['player_name'].eq(pname)
        for col, val in patch.items():
            if col in career.columns:
                career.loc[m, col] = val

    v4 = pd.read_csv(GOAT_V4_PATH) if GOAT_V4_PATH.exists() else pd.DataFrame()
    active_proj_v4 = pd.read_csv(ACTIVE_PROJ_V4_PATH) if ACTIVE_PROJ_V4_PATH.exists() else pd.DataFrame()
    comps_v4 = pd.read_csv(COMPS_V4_PATH) if COMPS_V4_PATH.exists() else pd.DataFrame()
    return career, seasons, v4, active_proj_v4, comps_v4


def aggregate_player_seasons(seasons: pd.DataFrame) -> pd.DataFrame:
    s = seasons.copy()
    s['season_start_year'] = pd.to_numeric(s['season_start_year'], errors='coerce')
    s['season_g'] = pd.to_numeric(s['season_g'], errors='coerce')
    rows = []
    for (player, year), grp in s.groupby(['player_name', 'season_start_year'], dropna=True):
        g = pd.to_numeric(grp['season_g'], errors='coerce').fillna(0.0)
        rows.append({
            'player_name': player,
            'season_start_year': int(year),
            'season_age': _weighted_mean(grp.get('season_age'), g),
            'season_g': float(g.sum()),
            'season_pts': float(pd.to_numeric(grp.get('season_pts'), errors='coerce').fillna(0.0).sum()),
            'season_trb': float(pd.to_numeric(grp.get('season_trb'), errors='coerce').fillna(0.0).sum()),
            'season_ast': float(pd.to_numeric(grp.get('season_ast'), errors='coerce').fillna(0.0).sum()),
            'season_stl': float(pd.to_numeric(grp.get('season_stl'), errors='coerce').fillna(0.0).sum()),
            'season_blk': float(pd.to_numeric(grp.get('season_blk'), errors='coerce').fillna(0.0).sum()),
            'adj_season_ppg_blended': _weighted_mean(grp.get('adj_season_ppg_blended'), g),
            'season_per': _weighted_mean(grp.get('season_per'), g),
            'season_ts_pct': _weighted_mean(grp.get('season_ts_pct'), g),
            'season_bpm': _weighted_mean(grp.get('season_bpm'), g),
            'adj_season_ws': float(pd.to_numeric(grp.get('adj_season_ws'), errors='coerce').fillna(0.0).sum()),
            'adj_season_vorp': float(pd.to_numeric(grp.get('adj_season_vorp'), errors='coerce').fillna(0.0).sum()),
        })
    return pd.DataFrame(rows).sort_values(['player_name', 'season_start_year']).reset_index(drop=True)


def compute_peak_v2(seasons: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_player_seasons(seasons).copy()
    agg['decade'] = (agg['season_start_year'] // 10 * 10).astype('Int64')
    feats = ['adj_season_ppg_blended', 'season_per', 'season_ts_pct', 'season_bpm', 'adj_season_ws', 'adj_season_vorp', 'season_g']
    for f in feats:
        med = agg.groupby('decade')[f].transform('median')
        agg[f] = pd.to_numeric(agg[f], errors='coerce').fillna(med).fillna(agg[f].median())
        agg[f'{f}_n'] = _minmax(agg[f])
    agg['season_peak_v2'] = (
        0.7 * agg[['adj_season_ppg_blended_n', 'season_per_n', 'season_ts_pct_n', 'season_bpm_n']].mean(axis=1)
        + 0.3 * agg[['adj_season_ws_n', 'adj_season_vorp_n', 'season_g_n']].mean(axis=1)
    )

    rows = []
    for player, grp in agg.groupby('player_name'):
        grp = grp.sort_values('season_start_year')
        vals = grp['season_peak_v2'].to_numpy(dtype=float)
        yrs = grp['season_start_year'].to_numpy(dtype=float)
        if len(vals) == 0:
            rows.append({'player_name': player, 'pillar_peak_v2_raw': np.nan, 'peak_v2_window': 'N/A'})
            continue
        if len(vals) < 3:
            rows.append({'player_name': player, 'pillar_peak_v2_raw': float(np.nanmean(vals)), 'peak_v2_window': f"{int(yrs[0])}-{int(yrs[-1])}"})
            continue
        best, bi = -1e9, 0
        for i in range(len(vals) - 2):
            s = float(np.nanmean(vals[i:i + 3]))
            if s > best:
                best, bi = s, i
        rows.append({'player_name': player, 'pillar_peak_v2_raw': best, 'peak_v2_window': f"{int(yrs[bi])}-{int(yrs[bi + 2])}"})
    out = pd.DataFrame(rows)
    out['pillar_peak_v2'] = _minmax(out['pillar_peak_v2_raw'])
    return out


def compute_scores(career: pd.DataFrame, seasons: pd.DataFrame, v4: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    df = career.copy()
    # Volume is intentionally totals-driven: cumulative box-score production only.
    volume_parts = [
        _minmax(df.get('career_pts', 0.0)),
        _minmax(df.get('career_trb', 0.0)),
        _minmax(df.get('career_ast', 0.0)),
        _minmax(df.get('career_stl', 0.0)),
        _minmax(df.get('career_blk', 0.0)),
    ]
    df['pillar_volume'] = pd.concat(volume_parts, axis=1).mean(axis=1)

    peak = compute_peak_v2(seasons)
    df = df.merge(peak[['player_name', 'pillar_peak_v2', 'peak_v2_window']], on='player_name', how='left')
    df['pillar_peak'] = pd.to_numeric(df['pillar_peak_v2'], errors='coerce').fillna(df['pillar_peak_v2'].median())
    # Mild reliability dampening for shorter careers so brief hot stretches do not dominate.
    seasons_played = pd.to_numeric(df.get('n_seasons', 0.0), errors='coerce').fillna(0.0)
    games_played = pd.to_numeric(df.get('career_g', 0.0), errors='coerce').fillna(0.0)
    peak_reliability = np.clip(0.7 * (seasons_played / 14.0) + 0.3 * (games_played / 1100.0), 0.45, 1.00)
    df['pillar_peak'] = df['pillar_peak'] * peak_reliability

    ws_above = pd.to_numeric(df['adj_career_ws'], errors='coerce').fillna(0.0) - pd.to_numeric(df['n_seasons'], errors='coerce').fillna(0.0) * 2.0
    df['pillar_context'] = pd.concat([_minmax(df['career_vorp']), _minmax(df['career_bpm']), _minmax(ws_above)], axis=1).mean(axis=1)

    all_nba_w = (
        pd.to_numeric(df['all_nba_1st'], errors='coerce').fillna(0.0) * 8.0
        + pd.to_numeric(df['all_nba_2nd'], errors='coerce').fillna(0.0) * 4.0
        + pd.to_numeric(df['all_nba_3rd'], errors='coerce').fillna(0.0)
    )
    df['pillar_honors'] = (
        _minmax(df['mvp']) * 0.35
        + _minmax(all_nba_w) * 0.30
        + _minmax(df['championships']) * 0.20
        + _minmax(df['allstar']) * 0.15
    )

    if not v4.empty and 'player_name' in v4.columns:
        add = [c for c in ['player_name', 'all_def_1st_v4', 'all_def_2nd_v4'] if c in v4.columns]
        df = df.merge(v4[add], on='player_name', how='left')
    for c in ['all_def_1st_v4', 'all_def_2nd_v4']:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    total = sum(max(v, 0.0) for v in weights.values())
    norm = {k: (max(v, 0.0) / total if total > 0 else 0.25) for k, v in weights.items()}
    df['goat_score'] = (
        df['pillar_volume'] * norm['volume_longevity']
        + df['pillar_peak'] * norm['peak_dominance']
        + df['pillar_context'] * norm['context_value']
        + df['pillar_honors'] * norm['honors_recognition']
    )
    df['eligible_all_time'] = (pd.to_numeric(df['n_seasons'], errors='coerce').fillna(0) >= 7) & (pd.to_numeric(df['career_g'], errors='coerce').fillna(0) >= 500)
    df = df.sort_values('goat_score', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df


def build_potential_projection(player: str, scored: pd.DataFrame, seasons: pd.DataFrame, active_proj_v4: pd.DataFrame, comps_v4: pd.DataFrame) -> dict:
    agg = aggregate_player_seasons(seasons).copy()
    agg['age'] = pd.to_numeric(agg['season_age'], errors='coerce').round().astype('Int64')
    agg = agg.dropna(subset=['age'])
    agg['age'] = agg['age'].astype(int)

    row = scored[scored['player_name'].eq(player)].iloc[0]
    me = agg[agg['player_name'].eq(player)].sort_values('age')
    if me.empty:
        return {}
    cur_age = int(me['age'].max())
    me3 = me[me['age'] <= cur_age].tail(3)
    base_games = float(me3['season_g'].mean()) if not me3.empty else 70.0
    base = {
        'ppg': float((me3['season_pts'] / me3['season_g'].replace(0, np.nan)).mean()),
        'rpg': float((me3['season_trb'] / me3['season_g'].replace(0, np.nan)).mean()),
        'apg': float((me3['season_ast'] / me3['season_g'].replace(0, np.nan)).mean()),
        'spg': float((me3['season_stl'] / me3['season_g'].replace(0, np.nan)).mean()),
        'bpg': float((me3['season_blk'] / me3['season_g'].replace(0, np.nan)).mean()),
    }
    for k in list(base.keys()):
        if np.isnan(base[k]):
            base[k] = _safe_float(row.get(f'career_{k}', 0.0), 0.0)

    # Cohort comps: historical players with similar age profile at current age.
    hist = set(scored[scored['player_status'].eq('historical')]['player_name'])
    comp_rows = []
    target_vec = np.array([base['ppg'], base['rpg'], base['apg'], base['spg'], base['bpg']])
    is_elite_rim = bool(base['bpg'] >= 2.5)
    for cand in hist:
        c = agg[(agg['player_name'].eq(cand)) & (agg['age'] <= cur_age)].sort_values('age')
        if len(c) < 3 or agg[(agg['player_name'].eq(cand)) & (agg['age'] >= cur_age + 1)].empty:
            continue
        c3 = c.tail(3)
        vec = np.array([
            float((c3['season_pts'] / c3['season_g'].replace(0, np.nan)).mean()),
            float((c3['season_trb'] / c3['season_g'].replace(0, np.nan)).mean()),
            float((c3['season_ast'] / c3['season_g'].replace(0, np.nan)).mean()),
            float((c3['season_stl'] / c3['season_g'].replace(0, np.nan)).mean()),
            float((c3['season_blk'] / c3['season_g'].replace(0, np.nan)).mean()),
        ])
        if np.isnan(vec).any():
            continue
        if is_elite_rim and vec[4] < 1.8:
            continue
        dist = float(np.sqrt(((vec - target_vec) ** 2 * np.array([1.0, 1.4, 0.6, 0.8, 3.0])).sum()))
        sim = float(np.exp(-0.2 * dist))
        comp_rows.append({'comp_player': cand, 'similarity': sim, 'similarity_pct': round(sim * 100, 1)})
    cohort = pd.DataFrame(comp_rows).sort_values('similarity', ascending=False).head(8).reset_index(drop=True) if comp_rows else pd.DataFrame()
    if not cohort.empty:
        cohort.insert(0, 'comp_rank', np.arange(1, len(cohort) + 1))

    inj_raw = 0.9
    cur_goat = float(row['goat_score'])
    proj_goat = cur_goat
    if not active_proj_v4.empty and player in active_proj_v4['player_name'].values:
        r = active_proj_v4[active_proj_v4['player_name'].eq(player)].iloc[0]
        inj_raw = _safe_float(r.get('injury_mult_v4', inj_raw), inj_raw)
        cur_goat = _safe_float(r.get('goat_score_v4_current', cur_goat), cur_goat)
    inj_used = float(np.clip(0.90 + 0.10 * inj_raw, 0.97, 0.995))

    if cur_age <= 22:
        ret_age = 40
    elif cur_age <= 25:
        ret_age = 38
    elif cur_age <= 29:
        ret_age = 37
    else:
        ret_age = 36
    ret_age = max(ret_age, cur_age + 1)

    cum = {
        'Points': float(row['career_pts']),
        'Rebounds': float(row['career_trb']),
        'Assists': float(row['career_ast']),
        'Steals': float(row['career_stl']),
        'Blocks': float(row['career_blk']),
    }
    traj = []
    comp_names = cohort['comp_player'].tolist() if not cohort.empty else []
    comp_w = cohort['similarity'].to_numpy(dtype=float) if not cohort.empty else np.array([])

    elite = np.clip((0.6 * _safe_float(row.get('pillar_peak', 50), 50) + 0.4 * _safe_float(row.get('pillar_context', 50), 50)) / 100.0, 0.0, 1.0)
    star_signal = np.clip((base['ppg'] - 18.0) / 10.0 + 0.45 * elite, 0.0, 1.4)
    prime_age = 27 if cur_age <= 24 else 29
    ppg_state, rpg_state, apg_state, spg_state, bpg_state = base['ppg'], base['rpg'], base['apg'], base['spg'], base['bpg']

    for i, age in enumerate(range(cur_age + 1, ret_age + 1), start=1):
        cvals = {}
        for col in ['season_g', 'season_pts', 'season_trb', 'season_ast', 'season_stl', 'season_blk']:
            vals, ws = [], []
            for w, cp in zip(comp_w, comp_names):
                x = agg[(agg['player_name'].eq(cp)) & (agg['age'].between(age - 1, age + 1))]
                if x.empty:
                    continue
                vals.append(float(pd.to_numeric(x[col], errors='coerce').mean()))
                ws.append(w)
            cvals[col] = float(np.average(vals, weights=ws)) if vals else np.nan

        # Development -> extended prime (into early 30s) -> sharper late-career cliff.
        if age <= prime_age:
            off_mult = 1.0 + min(0.20, 0.04 * (age - cur_age) + 0.06 * star_signal)
        elif age <= prime_age + 2:
            off_mult = 1.00 - 0.004 * (age - prime_age)
        elif age <= 33:
            off_mult = 0.99 - 0.012 * (age - (prime_age + 2))
        elif age <= 35:
            off_mult = 0.94 - 0.028 * (age - 33)
        elif age <= 37:
            off_mult = 0.86 - 0.060 * (age - 35)
        else:
            off_mult = 0.74 - 0.085 * (age - 37)
        off_mult = float(np.clip(off_mult, 0.40, 1.28))

        def_mult = off_mult if not is_elite_rim else float(np.clip(off_mult + 0.05, 0.50, 1.33))

        if age <= 30:
            g_age_mult = 1.00
        elif age <= 33:
            g_age_mult = 0.97
        elif age <= 35:
            g_age_mult = 0.91
        elif age <= 37:
            g_age_mult = 0.82
        else:
            g_age_mult = 0.70

        inj_decay = inj_used ** (i / 6.0)
        games_c = cvals['season_g'] if not np.isnan(cvals['season_g']) else base_games
        low_bound = 56 if age <= 32 else (44 if age <= 35 else 28)
        games = float(np.clip((0.58 * games_c + 0.42 * base_games) * g_age_mult * inj_decay, low_bound, 82))

        ppg_c = (cvals['season_pts'] / max(cvals['season_g'], 1.0)) if not np.isnan(cvals['season_pts']) and not np.isnan(cvals['season_g']) else ppg_state
        rpg_c = (cvals['season_trb'] / max(cvals['season_g'], 1.0)) if not np.isnan(cvals['season_trb']) and not np.isnan(cvals['season_g']) else rpg_state
        apg_c = (cvals['season_ast'] / max(cvals['season_g'], 1.0)) if not np.isnan(cvals['season_ast']) and not np.isnan(cvals['season_g']) else apg_state
        spg_c = (cvals['season_stl'] / max(cvals['season_g'], 1.0)) if not np.isnan(cvals['season_stl']) and not np.isnan(cvals['season_g']) else spg_state
        bpg_c = (cvals['season_blk'] / max(cvals['season_g'], 1.0)) if not np.isnan(cvals['season_blk']) and not np.isnan(cvals['season_g']) else bpg_state

        # Mild pace-era continuation plus controlled year-to-year variance (no random noise).
        pace_mult = float(np.clip(1.0 + 0.004 * i, 1.0, 1.07))
        var_amp = 0.030 if age <= 30 else (0.026 if age <= 34 else 0.050)

        def _var(phase: float, scale: float = 1.0) -> float:
            raw = math.sin(1.22 * i + phase) + 0.55 * math.sin(2.05 * i + 0.6 * phase)
            return float(np.clip(1.0 + scale * var_amp * raw, 0.90, 1.10))

        if age <= 28:
            block_age_decline = 1.02
        elif age <= 31:
            block_age_decline = 0.98
        elif age <= 34:
            block_age_decline = 0.91
        elif age <= 36:
            block_age_decline = 0.80
        else:
            block_age_decline = 0.66

        if age <= 29:
            steal_age_decline = 1.00
        elif age <= 33:
            steal_age_decline = 0.95
        elif age <= 36:
            steal_age_decline = 0.83
        else:
            steal_age_decline = 0.70

        ppg_state = max(0.0, (0.50 * (ppg_state * off_mult) + 0.35 * ppg_c + 0.15 * base['ppg']) * pace_mult * _var(0.2, 0.9))
        rpg_state = max(0.0, (0.53 * (rpg_state * (0.98 + 0.02 * off_mult)) + 0.32 * rpg_c + 0.15 * base['rpg']) * (1.0 + 0.0015 * i) * _var(1.1, 1.00))
        apg_state = max(0.0, (0.49 * (apg_state * off_mult) + 0.36 * apg_c + 0.15 * base['apg']) * pace_mult * _var(2.0, 1.10))
        spg_state = max(0.0, (0.52 * (spg_state * def_mult * steal_age_decline) + 0.33 * spg_c + 0.15 * base['spg']) * _var(2.8, 1.35))
        bpg_state = max(0.0, (0.50 * (bpg_state * def_mult * block_age_decline) + 0.35 * bpg_c + 0.15 * base['bpg']) * _var(3.5, 1.50))

        if is_elite_rim:
            elite_floor = base['bpg'] * (1.10 if age <= 27 else (1.00 if age <= 31 else (0.82 if age <= 35 else 0.62)))
            bpg_state = max(bpg_state, elite_floor)

        cum['Points'] += ppg_state * games
        cum['Rebounds'] += rpg_state * games
        cum['Assists'] += apg_state * games
        cum['Steals'] += spg_state * games
        cum['Blocks'] += bpg_state * games

        growth = 0.14 if cur_age <= 24 else (0.07 if cur_age <= 27 else 0.0)
        mvp_sig = np.clip((elite - 0.50 + growth) * 1.50, 0.0, 0.84)
        mvp_sig += np.clip((ppg_state - 24.0) / 12.0, 0.0, 0.25)
        mvp_sig += np.clip((apg_state - 6.0) / 8.0, 0.0, 0.08)
        mvp_p = mvp_sig * (games / 70.0) * (1.0 if age <= 33 else max(0.25, 1.0 - 0.10 * (age - 33)))

        dpoy_sig = np.clip((bpg_state / 2.5) * 0.78 + (spg_state / 1.7) * 0.22, 0.0, 1.8)
        dpoy_sig_main = np.clip((elite - 0.45 + growth * 0.7) * 1.24 * dpoy_sig, 0.0, 0.90)
        if is_elite_rim:
            dpoy_sig_main += np.clip((bpg_state - 2.6) / 2.0, 0.0, 0.26)
        dpoy_p = dpoy_sig_main * (1.0 if age <= 34 else max(0.25, 1.0 - 0.11 * (age - 34)))

        as_core = 0.30 + 0.72 * elite + 0.10 * np.clip((ppg_state - 20.0) / 10.0, 0.0, 1.0)
        as_p = np.clip(as_core, 0.20, 0.995) * (1.0 if age <= 36 else max(0.35, 1.0 - 0.09 * (age - 36)))

        traj.append({
            'Age': age,
            'Projected Games': games,
            'Projected PPG': ppg_state,
            'Projected RPG': rpg_state,
            'Projected APG': apg_state,
            'Projected SPG': spg_state,
            'Projected BPG': bpg_state,
            'Cumulative Points': cum['Points'],
            'Cumulative Rebounds': cum['Rebounds'],
            'Cumulative Assists': cum['Assists'],
            'Cumulative Steals': cum['Steals'],
            'Cumulative Blocks': cum['Blocks'],
            'mvp_p': mvp_p,
            'dpoy_p': dpoy_p,
            'as_p': as_p,
        })

    tdf = pd.DataFrame(traj)
    mvp_proj = _safe_float(pd.to_numeric(tdf.get('mvp_p', pd.Series(dtype=float)), errors='coerce').sum(), 0.0)
    dpoy_proj = _safe_float(pd.to_numeric(tdf.get('dpoy_p', pd.Series(dtype=float)), errors='coerce').sum(), 0.0)
    allstar_proj = _safe_float(pd.to_numeric(tdf.get('as_p', pd.Series(dtype=float)), errors='coerce').sum(), 0.0)
    final_acc = {
        'MVP': _safe_float(row.get('mvp', 0.0), 0.0) + mvp_proj,
        'DPOY': _safe_float(row.get('dpoy', 0.0), 0.0) + dpoy_proj,
        'All Star': _safe_float(row.get('allstar', 0.0), 0.0) + allstar_proj,
    }

    years_left = max(1, ret_age - cur_age)
    max_pts = _safe_float(scored['career_pts'].max(), 1.0)
    max_trb = _safe_float(scored['career_trb'].max(), 1.0)
    max_ast = _safe_float(scored['career_ast'].max(), 1.0)
    max_stl = _safe_float(scored['career_stl'].max(), 1.0)
    max_blk = _safe_float(scored['career_blk'].max(), 1.0)
    vol_gain = np.mean([
        max(0.0, cum['Points'] - _safe_float(row.get('career_pts', 0.0), 0.0)) / max_pts,
        max(0.0, cum['Rebounds'] - _safe_float(row.get('career_trb', 0.0), 0.0)) / max_trb,
        max(0.0, cum['Assists'] - _safe_float(row.get('career_ast', 0.0), 0.0)) / max_ast,
        max(0.0, cum['Steals'] - _safe_float(row.get('career_stl', 0.0), 0.0)) / max_stl,
        max(0.0, cum['Blocks'] - _safe_float(row.get('career_blk', 0.0), 0.0)) / max_blk,
    ])
    award_rate = (mvp_proj * 1.9 + dpoy_proj * 1.5 + allstar_proj * 0.35) / years_left
    upside = 24.0 * vol_gain + 7.0 * np.clip(award_rate, 0.0, 2.0) + 4.0 * max(0.0, elite - 0.52)
    proj_goat = float(np.clip(max(cur_goat, cur_goat + upside), 0.0, 99.9))

    v4_subset = pd.DataFrame()
    if not comps_v4.empty and 'active_player' in comps_v4.columns:
        v4_subset = comps_v4[comps_v4['active_player'].eq(player)].sort_values('comp_rank')

    return {
        'current_age': cur_age,
        'retirement_age': ret_age,
        'current_goat': cur_goat,
        'projected_goat': proj_goat,
        'injury_multiplier_raw_v4': inj_raw,
        'injury_multiplier_used': inj_used,
        'trajectory': tdf,
        'final_totals': cum,
        'final_accolades': final_acc,
        'cohort_comparables': cohort,
        'v4_comparables': v4_subset,
    }


def _build_record_book(scored: pd.DataFrame) -> dict:
    out = {}
    for label, (career_col, cumulative_col) in RECORD_MAP.items():
        idx = scored[career_col].astype(float).idxmax()
        out[label] = {
            'career_col': career_col,
            'cumulative_col': cumulative_col,
            'holder': str(scored.loc[idx, 'player_name']),
            'record': float(scored.loc[idx, career_col]),
        }
    return out


def main() -> None:
    if not CAREER_PATH.exists() or not SEASONS_PATH.exists():
        st.error('Missing CSV outputs. Run: python run_pipeline.py --mode full')
        st.stop()

    st.title('NBA GOAT Analyzer')
    st.caption('Four-pillar GOAT ranking + age-trajectory Potential GOAT forecasts.')

    career, seasons, v4, active_proj_v4, comps_v4 = load_data()
    with st.sidebar:
        st.header('Pillar Weights')
        w_volume = st.slider('Volume/Longevity', 0.0, 1.0, 0.25, 0.01)
        w_peak = st.slider('Peak Dominance', 0.0, 1.0, 0.25, 0.01)
        w_context = st.slider('Context Value', 0.0, 1.0, 0.25, 0.01)
        w_honors = st.slider('Honors/Recognition', 0.0, 1.0, 0.25, 0.01)
        gate = st.checkbox('Use all-time eligibility gate', value=True, help='7+ seasons and 500+ games.')
        active_only = st.checkbox('Show active players only', value=False)

    scored = compute_scores(career, seasons, v4, {
        'volume_longevity': w_volume,
        'peak_dominance': w_peak,
        'context_value': w_context,
        'honors_recognition': w_honors,
    })
    records = _build_record_book(scored)

    view = scored.copy()
    if gate:
        view = view[view['eligible_all_time']]
    if active_only:
        view = view[view['player_status'].eq('active')]
    view = view.reset_index(drop=True)
    view['rank'] = np.arange(1, len(view) + 1)

    t = view.iloc[0] if not view.empty else None
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Players Ranked', f'{len(view)}')
    c2.metric('Top Score', f"{t['goat_score']:.1f}" if t is not None else '-')
    c3.metric('Current #1', str(t['player_name']) if t is not None else '-')
    c4.metric('Active In Top 10', f"{int(view.head(10)['player_status'].eq('active').sum()) if not view.empty else 0}")
    c5.metric('Median Score', f"{view['goat_score'].median():.1f}" if not view.empty else '-')

    tab1, tab2, tab3, tab4 = st.tabs(['GOAT Rankings', 'Player Deep Dive', 'Potential GOAT', 'Methodology'])

    with tab1:
        cols = [c for c in [
            'rank', 'player_name', 'player_status', 'goat_score', 'pillar_volume', 'pillar_peak', 'pillar_context', 'pillar_honors',
            'mvp', 'dpoy', 'all_nba_1st', 'all_nba_2nd', 'all_nba_3rd', 'all_def_1st_v4', 'all_def_2nd_v4',
            'championships', 'allstar', 'career_ppg', 'career_rpg', 'career_apg', 'career_spg', 'career_bpg'
        ] if c in view.columns]
        show = view[cols].copy()
        for c in ['goat_score', 'pillar_volume', 'pillar_peak', 'pillar_context', 'pillar_honors']:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors='coerce').round(2)
        for c in ['career_ppg', 'career_rpg', 'career_apg', 'career_spg', 'career_bpg']:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors='coerce').round(1)
        show = show.rename(columns={k: k.replace('_', ' ').title() for k in show.columns})
        show = show.rename(columns={
            'Goat Score': 'GOAT Score',
            'Mvp': 'MVP',
            'Dpoy': 'DPOY',
            'All Nba 1St': 'All-NBA 1st Team',
            'All Nba 2Nd': 'All-NBA 2nd Team',
            'All Nba 3Rd': 'All-NBA 3rd Team',
            'All Def 1St V4': 'All-Def 1st Team',
            'All Def 2Nd V4': 'All-Def 2nd Team',
        })
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Pillar Volume': st.column_config.NumberColumn(
                    'Pillar Volume',
                    help='Career production totals: points, rebounds, assists, steals, and blocks (normalized 0-100).',
                    format='%.2f',
                ),
                'Pillar Peak': st.column_config.NumberColumn(
                    'Pillar Peak',
                    help='Best 3-year prime based on season production + impact, with reliability adjustment for shorter careers.',
                    format='%.2f',
                ),
                'Pillar Context': st.column_config.NumberColumn(
                    'Pillar Context',
                    help='Era-adjusted impact profile using BPM, VORP, and win-share-above-baseline signals.',
                    format='%.2f',
                ),
                'Pillar Honors': st.column_config.NumberColumn(
                    'Pillar Honors',
                    help='Recognition blend of MVP, weighted All-NBA teams, championships, and All-Star selections.',
                    format='%.2f',
                ),
            },
        )
        st.bar_chart(view.head(25).sort_values('goat_score').set_index('player_name')['goat_score'])

    with tab2:
        if view.empty:
            st.warning('No players available for current filter.')
            st.stop()
        p = st.selectbox('Select a player', options=view['player_name'].tolist(), index=0)
        r = view[view['player_name'].eq(p)].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric('Rank', f"#{int(r['rank'])}")
        d2.metric('GOAT Score', f"{r['goat_score']:.2f}")
        d3.metric('Status', str(r['player_status']).title())
        d4.metric('Seasons', f"{int(r['n_seasons'])}")
        st.caption(f"Peak v2 window: {r.get('peak_v2_window', 'N/A')}")
        snap_cols = [c for c in [
            'career_ppg', 'career_rpg', 'career_apg', 'career_spg', 'career_bpg', 'career_pts', 'career_trb', 'career_ast', 'career_stl', 'career_blk',
            'career_per', 'career_bpm', 'career_vorp', 'mvp', 'dpoy', 'finals_mvp', 'all_nba_1st', 'all_nba_2nd', 'all_nba_3rd',
            'all_def_1st_v4', 'all_def_2nd_v4', 'championships', 'allstar'
        ] if c in view.columns]
        snap = r[snap_cols].to_frame('value')
        for c in ['career_ppg', 'career_rpg', 'career_apg', 'career_spg', 'career_bpg']:
            if c in snap.index:
                snap.loc[c, 'value'] = round(float(snap.loc[c, 'value']), 1)
        snap.index = [i.replace('_', ' ').title() for i in snap.index]
        st.dataframe(snap, use_container_width=True)

    with tab3:
        actives = scored[scored['player_status'].eq('active')]['player_name'].sort_values().tolist()
        if not actives:
            st.warning('No active players found.')
            st.stop()
        p = st.selectbox('Select active player for projection', options=actives, index=0)
        proj = build_potential_projection(p, scored, seasons, active_proj_v4, comps_v4)
        if not proj:
            st.warning('Projection unavailable.')
            st.stop()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric('Current Age', f"{proj['current_age']}")
        p2.metric('Est. Retirement Age', f"{proj['retirement_age']}")
        p3.metric('Current GOAT Score', f"{proj['current_goat']:.1f}")
        p4.metric('Projected GOAT Score', f"{proj['projected_goat']:.1f}")
        st.caption(f"Injury multiplier (v4 raw): {proj['injury_multiplier_raw_v4']:.3f} | Used (conservative): {proj['injury_multiplier_used']:.3f}")

        totals = pd.DataFrame({'Metric': list(proj['final_totals'].keys()), 'Projected Career Total': list(proj['final_totals'].values())})
        totals['Projected Career Total'] = totals['Projected Career Total'].round(0)
        st.dataframe(totals, use_container_width=True, hide_index=True)

        acc = pd.DataFrame({'Award': list(proj['final_accolades'].keys()), 'Projected Career Total': list(proj['final_accolades'].values())})
        acc['Projected Career Total'] = pd.to_numeric(acc['Projected Career Total'], errors='coerce').round(0).astype('Int64')
        st.dataframe(acc, use_container_width=True, hide_index=True)

        metric = st.selectbox('Record metric', options=['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks'], index=4)
        rec = records[metric]
        traj = proj['trajectory'].copy()
        if not traj.empty:
            plot_df = traj[['Age', rec['cumulative_col']]].copy()
            plot_df[f"All-Time {metric} Record ({rec['holder']})"] = rec['record']
            st.line_chart(plot_df.set_index('Age'))

        cur_val = float(scored.loc[scored['player_name'].eq(p), rec['career_col']].iloc[0])
        proj_val = float(proj['final_totals'][metric])
        if cur_val >= rec['record'] - 1e-6:
            prob = 1.0
        else:
            gain = max(0.0, proj_val - cur_val)
            sigma = 700.0 if metric == 'Points' else (220.0 if metric in ['Rebounds', 'Assists'] else 90.0)
            sigma = max(sigma, 0.3 * gain)
            prob = float(np.clip(1.0 - _normal_cdf(rec['record'], proj_val, sigma), 0.0, 1.0))
        st.metric(f'Chance to finish as all-time {metric.lower()} leader', f"{prob * 100:.1f}%")

        st.line_chart(traj.set_index('Age')[['Projected PPG', 'Projected RPG', 'Projected APG', 'Projected SPG', 'Projected BPG']])

        st.write('Cohort comparables used by trajectory model')
        if proj['cohort_comparables'].empty:
            st.info('No cohort comparables found.')
        else:
            st.dataframe(proj['cohort_comparables'], use_container_width=True, hide_index=True)

        with st.expander('Why v4 comparables can look odd (debug)'):
            st.caption('v4 comparables are cosine similarity in GOAT feature space, not age-trajectory matching.')
            if proj['v4_comparables'].empty:
                st.info('No v4 comps found.')
            else:
                st.dataframe(proj['v4_comparables'], use_container_width=True, hide_index=True)

    with tab4:
        st.markdown(
            '- Peak is the best 3-year prime based on season production and impact.\n'
            '- Potential GOAT uses age-matched historical cohorts to learn career trajectories.\n'
            '- Injury assumptions are conservative, with stronger decline near retirement ages.\n'
            '- Projection includes mild pace-era continuation for offensive environment drift.\n'
            '- Record chart supports points, rebounds, assists, steals, and blocks.'
        )


if __name__ == '__main__':
    main()
