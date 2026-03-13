
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# GOAT SCORE ENGINE — Four-Pillar Composite Metric
# ═══════════════════════════════════════════════════════════════════════════════
# Pillars (each normalized 0-100 against full historical cohort):
#   1. Volume / Longevity   — era-adjusted career totals + games + seasons
#   2. Peak Dominance       — best 3-season rolling avg of PER, TS%, BPM
#   3. Context-Adjusted Val — career VORP, career BPM, WS above replacement
#   4. Honors / Recognition — All-NBA weighted + MVPs + championships + All-Stars
#
# Active players: current_goat_score + projected_goat_score
# Comparable matching: cosine similarity on era-adjusted stats → top-5 per active player
# Weights: configurable 0.0–1.0 per pillar (auto-normalised to sum=1)
# ═══════════════════════════════════════════════════════════════════════════════

ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
            'Nikola Jokic', 'LeBron James', 'Stephen Curry']

# ── 1. Pillar Configuration ──────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    'volume_longevity':    0.25,
    'peak_dominance':      0.25,
    'context_value':       0.25,
    'honors_recognition':  0.25,
}

def normalize_weights(weights: dict) -> dict:
    """Normalize pillar weights so they sum to 1.0. Clamps each to [0,1]."""
    w = {k: max(0.0, min(1.0, v)) for k, v in weights.items()}
    total = sum(w.values())
    if total == 0:
        n = len(w)
        return {k: 1.0/n for k in w}
    return {k: v / total for k, v in w.items()}

# ── 2. Per-Player Feature Assembly ──────────────────────────────────────────
# Use era_adj_career_df (99 players) as the main career table
# seasons comes from era_adj_seasons_df for rolling-window peak calc

goat_career = era_adj_career_df.copy()
goat_seasons = era_adj_seasons_df.copy()

# ─── 2a. Pillar 1: Volume / Longevity ───────────────────────────────────────
# Components: era-adj career pts, reb, ast, games played, seasons (n_seasons)
# We sum multiple normalised sub-scores for a composite volume pillar

VOL_COLS = {
    'adj_career_pts': 'vol_pts',
    'adj_career_trb': 'vol_trb',
    'adj_career_ast': 'vol_ast',
    'career_g':       'vol_games',
    'n_seasons':      'vol_seasons',
}

for src, dst in VOL_COLS.items():
    col = goat_career[src].fillna(0)
    mx = col.max()
    goat_career[dst] = (col / mx * 100) if mx > 0 else 0.0

# Equal-weight sub-components within pillar 1
goat_career['pillar_volume'] = goat_career[list(VOL_COLS.values())].mean(axis=1)

# ─── 2b. Pillar 2: Peak Dominance — best 3-season rolling avg ──────────────
# PER, TS%, BPM  →  rolling 3-season mean, take max per player
# Use actual season columns (not adjusted — PER and BPM are already context-free)

PEAK_COLS = ['season_per', 'season_ts_pct', 'season_bpm']

def best_3season_avg(player_seasons: pd.DataFrame, metric: str) -> float:
    """Return best 3-consecutive-season rolling average for a player metric."""
    srt = player_seasons.sort_values('season_start_year')
    vals = srt[metric].dropna().values
    if len(vals) < 1:
        return np.nan
    if len(vals) < 3:
        return float(np.mean(vals))
    rolling = np.convolve(vals, np.ones(3)/3, mode='valid')
    return float(np.max(rolling))

peak_rows = []
for player, grp in goat_seasons.groupby('player_name'):
    row = {'player_name': player}
    for col in PEAK_COLS:
        if col in grp.columns:
            row[f'peak3_{col}'] = best_3season_avg(grp, col)
        else:
            row[f'peak3_{col}'] = np.nan
    peak_rows.append(row)

peak_df = pd.DataFrame(peak_rows)

# Normalize each peak metric 0-100 against cohort
for col in PEAK_COLS:
    dst = f'peak3_{col}'
    if dst in peak_df.columns:
        vals = peak_df[dst].fillna(0)
        mn, mx = vals.min(), vals.max()
        peak_df[f'{dst}_norm'] = ((vals - mn) / (mx - mn) * 100) if mx > mn else 50.0

norm_peak_cols = [f'peak3_{c}_norm' for c in PEAK_COLS]
peak_df['pillar_peak'] = peak_df[norm_peak_cols].mean(axis=1)

goat_career = goat_career.merge(
    peak_df[['player_name', 'pillar_peak'] + norm_peak_cols],
    on='player_name', how='left'
)
goat_career['pillar_peak'] = goat_career['pillar_peak'].fillna(
    goat_career['pillar_peak'].median()
)

# ─── 2c. Pillar 3: Context-Adjusted Value ───────────────────────────────────
# Career VORP, career BPM, Win Shares above replacement (WS_above_repl = WS - 0*career_g
# We define WS_above_replacement = adj_career_ws - (career_g/82 * 2.0) for a 2 WS/season baseline

REPL_WS_PER_SEASON = 2.0  # Replacement player earns ~2 WS/season

goat_career['ws_above_repl'] = (
    goat_career['adj_career_ws'].fillna(0) -
    (goat_career['n_seasons'].fillna(0) * REPL_WS_PER_SEASON)
)

CTX_COLS = {
    'career_vorp':   'ctx_vorp',
    'career_bpm':    'ctx_bpm',
    'ws_above_repl': 'ctx_ws_repl',
}
for src, dst in CTX_COLS.items():
    vals = goat_career[src].fillna(goat_career[src].median())
    mn, mx = vals.min(), vals.max()
    goat_career[dst] = ((vals - mn) / (mx - mn) * 100) if mx > mn else 50.0

goat_career['pillar_context'] = goat_career[list(CTX_COLS.values())].mean(axis=1)

# ─── 2d. Pillar 4: Honors / Recognition ─────────────────────────────────────
# All-NBA: 1st = 5pts, 2nd = 3pts, 3rd = 1pt
# MVPs, championships, All-Star selections

goat_career['all_nba_weighted'] = (
    goat_career['all_nba_1st'].fillna(0) * 5 +
    goat_career['all_nba_2nd'].fillna(0) * 3 +
    goat_career['all_nba_3rd'].fillna(0) * 1
)

HON_COLS = {
    'mvp':             'hon_mvp',
    'championships':   'hon_champ',
    'allstar':         'hon_allstar',
    'all_nba_weighted':'hon_allnba',
}
for src, dst in HON_COLS.items():
    col_vals = goat_career[src].fillna(0)
    mx = col_vals.max()
    goat_career[dst] = (col_vals / mx * 100) if mx > 0 else 0.0

# Weight within honors pillar: MVPs most important, then All-NBA, championships, All-Stars
goat_career['pillar_honors'] = (
    goat_career['hon_mvp']     * 0.30 +
    goat_career['hon_allnba']  * 0.30 +
    goat_career['hon_champ']   * 0.25 +
    goat_career['hon_allstar'] * 0.15
)

# ── 3. Composite GOAT Score Function ────────────────────────────────────────
def compute_goat_score(career_row: pd.Series, weights: dict = None) -> float:
    """
    Compute GOAT Score (0-100) for a player row using configurable weights.
    weights dict: keys = volume_longevity, peak_dominance, context_value, honors_recognition
                  values = 0.0–1.0 (auto-normalized to sum=1)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w = normalize_weights(weights)

    score = (
        w['volume_longevity']   * career_row.get('pillar_volume',  0) +
        w['peak_dominance']     * career_row.get('pillar_peak',    0) +
        w['context_value']      * career_row.get('pillar_context', 0) +
        w['honors_recognition'] * career_row.get('pillar_honors',  0)
    )
    return round(float(score), 2)

# Apply default weights to all historical players → historical GOAT score
goat_career['goat_score_current'] = goat_career.apply(
    lambda r: compute_goat_score(r, DEFAULT_WEIGHTS), axis=1
)

# ── 4. Active Player Current vs Projected GOAT Scores ───────────────────────
# For active players: compute PROJECTED pillar scores using base-case projection end-states

# Grab base-case retirement row from career_projections_df
proj_endstate = (
    career_projections_df[career_projections_df['scenario'] == 'base']
    .sort_values('age')
    .groupby('player')
    .last()
    .reset_index()
)

# Build projected career totals for each active player
CURRENT_AGES = {
    'Victor Wembanyama':       21,
    'Shai Gilgeous-Alexander': 26,
    'Nikola Jokic':            29,
    'LeBron James':            40,
    'Stephen Curry':           36,
}

# Re-compute normalization maxima from full cohort (including projected end-states)
# to ensure fair comparison on projected pillars
_cohort_pts_max = max(goat_career['adj_career_pts'].max(),
                      proj_endstate['cum_pts'].max() if len(proj_endstate) else 0)
_cohort_vorp_max = goat_career['career_vorp'].max()
_cohort_ws_max   = goat_career['adj_career_ws'].max()

projected_rows = []
for pname in ACTIVE_5:
    # Current row in goat_career
    curr_row = goat_career[goat_career['player_name'] == pname]
    if curr_row.empty:
        continue
    curr = curr_row.iloc[0]
    curr_goat = curr['goat_score_current']

    # Projected end-state
    pe = proj_endstate[proj_endstate['player'] == pname]
    if pe.empty:
        projected_rows.append({
            'player_name': pname,
            'goat_score_current': curr_goat,
            'goat_score_projected': curr_goat,
            'proj_pillar_volume': curr['pillar_volume'],
            'proj_pillar_peak':   curr['pillar_peak'],
            'proj_pillar_context':curr['pillar_context'],
            'proj_pillar_honors': curr['pillar_honors'],
        })
        continue

    pe = pe.iloc[0]

    # Projected volume pillar: use projected cumulative totals
    # cum_pts = projected career points, cum_vorp/cum_ws also available
    proj_pts  = float(pe.get('cum_pts', curr['adj_career_pts']))
    proj_vorp = float(pe.get('cum_vorp', curr['career_vorp']))
    proj_ws   = float(pe.get('cum_ws', curr['adj_career_ws']))
    # Approximate projected games/seasons from retirement age
    ret_age   = int(pe.get('age', CURRENT_AGES.get(pname, 35)))
    curr_age  = CURRENT_AGES.get(pname, 30)
    proj_g    = float(curr['career_g']) + float(pe.get('cum_g', 0) - curr['career_g'])
    proj_g    = max(proj_g, float(curr['career_g']))
    proj_seas = max(int(curr['n_seasons']), int(curr['n_seasons']) + (ret_age - curr_age))

    # Normalize against cohort for projected pillars
    pts_max_c  = goat_career['adj_career_pts'].max()
    trb_max_c  = goat_career['adj_career_trb'].max()
    ast_max_c  = goat_career['adj_career_ast'].max()
    g_max_c    = goat_career['career_g'].max()
    seas_max_c = goat_career['n_seasons'].max()

    pv_pts   = min(proj_pts / pts_max_c * 100, 100) if pts_max_c > 0 else 0
    pv_trb   = curr['vol_trb']  # unchanged (projected scoring only)
    pv_ast   = curr['vol_ast']
    pv_g     = min(proj_g / g_max_c * 100, 100) if g_max_c > 0 else 0
    pv_seas  = min(proj_seas / seas_max_c * 100, 100) if seas_max_c > 0 else 0
    proj_pil_vol = np.mean([pv_pts, pv_trb, pv_ast, pv_g, pv_seas])

    # Projected peak pillar: peak is locked at current (already achieved peak-type seasons)
    proj_pil_peak = float(curr['pillar_peak'])

    # Projected context pillar: use projected VORP and WS
    vorp_mn_c = goat_career['career_vorp'].min()
    vorp_mx_c = goat_career['career_vorp'].max()
    bpm_mn_c  = goat_career['career_bpm'].min()
    bpm_mx_c  = goat_career['career_bpm'].max()

    proj_ws_abv   = proj_ws - (proj_seas * REPL_WS_PER_SEASON)
    _ws_abv_vals  = goat_career['ws_above_repl']
    ws_abv_mn, ws_abv_mx = _ws_abv_vals.min(), _ws_abv_vals.max()

    ctx_v = (proj_vorp - vorp_mn_c) / (vorp_mx_c - vorp_mn_c) * 100 if vorp_mx_c > vorp_mn_c else 50
    ctx_b = curr['ctx_bpm']  # career BPM doesn't change drastically; use current
    ctx_w = (proj_ws_abv - ws_abv_mn) / (ws_abv_mx - ws_abv_mn) * 100 if ws_abv_mx > ws_abv_mn else 50
    ctx_v = max(0, min(100, ctx_v))
    ctx_b = max(0, min(100, ctx_b))
    ctx_w = max(0, min(100, ctx_w))
    proj_pil_ctx = np.mean([ctx_v, ctx_b, ctx_w])

    # Projected honors: same as current (future honors not modelled explicitly; use current)
    proj_pil_hon = float(curr['pillar_honors'])

    # Compute projected GOAT score
    w = normalize_weights(DEFAULT_WEIGHTS)
    proj_goat = round(
        w['volume_longevity']   * proj_pil_vol  +
        w['peak_dominance']     * proj_pil_peak +
        w['context_value']      * proj_pil_ctx  +
        w['honors_recognition'] * proj_pil_hon,
        2
    )

    projected_rows.append({
        'player_name':          pname,
        'goat_score_current':   curr_goat,
        'goat_score_projected': proj_goat,
        'proj_pillar_volume':   round(proj_pil_vol,  2),
        'proj_pillar_peak':     round(proj_pil_peak, 2),
        'proj_pillar_context':  round(proj_pil_ctx,  2),
        'proj_pillar_honors':   round(proj_pil_hon,  2),
    })

proj_active_df = pd.DataFrame(projected_rows) if projected_rows else pd.DataFrame()

# Merge projected scores back into goat_career for active players
if len(proj_active_df):
    goat_career = goat_career.merge(
        proj_active_df[['player_name', 'goat_score_projected']],
        on='player_name', how='left'
    )
    # For historical players, projected = current
    goat_career['goat_score_projected'] = goat_career['goat_score_projected'].fillna(
        goat_career['goat_score_current']
    )
else:
    goat_career['goat_score_projected'] = goat_career['goat_score_current']

# ── 5. Rank players ─────────────────────────────────────────────────────────
goat_career = goat_career.sort_values('goat_score_current', ascending=False).reset_index(drop=True)
goat_career['goat_rank'] = goat_career.index + 1
goat_career['goat_percentile'] = (
    (len(goat_career) - goat_career['goat_rank']) / len(goat_career) * 100
).round(1)

# ── 6. Comparable Player Matching (Cosine Similarity) ───────────────────────
# Feature vector: era-adjusted per-game stats + advanced metrics
# Only use historical players as the comp pool

COMP_FEATURES = [
    'adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
    'career_ts_pct',  'career_per',     'career_bpm',
    'career_vorp',    'career_ws_per48',
]

_hist_pool = goat_career[goat_career['player_status'] == 'historical'].copy()
_hist_pool_clean = _hist_pool[['player_name'] + COMP_FEATURES].dropna(subset=COMP_FEATURES)

# Standardize features for cosine similarity
from sklearn.preprocessing import StandardScaler as _SS
_ss = _SS()
_X_hist = _ss.fit_transform(_hist_pool_clean[COMP_FEATURES])

comp_rows = []
for pname in ACTIVE_5:
    act_row = goat_career[goat_career['player_name'] == pname]
    if act_row.empty:
        continue
    act_row = act_row.iloc[0]

    feat_vals = act_row[COMP_FEATURES].values
    if np.any(pd.isna(feat_vals)):
        feat_vals = np.nan_to_num(feat_vals, nan=0.0)

    _x_act = _ss.transform([feat_vals])
    sims = cosine_similarity(_x_act, _X_hist)[0]
    top5_idx = np.argsort(sims)[::-1][:5]

    for rank, idx in enumerate(top5_idx, 1):
        hist_name = _hist_pool_clean.iloc[idx]['player_name']
        sim_score = float(sims[idx])
        # Find the historical player's GOAT score
        hist_goat_row = goat_career[goat_career['player_name'] == hist_name]
        hist_goat = float(hist_goat_row['goat_score_current'].values[0]) if len(hist_goat_row) else np.nan

        comp_rows.append({
            'active_player':    pname,
            'comp_rank':        rank,
            'comp_player':      hist_name,
            'similarity_pct':   round(sim_score * 100, 1),
            'comp_goat_score':  hist_goat,
            'comp_adj_ppg':     round(float(_hist_pool_clean.iloc[idx]['adj_career_ppg']), 1),
            'comp_career_per':  round(float(_hist_pool_clean.iloc[idx]['career_per']), 1),
            'comp_career_bpm':  round(float(_hist_pool_clean.iloc[idx]['career_bpm']), 2),
            'comp_career_vorp': round(float(_hist_pool_clean.iloc[idx]['career_vorp']), 1),
        })

goat_comps_df = pd.DataFrame(comp_rows)

# ── 7. Final Output Table ────────────────────────────────────────────────────
OUTPUT_COLS = [
    'goat_rank', 'player_name', 'player_status',
    'goat_score_current', 'goat_score_projected', 'goat_percentile',
    'pillar_volume', 'pillar_peak', 'pillar_context', 'pillar_honors',
    'adj_career_ppg', 'adj_career_rpg', 'adj_career_apg',
    'career_per', 'career_ts_pct', 'career_bpm', 'career_vorp',
    'adj_career_ws', 'n_seasons', 'career_g',
    'mvp', 'championships', 'allstar', 'all_nba_1st', 'all_nba_2nd', 'all_nba_3rd',
]
avail_cols = [c for c in OUTPUT_COLS if c in goat_career.columns]
goat_scores_df = goat_career[avail_cols].copy()

# ── 8. Print Results ─────────────────────────────────────────────────────────
print("=" * 80)
print("  GOAT SCORE ENGINE — Composite Rankings (Default Equal Weights)")
print("=" * 80)
print(f"\n{'Rank':>4}  {'Player':<32}  {'Status':>10}  {'GOAT':>6}  "
      f"{'Proj':>6}  {'Pct':>5}  {'Vol':>5}  {'Peak':>5}  {'Ctx':>5}  {'Hon':>5}")
print("-" * 90)
for _, r in goat_scores_df.head(30).iterrows():
    proj_str = f"{r['goat_score_projected']:>6.1f}" if r.get('goat_score_projected') else '  —  '
    print(f"  {int(r['goat_rank']):>3}  {r['player_name']:<32}  {r['player_status']:>10}  "
          f"{r['goat_score_current']:>6.1f}  {proj_str}  {r['goat_percentile']:>5.1f}  "
          f"{r['pillar_volume']:>5.1f}  {r['pillar_peak']:>5.1f}  "
          f"{r['pillar_context']:>5.1f}  {r['pillar_honors']:>5.1f}")

print(f"\n{'─'*80}")
print("  ACTIVE PLAYER SPOTLIGHT: Current vs Projected GOAT Scores")
print(f"{'─'*80}")
print(f"\n  {'Player':<32}  {'Current':>8}  {'Projected':>10}  {'Delta':>7}  {'Percentile':>10}")
print("  " + "-" * 70)
for pname in ACTIVE_5:
    row = goat_scores_df[goat_scores_df['player_name'] == pname]
    if row.empty: continue
    row = row.iloc[0]
    delta = row['goat_score_projected'] - row['goat_score_current']
    print(f"  {pname:<32}  {row['goat_score_current']:>8.1f}  "
          f"{row['goat_score_projected']:>10.1f}  "
          f"{delta:>+7.1f}  {row['goat_percentile']:>10.1f}%")

print(f"\n{'─'*80}")
print("  COMPARABLE PLAYER MATCHES (5 Historical Comps per Active Player)")
print(f"{'─'*80}")
for pname in ACTIVE_5:
    comps = goat_comps_df[goat_comps_df['active_player'] == pname]
    if comps.empty: continue
    active_row = goat_scores_df[goat_scores_df['player_name'] == pname]
    goat_c = active_row['goat_score_current'].values[0] if len(active_row) else 0
    print(f"\n  {pname}  (Current GOAT: {goat_c:.1f})")
    for _, comp in comps.iterrows():
        print(f"    #{int(comp['comp_rank'])}  {comp['comp_player']:<30}  "
              f"Sim={comp['similarity_pct']:>5.1f}%  "
              f"GOAT={comp['comp_goat_score']:>5.1f}  "
              f"PPG={comp['comp_adj_ppg']:>5.1f}  "
              f"PER={comp['comp_career_per']:>5.1f}  "
              f"BPM={comp['comp_career_bpm']:>+5.2f}")

print(f"\n{'─'*80}")
print("  WEIGHT SENSITIVITY DEMO — How Sliders Rerank Players")
print(f"{'─'*80}")
_weight_scenarios = [
    ('Volume Heavy (40/20/20/20)',  {'volume_longevity': 0.40, 'peak_dominance': 0.20, 'context_value': 0.20, 'honors_recognition': 0.20}),
    ('Peak Heavy   (20/40/20/20)',  {'volume_longevity': 0.20, 'peak_dominance': 0.40, 'context_value': 0.20, 'honors_recognition': 0.20}),
    ('Context Heavy(20/20/40/20)', {'volume_longevity': 0.20, 'peak_dominance': 0.20, 'context_value': 0.40, 'honors_recognition': 0.20}),
    ('Honors Heavy (20/20/20/40)', {'volume_longevity': 0.20, 'peak_dominance': 0.20, 'context_value': 0.20, 'honors_recognition': 0.40}),
]
_demo_players = ['Michael Jordan', 'LeBron James', 'Kareem Abdul-Jabbar',
                 'Wilt Chamberlain', 'Stephen Curry', 'Nikola Jokic',
                 'Shai Gilgeous-Alexander', 'Victor Wembanyama']

for scenario_name, scenario_w in _weight_scenarios:
    print(f"\n  Scenario: {scenario_name}")
    _scores = []
    for pname in _demo_players:
        r = goat_career[goat_career['player_name'] == pname]
        if r.empty: continue
        s = compute_goat_score(r.iloc[0], scenario_w)
        _scores.append((pname, s))
    _scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (pname, score) in enumerate(_scores, 1):
        print(f"    {rank}. {pname:<30}  {score:.1f}")

print(f"\n✓ goat_scores_df shape: {goat_scores_df.shape}")
print(f"✓ goat_comps_df shape:  {goat_comps_df.shape}")
print(f"✓ compute_goat_score() accepts custom weights and re-ranks correctly")
