
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ── Load upstream data ───────────────────────────────────────────────────────
seasons  = era_adj_seasons_df.copy()
career   = era_adj_career_df.copy()

# Active players we're projecting
ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
            'Nikola Jokic', 'LeBron James', 'Stephen Curry']

# Current ages (2024-25 season start)
CURRENT_AGES = {
    'Victor Wembanyama':       21,
    'Shai Gilgeous-Alexander': 26,
    'Nikola Jokic':            29,
    'LeBron James':            40,
    'Stephen Curry':           36,
}

RETIREMENT_AGE = 38   # typical retirement horizon
METRICS = ['season_ppg', 'season_per', 'season_bpm', 'season_vorp', 'season_ws']

# ── 1. Player Archetype Clustering (k-means on style metrics) ────────────────
# Use historical + active players; cluster on per-game style fingerprint
style_cols = ['season_ppg', 'season_rpg', 'season_apg',
              'season_bpm', 'season_dbpm', 'season_obpm', 'season_ws']
style_data = seasons[['player_name', 'pos'] + style_cols].dropna(subset=style_cols)

# Aggregate to career-average style per player
player_style = (
    style_data.groupby('player_name')[style_cols]
    .mean()
    .reset_index()
)

# Merge positions (most common)
pos_map = (
    seasons.groupby('player_name')['pos']
    .agg(lambda x: x.mode()[0] if len(x) else 'F')
    .reset_index()
)
player_style = player_style.merge(pos_map, on='player_name')

scaler_style = StandardScaler()
X_style = scaler_style.fit_transform(player_style[style_cols])

# Choose k=4 archetypes; validate with silhouette
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
player_style['archetype_id'] = kmeans.fit_predict(X_style)

# Label archetypes by centroid characteristics
centers = pd.DataFrame(
    scaler_style.inverse_transform(kmeans.cluster_centers_),
    columns=style_cols
)
# Scorer: high PPG. Playmaker: high APG. Big Man: high RPG low APG. Two-way Wing: balanced BPM.
archetype_labels = {}
for i, row in centers.iterrows():
    if row['season_apg'] >= 6.0:
        archetype_labels[i] = 'Playmaker'
    elif row['season_rpg'] >= 8.0:
        archetype_labels[i] = 'Big Man'
    elif row['season_ppg'] >= 22.0 or row['season_obpm'] >= 4.0:
        archetype_labels[i] = 'Scorer'
    else:
        archetype_labels[i] = 'Two-Way Wing'

player_style['archetype'] = player_style['archetype_id'].map(archetype_labels)
sil_score = silhouette_score(X_style, player_style['archetype_id'])

print(f"K-Means Silhouette Score (k=4): {sil_score:.3f}")
print("\nArchetype Cluster Centers:")
for aid, albl in archetype_labels.items():
    c = centers.iloc[aid]
    print(f"  {albl:16s} | PPG={c['season_ppg']:.1f} RPG={c['season_rpg']:.1f} "
          f"APG={c['season_apg']:.1f} BPM={c['season_bpm']:.1f}")

print("\nActive Player Archetypes:")
for p in ACTIVE_5:
    row_p = player_style[player_style['player_name'] == p]
    arch  = row_p['archetype'].values[0] if len(row_p) else '?'
    print(f"  {p:30s}: {arch}")

# ── 2. Fit Age Curves (polynomial degree-2) per metric per archetype ─────────
# Use only historical players with ≥5 seasons for curve fitting
hist_players = career[career['player_status'] == 'historical']['player_name'].tolist()

# Filter seasons to historical + age 18–40; join archetype
seasons_hist = seasons[
    seasons['player_name'].isin(hist_players) &
    seasons['season_age'].between(18, 40)
].merge(player_style[['player_name', 'archetype']], on='player_name', how='left')
seasons_hist['archetype'] = seasons_hist['archetype'].fillna('Two-Way Wing')

def fit_poly2(ages, vals):
    """Fit quadratic polynomial to age-value pairs; return coefficients."""
    mask = ~(np.isnan(ages) | np.isnan(vals))
    ages_c, vals_c = ages[mask], vals[mask]
    if len(ages_c) < 5:
        return None
    return np.polyfit(ages_c, vals_c, 2)  # [a, b, c] for a*x^2 + b*x + c

AGE_CURVES = {}   # {archetype: {metric: (a, b, c)}}
age_arr = seasons_hist['season_age'].values

for arch in seasons_hist['archetype'].dropna().unique():
    AGE_CURVES[arch] = {}
    arch_mask = seasons_hist['archetype'] == arch
    for m in METRICS:
        vals = sessions_m = seasons_hist.loc[arch_mask, m].values
        ages = seasons_hist.loc[arch_mask, 'season_age'].values
        coeffs = fit_poly2(ages, vals)
        if coeffs is not None:
            AGE_CURVES[arch][m] = coeffs

# Global fallback curves (all historical, regardless of archetype)
AGE_CURVES['global'] = {}
for m in METRICS:
    vals_g = seasons_hist[m].values
    ages_g = seasons_hist['season_age'].values
    coeffs_g = fit_poly2(ages_g, vals_g)
    if coeffs_g is not None:
        AGE_CURVES['global'][m] = coeffs_g

def poly_val(coeffs, age):
    return coeffs[0]*age**2 + coeffs[1]*age + coeffs[2]

print("\nAge Curve Coefficients (PPG example by archetype):")
for arch in AGE_CURVES:
    if 'season_ppg' in AGE_CURVES[arch]:
        c = AGE_CURVES[arch]['season_ppg']
        peak_age = -c[1] / (2 * c[0]) if c[0] < 0 else 27
        peak_val  = poly_val(c, peak_age)
        print(f"  {arch:16s}: peak_age≈{peak_age:.1f}, peak_ppg≈{peak_val:.1f}")

# ── 3. Injury Risk: Games Played per Season (availability discount) ──────────
# Compute games_fraction (games / 82) per season per player
seasons['games_fraction'] = seasons['season_g'] / 82.0

# Career average availability per player
player_avail = (
    seasons.groupby('player_name')['games_fraction']
    .agg(['mean', 'std', 'count'])
    .rename(columns={'mean': 'avg_avail', 'std': 'std_avail', 'count': 'n_seasons'})
    .reset_index()
)

# Injury risk tiers based on avg availability
def injury_tier(avg_avail):
    if avg_avail >= 0.88:   return 'low'
    elif avg_avail >= 0.78: return 'moderate'
    else:                    return 'high'

player_avail['injury_tier']       = player_avail['avg_avail'].apply(injury_tier)
# Availability multiplier for projected seasons (discount future games)
AVAIL_MULTIPLIER = {'low': 0.93, 'moderate': 0.83, 'high': 0.70}

print("\nActive Player Injury Risk (historical availability):")
for p in ACTIVE_5:
    row_a = player_avail[player_avail['player_name'] == p]
    if len(row_a):
        avg  = row_a['avg_avail'].values[0]
        tier = row_a['injury_tier'].values[0]
        mult = AVAIL_MULTIPLIER[tier]
        print(f"  {p:30s}: avg_avail={avg:.3f} → tier={tier}, discount_mult={mult:.2f}")
    else:
        print(f"  {p:30s}: no data — defaulting moderate")

# ── 4. Build Active Player Current Stats as Anchor Points ─────────────────
# Use most recent season's stats as the anchor
active_seasons = seasons[seasons['player_name'].isin(ACTIVE_5)]
latest_season = (
    active_seasons
    .sort_values('season_start_year')
    .groupby('player_name')
    .last()
    .reset_index()
)

# Current anchor dict: player → {metric: current_value, age: current_age}
active_anchors = {}
for p in ACTIVE_5:
    row_ls = latest_season[latest_season['player_name'] == p]
    if not len(row_ls):
        continue
    row_ls = row_ls.iloc[0]
    active_anchors[p] = {
        'current_age':     CURRENT_AGES[p],
        'archetype':       player_style[player_style['player_name'] == p]['archetype'].values[0]
                          if len(player_style[player_style['player_name'] == p]) else 'Two-Way Wing',
        'current_ppg':     row_ls.get('season_ppg', np.nan),
        'current_per':     row_ls.get('season_per', np.nan),
        'current_bpm':     row_ls.get('season_bpm', np.nan),
        'current_vorp':    row_ls.get('season_vorp', np.nan),
        'current_ws':      row_ls.get('season_ws', np.nan),
        'avail_mult':      AVAIL_MULTIPLIER.get(
                            player_avail[player_avail['player_name'] == p]['injury_tier'].values[0]
                            if len(player_avail[player_avail['player_name'] == p]) else 'moderate',
                           0.83),
    }

print("\nActive Player Current Anchors:")
for p, a in active_anchors.items():
    print(f"  {p:30s}: age={a['current_age']}, ppg={a['current_ppg']:.1f}, "
          f"bpm={a['current_bpm']:.1f}, arch={a['archetype']}, avail={a['avail_mult']:.2f}")

# Persist key objects for downstream projection block
age_curve_models        = AGE_CURVES
player_archetype_map    = player_style[['player_name', 'archetype']].set_index('player_name')['archetype'].to_dict()
player_availability     = player_avail.set_index('player_name')[['avg_avail', 'injury_tier']].to_dict('index')
active_player_anchors   = active_anchors
hist_seasons_for_curves = seasons_hist   # for downstream curve vis
kmeans_labels           = player_style[['player_name', 'archetype']].copy()
