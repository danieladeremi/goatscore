
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Zerve Design System ──────────────────────────────────────────────────────
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
_C_PINK   = '#F7B6D2'

plt.rcParams.update({
    'figure.facecolor': _BG,
    'axes.facecolor':   _BG,
    'axes.edgecolor':   _MUTED,
    'axes.labelcolor':  _FG,
    'xtick.color':      _MUTED,
    'ytick.color':      _MUTED,
    'text.color':       _FG,
    'grid.color':       '#333337',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})

ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
            'Nikola Jokic', 'LeBron James', 'Stephen Curry']

PLAYER_COLORS = {
    'Victor Wembanyama':       _C_BLUE,
    'Shai Gilgeous-Alexander': _C_ORANGE,
    'Nikola Jokic':            _C_GREEN,
    'LeBron James':            _C_CORAL,
    'Stephen Curry':           _C_LAVEN,
}

# ── Chart 1: Top-25 GOAT Score Leaderboard (Horizontal Bar) ─────────────────
top25 = goat_scores_df.head(25).copy().sort_values('goat_score_current', ascending=True)

fig_lb, ax_lb = plt.subplots(figsize=(13, 11))
fig_lb.patch.set_facecolor(_BG)
ax_lb.set_facecolor(_BG)

_bar_colors = []
for _, row in top25.iterrows():
    if row['player_status'] == 'active':
        _bar_colors.append(PLAYER_COLORS.get(row['player_name'], _C_GOLD))
    else:
        _bar_colors.append(_C_BLUE)

_ypos = np.arange(len(top25))
bars_lb = ax_lb.barh(_ypos, top25['goat_score_current'], color=_bar_colors,
                     alpha=0.88, height=0.72)

# Add projected score as outline for active players
for idx, (_, row) in enumerate(top25.iterrows()):
    if row['player_status'] == 'active' and row['goat_score_projected'] > row['goat_score_current']:
        ax_lb.barh(idx, row['goat_score_projected'],
                   color='none', edgecolor=PLAYER_COLORS.get(row['player_name'], _C_GOLD),
                   linewidth=1.5, height=0.72, linestyle='--', alpha=0.7)

# Value labels
for bar, (_, row) in zip(bars_lb, top25.iterrows()):
    x = bar.get_width()
    status_tag = ' ★' if row['player_status'] == 'active' else ''
    ax_lb.text(x + 0.5, bar.get_y() + bar.get_height()/2,
               f"{x:.1f}{status_tag}", va='center', ha='left', fontsize=8.5, color=_FG)

ax_lb.set_yticks(_ypos)
ax_lb.set_yticklabels([f"#{int(r['goat_rank'])}  {r['player_name']}" for _, r in top25.iterrows()],
                       fontsize=9)
ax_lb.set_xlabel('GOAT Score (0–100)', color=_FG, fontsize=11)
ax_lb.set_title('NBA All-Time GOAT Score Leaderboard\nTop 25 | Equal Weights | ★ = Active Player',
                fontsize=14, fontweight='bold', color=_FG, pad=14)
ax_lb.set_xlim(0, 95)
ax_lb.spines['top'].set_visible(False)
ax_lb.spines['right'].set_visible(False)
ax_lb.spines['left'].set_color(_MUTED)
ax_lb.spines['bottom'].set_color(_MUTED)
ax_lb.grid(axis='x', alpha=0.25)

_legend_patches = [
    mpatches.Patch(color=_C_BLUE, label='Historical Player'),
    mpatches.Patch(color=_C_GOLD, label='Active Player'),
]
ax_lb.legend(handles=_legend_patches, fontsize=9, facecolor=_BG,
             edgecolor=_MUTED, labelcolor=_FG, loc='lower right')
plt.tight_layout()
goat_leaderboard_chart = fig_lb

# ── Chart 2: Current vs Projected GOAT — Active 5 ───────────────────────────
fig_cv, ax_cv = plt.subplots(figsize=(12, 7))
fig_cv.patch.set_facecolor(_BG)
ax_cv.set_facecolor(_BG)

_short = {
    'Victor Wembanyama': 'Wemby',
    'Shai Gilgeous-Alexander': 'SGA',
    'Nikola Jokic': 'Jokic',
    'LeBron James': 'LeBron',
    'Stephen Curry': 'Curry',
}

_act5_data = goat_scores_df[goat_scores_df['player_name'].isin(ACTIVE_5)].copy()
_act5_data = _act5_data.set_index('player_name').loc[ACTIVE_5].reset_index()

_xpos2 = np.arange(5)
_width = 0.35

bars_curr = ax_cv.bar(_xpos2 - _width/2, _act5_data['goat_score_current'],
                      _width, color=[PLAYER_COLORS[p] for p in _act5_data['player_name']],
                      alpha=0.9, label='Current GOAT Score')
bars_proj = ax_cv.bar(_xpos2 + _width/2, _act5_data['goat_score_projected'],
                      _width, color=[PLAYER_COLORS[p] for p in _act5_data['player_name']],
                      alpha=0.55, label='Projected GOAT Score (Base Case)', edgecolor=_FG,
                      linewidth=1.0, linestyle='--')

for b, v in zip(bars_curr, _act5_data['goat_score_current']):
    ax_cv.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
               f'{v:.1f}', ha='center', va='bottom', fontsize=9.5, color=_FG, fontweight='bold')
for b, v, c in zip(bars_proj, _act5_data['goat_score_projected'], _act5_data['goat_score_current']):
    ax_cv.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
               f'{v:.1f}', ha='center', va='bottom', fontsize=9.5, color=_MUTED)
    # Delta annotation
    delta = v - c
    ax_cv.text(b.get_x() + b.get_width()/2, b.get_height() + 2.5,
               f'+{delta:.1f}', ha='center', va='bottom', fontsize=8, color=_C_GOLD)

# Historical greats reference lines
for ref_name, ref_color in [('Michael Jordan', _C_LAVEN), ('Kareem Abdul-Jabbar', _C_TEAL)]:
    ref_row = goat_scores_df[goat_scores_df['player_name'] == ref_name]
    if len(ref_row):
        ref_val = ref_row['goat_score_current'].values[0]
        ax_cv.axhline(ref_val, color=ref_color, linestyle=':', linewidth=1.3, alpha=0.7)
        ax_cv.text(4.7, ref_val + 0.4, ref_name.split(' ')[-1], fontsize=8,
                   color=ref_color, ha='right', va='bottom')

ax_cv.set_xticks(_xpos2)
ax_cv.set_xticklabels([_short[p] for p in _act5_data['player_name']], fontsize=11)
ax_cv.set_ylabel('GOAT Score (0–100)', color=_FG, fontsize=11)
ax_cv.set_title('Active Stars: Current vs Projected GOAT Score\n(Base-Case Career Projection | Gold = Projected Gain)',
                fontsize=13, fontweight='bold', color=_FG, pad=12)
ax_cv.set_ylim(0, 100)
ax_cv.spines['top'].set_visible(False)
ax_cv.spines['right'].set_visible(False)
ax_cv.spines['left'].set_color(_MUTED)
ax_cv.spines['bottom'].set_color(_MUTED)
ax_cv.grid(axis='y', alpha=0.2)
ax_cv.legend(fontsize=9, facecolor=_BG, edgecolor=_MUTED, labelcolor=_FG)
plt.tight_layout()
goat_current_vs_projected_chart = fig_cv

# ── Chart 3: Pillar Breakdown Radar — Key Players ────────────────────────────
def radar_chart(players_data, title, colors):
    """Create a radar/spider chart for pillar scores."""
    categories = ['Volume\nLongevity', 'Peak\nDominance', 'Context\nValue', 'Honors\nRecognition']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color=_FG)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7, color=_MUTED)
    ax.grid(color='#333337', linewidth=0.8, alpha=0.6)
    ax.spines['polar'].set_color(_MUTED)

    for pname, color in zip(players_data, colors):
        row = goat_scores_df[goat_scores_df['player_name'] == pname]
        if row.empty:
            continue
        row = row.iloc[0]
        values = [row['pillar_volume'], row['pillar_peak'],
                  row['pillar_context'], row['pillar_honors']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.2, color=color, label=pname, markersize=5)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.15), fontsize=9,
              facecolor=_BG, edgecolor=_MUTED, labelcolor=_FG)
    ax.set_title(title, fontsize=13, fontweight='bold', color=_FG, pad=18)
    plt.tight_layout()
    return fig

# Radar 1: Top historical greats
_hist_top6 = ['Michael Jordan', 'LeBron James', 'Kareem Abdul-Jabbar',
              'Wilt Chamberlain', 'Magic Johnson', 'Tim Duncan']
_hist_colors = [_C_LAVEN, _C_CORAL, _C_TEAL, _C_ORANGE, _C_GREEN, _C_BLUE]
fig_radar_hist = radar_chart(
    _hist_top6,
    'GOAT Score Pillar Breakdown — Historical Greats',
    _hist_colors
)
goat_radar_historical_chart = fig_radar_hist

# Radar 2: Active stars
fig_radar_act = radar_chart(
    ACTIVE_5,
    'GOAT Score Pillar Breakdown — Active Stars',
    list(PLAYER_COLORS.values())
)
goat_radar_active_chart = fig_radar_act

# ── Chart 4: Comparable Player Similarity — Active Stars ────────────────────
fig_sim, axes_sim = plt.subplots(1, 5, figsize=(18, 7))
fig_sim.patch.set_facecolor(_BG)

for col_i, pname in enumerate(ACTIVE_5):
    ax_s = axes_sim[col_i]
    ax_s.set_facecolor(_BG)
    comps = goat_comps_df[goat_comps_df['active_player'] == pname].sort_values(
        'comp_rank', ascending=True
    )
    if comps.empty:
        ax_s.axis('off')
        continue

    _short_name = _short[pname]
    _comp_names = [r['comp_player'].split(' ')[-1] for _, r in comps.iterrows()]
    _comp_sims  = comps['similarity_pct'].values
    _comp_goat  = comps['comp_goat_score'].values

    _y = np.arange(len(_comp_names))
    _bar_c = [PLAYER_COLORS[pname]] * len(_comp_names)
    bars_s = ax_s.barh(_y, _comp_sims, color=_bar_c, alpha=0.82, height=0.65)

    # Overlay GOAT score as marker
    ax_s_twin = ax_s.twiny()
    ax_s_twin.set_facecolor(_BG)
    ax_s_twin.scatter(_comp_goat, _y + 0.15, color=_C_GOLD, s=60, zorder=5, marker='D',
                      label='GOAT Score', alpha=0.9)
    ax_s_twin.set_xlim(0, 100)
    ax_s_twin.tick_params(colors=_MUTED, labelsize=7)
    ax_s_twin.spines['top'].set_color(_MUTED)
    ax_s_twin.spines['right'].set_color(_MUTED)
    ax_s_twin.set_xlabel('Comp GOAT', fontsize=7, color=_C_GOLD)

    # Similarity % labels
    for bar, sv in zip(bars_s, _comp_sims):
        ax_s.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                  f'{sv:.0f}%', va='center', ha='left', fontsize=8, color=_FG)

    ax_s.set_yticks(_y)
    ax_s.set_yticklabels(_comp_names, fontsize=9)
    ax_s.set_xlim(0, 110)
    ax_s.set_xlabel('Similarity %', fontsize=9, color=_FG)
    ax_s.set_title(_short_name, fontsize=11, fontweight='bold',
                   color=PLAYER_COLORS[pname], pad=8)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)
    ax_s.spines['left'].set_color(_MUTED)
    ax_s.spines['bottom'].set_color(_MUTED)

fig_sim.suptitle('Historical Comparable Players per Active Star\n(Cosine Similarity on Era-Adj Stats + Advanced Metrics | ◆ = Comp GOAT Score)',
                  fontsize=12, fontweight='bold', color=_FG, y=1.02)
plt.tight_layout()
goat_comps_similarity_chart = fig_sim

# ── Chart 5: Weight Sensitivity — How Sliders Shift Top-8 Rankings ──────────
_key8 = ['Michael Jordan', 'LeBron James', 'Kareem Abdul-Jabbar', 'Nikola Jokic',
          'Wilt Chamberlain', 'Stephen Curry', 'Shai Gilgeous-Alexander', 'Victor Wembanyama']
_k8_colors = [_C_LAVEN, _C_CORAL, _C_TEAL, _C_GREEN, _C_ORANGE, _C_LAVEN, _C_BLUE, _C_PINK]

_weight_scenarios = {
    'Equal\n(25/25/25/25)': {'volume_longevity': 0.25, 'peak_dominance': 0.25,
                              'context_value': 0.25, 'honors_recognition': 0.25},
    'Volume\nHeavy': {'volume_longevity': 0.55, 'peak_dominance': 0.15,
                      'context_value': 0.15, 'honors_recognition': 0.15},
    'Peak\nHeavy': {'volume_longevity': 0.15, 'peak_dominance': 0.55,
                    'context_value': 0.15, 'honors_recognition': 0.15},
    'Context\nHeavy': {'volume_longevity': 0.15, 'peak_dominance': 0.15,
                       'context_value': 0.55, 'honors_recognition': 0.15},
    'Honors\nHeavy': {'volume_longevity': 0.15, 'peak_dominance': 0.15,
                      'context_value': 0.15, 'honors_recognition': 0.55},
}

fig_ws, ax_ws = plt.subplots(figsize=(14, 7))
fig_ws.patch.set_facecolor(_BG)
ax_ws.set_facecolor(_BG)

_sc_names = list(_weight_scenarios.keys())
_xw = np.arange(len(_sc_names))
_bar_width = 0.09
_offset_map = {p: (i - len(_key8)/2 + 0.5) * _bar_width for i, p in enumerate(_key8)}

for p, color in zip(_key8, _k8_colors):
    _sc_scores = []
    row_p = goat_career[goat_career['player_name'] == p]
    if row_p.empty:
        continue
    row_p = row_p.iloc[0]
    for sc_name, sc_w in _weight_scenarios.items():
        _sc_scores.append(compute_goat_score(row_p, sc_w))
    _x_off = _offset_map[p]
    ax_ws.bar(_xw + _x_off, _sc_scores, _bar_width,
              color=color, alpha=0.85,
              label=p.split(' ')[-1] if p != 'Shai Gilgeous-Alexander' else 'SGA')

ax_ws.set_xticks(_xw)
ax_ws.set_xticklabels(_sc_names, fontsize=10)
ax_ws.set_ylabel('GOAT Score', color=_FG, fontsize=11)
ax_ws.set_title('Weight Sensitivity Analysis — How Pillar Sliders Shift Scores\n(Each bar group = one weight scenario; bars within = individual players)',
                fontsize=12, fontweight='bold', color=_FG, pad=12)
ax_ws.spines['top'].set_visible(False)
ax_ws.spines['right'].set_visible(False)
ax_ws.spines['left'].set_color(_MUTED)
ax_ws.spines['bottom'].set_color(_MUTED)
ax_ws.grid(axis='y', alpha=0.2)
ax_ws.legend(fontsize=8.5, facecolor=_BG, edgecolor=_MUTED, labelcolor=_FG,
             ncol=4, loc='upper right')
plt.tight_layout()
goat_weight_sensitivity_chart = fig_ws

print("✓ 5 GOAT Score charts rendered:")
print("  1. goat_leaderboard_chart          — Top-25 horizontal bar with current/projected")
print("  2. goat_current_vs_projected_chart — Active 5 current vs projected GOAT score")
print("  3. goat_radar_historical_chart     — Spider/radar pillar breakdown for historical greats")
print("  4. goat_radar_active_chart         — Spider/radar pillar breakdown for active stars")
print("  5. goat_comps_similarity_chart     — Comparable player similarity bars per active player")
print("  6. goat_weight_sensitivity_chart   — How slider weights shift rankings across key players")
