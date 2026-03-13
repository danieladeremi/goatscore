
"""
Visualization block for GOAT Score Bias Corrections.
Charts:
  1. Corrected Top-25 leaderboard (with before/after comparison)
  2. Key players delta chart (who rose, who fell)
  3. Individual Brilliance vs Team Win% scatter
  4. Era Credibility factor impact chart
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
_C_RED    = '#f04438'
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
PLAYER_COLORS_VIZ = {
    'Victor Wembanyama':       _C_BLUE,
    'Shai Gilgeous-Alexander': _C_ORANGE,
    'Nikola Jokic':            _C_GREEN,
    'LeBron James':            _C_CORAL,
    'Stephen Curry':           _C_LAVEN,
}

# ── Chart 1: Corrected Top-25 Leaderboard ───────────────────────────────────
top25_corr = corrected_goat_scores_df.head(25).copy().sort_values('goat_score_gated', ascending=True)

fig_corr_lb, ax_clb = plt.subplots(figsize=(14, 11))
fig_corr_lb.patch.set_facecolor(_BG)
ax_clb.set_facecolor(_BG)

_bar_c2 = []
for _, rr in top25_corr.iterrows():
    if rr['player_status'] == 'active':
        _bar_c2.append(PLAYER_COLORS_VIZ.get(rr['player_name'], _C_GOLD))
    else:
        _bar_c2.append(_C_BLUE)

_ypos2 = np.arange(len(top25_corr))
_bars_corr = ax_clb.barh(_ypos2, top25_corr['goat_score_gated'],
                         color=_bar_c2, alpha=0.88, height=0.72)

# Original score as ghost bar
for idx2, (_, rr) in enumerate(top25_corr.iterrows()):
    orig = rr['goat_score_current']
    corr = rr['goat_score_gated']
    # Show original as thin outline
    ax_clb.barh(idx2, orig, color='none',
                edgecolor=_MUTED, linewidth=1.0, height=0.72, alpha=0.45, linestyle=':')
    # Delta label
    delta2 = corr - orig
    delta_color = _C_TEAL if delta2 > 0.3 else (_C_RED if delta2 < -0.3 else _MUTED)
    ax_clb.text(corr + 0.4, idx2 + 0.02, f'{corr:.1f}', va='center', ha='left',
                fontsize=8.5, color=_FG, fontweight='bold')
    if abs(delta2) > 0.1:
        ax_clb.text(corr + 3.0, idx2 + 0.02, f'{delta2:+.1f}',
                    va='center', ha='left', fontsize=7.5, color=delta_color)

ax_clb.set_yticks(_ypos2)
ax_clb.set_yticklabels(
    [f"#{int(r['corrected_rank'])}  {r['player_name']}"
     for _, r in top25_corr.iterrows()], fontsize=9
)
ax_clb.set_xlabel('Corrected GOAT Score (0–100)', color=_FG, fontsize=11)
ax_clb.set_title(
    'NBA All-Time GOAT Score Leaderboard — Bias-Corrected Rankings\n'
    'Era Credibility + Supporting Cast + Career Gate | ★ = Active | dotted = original score',
    fontsize=13, fontweight='bold', color=_FG, pad=14
)
ax_clb.set_xlim(0, 100)
ax_clb.spines['top'].set_visible(False)
ax_clb.spines['right'].set_visible(False)
ax_clb.spines['left'].set_color(_MUTED)
ax_clb.spines['bottom'].set_color(_MUTED)
ax_clb.grid(axis='x', alpha=0.2)

_leg_patches = [
    mpatches.Patch(color=_C_BLUE,  label='Historical Player'),
    mpatches.Patch(color=_C_GOLD,  label='Active Player'),
    mpatches.Patch(color=_C_TEAL,  label='Score rose after correction'),
    mpatches.Patch(color=_C_RED,   label='Score fell after correction'),
]
ax_clb.legend(handles=_leg_patches, fontsize=8.5, facecolor=_BG, edgecolor=_MUTED,
              labelcolor=_FG, loc='lower right')
plt.tight_layout()
corrected_goat_leaderboard_chart = fig_corr_lb

# ── Chart 2: Before vs After — Key Players Delta ─────────────────────────────
_delta_players = [
    'LeBron James', 'Kareem Abdul-Jabbar', 'Michael Jordan',
    'Magic Johnson', 'Larry Bird', 'Nikola Jokic',
    'John Stockton', 'Karl Malone', 'Hakeem Olajuwon',
    'Charles Barkley', 'Kevin Garnett', 'Tim Duncan',
    'Shai Gilgeous-Alexander', 'Stephen Curry', 'Kevin Durant',
    'Wilt Chamberlain', 'Bill Russell', 'Dirk Nowitzki',
]

_delta_df = corrected_goat_scores_df[
    corrected_goat_scores_df['player_name'].isin(_delta_players)
].copy()
_delta_df = _delta_df.sort_values('score_delta', ascending=True)

fig_delta, ax_d = plt.subplots(figsize=(12, 9))
fig_delta.patch.set_facecolor(_BG)
ax_d.set_facecolor(_BG)

_delta_colors = [_C_TEAL if v > 0.1 else (_C_RED if v < -0.1 else _MUTED)
                 for v in _delta_df['score_delta']]
_dy = np.arange(len(_delta_df))
_dbars = ax_d.barh(_dy, _delta_df['score_delta'],
                   color=_delta_colors, alpha=0.88, height=0.7)

for bar_d, val_d in zip(_dbars, _delta_df['score_delta']):
    xl = bar_d.get_width()
    ha = 'left' if xl >= 0 else 'right'
    offset = 0.08 if xl >= 0 else -0.08
    ax_d.text(xl + offset, bar_d.get_y() + bar_d.get_height()/2,
              f'{val_d:+.2f}', va='center', ha=ha, fontsize=9, color=_FG)

ax_d.set_yticks(_dy)
ax_d.set_yticklabels(_delta_df['player_name'], fontsize=10)
ax_d.axvline(0, color=_MUTED, linewidth=1.2, alpha=0.8)
ax_d.set_xlabel('Score Change (Corrected − Original)', color=_FG, fontsize=11)
ax_d.set_title(
    'GOAT Score Impact: Who Rose & Who Fell After Bias Corrections\n'
    'Era Credibility Discount + Individual Brilliance + Career Gate',
    fontsize=13, fontweight='bold', color=_FG, pad=12
)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.spines['left'].set_color(_MUTED)
ax_d.spines['bottom'].set_color(_MUTED)
ax_d.grid(axis='x', alpha=0.2)

_rise_patch = mpatches.Patch(color=_C_TEAL, label='Score rises — carrying weak teams / pre-1980 era boost')
_fall_patch  = mpatches.Patch(color=_C_RED,  label='Score falls — dynasty team inflation / recency bias / career gate')
ax_d.legend(handles=[_rise_patch, _fall_patch], fontsize=8.5, facecolor=_BG,
            edgecolor=_MUTED, labelcolor=_FG, loc='lower right')
plt.tight_layout()
bias_correction_delta_chart = fig_delta

# ── Chart 3: Individual Brilliance vs Average Team Win% Scatter ──────────────
fig_ib, ax_ib = plt.subplots(figsize=(12, 8))
fig_ib.patch.set_facecolor(_BG)
ax_ib.set_facecolor(_BG)

_plot_players = corrected_goat_scores_df[
    corrected_goat_scores_df['corrected_rank'] <= 35
].copy()

_colors_sc = [PLAYER_COLORS_VIZ.get(p, _C_BLUE) if s == 'active' else _C_LAVEN
              for p, s in zip(_plot_players['player_name'], _plot_players['player_status'])]

_sc_sizes = (_plot_players['goat_score_gated'] / 80 * 300).clip(80, 400)

scatter = ax_ib.scatter(
    _plot_players['avg_team_win_pct'],
    _plot_players['individual_brilliance_score'],
    c=_colors_sc, s=_sc_sizes, alpha=0.85, zorder=5, edgecolors='none'
)

# Annotate key players
_key_annotate = [
    'Hakeem Olajuwon', 'Kevin Garnett', 'Charles Barkley',
    'John Stockton', 'Magic Johnson', 'Michael Jordan',
    'Nikola Jokic', 'LeBron James', 'Tim Duncan', 'Larry Bird',
    'Shai Gilgeous-Alexander', 'Karl Malone', 'Kareem Abdul-Jabbar',
]
for _, pr in _plot_players.iterrows():
    if pr['player_name'] in _key_annotate:
        _short_n = pr['player_name'].split(' ')[-1]
        ax_ib.annotate(
            _short_n,
            xy=(pr['avg_team_win_pct'], pr['individual_brilliance_score']),
            xytext=(7, 4), textcoords='offset points',
            fontsize=8.5, color=_FG, alpha=0.9,
        )

# Reference lines
ax_ib.axvline(0.42, color=_C_TEAL,   linewidth=1.2, linestyle='--', alpha=0.6, label='Weak team threshold (< .420)')
ax_ib.axvline(0.65, color=_C_ORANGE, linewidth=1.2, linestyle='--', alpha=0.6, label='Dynasty team threshold (> .650)')

ax_ib.set_xlabel('Average Team Win% During Career', color=_FG, fontsize=11)
ax_ib.set_ylabel('Individual Brilliance Score (0–100)', color=_FG, fontsize=11)
ax_ib.set_title(
    'Individual Brilliance vs Supporting Cast Strength\n'
    '(Bubble size = Corrected GOAT Score | Players who carried weak teams score higher)',
    fontsize=12, fontweight='bold', color=_FG, pad=12
)
ax_ib.spines['top'].set_visible(False)
ax_ib.spines['right'].set_visible(False)
ax_ib.spines['left'].set_color(_MUTED)
ax_ib.spines['bottom'].set_color(_MUTED)
ax_ib.grid(alpha=0.2)
ax_ib.legend(fontsize=9, facecolor=_BG, edgecolor=_MUTED, labelcolor=_FG)
plt.tight_layout()
individual_brilliance_scatter_chart = fig_ib

# ── Chart 4: Rankings Comparison — Original vs Corrected (Top 15) ─────────
# Show both old and new rank side-by-side for key names
_rank_players = [
    'LeBron James', 'Kareem Abdul-Jabbar', 'Michael Jordan',
    'Magic Johnson', 'Larry Bird', 'Nikola Jokic',
    'John Stockton', 'Karl Malone', 'Hakeem Olajuwon',
    'Tim Duncan', 'Kevin Durant', 'Stephen Curry',
]

# Get original ranks from goat_scores_df
_orig_ranks = goat_scores_df[['player_name', 'goat_rank']].copy()
_new_ranks  = corrected_goat_scores_df[['player_name', 'corrected_rank']].copy()
_rank_comp  = _orig_ranks.merge(_new_ranks, on='player_name')
_rank_comp  = _rank_comp[_rank_comp['player_name'].isin(_rank_players)].copy()
_rank_comp['rank_change'] = _rank_comp['goat_rank'] - _rank_comp['corrected_rank']
_rank_comp = _rank_comp.sort_values('corrected_rank')

fig_rc, ax_rc = plt.subplots(figsize=(11, 7))
fig_rc.patch.set_facecolor(_BG)
ax_rc.set_facecolor(_BG)

_xrc = np.arange(len(_rank_comp))
_w_rc = 0.35

_bars_orig = ax_rc.bar(_xrc - _w_rc/2, _rank_comp['goat_rank'],
                       _w_rc, color=_C_BLUE, alpha=0.7, label='Original Rank')
_bars_new  = ax_rc.bar(_xrc + _w_rc/2, _rank_comp['corrected_rank'],
                       _w_rc, color=_C_ORANGE, alpha=0.85, label='Corrected Rank')

# Invert y-axis so rank 1 is at top
ax_rc.invert_yaxis()

for b_o, r_o in zip(_bars_orig, _rank_comp['goat_rank']):
    ax_rc.text(b_o.get_x() + b_o.get_width()/2, b_o.get_height() - 0.3,
               f'#{int(r_o)}', ha='center', va='bottom', fontsize=9, color=_FG, alpha=0.9)
for b_n, r_n in zip(_bars_new, _rank_comp['corrected_rank']):
    ax_rc.text(b_n.get_x() + b_n.get_width()/2, b_n.get_height() - 0.3,
               f'#{int(r_n)}', ha='center', va='bottom', fontsize=9.5, color=_FG, fontweight='bold')

ax_rc.set_xticks(_xrc)
ax_rc.set_xticklabels(
    [r['player_name'].split(' ')[-1] for _, r in _rank_comp.iterrows()],
    fontsize=10, rotation=20, ha='right'
)
ax_rc.set_ylabel('All-Time Rank (#1 = best)', color=_FG, fontsize=11)
ax_rc.set_title(
    'Ranking Changes: Original vs Bias-Corrected GOAT Scores\n'
    '(lower bar = better rank | orange bars = corrected ranking)',
    fontsize=12, fontweight='bold', color=_FG, pad=12
)
ax_rc.spines['top'].set_visible(False)
ax_rc.spines['right'].set_visible(False)
ax_rc.spines['left'].set_color(_MUTED)
ax_rc.spines['bottom'].set_color(_MUTED)
ax_rc.grid(axis='y', alpha=0.2)
ax_rc.legend(fontsize=9.5, facecolor=_BG, edgecolor=_MUTED, labelcolor=_FG)
plt.tight_layout()
ranking_comparison_chart = fig_rc

print("✓ 4 bias-correction charts rendered:")
print("  1. corrected_goat_leaderboard_chart  — Top-25 corrected vs original scores")
print("  2. bias_correction_delta_chart       — Who rose/fell after corrections")
print("  3. individual_brilliance_scatter_chart — IB score vs team win%")
print("  4. ranking_comparison_chart          — Original vs corrected rank side-by-side")
print()
print("── Top 10 Corrected GOAT Rankings ──")
for _, rr in corrected_goat_scores_df.head(10).iterrows():
    orig_rk = goat_scores_df[goat_scores_df['player_name'] == rr['player_name']]['goat_rank'].values
    orig_rk_str = f"was #{int(orig_rk[0])}" if len(orig_rk) else ""
    print(f"  #{int(rr['corrected_rank'])}  {rr['player_name']:<30}  "
          f"Score={rr['goat_score_gated']:>5.1f}  IB={rr['individual_brilliance_score']:>5.1f}  "
          f"{orig_rk_str}")
