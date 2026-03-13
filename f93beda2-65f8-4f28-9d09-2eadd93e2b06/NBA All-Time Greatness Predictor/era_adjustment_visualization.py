
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Zerve design system
BG       = '#1D1D20'
FG       = '#fbfbff'
MUTED    = '#909094'
C_BLUE   = '#A1C9F4'
C_ORANGE = '#FFB482'
C_GREEN  = '#8DE5A1'
C_CORAL  = '#FF9F9B'
C_LAVEN  = '#D0BBFF'
C_GOLD   = '#ffd400'
C_TEAL   = '#17b26a'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG,
    'axes.edgecolor':    MUTED,
    'axes.labelcolor':   FG,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        FG,
    'grid.color':        '#333337',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
})

# ── Chart 1: Pace & Key Era Factors by Decade ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 6))
fig1.patch.set_facecolor(BG)

_dsum = era_factors_df.groupby('decade').agg(
    lg_pace     = ('lg_pace',     'mean'),
    efg_factor  = ('efg_factor',  'mean'),
    ppg_factor  = ('ppg_factor',  'mean'),
).reset_index().sort_values('decade')

_x    = range(len(_dsum))
_lbls = _dsum['decade'].tolist()

ax1.bar(_x, _dsum['lg_pace'], color=C_BLUE, alpha=0.85, label='League Avg Pace (poss/48)', width=0.55)

ax2 = ax1.twinx()
ax2.set_facecolor(BG)
ax2.tick_params(colors=MUTED)
ax2.spines['right'].set_color(MUTED)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_color(MUTED)
ax2.plot(_x, _dsum['efg_factor'],  color=C_ORANGE, linewidth=2.5, marker='o', markersize=7,
         label='eFG% Adj Factor (right axis)')
ax2.plot(_x, _dsum['ppg_factor'],  color=C_GREEN,  linewidth=2.5, marker='s', markersize=7,
         label='PPG Adj Factor (right axis)')
ax2.axhline(1.0, color=MUTED, linestyle='--', linewidth=1.2, alpha=0.7)
ax2.set_ylabel('Adjustment Factor (1.0 = modern baseline)', color=FG, fontsize=10)
ax2.yaxis.label.set_color(FG)
ax2.tick_params(axis='y', colors=MUTED)

ax1.set_xticks(list(_x))
ax1.set_xticklabels(_lbls, fontsize=10)
ax1.set_ylabel('Avg Possessions per 48 min', color=FG, fontsize=11)
ax1.set_xlabel('Decade', color=FG, fontsize=11)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_color(MUTED)

_handles1 = [mpatches.Patch(color=C_BLUE, label='League Avg Pace (left axis)')]
_lines2   = [
    plt.Line2D([0],[0], color=C_ORANGE, linewidth=2.5, marker='o', markersize=7, label='eFG% Adj Factor (right)'),
    plt.Line2D([0],[0], color=C_GREEN,  linewidth=2.5, marker='s', markersize=7, label='PPG Adj Factor (right)'),
    plt.Line2D([0],[0], color=MUTED,    linewidth=1.2, linestyle='--', label='Modern Baseline (1.0)'),
]
ax1.legend(handles=_handles1 + _lines2, loc='upper right', fontsize=9,
           facecolor=BG, edgecolor=MUTED, labelcolor=FG)

ax1.set_title('NBA Era Adjustment Factors by Decade\n1940s → 2020s', fontsize=14, fontweight='bold', color=FG, pad=14)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()

era_factors_by_decade_chart = fig1

# ── Chart 2: Raw vs Era-Adjusted PPG — Key Players ─────────────────────────
_key_players = [
    'Bill Russell', 'Wilt Chamberlain', 'Elgin Baylor',
    'Kareem Abdul-Jabbar', 'Magic Johnson', 'Michael Jordan',
    'LeBron James', 'Nikola Jokic', 'Stephen Curry',
    'Shai Gilgeous-Alexander', 'Victor Wembanyama'
]

_ppg_data = era_adj_career_df[
    era_adj_career_df['player_name'].isin(_key_players) &
    era_adj_career_df['adj_career_ppg'].notna()
].copy()
_ppg_data = _ppg_data.sort_values('min_season_year')

# Build labels with era
_ppg_data['label'] = _ppg_data.apply(
    lambda r: f"{r['player_name'].split(' ')[-1]}\n({int(r['min_season_year'])}s)"
    if pd.notna(r['min_season_year']) else r['player_name'].split(' ')[-1],
    axis=1
)

fig2, ax = plt.subplots(figsize=(14, 6))
fig2.patch.set_facecolor(BG)
ax.set_facecolor(BG)

_n     = len(_ppg_data)
_xpos  = np.arange(_n)
_width = 0.38

_raw_vals = _ppg_data['career_ppg'].fillna(0).values
_adj_vals = _ppg_data['adj_career_ppg'].values
_lbls     = _ppg_data['label'].values

bars1 = ax.bar(_xpos - _width/2, _raw_vals, _width, color=C_BLUE,   alpha=0.85, label='Raw Career PPG')
bars2 = ax.bar(_xpos + _width/2, _adj_vals, _width, color=C_ORANGE, alpha=0.85, label='Era-Adjusted PPG (modern equiv.)')

# Value labels on bars
for _b, _v in zip(bars1, _raw_vals):
    if _v > 0:
        ax.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 0.3,
                f'{_v:.0f}', ha='center', va='bottom', fontsize=8, color=C_BLUE)
for _b, _v in zip(bars2, _adj_vals):
    ax.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 0.3,
            f'{_v:.0f}', ha='center', va='bottom', fontsize=8, color=C_ORANGE)

ax.set_xticks(_xpos)
ax.set_xticklabels(_lbls, fontsize=9)
ax.set_ylabel('Points Per Game', color=FG, fontsize=11)
ax.set_title('Raw vs Era-Adjusted Career PPG\n(Adjusted to 2015–2024 Modern Baseline)', fontsize=13, fontweight='bold', color=FG, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=10, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
plt.tight_layout()

raw_vs_adj_ppg_chart = fig2

# ── Chart 3: Pace Timeline (1946–2024) ─────────────────────────────────────
fig3, ax = plt.subplots(figsize=(13, 5))
fig3.patch.set_facecolor(BG)
ax.set_facecolor(BG)

_ef_sorted = era_factors_df.sort_values('season_start_year')
ax.plot(_ef_sorted['season_start_year'], _ef_sorted['lg_pace'],
        color=C_BLUE, linewidth=2.2, alpha=0.9)
ax.fill_between(_ef_sorted['season_start_year'], _ef_sorted['lg_pace'],
                alpha=0.18, color=C_BLUE)

# Annotate key rule changes
_events = [
    (1954, 'Shot clock\nintroduced', 120),
    (1979, '3-pt line\ndebuts',      95),
    (1994, 'Hand-check\nenforced',   105),
    (2004, 'Post-Jordan\nrules',     88),
    (2015, '3-pt\nrevolution',       88),
]
for _yr, _lbl, _ypos in _events:
    ax.axvline(_yr, color=MUTED, linestyle=':', linewidth=1.1, alpha=0.6)
    ax.text(_yr + 0.3, _ypos, _lbl, fontsize=7.5, color=MUTED, va='bottom')

ax.axhline(MODERN_PACE, color=C_GOLD, linestyle='--', linewidth=1.4, alpha=0.8)
ax.text(2024.5, MODERN_PACE + 0.5, f'Modern avg\n{MODERN_PACE:.1f}', fontsize=8, color=C_GOLD, va='bottom')

ax.set_xlabel('Season Start Year', color=FG, fontsize=11)
ax.set_ylabel('League Avg Pace (Poss/48 min)', color=FG, fontsize=11)
ax.set_title('NBA League-Average Pace 1946–2024\n(Possessions per 48 minutes)', fontsize=13, fontweight='bold', color=FG, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.grid(alpha=0.25)
plt.tight_layout()

pace_timeline_chart = fig3

# ── Chart 4: eFG% over Time ─────────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(13, 5))
fig4.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(_ef_sorted['season_start_year'], _ef_sorted['lg_efg_pct'],
        color=C_TEAL, linewidth=2.2, alpha=0.9, label='League eFG%')
ax.fill_between(_ef_sorted['season_start_year'], _ef_sorted['lg_efg_pct'],
                alpha=0.15, color=C_TEAL)
ax.plot(_ef_sorted['season_start_year'], _ef_sorted['lg_tpar'],
        color=C_CORAL, linewidth=2.0, alpha=0.85, linestyle='-', label='3-Point Attempt Rate (3PAr)')

ax.axhline(MODERN_EFG, color=C_GOLD, linestyle='--', linewidth=1.3, alpha=0.7)
ax.text(2024.5, MODERN_EFG, f'Modern\neFG%\n{MODERN_EFG:.3f}', fontsize=7.5, color=C_GOLD)

ax.set_xlabel('Season Start Year', color=FG, fontsize=11)
ax.set_ylabel('Rate', color=FG, fontsize=11)
ax.set_title('NBA League eFG% & 3-Point Attempt Rate 1946–2024\n(Era Shooting Environment Context)', fontsize=13, fontweight='bold', color=FG, pad=12)
ax.legend(fontsize=10, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.grid(alpha=0.25)
plt.tight_layout()

efg_timeline_chart = fig4

print("✓ All 4 era adjustment charts rendered successfully.")
print("  1. era_factors_by_decade_chart — Pace bars + eFG/PPG factor lines by decade")
print("  2. raw_vs_adj_ppg_chart        — Side-by-side raw vs adjusted PPG for key players")
print("  3. pace_timeline_chart         — League pace 1946–2024 with rule-change annotations")
print("  4. efg_timeline_chart          — eFG% and 3PAr trend 1946–2024")
