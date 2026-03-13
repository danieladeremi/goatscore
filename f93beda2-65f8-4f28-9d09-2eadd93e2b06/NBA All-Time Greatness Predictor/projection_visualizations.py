
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Zerve Design System ──────────────────────────────────────────────────────
BG    = '#1D1D20'
FG    = '#fbfbff'
MUTED = '#909094'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': MUTED, 'axes.labelcolor': FG,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': FG, 'grid.color': '#333337',
    'grid.alpha': 0.4, 'font.family': 'DejaVu Sans',
})

# Per-player signature colors
PLAYER_COLORS = {
    'Victor Wembanyama':       '#A1C9F4',  # sky blue
    'Shai Gilgeous-Alexander': '#FFB482',  # orange
    'Nikola Jokic':            '#8DE5A1',  # green
    'LeBron James':            '#FF9F9B',  # coral
    'Stephen Curry':           '#D0BBFF',  # lavender
}
PLAYER_SHORT = {
    'Victor Wembanyama':       'Wembanyama',
    'Shai Gilgeous-Alexander': 'SGA',
    'Nikola Jokic':            'Jokic',
    'LeBron James':            'LeBron',
    'Stephen Curry':           'Curry',
}
SCENARIO_COLORS = {
    'optimistic':  '#8DE5A1',
    'base':        '#A1C9F4',
    'pessimistic': '#FF9F9B',
}
SCENARIO_ALPHA  = {'optimistic': 0.7, 'base': 0.9, 'pessimistic': 0.6}
SCENARIO_DASH   = {'optimistic': '--', 'base': '-', 'pessimistic': ':'}

proj = career_projections_df.copy()
active_anchors_local = active_player_anchors
ACTIVE_5 = ['Victor Wembanyama', 'Shai Gilgeous-Alexander',
            'Nikola Jokic', 'LeBron James', 'Stephen Curry']

# ── Chart 1: PPG Career Trajectory (base case) ──────────────────────────────
fig_ppg, ax = plt.subplots(figsize=(13, 7))
fig_ppg.patch.set_facecolor(BG)
ax.set_facecolor(BG)

base_proj = proj[proj['scenario'] == 'base']
for pname in ACTIVE_5:
    pdata = base_proj[base_proj['player'] == pname]
    if pdata.empty:
        continue
    col = PLAYER_COLORS[pname]
    sname = PLAYER_SHORT[pname]
    ax.plot(pdata['age'], pdata['proj_ppg'], color=col, linewidth=2.5, label=sname)
    # Anchor dot
    anchor = active_anchors_local[pname]
    ax.scatter([anchor['current_age']], [anchor['current_ppg']],
               color=col, s=90, zorder=5, edgecolors=FG, linewidths=0.8)
    # End label
    last_row = pdata.iloc[-1]
    ax.text(last_row['age'] + 0.2, last_row['proj_ppg'], sname,
            color=col, fontsize=8.5, va='center')

ax.axvline(27, color=MUTED, linestyle=':', linewidth=1.2, alpha=0.6)
ax.text(27.2, ax.get_ylim()[1] if ax.get_ylim()[1] > 10 else 35,
        'Typical peak\nage 27', fontsize=7.5, color=MUTED, va='top')
ax.set_xlabel('Age', color=FG, fontsize=12)
ax.set_ylabel('Projected PPG', color=FG, fontsize=12)
ax.set_title('Career PPG Trajectory — Active Stars (Base Case)\n'
             'Anchored to 2024-25 season stats', fontsize=13, fontweight='bold', color=FG, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.legend(fontsize=10, facecolor=BG, edgecolor=MUTED, labelcolor=FG, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
ppg_trajectory_chart = fig_ppg

# ── Chart 2: Scenario Fan (3 scenarios) for each player — PPG ───────────────
fig_fan, axes = plt.subplots(2, 3, figsize=(16, 9))
fig_fan.patch.set_facecolor(BG)
fig_fan.suptitle('Career PPG Projections: Optimistic / Base / Pessimistic Scenarios',
                 fontsize=13, fontweight='bold', color=FG, y=0.98)

for idx, pname in enumerate(ACTIVE_5):
    row_i, col_i = divmod(idx, 3)
    ax_s = axes[row_i][col_i]
    ax_s.set_facecolor(BG)
    anchor = active_anchors_local[pname]
    pcolor = PLAYER_COLORS[pname]
    sname  = PLAYER_SHORT[pname]

    for sc in ['pessimistic', 'base', 'optimistic']:
        sc_data = proj[(proj['player'] == pname) & (proj['scenario'] == sc)]
        if sc_data.empty:
            continue
        ages = sc_data['age'].values
        ppgs = sc_data['proj_ppg'].values
        ax_s.plot(ages, ppgs,
                  color=SCENARIO_COLORS[sc], linewidth=1.8 if sc == 'base' else 1.3,
                  linestyle=SCENARIO_DASH[sc], alpha=SCENARIO_ALPHA[sc],
                  label=sc.capitalize())

    # Shade confidence band (pessimistic to optimistic)
    pess = proj[(proj['player'] == pname) & (proj['scenario'] == 'pessimistic')]['proj_ppg'].values
    opti = proj[(proj['player'] == pname) & (proj['scenario'] == 'optimistic')]['proj_ppg'].values
    ages_band = proj[(proj['player'] == pname) & (proj['scenario'] == 'base')]['age'].values
    if len(pess) == len(opti) == len(ages_band):
        ax_s.fill_between(ages_band, pess, opti, alpha=0.12, color=pcolor)

    # Current anchor
    ax_s.scatter([anchor['current_age']], [anchor['current_ppg']],
                 color=pcolor, s=80, zorder=6, edgecolors=FG, linewidths=0.8)

    inj_tier = 'high' if anchor['avail_mult'] <= 0.70 else 'moderate' if anchor['avail_mult'] <= 0.83 else 'low'
    ax_s.set_title(f"{sname}  (age {anchor['current_age']}) | inj: {inj_tier}",
                   fontsize=10, color=pcolor, pad=6)
    ax_s.set_xlabel('Age', fontsize=9, color=MUTED)
    ax_s.set_ylabel('PPG', fontsize=9, color=MUTED)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)
    ax_s.spines['left'].set_color(MUTED)
    ax_s.spines['bottom'].set_color(MUTED)
    ax_s.grid(axis='y', alpha=0.25)

    if idx == 0:
        ax_s.legend(fontsize=8, facecolor=BG, edgecolor=MUTED, labelcolor=FG,
                    loc='upper right')

# Hide empty 6th subplot
axes[1][2].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
scenario_fan_chart = fig_fan

# ── Chart 3: Injury-Discounted Career Totals — Cumulative Points ─────────────
fig_cum, ax = plt.subplots(figsize=(13, 7))
fig_cum.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for pname in ACTIVE_5:
    for sc in ['pessimistic', 'base', 'optimistic']:
        sc_data = proj[(proj['player'] == pname) & (proj['scenario'] == sc)]
        if sc_data.empty:
            continue
        col = PLAYER_COLORS[pname]
        ax.plot(sc_data['age'], sc_data['cum_pts'] / 1000,
                color=col,
                linewidth=2.0 if sc == 'base' else 1.0,
                linestyle=SCENARIO_DASH[sc],
                alpha=SCENARIO_ALPHA[sc],
                label=f"{PLAYER_SHORT[pname]} ({sc})" if sc == 'base' else None)

# Historical greats reference lines
refs = [
    ('LeBron (actual)',  38984, '#FFB482', 0.5),
    ('Jordan (actual)',  32292, '#A1C9F4', 0.5),
    ('Kareem (actual)',  38387, '#8DE5A1', 0.5),
]
for label, pts, col, alpha in refs:
    ax.axhline(pts / 1000, color=col, linestyle='-.', linewidth=1.2, alpha=alpha)
    ax.text(22, pts / 1000 + 0.3, label, fontsize=7.5, color=col, alpha=0.8)

ax.set_xlabel('Age', color=FG, fontsize=12)
ax.set_ylabel('Career Points (thousands)', color=FG, fontsize=12)
ax.set_title('Projected Career Points — All Scenarios + Injury Discount\n'
             'Dashed = reference benchmarks (historical greats)',
             fontsize=12, fontweight='bold', color=FG, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.legend(fontsize=9, facecolor=BG, edgecolor=MUTED, labelcolor=FG, loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
cumulative_pts_chart = fig_cum

# ── Chart 4: Career End-State Comparison — Radar-style bar chart ─────────────
# Compare base-case projected totals vs historical greats
fig_end, ax = plt.subplots(figsize=(13, 6))
fig_end.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Historical greats' career VORP from era_adj_career_df
career_local = era_adj_career_df.copy()

compare_players = {
    'Michael Jordan':    {'pts': 32292, 'vorp': 100.0, 'ws': 214.0, 'is_hist': True},
    'LeBron James':      {'pts': 39787, 'vorp': 150.0, 'ws': 265.0, 'is_hist': False},
    'Kareem A-J':        {'pts': 38387, 'vorp': 94.0,  'ws': 273.4, 'is_hist': True},
    'Nikola Jokic (proj)': None,
    'SGA (proj)':          None,
    'Wembanyama (proj)':   None,
}

# Fill proj player end-states (base case last row)
proj_endstates = {}
for pname in ACTIVE_5:
    sc_data = proj[(proj['player'] == pname) & (proj['scenario'] == 'base')]
    if not sc_data.empty:
        last_r = sc_data.iloc[-1]
        proj_endstates[PLAYER_SHORT[pname]] = {
            'pts':  last_r['cum_pts'],
            'vorp': last_r['cum_vorp'],
            'ws':   last_r['cum_ws'],
        }

# Bar chart comparing projected career Win Shares (comparable metric)
bar_labels = [
    'Jordan\n(actual)', 'Kareem\n(actual)', 'LeBron\n(actual)',
    'Jokic\n(proj.)', 'SGA\n(proj.)', 'Wemby\n(proj.)',
    'Curry\n(proj.)'
]
bar_ws_vals = [
    214.0, 273.4, 265.0,
    proj_endstates.get('Jokic', {}).get('ws', 0),
    proj_endstates.get('SGA', {}).get('ws', 0),
    proj_endstates.get('Wembanyama', {}).get('ws', 0),
    proj_endstates.get('Curry', {}).get('ws', 0),
]
bar_colors = [
    '#909094', '#909094', '#909094',
    PLAYER_COLORS['Nikola Jokic'],
    PLAYER_COLORS['Shai Gilgeous-Alexander'],
    PLAYER_COLORS['Victor Wembanyama'],
    PLAYER_COLORS['Stephen Curry'],
]
bar_alphas = [0.55, 0.55, 0.55, 0.9, 0.9, 0.9, 0.9]

xpos = np.arange(len(bar_labels))
bars = ax.bar(xpos, bar_ws_vals, color=bar_colors,
              alpha=0.85, width=0.65, edgecolor='none')
for _b, _v, _a in zip(bars, bar_ws_vals, bar_alphas):
    _b.set_alpha(_a)
    ax.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 2,
            f'{_v:.0f}', ha='center', va='bottom', fontsize=9, color=FG)

ax.set_xticks(xpos)
ax.set_xticklabels(bar_labels, fontsize=10)
ax.set_ylabel('Career Win Shares', color=FG, fontsize=11)
ax.set_title('Projected Career Win Shares vs Historical Greats (Base Case)\n'
             'Grey = retired actuals | Colored = projected to retirement',
             fontsize=12, fontweight='bold', color=FG, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.grid(axis='y', alpha=0.3)

# Legend
_handle_hist = mpatches.Patch(color='#909094', alpha=0.55, label='Historical (actual)')
_handles_proj = [
    mpatches.Patch(color=PLAYER_COLORS[p], alpha=0.9, label=PLAYER_SHORT[p] + ' (projected)')
    for p in ['Nikola Jokic', 'Shai Gilgeous-Alexander', 'Victor Wembanyama', 'Stephen Curry']
]
ax.legend(handles=[_handle_hist] + _handles_proj, fontsize=8.5,
          facecolor=BG, edgecolor=MUTED, labelcolor=FG, loc='upper left')
plt.tight_layout()
career_endstate_comparison_chart = fig_end

# ── Chart 5: BPM Decline Curve — showing age effect ─────────────────────────
fig_bpm, ax = plt.subplots(figsize=(13, 6))
fig_bpm.patch.set_facecolor(BG)
ax.set_facecolor(BG)

base_proj_bpm = proj[proj['scenario'] == 'base']
for pname in ACTIVE_5:
    pdata = base_proj_bpm[base_proj_bpm['player'] == pname]
    if pdata.empty:
        continue
    col   = PLAYER_COLORS[pname]
    sname = PLAYER_SHORT[pname]
    anchor = active_anchors_local[pname]
    all_ages = [anchor['current_age']] + pdata['age'].tolist()
    all_bpm  = [anchor['current_bpm']] + pdata['proj_bpm'].tolist()
    ax.plot(all_ages, all_bpm, color=col, linewidth=2.2, label=sname)
    ax.scatter([anchor['current_age']], [anchor['current_bpm']],
               color=col, s=80, zorder=5, edgecolors=FG, linewidths=0.8)

ax.axhline(0, color=MUTED, linestyle='--', linewidth=1.0, alpha=0.6)
ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 20 else 21, 0.4,
        'League avg (BPM=0)', fontsize=8, color=MUTED, alpha=0.8)
ax.set_xlabel('Age', color=FG, fontsize=12)
ax.set_ylabel('Projected BPM', color=FG, fontsize=12)
ax.set_title('Career BPM Decline Curves — Active Stars (Base Case)\n'
             'Dots = current anchor; lines = projected trajectory',
             fontsize=12, fontweight='bold', color=FG, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MUTED)
ax.spines['bottom'].set_color(MUTED)
ax.legend(fontsize=10, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
bpm_decline_chart = fig_bpm

print("✓ 5 projection charts rendered:")
print("  1. ppg_trajectory_chart         — Career PPG base-case all 5 players")
print("  2. scenario_fan_chart           — Per-player 3-scenario fan plots")
print("  3. cumulative_pts_chart         — Injury-discounted career pts vs benchmarks")
print("  4. career_endstate_comparison_chart — Win Shares vs historical greats")
print("  5. bpm_decline_chart            — BPM decline trajectories")
