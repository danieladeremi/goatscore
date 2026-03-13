import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title='NBA GOAT Analyzer', page_icon='??', layout='wide')

BASE_DIR = Path(__file__).resolve().parent
CAREER_PATH = BASE_DIR / 'nba_era_adjusted_career.csv'
SEASONS_PATH = BASE_DIR / 'nba_era_adjusted_seasons.csv'


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return (s - lo) / (hi - lo) * 100.0


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    career = pd.read_csv(CAREER_PATH)
    seasons = pd.read_csv(SEASONS_PATH)
    return career, seasons


def compute_peak_pillar(seasons: pd.DataFrame) -> pd.DataFrame:
    use_cols = ['season_per', 'season_ts_pct', 'season_bpm']
    tmp = seasons[['player_name', 'season_start_year'] + use_cols].copy()
    tmp['season_start_year'] = pd.to_numeric(tmp['season_start_year'], errors='coerce')

    rows = []
    for player, grp in tmp.groupby('player_name', dropna=True):
        grp = grp.sort_values('season_start_year')
        row = {'player_name': player}
        for col in use_cols:
            vals = pd.to_numeric(grp[col], errors='coerce').dropna().to_numpy()
            if len(vals) == 0:
                best = np.nan
            elif len(vals) < 3:
                best = float(np.mean(vals))
            else:
                best = float(np.max(np.convolve(vals, np.ones(3) / 3, mode='valid')))
            row[f'peak3_{col}'] = best
        rows.append(row)

    peak = pd.DataFrame(rows)
    for col in use_cols:
        peak[f'peak3_{col}_norm'] = _minmax(peak[f'peak3_{col}'])

    peak['pillar_peak'] = peak[[f'peak3_{c}_norm' for c in use_cols]].mean(axis=1)
    return peak[['player_name', 'pillar_peak']]


def compute_scores(career: pd.DataFrame, seasons: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    df = career.copy()

    # Volume/Longevity
    df['vol_pts'] = _minmax(df['adj_career_pts'])
    df['vol_trb'] = _minmax(df['adj_career_trb'])
    df['vol_ast'] = _minmax(df['adj_career_ast'])
    df['vol_games'] = _minmax(df['career_g'])
    df['vol_seasons'] = _minmax(df['n_seasons'])
    df['pillar_volume'] = df[['vol_pts', 'vol_trb', 'vol_ast', 'vol_games', 'vol_seasons']].mean(axis=1)

    # Peak Dominance
    peak = compute_peak_pillar(seasons)
    df = df.merge(peak, on='player_name', how='left')
    df['pillar_peak'] = df['pillar_peak'].fillna(df['pillar_peak'].median())

    # Context Value
    df['ws_above_repl'] = pd.to_numeric(df['adj_career_ws'], errors='coerce').fillna(0.0) - (
        pd.to_numeric(df['n_seasons'], errors='coerce').fillna(0.0) * 2.0
    )
    df['ctx_vorp'] = _minmax(df['career_vorp'])
    df['ctx_bpm'] = _minmax(df['career_bpm'])
    df['ctx_ws_repl'] = _minmax(df['ws_above_repl'])
    df['pillar_context'] = df[['ctx_vorp', 'ctx_bpm', 'ctx_ws_repl']].mean(axis=1)

    # Honors/Recognition
    all_nba_weighted = (
        pd.to_numeric(df['all_nba_1st'], errors='coerce').fillna(0.0) * 8.0
        + pd.to_numeric(df['all_nba_2nd'], errors='coerce').fillna(0.0) * 4.0
        + pd.to_numeric(df['all_nba_3rd'], errors='coerce').fillna(0.0) * 1.0
    )
    df['hon_mvp'] = _minmax(df['mvp'])
    df['hon_allnba'] = _minmax(all_nba_weighted)
    df['hon_champ'] = _minmax(df['championships'])
    df['hon_allstar'] = _minmax(df['allstar'])
    df['pillar_honors'] = (
        df['hon_mvp'] * 0.35
        + df['hon_allnba'] * 0.30
        + df['hon_champ'] * 0.20
        + df['hon_allstar'] * 0.15
    )

    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        norm = {k: 0.25 for k in weights}
    else:
        norm = {k: max(v, 0.0) / total for k, v in weights.items()}

    df['goat_score'] = (
        df['pillar_volume'] * norm['volume_longevity']
        + df['pillar_peak'] * norm['peak_dominance']
        + df['pillar_context'] * norm['context_value']
        + df['pillar_honors'] * norm['honors_recognition']
    )

    df = df.sort_values('goat_score', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df


def main() -> None:
    st.title('NBA GOAT Analyzer')
    st.caption('Interactive four-pillar ranking built from your local pipeline outputs.')

    if not CAREER_PATH.exists() or not SEASONS_PATH.exists():
        st.error('Missing CSV outputs. Run: python run_pipeline.py --mode full')
        st.stop()

    career, seasons = load_data()

    with st.sidebar:
        st.header('Pillar Weights')
        w_volume = st.slider('Volume/Longevity', 0.0, 1.0, 0.25, 0.01)
        w_peak = st.slider('Peak Dominance', 0.0, 1.0, 0.25, 0.01)
        w_context = st.slider('Context Value', 0.0, 1.0, 0.25, 0.01)
        w_honors = st.slider('Honors/Recognition', 0.0, 1.0, 0.25, 0.01)
        show_active_only = st.checkbox('Show active players only', value=False)

    weights = {
        'volume_longevity': w_volume,
        'peak_dominance': w_peak,
        'context_value': w_context,
        'honors_recognition': w_honors,
    }

    scored = compute_scores(career, seasons, weights)
    view = scored.copy()
    if show_active_only:
        view = view[view['player_status'].eq('active')].reset_index(drop=True)
        view['rank'] = np.arange(1, len(view) + 1)

    top = view.iloc[0] if not view.empty else None
    active_top10 = int(view.head(10)['player_status'].eq('active').sum()) if not view.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Players Ranked', f"{len(view)}")
    c2.metric('Top Score', f"{top['goat_score']:.1f}" if top is not None else '-')
    c3.metric('Current #1', str(top['player_name']) if top is not None else '-')
    c4.metric('Active In Top 10', f"{active_top10}")
    c5.metric('Median Score', f"{view['goat_score'].median():.1f}" if not view.empty else '-')

    tab1, tab2, tab3 = st.tabs(['GOAT Rankings', 'Player Deep Dive', 'Methodology'])

    with tab1:
        display_cols = [
            'rank', 'player_name', 'player_status', 'goat_score',
            'pillar_volume', 'pillar_peak', 'pillar_context', 'pillar_honors',
            'mvp', 'championships', 'allstar'
        ]
        df_show = view[display_cols].copy()
        for col in ['goat_score', 'pillar_volume', 'pillar_peak', 'pillar_context', 'pillar_honors']:
            df_show[col] = df_show[col].round(2)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.subheader('Top 25 Score Distribution')
        top25 = view.head(25).sort_values('goat_score')
        st.bar_chart(top25.set_index('player_name')['goat_score'])

    with tab2:
        player = st.selectbox('Select a player', options=view['player_name'].tolist(), index=0)
        row = view[view['player_name'] == player].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric('Rank', f"#{int(row['rank'])}")
        d2.metric('GOAT Score', f"{row['goat_score']:.2f}")
        d3.metric('Status', str(row['player_status']).title())
        d4.metric('Seasons', f"{int(row['n_seasons'])}")

        st.write('Pillar profile')
        pillar_df = pd.DataFrame({
            'pillar': ['Volume', 'Peak', 'Context', 'Honors'],
            'score': [row['pillar_volume'], row['pillar_peak'], row['pillar_context'], row['pillar_honors']],
        }).set_index('pillar')
        st.bar_chart(pillar_df)

        st.write('Career snapshot')
        snap_cols = [
            'career_ppg', 'career_rpg', 'career_apg', 'career_per', 'career_bpm',
            'career_vorp', 'mvp', 'championships', 'allstar'
        ]
        snap = row[snap_cols].to_frame('value')
        st.dataframe(snap, use_container_width=True)

    with tab3:
        st.markdown(
            '- Volume/Longevity: normalized era-adjusted points/rebounds/assists plus games and seasons.\n'
            '- Peak Dominance: best rolling 3-season average of PER, TS%, BPM.\n'
            '- Context Value: career BPM, VORP, and Win Shares above replacement baseline.\n'
            '- Honors/Recognition: MVP, weighted All-NBA, championships, and All-Star selections.\n'
            '- Slider weights are normalized to sum to 1.0 before scoring.'
        )


if __name__ == '__main__':
    main()
