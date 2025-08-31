import pandas as pd
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import (commonplayerinfo, commonteamroster, playergamelog)
from nba_api.stats.static import teams
from tenacity import retry, stop_after_attempt, wait_fixed # <<< ADDED >>> Import the tenacity library

# --- Page Config & Custom CSS (Same as before) ---
st.set_page_config(layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700&display=swap');
body {font-family: 'Roboto Condensed', sans-serif;}
.st-emotion-cache-1g6xema {text-transform: uppercase;}
</style>
""", unsafe_allow_html=True)

# --- API Headers (Same as before) ---
CUSTOM_HEADERS = {
    'Host': 'stats.nba.com', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*', 'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br', 'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true', 'Connection': 'keep-alive', 'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache', 'Cache-Control': 'no-cache',
}

# --- Caching Data Fetching Functions ---

# <<< MODIFIED >>> This function now has a retry decorator!
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_team_roster_with_retries(team_id, season):
    """
    This is a wrapper function that attempts to fetch the team roster up to 3 times,
    waiting 2 seconds between each attempt. This handles timeouts and throttling.
    """
    st.write(f"Fetching roster for team {team_id} (attempt)...") # Temporary message to see it work
    roster = commonteamroster.CommonTeamRoster(
        team_id=team_id,
        season=season,
        headers=CUSTOM_HEADERS,
        timeout=90  # Increased timeout slightly for more buffer
    )
    return roster.get_data_frames()[0]

@st.cache_data(ttl=3600)
def get_team_roster(team_id, season):
    # This cached function now calls our new, resilient retry function
    return get_team_roster_with_retries(team_id, season)

# (The rest of the data functions are the same)
@st.cache_data(ttl=86400)
def get_nba_teams(): return teams.get_teams()

@st.cache_data(ttl=3600)
def fetch_player_game_logs(player_id, season):
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season, headers=CUSTOM_HEADERS, timeout=60)
    return log.get_data_frames()[0]

@st.cache_data(ttl=3600)
def fetch_player_info(player_id):
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, headers=CUSTOM_HEADERS, timeout=60)
    data = info.get_data_frames()[0]
    return data['DISPLAY_FIRST_LAST'][0], data['POSITION'][0].strip()

# (The rest of the app is identical to the last version)
@st.cache_data
def get_team_logo_url(team_id): return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"
@st.cache_data
def get_player_headshot_url(player_id): return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

def calculate_averages(df):
    if df.empty: return {}
    stats_to_avg = ['PTS', 'REB', 'AST']
    averages = df[stats_to_avg].mean()
    p, r, a = averages.get('PTS', 0), averages.get('REB', 0), averages.get('AST', 0)
    return {'PTS': round(p, 1), 'REB': round(r, 1), 'AST': round(a, 1), 'P+R+A': round(p + r + a, 1)}

def plot_performance_chart(df, player_name, stat, prop_line):
    fig = px.line(df, x='GAME_DATE', y=stat, markers=True, labels={'GAME_DATE': 'Game Date', stat: f'{stat} Count'})
    fig.add_hline(y=prop_line, line_dash="dash", line_color="#FF4B2B", annotation_text=f"Prop Line: {prop_line}", annotation_position="bottom right")
    fig.update_layout(template="plotly_dark", title={'text': f'{player_name} - {stat} Over Time', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, font=dict(family="Roboto Condensed, sans-serif"))
    st.plotly_chart(fig, use_container_width=True)

def get_recent_games_df(df):
    if df.empty: return pd.DataFrame()
    display_cols = {'GAME_DATE': 'Date', 'MIN': 'Min', 'PTS': 'Pts', 'REB': 'Reb', 'AST': 'Ast', 'STL': 'Stl', 'BLK': 'Blk', 'TOV': 'TO', 'FG_PCT': 'FG%', 'FG3_PCT': '3P%', 'FT_PCT': 'FT%'}
    recent = df[display_cols.keys()].rename(columns=display_cols)
    recent['Date'] = recent['Date'].dt.strftime('%b %d')
    for col in ['FG%', '3P%', 'FT%']: recent[col] = (recent[col] * 100).map('{:.1f}%'.format)
    return recent

def main():
    st.title("🏀 NBA Prop Performance Analyzer")
    st.sidebar.header("CONTROLS")
    season_select = st.sidebar.selectbox("Season:", [f"{yr}-{str(yr + 1)[-2:]}" for yr in range(2023, 2010, -1)])
    nba_teams = get_nba_teams()
    team_names = sorted([team['full_name'] for team in nba_teams])
    selected_team_name = st.sidebar.selectbox("Team:", team_names, index=team_names.index("Denver Nuggets"))
    selected_team_id = next((t['id'] for t in nba_teams if t['full_name'] == selected_team_name), None)
    if selected_team_id:
        st.sidebar.image(get_team_logo_url(selected_team_id), width=150)
        roster_df = get_team_roster(selected_team_id, season_select) # This call is now super resilient
        if not roster_df.empty:
            player_names = sorted(roster_df['PLAYER'].tolist())
            selected_player_name = st.sidebar.selectbox("Player:", player_names)
            player_id_map = dict(zip(roster_df['PLAYER'], roster_df['PLAYER_ID']))
            selected_player_id = player_id_map.get(selected_player_name)
        else:
            st.sidebar.warning(f"No roster for {season_select}.")
            selected_player_id = None
    if not selected_player_id:
        st.info("Select a Team and Player to begin.")
        return
    full_name, position = fetch_player_info(selected_player_id)
    game_log_df = fetch_player_game_logs(selected_player_id, season_select)
    game_log_df['GAME_DATE'] = pd.to_datetime(game_log_df['GAME_DATE'])
    game_log_df = game_log_df.sort_values('GAME_DATE', ascending=False)
    col1, col2 = st.columns([1, 4])
    with col1: st.image(get_player_headshot_url(selected_player_id))
    with col2:
        st.header(f"{full_name}")
        st.subheader(f"*{position}*")
    if game_log_df.empty:
        st.warning(f"No game logs found for {full_name} in {season_select}.")
        return
    num_games = st.sidebar.slider("Games to Analyze:", 1, 30, 10)
    recent_games_df = game_log_df.head(num_games)
    st.subheader(f"AVERAGES (LAST {num_games} GAMES)")
    averages = calculate_averages(recent_games_df)
    cols = st.columns(4)
    cols[0].metric("Points", averages.get('PTS', 0))
    cols[1].metric("Rebounds", averages.get('REB', 0))
    cols[2].metric("Assists", averages.get('AST', 0))
    cols[3].metric("P+R+A", averages.get('P+R+A', 0))
    st.markdown("---")
    c1, c2 = st.columns((2, 1))
    with c1:
        st.subheader("PERFORMANCE CHART")
        available_stats = ['PTS', 'REB', 'AST']
        selected_stat = st.selectbox("Select Stat:", available_stats, index=0)
        projected_stat_value = averages.get(selected_stat, 0)
        prop_line = st.number_input("Enter Prop Line:", value=float(projected_stat_value), step=0.5)
        plot_performance_chart(recent_games_df.sort_values('GAME_DATE', ascending=True), full_name, selected_stat, prop_line)
    with c2:
        st.subheader(f"RECENT GAME LOGS")
        display_df = get_recent_games_df(recent_games_df)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()