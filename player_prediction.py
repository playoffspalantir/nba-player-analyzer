import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
import logging
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog, teamgamelogs, commonteamroster, commonplayerinfo
from nba_api.stats.static import players, teams
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# <<< ADDED >>> Headers to mimic a real browser request, preventing API blocks on cloud servers.
CUSTOM_HEADERS = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
}


# --- Caching Functions to Improve Performance ---
@st.cache_data(ttl=3600)
def fetch_player_game_logs(player_id, season):
    logging.info(f"Fetching game logs for Player ID: {player_id}, Season: {season}")
    if season == "All":
        # <<< CHANGED >>> Added headers and timeout
        log = playergamelog.PlayerGameLog(player_id=player_id, headers=CUSTOM_HEADERS, timeout=60)
    else:
        # <<< CHANGED >>> Added headers and timeout
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season, headers=CUSTOM_HEADERS, timeout=60)
    return log.get_data_frames()[0]


@st.cache_data(ttl=86400)
def get_league_team_rankings(season="2023-24"):
    logging.info(f"Fetching league-wide team statistics for rankings for season {season}...")
    try:
        # <<< CHANGED >>> Added headers and timeout
        logs = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            headers=CUSTOM_HEADERS,
            timeout=60
        )
        df = logs.get_data_frames()[0]
        if df.empty:
            logging.warning(f"No team game logs found for season {season}.")
            return {}

        aggregated = df.groupby('TEAM_ID').agg({
            'PTS': 'mean', 'REB': 'mean', 'AST': 'mean', 'STL': 'mean', 'BLK': 'mean'
        }).reset_index()

        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        rankings = {}
        for stat in stats:
            aggregated[f'{stat}_RANK'] = aggregated[stat].rank(ascending=False, method='min')
            rankings[stat] = dict(zip(aggregated['TEAM_ID'], aggregated[f'{stat}_RANK']))

        logging.info("League-wide team rankings calculated.")
        return rankings
    except Exception as e:
        logging.error(f"Failed to fetch league team rankings for season {season}: {e}")
        return {}


# Function to visualize correlation
def visualize_correlation(data, title):
    correlation_matrix = data.corr().round(2)
    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="Viridis", title=title)
    st.plotly_chart(fig, use_container_width=True)


# Function to get recent games with critical stats only
def get_recent_games(df, entity_name, num_games=5):
    if df.empty or 'GAME_DATE' not in df.columns:
        st.warning("No game data available to show recent games.")
        return

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    recent_games = df.nlargest(num_games, 'GAME_DATE')[[
        'GAME_DATE', 'MIN', 'FGM', 'FGA', 'FG_PCT',
        'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
        'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
        'STL', 'BLK', 'TOV', 'PTS'
    ]]

    recent_games['GAME_DATE'] = recent_games['GAME_DATE'].dt.strftime('%Y-%m-%d')
    recent_games.columns = [
        'Date', 'Minutes', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%',
        'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS'
    ]

    st.subheader(f"Last {num_games} Games for {entity_name}:")
    st.dataframe(recent_games)


def plot_points_over_time(df, entity_name, y_axis_stat, hline_value):
    if df.empty or y_axis_stat not in df.columns:
        st.warning(f"Not enough data to plot '{y_axis_stat}' over time.")
        return

    plot_df = df.copy()
    plot_df['GAME_DATE'] = pd.to_datetime(plot_df['GAME_DATE'])
    plot_df = plot_df.sort_values(by='GAME_DATE', ascending=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(plot_df['GAME_DATE'], plot_df[y_axis_stat], marker='o', linestyle='-',
            label=f'{y_axis_stat}', color='dodgerblue', zorder=2)

    if len(plot_df) > 1:
        x_numeric = plot_df['GAME_DATE'].map(pd.Timestamp.toordinal)
        y = plot_df[y_axis_stat]
        coefficients = np.polyfit(x_numeric, y, 1)
        polynomial = np.poly1d(coefficients)
        trendline = polynomial(x_numeric)
        ax.plot(plot_df['GAME_DATE'], trendline, color='red', linestyle='--',
                label='Trend Line', zorder=3)

    ax.axhline(y=hline_value, color='green', linestyle='-', lw=2,
               label=f'Prop Line: {hline_value}', zorder=1)

    ax.set_title(f"{entity_name} {y_axis_stat} Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel('Game Date', fontsize=12)
    ax.set_ylabel(y_axis_stat, fontsize=12)
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)


@st.cache_data(ttl=3600)
def fetch_player_info(player_id):
    logging.info(f"Fetching info for player ID: {player_id}")
    # <<< CHANGED >>> Added headers and timeout
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, headers=CUSTOM_HEADERS, timeout=60)
    data = info.get_data_frames()[0]
    full_name = data['DISPLAY_FIRST_LAST'][0]
    position = data['POSITION'][0].strip()
    return full_name, position


# --- Other Helper Functions ---
@st.cache_data
def get_team_id_by_abbreviation(abbreviation):
    nba_teams = teams.get_teams()
    team = next((team for team in nba_teams if team['abbreviation'] == abbreviation), None)
    return team['id'] if team else None


@st.cache_data
def find_team_name_by_id(team_id):
    nba_teams = teams.get_teams()
    team = next((team for team in nba_teams if team['id'] == team_id), None)
    return team['full_name'] if team else "Unknown Team"


def get_opponent_team_id(df):
    if df.empty or 'MATCHUP' not in df.columns:
        return None
    opponent_abbreviations = df['MATCHUP'].apply(lambda x: x.split(' ')[-1]).unique()
    team_dict = {}
    for abbrev in opponent_abbreviations:
        team_id = get_team_id_by_abbreviation(abbrev)
        if team_id:
            team_name = find_team_name_by_id(team_id)
            team_dict[team_name] = team_id
    if not team_dict:
        return None

    opponent_team_name = st.selectbox("Select Opponent Team (Optional)", options=[''] + sorted(list(team_dict.keys())))
    return team_dict.get(opponent_team_name)


@st.cache_data(ttl=3600)
def get_opponent_team_stats(team_id, season="2023-24", num_games=5):
    # <<< CHANGED >>> Added headers and timeout
    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
        headers=CUSTOM_HEADERS,
        timeout=60
    )
    df = logs.get_data_frames()[0]
    if df.empty: return None
    numeric_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    team_games = df[numeric_stats].head(num_games).apply(pd.to_numeric, errors='coerce')
    return team_games.mean().round(1).to_dict()


def calculate_averages(df, num_games=5):
    if df.empty:
        return {}
    recent_df = df.head(num_games)
    stats_to_avg = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    averages = recent_df[stats_to_avg].mean()

    p = averages.get('PTS', 0)
    r = averages.get('REB', 0)
    a = averages.get('AST', 0)

    return {
        'PTS': round(p, 1), 'REB': round(r, 1), 'AST': round(a, 1),
        'STL': round(averages.get('STL', 0), 1), 'BLK': round(averages.get('BLK', 0), 1),
        'P+R': round(p + r, 1), 'P+A': round(p + a, 1), 'R+A': round(r + a, 1), 'P+R+A': round(p + r + a, 1)
    }


# Main app function
def app():
    st.set_page_config(layout="wide")
    st.title("🏀 NBA Player Performance Analyzer")

    st.sidebar.header("Selections")

    season_options = [f"{yr}-{str(yr + 1)[-2:]}" for yr in range(2023, 2000, -1)]
    season_select = st.sidebar.selectbox("Select Season:", season_options)

    nba_teams = teams.get_teams()
    team_names = sorted([team['full_name'] for team in nba_teams])
    selected_team_name = st.sidebar.selectbox("Select a Team:", team_names, index=team_names.index("Denver Nuggets"))

    selected_team_id = next((team['id'] for team in nba_teams if team['full_name'] == selected_team_name), None)

    selected_player_id = None
    if selected_team_id:
        try:
            # <<< CHANGED >>> Added headers and timeout to the roster call (the one that was failing)
            team_roster = commonteamroster.CommonTeamRoster(
                team_id=selected_team_id,
                season=season_select,
                headers=CUSTOM_HEADERS,
                timeout=60
            )
            roster_df = team_roster.get_data_frames()[0]
            if not roster_df.empty:
                player_names = sorted(roster_df['PLAYER'].tolist())
                selected_player_name = st.sidebar.selectbox("Select a Player:", player_names)
                player_id_map = dict(zip(roster_df['PLAYER'], roster_df['PLAYER_ID']))
                selected_player_id = player_id_map.get(selected_player_name)
            else:
                st.sidebar.warning(
                    f"No player roster found for the {selected_team_name} in the {season_select} season.")
        except Exception as e:
            st.sidebar.error(f"Error fetching player roster: {e}")

    if not selected_player_id:
        st.info("Please select a team and player from the sidebar to begin analysis.")
        return

    try:
        full_name, position = fetch_player_info(selected_player_id)
        st.header(f"Analysis for {full_name} ({position})")
    except Exception as e:
        st.error(f"Could not fetch player information: {e}")
        return

    game_log_df = fetch_player_game_logs(selected_player_id, season_select)

    if game_log_df.empty:
        st.warning(f"No game logs found for {full_name} in the {season_select} season.")
        return

    st.subheader("Analysis Options")
    num_games = st.slider("Number of Recent Games to Analyze:", 1, 20, 10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Projected Stats (Last {num_games} Games)")
        projected_line = calculate_averages(game_log_df, num_games=num_games)
        projected_df = pd.DataFrame([projected_line])
        st.dataframe(projected_df)

        csv = projected_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Projections as CSV",
            data=csv,
            file_name=f"{full_name.replace(' ', '_')}_projections.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("Opponent Defense Ranking")
        rankings = get_league_team_rankings(season=season_select)
        opponent_team_id = get_opponent_team_id(game_log_df)

        if opponent_team_id and rankings:
            opponent_name = find_team_name_by_id(opponent_team_id)
            opponent_ranks = {
                stat: ranks.get(opponent_team_id, 'N/A')
                for stat, ranks in rankings.items()
            }
            st.metric(label=f"vs {opponent_name} | Points Allowed Rank", value=f"#{int(opponent_ranks.get('PTS', 0))}")
            st.metric(label=f"vs {opponent_name} | Rebounds Allowed Rank",
                      value=f"#{int(opponent_ranks.get('REB', 0))}")
            st.metric(label=f"vs {opponent_name} | Assists Allowed Rank", value=f"#{int(opponent_ranks.get('AST', 0))}")
        elif not rankings:
            st.warning(f"Could not retrieve league rankings for the {season_select} season.")
        else:
            st.info("Select an opponent team to see their defensive rankings.")

    st.markdown("---")

    available_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV']
    selected_stat = st.selectbox("Select Stat to Visualize:", available_stats, index=0)

    projected_stat_value = projected_line.get(selected_stat, 0)
    hline_value = st.number_input("Enter Prop Line:", value=float(projected_stat_value), step=0.5)

    plot_points_over_time(game_log_df, full_name, selected_stat, hline_value)
    get_recent_games(game_log_df, full_name, num_games)

    numeric_df = game_log_df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        visualize_correlation(numeric_df, "Correlation Matrix of Player Statistics")


if __name__ == "__main__":
    app()