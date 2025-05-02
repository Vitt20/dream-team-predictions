import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the data
bbb_df = pd.read_csv('bbb.csv')
player_at_venue = pd.read_csv('player_at_venue.csv')
bowler_at_venue = pd.read_csv('bowler_at_venue.csv')

# Load team player data
teams = ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Royal Challengers Bangalore', 'Rajasthan Royals', 'Sunrisers Hyderabad']
team_data = {}
for team in teams:
    file_name = f'{team}.xlsx'
    team_data[team] = pd.read_excel(file_name)

# Preprocess the data
bbb_df['Date'] = pd.to_datetime(bbb_df['Date'])
bbb_df.fillna(0, inplace=True)

# Feature Engineering
player_stats = bbb_df.groupby(['Batter']).agg({
    'Batter Runs': ['sum', 'mean'],
    'Ball': 'count'
}).reset_index()
player_stats.columns = ['Batter', 'Total Runs', 'Avg Runs', 'Balls Faced']
player_stats['Strike Rate'] = (player_stats['Total Runs'] / player_stats['Balls Faced']) * 100

bowler_stats = bbb_df.groupby(['Bowler']).agg({
    'Bowler Runs Conceded': ['sum', 'mean'],
    'Valid Ball': 'count'
}).reset_index()
bowler_stats.columns = ['Bowler', 'Total Runs Conceded', 'Avg Runs Conceded', 'Balls Bowled']
bowler_stats['Economy Rate'] = (bowler_stats['Total Runs Conceded'] / bowler_stats['Balls Bowled']) * 6

player_performance = pd.merge(player_stats, bowler_stats, left_on='Batter', right_on='Bowler', how='outer').fillna(0)
player_performance.drop(columns=['Bowler'], inplace=True)

# Venue-Specific Performance
venue_batting = player_at_venue.groupby(['Batter', 'Venue']).agg({
    'total_runs': 'sum',
    'balls_faced': 'sum',
    'strike_rate': 'mean'
}).reset_index()

venue_bowling = bowler_at_venue.groupby(['Bowler', 'Venue']).agg({
    'wickets_taken': 'sum',
    'runs_conceded': 'sum',
    'balls_bowled': 'sum',
    'economy_rate': 'mean'
}).reset_index()

# Encode categorical features
encoder = LabelEncoder()
bbb_df['Venue'] = encoder.fit_transform(bbb_df['Venue'])

# Model Training
data = player_performance.drop(columns=['Batter'])
target = (player_performance['Avg Runs'] > player_performance['Avg Runs'].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define function to predict best 11
def predict_best_11(selected_players, venue):
    selected_players_stats = player_performance[player_performance['Batter'].isin(selected_players)]
    venue_stats_filtered = venue_batting[venue_batting['Batter'].isin(selected_players) & (venue_batting['Venue'] == venue)]
    
    combined_stats = selected_players_stats.merge(venue_stats_filtered, on='Batter', how='left').fillna(0)
    combined_stats['Final Score'] = combined_stats['Total Runs'] * 0.7 + combined_stats['total_runs'] * 0.3
    
    ranked_players = combined_stats.sort_values(by='Final Score', ascending=False).reset_index()
    return ranked_players['Batter'].head(11).tolist()

# Streamlit UI
st.title('Cricket Team Selector')

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# Dropdowns for players selection
team1_players = team_data[team1]['player_name'].tolist()
team2_players = team_data[team2]['player_name'].tolist()

selected_team1_players = st.multiselect(f"Select 11 players for {team1}", team1_players, max_selections=11)
selected_team2_players = st.multiselect(f"Select 11 players for {team2}", team2_players, max_selections=11)

# Venue input
venue = st.text_input("Enter Venue")

if st.button("Predict Best 11"):
    # Ensure exactly 11 players are selected from each team
    if len(selected_team1_players) == 11 and len(selected_team2_players) == 11:
        selected_players = selected_team1_players + selected_team2_players
        best_11 = predict_best_11(selected_players, venue)
        st.write(f"Best 11 Players: {best_11}")
    else:
        st.warning("Please select exactly 11 players from each team.")
