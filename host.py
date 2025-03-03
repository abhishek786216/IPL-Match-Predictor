import streamlit as st

import pandas as pd

import gzip
import pickle

import joblib

pipe = joblib.load('pipe_compressed.pkl')



# Set page title and layout
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

# Title with emoji
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🏏 IPL Win Predictor 🏆</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# IPL Teams
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

# Cities where IPL is played
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Two-column layout for team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('🏏 Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Select the Bowling Team', sorted(teams))

# Prevent selecting the same team
if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling teams cannot be the same. Please select different teams.")

# Host city selection
selected_city = st.selectbox('📍 Select Match Location', sorted(cities))

# Target Score Input
target = st.number_input('🎯 Target Score', min_value=0, max_value=500, value=0, step=1, format="%d")

# Three-column layout for match details
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('🏏 Current Score', min_value=0, max_value=500, step=1)

with col4:
    overs = st.number_input('⏳ Overs Completed', min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")

with col5:
    wickets = st.number_input('❌ Wickets Lost', min_value=0, max_value=10, value=0, step=1, format="%d")

# Prediction Button
if st.button('🔮 Predict Winning Probability'):
    
    # Check if all required fields are entered
    if target == 0:
        st.error("❌ Please enter a valid target score.")
    elif overs == 0 and score > 0:
        st.error("❌ Overs cannot be 0 if a score is already set. Please enter a valid number of overs.")
    elif overs == 0:
        st.warning("⚠️ Please enter the number of overs to calculate probability.")
    elif batting_team == bowling_team:
        st.error("❌ Batting and Bowling teams cannot be the same.")
    else:
        # Compute derived features
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Create DataFrame for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team], 'bowling_team': [bowling_team],
            'city': [selected_city], 'run_left': [runs_left], 'ball_left': [balls_left],
            'wicket_left': [wickets_left], 'total_runs_y': [target], 'curr': [crr], 'req': [rrr]
        })

        # Predict probabilities
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Display results with progress bars
        st.markdown(f"<h2 style='text-align: center; color: #0078FF;'>{batting_team} - {round(win_prob * 100)}% Chance to Win</h2>", unsafe_allow_html=True)
        st.progress(win_prob)

        st.markdown(f"<h2 style='text-align: center; color: #FF0000;'>{bowling_team} - {round(loss_prob * 100)}% Chance to Win</h2>", unsafe_allow_html=True)
        st.progress(loss_prob)
