import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load the trained model ---
try:
    with open('best_taxi_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: The 'best_taxi_model.pkl' file was not found. Please ensure it's in the same directory as this app file.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- App Title and Description ---
st.title("🚖 Taxi Trip Price Predictor")
st.markdown("Enter the trip details below to get a price prediction.")
st.markdown("---")

# --- User Input Widgets ---

st.header("Trip Details")

# Numerical inputs
col1, col2 = st.columns(2)
with col1:
    trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, value=10.0)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1)
    base_fare = st.number_input("Base Fare ($)", min_value=0.0, value=2.5)
    per_km_rate = st.number_input("Per Km Rate ($)", min_value=0.0, value=0.8)

with col2:
    per_minute_rate = st.number_input("Per Minute Rate ($)", min_value=0.0, value=0.3)
    trip_duration = st.number_input("Trip Duration (minutes)", min_value=1.0, value=15.0)
    
    # Categorical inputs
    time_of_day = st.selectbox(
        "Time of Day",
        ('Morning', 'Afternoon', 'Evening', 'Night')
    )
    day_of_week = st.selectbox(
        "Day of Week",
        ('Weekday', 'Weekend')
    )

traffic_conditions = st.selectbox(
    "Traffic Conditions",
    ('Low', 'Medium', 'High')
)

weather = st.selectbox(
    "Weather",
    ('Clear', 'Rain', 'Snow')
)

# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Trip Price"):
    
    # --- Data Preprocessing ---
    # Create a DataFrame from user inputs
    input_data = {
        'Trip_Distance_km': [trip_distance],
        'Passenger_Count': [passenger_count],
        'Base_Fare': [base_fare],
        'Per_Km_Rate': [per_km_rate],
        'Per_Minute_Rate': [per_minute_rate],
        'Trip_Duration_Minutes': [trip_duration],
        'Time_of_Day': [time_of_day],
        'Day_of_Week': [day_of_week],
        'Traffic_Conditions': [traffic_conditions],
        'Weather': [weather]
    }
    input_df = pd.DataFrame(input_data)

    # One-hot encode the categorical features from user input
    categorical_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
    encoded_input = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Get the list of columns the model was trained on
    # This is the key to preventing prediction errors.
    model_features = [
        'Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate', 
        'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Time_of_Day_Evening', 
        'Time_of_Day_Morning', 'Time_of_Day_Night', 'Day_of_Week_Weekend', 
        'Traffic_Conditions_Low', 'Traffic_Conditions_Medium', 
        'Weather_Rain', 'Weather_Snow'
    ]
    
    # Reindex the encoded user input to match the training data columns
    encoded_input_aligned = encoded_input.reindex(columns=model_features, fill_value=0)

    # --- Make Prediction ---
    try:
        prediction = model.predict(encoded_input_aligned)[0]
        st.header(f"💰 Predicted Trip Price: ${prediction:.2f}")
    except Exception as e:

        st.error(f"Prediction failed. Please check the input values and model compatibility. Error: {e}")
