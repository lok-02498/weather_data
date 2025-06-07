import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Load models
work_model = joblib.load("weather_model.pkl")
weather_desc_model = joblib.load("weather_description_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load dataset with location
df = pd.read_csv("updated_weather_data.csv")

st.title("🌦️ Work Suitability and Weather Prediction System")

# 📍 Step 1: Select Location
if 'Location' in df.columns:
    locations = df['Location'].dropna().unique()
    selected_location = st.selectbox("📍 Choose a Location", sorted(locations), key="location_select")
    
    # Show recent data for that location
    loc_data = df[df['Location'] == selected_location]
    st.write(f"📊 Recent data from **{selected_location}**:")
    st.dataframe(loc_data.tail(5))
else:
    st.error("❌ Your data does not contain a 'Location' column.")

# User manual input sliders with unique keys
temp = st.slider("🌡️ Temperature (°C)", -20.0, 50.0, 25.0, key="temp_slider")
humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 50.0, key="humidity_slider")
precip = st.slider("🌧️ Precipitation (mm)", 0.0, 100.0, 10.0, key="precip_slider")
wind = st.slider("💨 Wind Speed (km/h)", 0.0, 100.0, 20.0, key="wind_slider")

# Prepare input data for manual prediction (needed later)
input_data = pd.DataFrame([{
    "Temperature_C": temp,
    "Humidity_pct": humidity,
    "Precipitation_mm": precip,
    "Wind_Speed_kmh": wind
}])

# Function to check flood risk based on predicted label and numeric conditions
#def check_flood_risk(predicted_label, precip, humidity, wind):
    # Check flood keywords in label
   # flood_keywords = ["flood", "heavy rainfall", "storm", "cyclone"]
  #  label_flood = any(k in predicted_label.lower() for k in flood_keywords)
    # Also use numeric thresholds to flag flood risk
   # numeric_flood = (precip >= 8) or (precip >= 6 and humidity > 70) or (precip >= 4 and wind > 25)
   # return label_flood or numeric_flood

# Manual Prediction Button
if st.button("🔍 Predict", key="manual_predict_btn"):
    predicted_weather_encoded = weather_desc_model.predict(input_data)[0]
    predicted_weather_label = le.inverse_transform([predicted_weather_encoded])[0]
    st.info(f"🌤️ **Predicted Weather:** {predicted_weather_label}")

    # Show flood risk for manual input
 #   if check_flood_risk(predicted_weather_label, precip, humidity, wind):
 #       st.error("🌊 Flood Risk (Manual Input): HIGH – Take precautions!")
 #   else:
 #       st.success("✅ Flood Risk (Manual Input): LOW")

    # Add predicted weather description encoded to input data for work suitability model
    input_data["Weather_Description"] = predicted_weather_encoded

    manual_strict = (
        temp < 5 or temp > 40 or
        precip > 8 or
        wind > 30 or
        predicted_weather_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
    )
    st.session_state['manual_strict'] = manual_strict

    if manual_strict:
        prediction = 0
        st.warning("⚠️ Conditions are harsh – work not suitable (rule-based override).")
    else:
        prediction = work_model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Work is Suitable under these conditions.")
    else:
        st.error("❌ Work is NOT Suitable under these conditions.")

    # Alert if both manual and realtime harsh conditions detected
    if st.session_state.get('realtime_strict', False) and manual_strict:
        st.warning("🚨 Alert: Harsh weather conditions detected BOTH in manual input and real-time data for today!")

# Function to get real-time weather from Tomorrow.io API
def get_realtime_weather(lat, lon, api_key):
    url = "https://api.tomorrow.io/v4/timelines"
    headers = {"apikey": api_key}
    params = {
        "location": f"{lat},{lon}",
        "fields": ["temperature", "humidity", "precipitationIntensity", "windSpeed"],
        "units": "metric",
        "timesteps": "current"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"❌ Failed to fetch weather data: {response.text}")
        return None
    data = response.json()
    try:
        values = data['data']['timelines'][0]['intervals'][0]['values']
        temperature = values['temperature']
        humidity = values['humidity']
        precipitation = values.get('precipitationIntensity', 0.0)
        wind_speed = values['windSpeed']

        return {
            "Temperature_C": temperature,
            "Humidity_pct": humidity,
            "Precipitation_mm": precipitation,
            "Wind_Speed_kmh": wind_speed
        }
    except (KeyError, IndexError):
        st.error("❌ Incomplete weather data received.")
        return None

st.markdown("---")
st.subheader("🌐 Get Real-Time Weather & Predict Work Suitability")

city_coords = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Agra": (27.1767, 78.0081),
    "Bangalore": (12.9716, 77.5946),
    "Shillong": (25.5788, 91.8933),
}

# Dropdown to select city dynamically
city_name = st.selectbox("🌍 Select City for Real-Time Weather", sorted(city_coords.keys()), key="city_select")

# Hardcoded API key (keep it here or use secrets management for safety)
api_key = "6OtYQy1ODeUOOH2UJJYwnglbW1AidxPr"

if st.button("🚀 Fetch & Predict Real-Time Weather", key="realtime_predict_btn"):
    lat, lon = city_coords[city_name]
    weather_data = get_realtime_weather(lat, lon, api_key)
    if weather_data:
        st.write(f"📊 Real-Time Weather Data for **{city_name}**:")
        st.json(weather_data)

        df_weather = pd.DataFrame([weather_data])
        predicted_weather_encoded = weather_desc_model.predict(df_weather)[0]
        predicted_weather_label = le.inverse_transform([predicted_weather_encoded])[0]

        st.info(f"🌤️ **Predicted Weather:** {predicted_weather_label}")

        df_input = df_weather.copy()
        df_input["Weather_Description"] = predicted_weather_encoded
        
        # Flood risk for realtime input
     #   if check_flood_risk(predicted_weather_label, 
     #                       weather_data["Precipitation_mm"], 
     #                       weather_data["Humidity_pct"], 
     #                       weather_data["Wind_Speed_kmh"]):
     #       st.error("🌊 Flood Risk (Real-Time): HIGH – Take precautions!")
     #   else:
      #      st.success("✅ Flood Risk (Real-Time): LOW")

        realtime_strict = (
            weather_data["Temperature_C"] < 5 or weather_data["Temperature_C"] > 40 or
            weather_data["Precipitation_mm"] > 8 or
            weather_data["Wind_Speed_kmh"] > 30 or
            predicted_weather_label.lower() in ["heavy rainfall", "flood", "storm", "cyclone"]
        )
        st.session_state['realtime_strict'] = realtime_strict

        if realtime_strict:
            st.warning("⚠️ Harsh weather detected – work not suitable (rule-based override).")
        else:
            prediction = work_model.predict(df_input)[0]
            if prediction == 1:
                st.success("✅ Work is Suitable under current conditions.")
            else:
                st.error("❌ Work is NOT Suitable under current conditions.")

        # Alert if harsh weather in both manual and realtime
        if st.session_state.get('manual_strict', False) and realtime_strict:
            st.warning("🚨 Alert: Harsh weather conditions detected BOTH in manual input and real-time data for today!")

st.markdown("---")
with st.expander("📊 Show Weather Visualizations"):

    # Convert Date_Time column to datetime format
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')

    # 📈 Temperature trend for selected location
    if not loc_data.empty:
        st.subheader(f"📈 Temperature Trend in {selected_location}")
        fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
        loc_data_sorted = loc_data.copy()
        loc_data_sorted['Date_Time'] = pd.to_datetime(loc_data_sorted['Date_Time'], errors='coerce')
        loc_data_sorted = loc_data_sorted.dropna(subset=['Date_Time'])
        loc_data_sorted = loc_data_sorted.sort_values("Date_Time")
        if 'Temperature_C' in loc_data_sorted.columns:
            sns.lineplot(data=loc_data_sorted, x='Date_Time', y='Temperature_C', ax=ax_temp, marker="o")
            ax_temp.set_xlabel("Date")
            ax_temp.set_ylabel("Temperature (°C)")
            plt.xticks(rotation=45)
            st.pyplot(fig_temp)
        else:
            st.warning("⚠️ 'Temperature_C' column is missing in the data.")

    # ✅ Work Suitability Pie Chart
    if 'Work_Suitability' in df.columns:
        st.subheader("✅ Work Suitability Distribution")
        suit_counts = df['Work_Suitability'].value_counts()
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(suit_counts, labels=suit_counts.index, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
    else:
        st.warning("⚠️ 'Work_Suitability' column not found.")

    # 🌦️ Weather Description Count (Bar chart)
    if 'Weather_Description' in df.columns:
        st.subheader("🌦️ Weather Description Frequency")
        fig_desc, ax_desc = plt.subplots(figsize=(10, 4))
        desc_counts = df['Weather_Description'].value_counts()
        sns.barplot(x=desc_counts.index, y=desc_counts.values, ax=ax_desc)
        ax_desc.set_ylabel("Count")
        ax_desc.set_xlabel("Weather Description")
        plt.xticks(rotation=45)
        st.pyplot(fig_desc)
    else:
        st.warning("⚠️ 'Weather_Description' column not found.")

    # ☔ Rainfall Distribution by Location (Box plot)
    if 'Precipitation_mm' in df.columns and 'Location' in df.columns:
        st.subheader("☔ Precipitation Distribution by Location")
        fig_box, ax_box = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df, x="Location", y="Precipitation_mm", ax=ax_box)
        plt.xticks(rotation=45)
        st.pyplot(fig_box)
    else:
        st.warning("⚠️ Columns 'Precipitation_mm' or 'Location' are missing.")

    # 🌊 Flood Risk Distribution (Bar chart)
    if 'Flood_Risk' in df.columns:
        st.subheader("🌊 Flood Risk Frequency")
        fig_flood, ax_flood = plt.subplots()
        flood_counts = df['Flood_Risk'].value_counts()
        sns.barplot(x=flood_counts.index, y=flood_counts.values, ax=ax_flood)
        ax_flood.set_xlabel("Flood Risk")
        ax_flood.set_ylabel("Count")
        st.pyplot(fig_flood)
    else:
        st.warning("⚠️ 'Flood_Risk' column not found.")








def set_navy_blue_background(image_file):
    import base64
    import streamlit as st

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .block-container {{
            background: rgba(0, 0, 128, 0.6);  /* Navy blue overlay */
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 0 20px 8px rgba(0, 0, 128, 0.5);
            border: 2px solid rgba(0, 0, 128, 0.7);
            margin-top: 2rem;
            color: #f0f0f0;
            font-weight: 600;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .stText, .stMarkdown {{
            color: #ffffff !important;
        }}

        .stButton > button {{
            background-color: #001F54 !important;
            color: #ffffff !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 0 10px 4px #001F54;
            transition: background-color 0.3s ease;
        }}
        .stButton > button:hover {{
            background-color: #003366 !important;
            box-shadow: 0 0 15px 6px #001F54;
        }}

        .stSlider > div {{
            background-color: rgba(0, 0, 128, 0.4);
            border-radius: 8px;
        }}
        </style>
    """, unsafe_allow_html=True)

# Call the function
set_navy_blue_background("gail_bg.jpg")








