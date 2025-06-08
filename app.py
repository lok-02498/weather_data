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

df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')

# Clean Location column: remove leading/trailing spaces (common cause for filtering bugs)
df['Location'] = df['Location'].str.strip()

# Location selector dropdown (must be outside visualization expander)
selected_location = st.selectbox("Select Location", options=df['Location'].unique())

# Filter data based on selected location
filtered_df = df[df['Location'] == selected_location]

with st.expander("📊 Weather Data Visualizations"):

    if filtered_df.empty:
        st.warning(f"No data available for location: {selected_location}")
    else:
        st.write(f"Showing data for: **{selected_location}**")

        # Example visualization 1: Temperature over time
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filtered_df.sort_values('Date_Time'), x='Date_Time', y='Temperature_C', marker='o', ax=ax1)
        ax1.set_title("Temperature Over Time")
        ax1.set_xlabel("Date and Time")
        ax1.set_ylabel("Temperature (°C)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # Example visualization 2: Humidity over time
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filtered_df.sort_values('Date_Time'), x='Date_Time', y='Humidity_pct', marker='o', ax=ax2, color='orange')
        ax2.set_title("Humidity Over Time")
        ax2.set_xlabel("Date and Time")
        ax2.set_ylabel("Humidity (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # Example visualization 3: Work Suitability distribution (if categorical)
        st.write("Work Suitability Distribution")
        st.bar_chart(filtered_df['Work_Suitability'].value_counts())
with st.expander("📈 More Weather Visualizations"):

    if filtered_df.empty:
        st.warning("No data available for this location.")
    else:
        # 1. Precipitation over time
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filtered_df.sort_values('Date_Time'), x='Date_Time', y='Precipitation_mm', marker='o', ax=ax3, color='blue')
        ax3.set_title("Precipitation Over Time (mm)")
        ax3.set_xlabel("Date and Time")
        ax3.set_ylabel("Precipitation (mm)")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        # 2. Wind Speed over time
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filtered_df.sort_values('Date_Time'), x='Date_Time', y='Wind_Speed_kmh', marker='o', ax=ax4, color='green')
        ax4.set_title("Wind Speed Over Time (km/h)")
        ax4.set_xlabel("Date and Time")
        ax4.set_ylabel("Wind Speed (km/h)")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        # 3. Weather Description count plot
        st.write("Weather Description Distribution")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=filtered_df, y='Weather_Description', order=filtered_df['Weather_Description'].value_counts().index, palette='viridis', ax=ax5)
        ax5.set_title("Frequency of Weather Descriptions")
        st.pyplot(fig5)

        # 4. Flood Risk Pie Chart (if categorical 'Yes'/'No')
        flood_counts = filtered_df['Flood_Risk'].value_counts()
        st.write("Flood Risk Distribution")
        fig6, ax6 = plt.subplots()
        ax6.pie(flood_counts, labels=flood_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'lightgrey'])
        ax6.axis('equal')
        st.pyplot(fig6)

        # 5. Work Suitability Over Time (if suitable for line plot)
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        # Convert 'Work_Suitability' Yes/No to 1/0
        work_numeric = filtered_df['Work_Suitability'].map({'Yes':1, 'No':0})
        sns.lineplot(x=filtered_df['Date_Time'], y=work_numeric, marker='o', ax=ax7)
        ax7.set_title("Work Suitability Over Time (Yes=1, No=0)")
        ax7.set_xlabel("Date and Time")
        ax7.set_ylabel("Work Suitability")
        plt.xticks(rotation=45)
        st.pyplot(fig7)

def set_purple_background(image_file, intensified=False):
    import base64
    import streamlit as st

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    # Overlay intensity settings
    overlay_opacity = 0.7 if not intensified else 0.92
    overlay_blur = 6 if not intensified else 14
    box_shadow_opacity = 0.45 if not intensified else 0.75
    border_opacity = 0.6 if not intensified else 0.9

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
            background: rgba(10, 10, 10, {overlay_opacity});
            backdrop-filter: blur({overlay_blur}px);
            -webkit-backdrop-filter: blur({overlay_blur}px);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 0 20px 8px rgba(128, 0, 128, {box_shadow_opacity});
            border: 2px solid rgba(186, 85, 211, {border_opacity});
            color: #ffffff;
            font-weight: 600;
            transition: background 0.5s ease, box-shadow 0.5s ease;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .stText, .stMarkdown {{
            color: #ffffff !important;
            text-shadow: 0 0 4px rgba(255, 255, 255, 0.2);
        }}

        .stButton > button {{
            background-color: #bb86fc !important;
            color: #000000 !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 0 12px 4px rgba(187, 134, 252, 0.7);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: #d0a6ff !important;
            box-shadow: 0 0 18px 6px rgba(187, 134, 252, 0.9);
        }}

        .stSlider > div {{
            background-color: rgba(128, 0, 128, 0.3);
            border-radius: 8px;
        }}

        .result-box {{
            background: rgba(75, 0, 130, 0.95) !important;
            color: #ffffff !important;
            font-weight: 900!important;
            padding: 1rem !important;
            border-radius: 16px !important;
            box-shadow: 0 0 15px 6px rgba(138, 43, 226, 0.9) !important;
            margin-top: 1rem !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# Initialize session state if needed
if "intensified" not in st.session_state:
    st.session_state.intensified = False

# Call the function
set_purple_background("gail_bg.jpg", intensified=st.session_state.intensified)
st.markdown("""
---
<center><span style="font-size:20px; font-weight:600;">Developed by Lokesh Deshwal | 2025</span></center>
""", unsafe_allow_html=True)


st.markdown("""
---
<center>
    <p style="font-size:16px; color:gray; font-weight:500; margin-top:20px;">
        &copy; 2025 Lokesh Deshwal. All rights reserved.
    </p>
</center>
""", unsafe_allow_html=True)
