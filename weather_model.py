import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("updated_weather_data.csv")
# Group similar weather descriptions
def group_weather(desc):
    desc = str(desc).lower()
    if "clear" in desc or "sunny" in desc or "fair" in desc:
        return "Clear"
    elif "moderate" in desc or "showers" in desc:
        return "Moderate Rainfall"
    elif "heavy" in desc or "flood" in desc:
        return "Heavy Rainfall"
    elif "storm" in desc or "cyclone" in desc or "thunder" in desc:
        return "Storm"
    else:
        return "Cloudy"

df['Weather_Description'] = df['Weather_Description'].apply(group_weather)


# Label encode Weather_Description
le = LabelEncoder()
df['Weather_Description'] = le.fit_transform(df['Weather_Description'])

# Save encoder
joblib.dump(le, "label_encoder.pkl")

# 🔄 Balance Weather_Description Classes
max_size = df['Weather_Description'].value_counts().max()

dfs = []
for class_index in df['Weather_Description'].unique():
    df_class = df[df['Weather_Description'] == class_index]
    df_upsampled = resample(df_class, replace=True, n_samples=max_size, random_state=42)
    dfs.append(df_upsampled)

df_balanced = pd.concat(dfs)

# 🔄 Re-map Work_Suitability
df_balanced['Work_Suitability'] = df_balanced['Work_Suitability'].map({'Yes': 1, 'No': 0})

# Features and targets
X = df_balanced[['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh', 'Weather_Description']]
y = df_balanced['Work_Suitability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
work_model = RandomForestClassifier(random_state=42)
work_model.fit(X_train, y_train)

# Save model
joblib.dump(work_model, "weather_model.pkl")

# Accuracy
y_pred = work_model.predict(X_test)
print(f"✅ Work Suitability Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# ----------------------------
# Train Weather Description Model
# ----------------------------
X_desc = df_balanced[['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh']]
y_desc = df_balanced['Weather_Description']

X_desc_train, X_desc_test, y_desc_train, y_desc_test = train_test_split(X_desc, y_desc, test_size=0.2, random_state=42)

weather_desc_model = RandomForestClassifier(random_state=42)
weather_desc_model.fit(X_desc_train, y_desc_train)

# Save model
joblib.dump(weather_desc_model, "weather_description_model.pkl")

# Evaluate
desc_pred = weather_desc_model.predict(X_desc_test)
print(f"✅ Weather Description Accuracy: {accuracy_score(y_desc_test, desc_pred)*100:.2f}%")
