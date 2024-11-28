import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import folium
from streamlit.components.v1 import html

# Function to load and preprocess data (use the same preprocessing logic)
def preprocess_data(data):
    # Define Target Variables
    data['high_risk'] = (data['incident_frequency'] > 5).astype(int)  # Threshold for high risk
    target_classification = data['high_risk']
    target_prediction = data['incident_count']  # Assuming 'incident_count' is your prediction target

    # Define Features
    features = data[['Latitude', 'Longitude', 'incident_frequency', 'avg_severity', 'proximity_to_hotspot', 'Month', 'Day_of_Week']]

    # Add Weather feature (replace with your actual weather data or simulation)
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
    data['Weather'] = np.random.choice(weather_conditions, size=len(data))

    # Include 'Weather' in features before creating dummies
    features = data[['Latitude', 'Longitude', 'incident_frequency', 'avg_severity', 'proximity_to_hotspot', 'Weather', 'Month', 'Day_of_Week']]
    features = pd.get_dummies(features, columns=['Weather'], drop_first=True)

    # Combine targets into a single array for multi-output model
    target = np.column_stack((target_classification, target_prediction))

    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)  # Use the same imputer fitted on training data

    return X_train, X_test, y_train, y_test, imputer

# Load pre-trained models (assumed models are saved using joblib or pickle)
def load_models():
    model_classification = joblib.load('modelclassification.pkl')  # Load your saved model for classification
    model_prediction = joblib.load('model_predict.pkl')  # Load your saved model for regression
    return model_classification, model_prediction

# Streamlit UI
st.title('Incident Prediction and Risk zone identification')
st.markdown("""
This application predicts the risk classification (high or low) of an incident, and estimates the number of incidents based on various features.
""")

# Inputs for prediction
latitude = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=0.0)
longitude = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=0.0)
incident_frequency = st.number_input("Enter Incident Frequency", min_value=0, value=0)
avg_severity = st.number_input("Enter Average Severity", min_value=0, value=0)
proximity_to_hotspot = st.number_input("Enter Proximity to Hotspot", min_value=0, value=0)
month = st.selectbox("Select Month", options=list(range(1, 13)), index=0)
day_of_week = st.selectbox("Select Day of the Week", options=list(range(1, 8)), index=0)

# Weather selection (optional, or simulate for now)
weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
weather = st.selectbox("Select Weather Condition", options=weather_conditions)

# Button to trigger prediction
if st.button('Predict'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Latitude': [latitude],
        'Longitude': [longitude],
        'incident_frequency': [incident_frequency],
        'avg_severity': [avg_severity],
        'proximity_to_hotspot': [proximity_to_hotspot],
        'Weather': [weather],
        'Month': [month],
        'Day_of_Week': [day_of_week]
    })

    # Apply the same preprocessing steps to the input data
    input_data = pd.get_dummies(input_data, columns=['Weather'], drop_first=True)
    
    # Ensure the correct columns are present in the input data
    required_columns = ['Latitude', 'Longitude', 'incident_frequency', 'avg_severity', 'proximity_to_hotspot', 'Month', 'Day_of_Week', 'Weather_Cloudy', 'Weather_Rainy', 'Weather_Snowy']
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Initialize the imputer (use the same one as trained)
    imputer = SimpleImputer(strategy='mean')
    input_data = imputer.fit_transform(input_data)

    # Load models
    model_classification, model_prediction = load_models()

    # Make predictions
    predicted_class = model_classification.predict(input_data)[0]
    predicted_incidents = model_prediction.predict(input_data)[0]

    # Show results
    risk_level = "High" if predicted_class == 1 else "Low"
    st.write(f"Risk Classification: {risk_level}")
    st.write(f"Predicted Incident Count: {predicted_incidents:.2f}")

    # Display the location on a map using folium
    m = folium.Map(location=[latitude, longitude], zoom_start=13)
    folium.Marker([latitude, longitude], popup=f"Prediction: {predicted_class}").add_to(m)
    
    # Save the map to an HTML file
    map_html = "map.html"
    m.save(map_html)
    
    # Display the map in Streamlit
    with open(map_html, 'r') as f:
        map_html_content = f.read()
    html(map_html_content, height=500, width=700)




