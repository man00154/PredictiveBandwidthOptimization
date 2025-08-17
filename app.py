import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI Bandwidth Optimizer",
    page_icon="📶",
    layout="wide",
)

st.title("📶 MANISH - Proactive Bandwidth Optimizer")
st.subheader("Leveraging simple predictive analytics to forecast network traffic and prevent congestion.")

# --- Data Generation and Prediction ---
# This function simulates generating historical network data.
# In a real-world scenario, this would come from a database or API.
def generate_historical_data(periods=100):
    """
    Generates a mock DataFrame with historical bandwidth usage data.
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='D')
    # Create a base sine wave to simulate daily/weekly patterns
    sine_wave = np.sin(np.linspace(0, 4 * np.pi, periods)) * 20
    # Add a linear trend and some random noise
    linear_trend = np.arange(periods) * 0.5
    noise = np.random.normal(0, 5, periods)
    
    usage = (100 + linear_trend + sine_wave + noise).astype(int)
    
    df = pd.DataFrame({
        'Date': dates,
        'Bandwidth Usage (Mbps)': usage
    })
    df.set_index('Date', inplace=True)
    return df

# Generate data for the app
df_historical = generate_historical_data()

# --- Predictive Model ---
# Using a very simple linear regression model for prediction
# This demonstrates the "predictive" part with very low overhead.
# In a real app, you would use a more sophisticated model (e.g., ARIMA, Prophet, etc.).
def get_predictions(df, future_days=7):
    """
    Trains a simple linear regression model and predicts future usage.
    """
    # Prepare the data
    df['Day_Index'] = np.arange(len(df))
    X = df[['Day_Index']]
    y = df['Bandwidth Usage (Mbps)']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future values
    future_indices = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    predictions = model.predict(future_indices)
    
    # Create a DataFrame for the predictions
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
    df_predictions = pd.DataFrame({
        'Date': future_dates,
        'Bandwidth Usage (Mbps)': predictions
    })
    df_predictions.set_index('Date', inplace=True)
    
    return df_predictions

df_future = get_predictions(df_historical)

# Combine historical and future data for plotting
df_combined = pd.concat([df_historical, df_future])

# --- GenAI Mock Function ---
# This function simulates a call to a GenAI model (e.g., Gemini)
# to generate human-readable optimization recommendations.
def generate_genai_recommendations(predictions):
    """
    Mocks a GenAI-powered recommendation engine.
    In a real app, this would use an LLM API to analyze predictions.
    """
    peak_prediction = predictions['Bandwidth Usage (Mbps)'].max()
    peak_date = predictions['Bandwidth Usage (Mbps)'].idxmax().strftime('%A, %B %d')

    recommendations = (
        f"Based on the predictive analysis, network usage is expected to peak at approximately "
        f"**{peak_prediction:.2f} Mbps** on **{peak_date}**. To proactively optimize, consider the following actions:\n\n"
        f"- **Resource Pre-allocation:** Increase bandwidth allocation for critical services and applications by 15-20% on {peak_date} to prevent slowdowns.\n"
        f"- **Load Shifting:** Schedule large data transfers or non-critical backups to off-peak hours (e.g., late night on {peak_date}).\n"
        f"- **Service Prioritization:** Temporarily adjust QoS (Quality of Service) policies to prioritize voice and video traffic during the predicted peak window."
    )
    return recommendations

# Get the mock recommendations
recommendations = generate_genai_recommendations(df_future)

# --- Streamlit UI Components ---
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical & Predicted Bandwidth Usage")
    st.line_chart(df_combined, use_container_width=True)
    st.info("The line chart shows historical bandwidth usage and future predictions.")

with col2:
    st.subheader("GenAI-Powered Optimization Recommendations")
    st.success(recommendations)
    st.info("These recommendations are generated automatically to prevent network congestion before it occurs.")

st.markdown("---")

st.info("This is a simple proof-of-concept. A production application would use a more complex predictive model and integrate with a real GenAI API for dynamic analysis and recommendations.")
