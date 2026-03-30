import streamlit as st
import pandas as pd
import numpy as np

# ---------------- TOP ----------------
st.title("AI Epidemic Spread Prediction Dashboard")
st.info("This tool predicts outbreak trends using machine learning")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/time_series_covid19_confirmed_global.csv")
df = df.drop(["Lat", "Long"], axis=1)
df = df.groupby("Country/Region").sum()

# ---------------- USER INPUT ----------------
country = st.selectbox("Select Country", df.index)
data = df.loc[country]

# ---------------- HISTORICAL DATA ----------------
st.header("📊 Historical Data")
st.line_chart(data)

# ---------------- FAKE PREDICTION ----------------
future_days = 7
last_value = data.values[-1]

predicted = [last_value + i*1000 for i in range(1, future_days+1)]

# ---------------- METRICS ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Current Cases", int(last_value))

with col2:
    st.metric("Predicted (7 days)", int(predicted[-1]))

# ---------------- PREDICTION SECTION ----------------
st.header("🔮 Prediction")

st.write("### Predicted Cases (Next 7 Days)")
st.write(predicted)

# Prediction Graph
future_dates = pd.date_range(start=pd.to_datetime(data.index[-1]), periods=8)[1:]

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Cases": predicted
}).set_index("Date")

st.line_chart(pred_df)

# ---------------- RISK LEVEL ----------------
growth_rate = (predicted[-1] - last_value) / last_value * 100

st.write("### Risk Level")

if growth_rate < 5:
    st.success("Low Risk")
elif growth_rate < 15:
    st.warning("Medium Risk")
else:
    st.error("High Risk")

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Built for Hackathon | AI Epidemic Prediction System")