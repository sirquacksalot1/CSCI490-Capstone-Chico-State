import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# ----------------------------
# Load model and scaler
# ----------------------------
model = tf.keras.models.load_model("5k_prediction_rnn.keras")
scaler = joblib.load("5k_scaler.pkl")

# ----------------------------
# Helper functions
# ----------------------------
def seconds_to_mmss(total_seconds):
    total_seconds = int(round(total_seconds))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"

def predict_finish_time(split_1000, split_2000, split_3000, split_4000):

    # Match training feature order exactly
    x = np.array([[split_1000, split_2000, split_3000, split_4000]], dtype=np.float32)

    # Apply same scaling used during training
    x_scaled = scaler.transform(x)

    # Reshape for LSTM input: (batch_size, timesteps, features)
    x_scaled = x_scaled.reshape((1, 4, 1))

    # Predict
    pred = model.predict(x_scaled, verbose=0)[0][0]
    return float(pred)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="5K Finish Time Predictor")

st.title("5K Finish Time Predictor")
st.write("Enter your 1K split times in seconds to predict total 5K finish time.")

with st.form("prediction_form"):
    split_1000 = st.number_input("1000m split (seconds)", min_value=0.0, value=175.2, step=1.0)
    split_2000 = st.number_input("2000m split (seconds)", min_value=0.0, value=354.2, step=1.0)
    split_3000 = st.number_input("3000m split (seconds)", min_value=0.0, value=531.9, step=1.0)
    split_4000 = st.number_input("4000m split (seconds)", min_value=0.0, value=709.1, step=1.0)

    submitted = st.form_submit_button("Predict 5K Time")

if submitted:
    if min(split_1000, split_2000, split_3000, split_4000) <= 0:
        st.error("Please enter positive split times.")
    else:
        prediction_seconds = predict_finish_time(
            split_1000, split_2000, split_3000, split_4000
        )

        st.success("Prediction complete.")
        st.metric("Predicted Finish Time (seconds)", f"{prediction_seconds:.1f}")
        st.metric("Predicted Finish Time (mm:ss)", seconds_to_mmss(prediction_seconds))
