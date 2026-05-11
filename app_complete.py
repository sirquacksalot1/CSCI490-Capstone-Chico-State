import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px

# Starter code generated with ChatGPT OpenAI

st.set_page_config(page_title="Live Race Finish Time Predictor", layout="centered")

RACE_CONFIG = {
    "5K": {
        "title": "Live 5K Finish Time Predictor",
        "description": (
            "Update the expected final 5K finish time during the race. "
            "Choose the current checkpoint, enter the cumulative splits known so far, "
            "and the app will use the stage-specific model."
        ),
        "prediction_label": "Predicted 5K Finish Time",
        "stages": {
            "1K": {
                "split_cols": ["1000_split"],
                "input_labels": ["1000m cumulative split (seconds)"],
                "model_path": "5k_prediction_rnn_1k.keras",
                "scaler_path": "5k_scaler_1k.pkl",
            },
            "2K": {
                "split_cols": ["1000_split", "2000_split"],
                "input_labels": [
                    "1000m cumulative split (seconds)",
                    "2000m cumulative split (seconds)",
                ],
                "model_path": "5k_prediction_rnn_2k.keras",
                "scaler_path": "5k_scaler_2k.pkl",
            },
            "3K": {
                "split_cols": ["1000_split", "2000_split", "3000_split"],
                "input_labels": [
                    "1000m cumulative split (seconds)",
                    "2000m cumulative split (seconds)",
                    "3000m cumulative split (seconds)",
                ],
                "model_path": "5k_prediction_rnn_3k.keras",
                "scaler_path": "5k_scaler_3k.pkl",
            },
            "4K": {
                "split_cols": ["1000_split", "2000_split", "3000_split", "4000_split"],
                "input_labels": [
                    "1000m cumulative split (seconds)",
                    "2000m cumulative split (seconds)",
                    "3000m cumulative split (seconds)",
                    "4000m cumulative split (seconds)",
                ],
                "model_path": "5k_prediction_rnn_4k.keras",
                "scaler_path": "5k_scaler_4k.pkl",
            },
        },
    },
    "Estimated 2 Mile": {
        "title": "Estimated 2 Mile Predictor",
        "description": (
            "Predict an estimated 2 mile / 3200m time using models trained from the same 5K split data. "
            
        ),
        "prediction_label": "Predicted Estimated 2 Mile Time",
        "stages": {
            "1K": {
                "split_cols": ["1000_split"],
                "input_labels": ["1000m cumulative split (seconds)"],
                "model_path": "2mile_prediction_rnn_1k.keras",
                "scaler_path": "2mile_scaler_1k.pkl",
            },
            "2K": {
                "split_cols": ["1000_split", "2000_split"],
                "input_labels": [
                    "1000m cumulative split (seconds)",
                    "2000m cumulative split (seconds)",
                ],
                "model_path": "2mile_prediction_rnn_2k.keras",
                "scaler_path": "2mile_scaler_2k.pkl",
            },
            "3K": {
                "split_cols": ["1000_split", "2000_split", "3000_split"],
                "input_labels": [
                    "1000m cumulative split (seconds)",
                    "2000m cumulative split (seconds)",
                    "3000m cumulative split (seconds)",
                ],
                "model_path": "2mile_prediction_rnn_3k.keras",
                "scaler_path": "2mile_scaler_3k.pkl",
            },
        },
    },
}

DEFAULT_SPLITS = {
    "1000_split": 175.2,
    "2000_split": 354.2,
    "3000_split": 531.9,
    "4000_split": 709.1,
}


def current_stage_config(race_name: str, stage_name: str) -> dict:
    return RACE_CONFIG[race_name]["stages"][stage_name]


def initialize_session_state():
    for col_name, default_value in DEFAULT_SPLITS.items():
        if col_name not in st.session_state:
            st.session_state[col_name] = float(default_value)

        widget_key = f"widget_{col_name}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = float(default_value)

    if "race_name" not in st.session_state:
        st.session_state["race_name"] = "5K"

    if "gender_label" not in st.session_state:
        st.session_state["gender_label"] = "Male"

    if "stage_name" not in st.session_state:
        st.session_state["stage_name"] = "4K"

    if "confirmed_splits" not in st.session_state:
        st.session_state["confirmed_splits"] = {}

    if "last_prediction_seconds" not in st.session_state:
        st.session_state["last_prediction_seconds"] = None

    if "last_prediction_stage" not in st.session_state:
        st.session_state["last_prediction_stage"] = None

    if "last_prediction_race" not in st.session_state:
        st.session_state["last_prediction_race"] = None


def reset_prediction_state():
    st.session_state["confirmed_splits"] = {}
    st.session_state["last_prediction_seconds"] = None
    st.session_state["last_prediction_stage"] = None
    st.session_state["last_prediction_race"] = None


@st.cache_resource
def load_stage_assets(race_name: str, stage_name: str):
    config = current_stage_config(race_name, stage_name)
    model_path = config["model_path"]
    scaler_path = config["scaler_path"]

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing files for {race_name} {stage_name}. Expected {model_path} and {scaler_path}."
        )

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def seconds_to_mmss(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds)))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def validate_cumulative_splits(splits: list[float]) -> list[str]:
    errors = []
    if any(value <= 0 for value in splits):
        errors.append("All entered split values must be positive.")

    for earlier, later in zip(splits, splits[1:]):
        if later <= earlier:
            errors.append("Cumulative split times must increase as distance increases.")
            break

    return errors


def prepare_model_input(splits: list[float], gender_value: int, scaler) -> np.ndarray:
    x_splits = np.array([splits], dtype=np.float32)
    x_splits_scaled = scaler.transform(x_splits)
    timesteps = x_splits_scaled.shape[1]
    x_splits_scaled = x_splits_scaled.reshape((1, timesteps, 1))
    x_gender = np.full((1, timesteps, 1), float(gender_value), dtype=np.float32)
    return np.concatenate([x_splits_scaled, x_gender], axis=2)


def predict_stage(race_name: str, stage_name: str, splits: list[float], gender_value: int) -> float:
    model, scaler = load_stage_assets(race_name, stage_name)
    x_final = prepare_model_input(splits, gender_value, scaler)
    prediction = model.predict(x_final, verbose=0)[0][0]
    return float(prediction)


def build_progressive_predictions(
    race_name: str,
    confirmed_splits: dict[str, float],
    gender_value: int,
) -> pd.DataFrame:
    rows = []
    for stage_name, config in RACE_CONFIG[race_name]["stages"].items():
        split_cols = config["split_cols"]
        if all(col in confirmed_splits for col in split_cols):
            splits = [confirmed_splits[col] for col in split_cols]
            try:
                pred_seconds = predict_stage(race_name, stage_name, splits, gender_value)
                rows.append(
                    {
                        "Stage": stage_name,
                        "Predicted Time (sec)": pred_seconds,
                        "Predicted Time": seconds_to_mmss(pred_seconds),
                    }
                )
            except FileNotFoundError:
                pass
    return pd.DataFrame(rows)


def parse_time_input(value, mode="Seconds"):
    if mode == "Seconds":
        try:
            return float(value)
        except Exception:
            return None
    try:
        parts = str(value).strip().split(":")
        if len(parts) != 2:
            return None
        minutes = int(parts[0])
        seconds = int(parts[1])
        if seconds < 0 or seconds >= 60:
            return None
        return minutes * 60 + seconds
    except Exception:
        return None


initialize_session_state()

st.title("Live Race Finish Time Predictor")

race_name = st.selectbox(
    "Prediction type",
    list(RACE_CONFIG.keys()),
    key="race_name",
    on_change=reset_prediction_state,
)

st.subheader(RACE_CONFIG[race_name]["title"])
st.write(RACE_CONFIG[race_name]["description"])


time_format = st.radio(
    "Split input format",
    ["Seconds", "MM:SS"],
    horizontal=True,
)

available_stages = list(RACE_CONFIG[race_name]["stages"].keys())
if st.session_state["stage_name"] not in available_stages:
    st.session_state["stage_name"] = available_stages[-1]

stage_name = st.selectbox(
    "Current race checkpoint",
    available_stages,
    key="stage_name",
)

gender_label = st.selectbox(
    "Gender",
    ["Male", "Female"],
    key="gender_label",
)

current_config = current_stage_config(race_name, stage_name)

if st.button("Reset App"):
    keys_to_reset = [
        "race_name",
        "gender_label",
        "stage_name",
        "confirmed_splits",
        "last_prediction_seconds",
        "last_prediction_stage",
        "last_prediction_race",
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    for col_name in DEFAULT_SPLITS:
        if col_name in st.session_state:
            del st.session_state[col_name]

        widget_key = f"widget_{col_name}"
        if widget_key in st.session_state:
            del st.session_state[widget_key]

    st.rerun()

# Inputs
for col_name, label in zip(current_config["split_cols"], current_config["input_labels"]):
    widget_key = f"widget_{col_name}"

    if time_format == "Seconds":
        if isinstance(st.session_state.get(widget_key), str):
            parsed = parse_time_input(st.session_state[widget_key], "MM:SS")
            st.session_state[widget_key] = float(parsed if parsed is not None else st.session_state[col_name])

        st.number_input(
            label,
            min_value=0.0,
            step=5.0,
            key=widget_key,
        )
    else:
        if isinstance(st.session_state.get(widget_key), float):
            st.session_state[widget_key] = seconds_to_mmss(st.session_state[widget_key])

        st.text_input(
            label.replace("(seconds)", "(MM:SS)"),
            key=widget_key,
        )

# Prediction input handling
if st.button("Update Live Prediction"):
    gender_value = 0 if st.session_state["gender_label"] == "Male" else 1

    ordered_splits = []

    for col in current_config["split_cols"]:
        raw_value = st.session_state[f"widget_{col}"]
        parsed_value = parse_time_input(raw_value, time_format)

        if parsed_value is None:
            st.error(f"Invalid time format for {col}. Use MM:SS like 5:42.")
            st.stop()

        ordered_splits.append(parsed_value)
        st.session_state[col] = parsed_value

    validation_errors = validate_cumulative_splits(ordered_splits)

    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        try:
            prediction_seconds = predict_stage(race_name, stage_name, ordered_splits, gender_value)

            st.session_state["last_prediction_seconds"] = prediction_seconds
            st.session_state["last_prediction_stage"] = stage_name
            st.session_state["last_prediction_race"] = race_name

            confirmed = dict(st.session_state["confirmed_splits"])
            for col in current_config["split_cols"]:
                confirmed[col] = st.session_state[col]
            st.session_state["confirmed_splits"] = confirmed

            st.success("Live prediction updated.")

        except FileNotFoundError as exc:
            st.error(str(exc))
            st.warning(
                "Run train_model_live_with_2mile.py first so the stage-specific model and scaler files exist before launching the live app."
            )

# Outputs
if st.session_state["last_prediction_seconds"] is not None:
    metric_label = RACE_CONFIG[st.session_state["last_prediction_race"]]["prediction_label"]
    st.metric(
        f"{metric_label} (seconds)",
        f"{st.session_state['last_prediction_seconds']:.1f}",
    )
    st.metric(
        f"{metric_label} (mm:ss)",
        seconds_to_mmss(st.session_state["last_prediction_seconds"]),
    )

last_race = st.session_state["last_prediction_race"]
last_stage = st.session_state["last_prediction_stage"]

if last_race is not None and last_stage is not None:
    final_stage = list(RACE_CONFIG[last_race]["stages"].keys())[-1]

    if last_stage == final_stage:
        confirmed_splits = st.session_state["confirmed_splits"]
        gender_value = 0 if st.session_state["gender_label"] == "Male" else 1
        final_split_cols = RACE_CONFIG[last_race]["stages"][final_stage]["split_cols"]

        if all(col in confirmed_splits for col in final_split_cols):
            progression_df = build_progressive_predictions(last_race, confirmed_splits, gender_value)

            if not progression_df.empty:
                st.subheader("Prediction progression")
                st.dataframe(progression_df, use_container_width=True, hide_index=True)

                fig = px.line(
                    progression_df,
                    x="Stage",
                    y="Predicted Time (sec)",
                    markers=True,
                    title="How the predicted time changes during the race",
                )
                st.plotly_chart(fig, use_container_width=True)
