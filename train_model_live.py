import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


DATA_PATH = "LapSplits_5k_clean.csv"
RANDOM_STATE = 2
TEST_SIZE = 0.30
VALIDATION_SPLIT = 0.30
EPOCHS = 120
BATCH_SIZE = 16
OUTPUT_DIR = "."

STAGES = {
    "1k": ["1000_split"],
    "2k": ["1000_split", "2000_split"],
    "3k": ["1000_split", "2000_split", "3000_split"],
    "4k": ["1000_split", "2000_split", "3000_split", "4000_split"],
}


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed_cols = [
        "1000_split",
        "2000_split",
        "3000_split",
        "4000_split",
        "ResultTime",
        "gender",
    ]
    df = df.dropna(subset=needed_cols).copy()

    numeric_cols = needed_cols
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    # Keep only positive elapsed times.
    for col in ["1000_split", "2000_split", "3000_split", "4000_split", "ResultTime"]:
        df = df[df[col] > 0]

    # Keep only binary gender values if that is how the current project is encoded.
    df = df[df["gender"].isin([0, 1])].copy()

    return df



def build_lstm_model(timesteps: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(timesteps, 2)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model



def prepare_sequence_inputs(
    X_splits: np.ndarray,
    X_gender: np.ndarray,
    scaler: StandardScaler,
    fit_scaler: bool,
) -> np.ndarray:
    if fit_scaler:
        X_splits_scaled = scaler.fit_transform(X_splits)
    else:
        X_splits_scaled = scaler.transform(X_splits)

    timesteps = X_splits_scaled.shape[1]
    X_splits_seq = X_splits_scaled.reshape((X_splits_scaled.shape[0], timesteps, 1))
    X_gender_seq = np.repeat(X_gender[:, np.newaxis, :], timesteps, axis=1)
    return np.concatenate([X_splits_seq, X_gender_seq], axis=2)



def train_stage_model(df: pd.DataFrame, stage_name: str, split_cols: list[str]) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Training stage: {stage_name.upper()} | Inputs: {split_cols}")
    print(f"{'=' * 60}")

    X_splits = df[split_cols].astype("float32").values
    X_gender = df["gender"].astype("float32").values.reshape(-1, 1)
    y = df["ResultTime"].astype("float32").values

    X_splits_train, X_splits_test, X_gender_train, X_gender_test, y_train, y_test = train_test_split(
        X_splits,
        X_gender,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = prepare_sequence_inputs(X_splits_train, X_gender_train, scaler, fit_scaler=True)
    X_test = prepare_sequence_inputs(X_splits_test, X_gender_test, scaler, fit_scaler=False)

    model = build_lstm_model(timesteps=len(split_cols))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=12,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping],
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    test_mae_check = mean_absolute_error(y_test, y_pred)

    model_path = os.path.join(OUTPUT_DIR, f"5k_prediction_rnn_{stage_name}.keras")
    scaler_path = os.path.join(OUTPUT_DIR, f"5k_scaler_{stage_name}.pkl")
    plot_path = os.path.join(OUTPUT_DIR, f"training_history_{stage_name}.png")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (seconds)")
    plt.title(f"Training vs Validation MAE ({stage_name.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved plot:   {plot_path}")
    print(f"Test MAE: {test_mae:.2f} seconds")

    return {
        "stage": stage_name,
        "inputs": ", ".join(split_cols),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_mae_seconds": round(float(test_mae), 2),
        "test_mae_check_seconds": round(float(test_mae_check), 2),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "plot_path": plot_path,
    }



def main() -> None:
    df = load_and_clean_data(DATA_PATH)
    print(f"Loaded {len(df)} cleaned rows from {DATA_PATH}")

    results = []
    for stage_name, split_cols in STAGES.items():
        results.append(train_stage_model(df, stage_name, split_cols))

    results_df = pd.DataFrame(results)
    summary_path = os.path.join(OUTPUT_DIR, "live_model_results.csv")
    results_df.to_csv(summary_path, index=False)

    print("\nStage-by-stage results")
    print(results_df[["stage", "inputs", "test_mae_seconds"]].to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
