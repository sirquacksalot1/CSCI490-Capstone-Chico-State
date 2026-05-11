import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load
df = pd.read_csv("LapSplits_5k_clean.csv")

# Drop potential missing rows
needed_cols = ["1000_split", "2000_split", "3000_split", "4000_split", "ResultTime", "gender"]
df = df.dropna(subset=needed_cols)

# Inputs / Output
df["ResultTime"] = pd.to_numeric(df["ResultTime"], errors="coerce")
df["gender"] = pd.to_numeric(df["gender"], errors="coerce")

# Drop rows where conversion fails
df = df.dropna(subset=["ResultTime", "gender"])

# X and y labels for training and testing
split_cols = ["1000_split", "2000_split", "3000_split", "4000_split"]

X_splits = df[split_cols].astype("float32").values
X_gender = df["gender"].astype("float32").values.reshape(-1, 1)
y = df["ResultTime"].astype("float32").values

# Train / Test split
X_splits_train, X_splits_test, X_gender_train, X_gender_test, y_train, y_test = train_test_split(
    X_splits, X_gender, y, test_size=0.3, random_state=2
)  # 30% test, 70% train

# Scale only the split values
scaler = StandardScaler()
X_splits_train = scaler.fit_transform(X_splits_train)
X_splits_test = scaler.transform(X_splits_test)

joblib.dump(scaler, "5k_scaler.pkl")

# Reshape split features into sequence format: (samples, 4, 1)
X_splits_train = X_splits_train.reshape((X_splits_train.shape[0], 4, 1))
X_splits_test = X_splits_test.reshape((X_splits_test.shape[0], 4, 1))

# Repeat gender across all 4 timesteps: (samples, 4, 1)
X_gender_train_seq = np.repeat(X_gender_train[:, np.newaxis, :], 4, axis=1)
X_gender_test_seq = np.repeat(X_gender_test[:, np.newaxis, :], 4, axis=1)

# Combine split feature + gender feature => (samples, 4, 2)
X_train = np.concatenate([X_splits_train, X_gender_train_seq], axis=2)
X_test = np.concatenate([X_splits_test, X_gender_test_seq], axis=2)

# Build Neural Network
model = models.Sequential([
    layers.Input(shape=(4, 2)),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.3,
    epochs=200,
    batch_size=16,
    verbose=1
)

# Evaluate Model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest MAE: {test_mae:.2f} seconds")

# Plot Training History
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE (seconds)")
plt.legend()
plt.title("Training vs Validation MAE")
plt.show()

# Save
model.save("5k_prediction_rnn.keras")

print("Model saved successfully.")
