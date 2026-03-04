# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("finish_time_with_splits.csv")

#Drop potenital missing rows
df = df.dropna()

#Inputs / Output
df["5000_finish_s"] = pd.to_numeric(df["5000_finish_s"], errors="coerce")

#DEBUG drop rows where conversion fails
df = df.dropna(subset=["5000_finish_s"])

X = df[["mile1_s", "mile2_s", "mile3_s"]].astype("float32").values
y = df["5000_finish_s"].astype("float32").values
#Train / Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2
) #30% Test 70% Train random_state to reproduce (2 is my favorite number)

# Scalars
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Build Neural Network

model = models.Sequential([
	layers.Input(shape=(3,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

#Compile Model

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae']
)

#Train Model

history = model.fit(
    X_train,
    y_train,
    validation_split=0.3,
    epochs=100,
    batch_size=16,
    verbose=1
)

#Evaluate Model

test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"\nTest MAE: {test_mae:.2f} seconds")

#Plot Training History

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE (seconds)")
plt.legend()
plt.title("Training vs Validation MAE")
plt.show()

#Save
model.save("5k_prediction_model.keras")

print("Model saved successfully.")
