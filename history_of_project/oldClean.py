import pandas as pd
import numpy as np

#created with ChatGPT openAI

def time_to_seconds(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float)):
        return float(x)

    x = str(x).strip()
    parts = x.split(":")

    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            return float(x)
    except:
        return np.nan

df = pd.read_csv("LapSplits_5kResults_with_splits.csv")

cols = ["1000_split", "2000_split", "3000_split", "4000_split", "ResultTime"]

for col in cols:
    df[col] = df[col].apply(time_to_seconds)

df = df.dropna(subset=cols)

df.to_csv("LapSplits_5k_clean.csv", index=False)

print("Saved cleaned dataset.")
