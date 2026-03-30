import pandas as pd
import numpy as np

# --------------------------
# REALISM SETTINGS (TUNE THESE)
# --------------------------
FINISH_NOISE_SD = 8.0       # seconds of race-day randomness (try 5–15)
SPLIT_MEASUREMENT_SD = 1.0  # GPS/timing error per mile
OUTLIER_PROB = 0.08         # chance of weird mile
OUTLIER_SD = 12.0           # seconds added/removed during weird mile

# Load your data
df = pd.read_csv("finish_time_matrix.csv").dropna()
df["5000_finish_s"] = pd.to_numeric(df["5000_finish_s"], errors="coerce")
df = df.dropna(subset=["5000_finish_s"]).copy()

T = df["5000_finish_s"].astype(float).values

DIST_MILES = 5000 / 1609.344
LAST_SEG = DIST_MILES - 3.0

rng = np.random.default_rng(42)

# --------------------------
# Finish kick mixture model
# --------------------------
def sample_last_multiplier():
    mode = rng.choice(["kick", "normal", "fade"], p=[0.25, 0.70, 0.05])

    if mode == "kick":
        m_last = rng.normal(0.90, 0.06)
    elif mode == "normal":
        m_last = rng.normal(0.98, 0.08)
    else:
        m_last = rng.normal(1.12, 0.06)

    return float(np.clip(m_last, 0.80, 1.25))


def sample_pattern_multipliers():
    archetype = rng.choice(
        ["even", "positive", "negative", "u_shape", "random"],
        p=[0.35, 0.20, 0.20, 0.15, 0.10]
    )

    sd = 0.03
    base = rng.normal(1.0, sd, size=3)

    if archetype == "positive":
        base[0] -= abs(rng.normal(0.03, 0.015))
        base[2] += abs(rng.normal(0.04, 0.02))
    elif archetype == "negative":
        base[0] += abs(rng.normal(0.03, 0.015))
        base[2] -= abs(rng.normal(0.04, 0.02))
    elif archetype == "u_shape":
        base[1] += abs(rng.normal(0.05, 0.02))

    base = np.clip(base, 0.85, 1.18)
    return base[0], base[1], base[2], sample_last_multiplier()


def make_splits(total_s: float):

    pace = total_s / DIST_MILES

    m1, m2, m3, m_last = sample_pattern_multipliers()

    t_last = pace * m_last * LAST_SEG
    remaining = total_s - t_last

    weights = np.array([m1, m2, m3])
    t1, t2, t3 = remaining * (weights / weights.sum())

    # Split measurement noise (GPS / human error)
    split_noise = rng.normal(0, SPLIT_MEASUREMENT_SD, size=3)
    t1 += split_noise[0]
    t2 += split_noise[1]
    t3 += split_noise[2]

    # Occasional weird mile (hill, surge, fade)
    if rng.random() < OUTLIER_PROB:
        j = rng.integers(0, 3)
        bump = rng.normal(0, OUTLIER_SD)
        if j == 0: t1 += bump
        if j == 1: t2 += bump
        if j == 2: t3 += bump

    # Race-day finish randomness (label noise)
    finish_noise = rng.normal(0, FINISH_NOISE_SD)

    final_finish = t1 + t2 + t3 + t_last + finish_noise

    return np.round(t1,1), np.round(t2,1), np.round(t3,1), np.round(final_finish,1)


# Generate dataset
splits = np.array([make_splits(float(t)) for t in T])
mile1_s, mile2_s, mile3_s, finish_s = splits.T

df_out = df.copy()
df_out["mile1_s"] = mile1_s
df_out["mile2_s"] = mile2_s
df_out["mile3_s"] = mile3_s
df_out["5000_finish_s"] = finish_s

df_out.to_csv("finish_time_with_splits.csv", index=False)

print("Saved as finish_time_with_splits.csv")
print(df_out[["mile1_s","mile2_s","mile3_s","5000_finish_s"]].head())
