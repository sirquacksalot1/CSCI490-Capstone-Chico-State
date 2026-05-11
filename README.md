# Documentation for CSCI490 Capstone

## Project Overview

This project focuses on predicting 5K race performance using machine learning models trained on both real and synthetically generated split data. It includes tools for data scraping, preprocessing, synthetic data generation, and model training/evaluation.

The repository contains:

* Scripts for generating synthetic race splits
* Machine learning models (MLP, RNN, etc.)
* Pretrained model artifacts and scalers
* Historical experiments and earlier iterations
* Scrapers for collecting race data

---

## Repository Structure

```
.
├── synthetic/             # Synthetic data + split generator
├── scrapers/              # Scripts to scrape race/split data
├── models/                # Saved models and training outputs
├── history_of_project/    # Older experiments and prototype code
├── oldRealData/           # Early real datasets and matrices
├── *.keras / *.pkl        # Trained models and scalers
```

---

## How to Run

* The following programs have been run on Python 3.10.12 and WSL version 2.6.3.0

1. Install dependencies:

```
pip install -r requirements.txt
```

* Uses `finish_time_matrix.csv` to generate realistic split times

2. Train the model:

```
python3 train_model_complete.py
```

* Outputs graphs and saves trained model artifacts

3. Run the app:

```
streamlit run app_complete.py
```

* Runs minimal streamlit app
---

## Data Sources

### Synthetic Data

* Located in `synthetic/`
* Generated to simulate realistic pacing/split behavior

### Real Data

* Final models trained from `LapSplits_5k_clean.csv` 
* Cleaned uses `cleanScript.py` from `LapSplits_5kResults_with_splits.csv`
* Stored in `oldRealData/`
* Includes early training matrices and raw race data

### Scraped Data

* Located in `scrapers/`
* Contains scripts for collecting race results and split information from external sources

---

## Models

The project includes several trained models:

* MLP-based models (`models/`)
* RNN-based model (`5k_prediction_rnn.keras`)
* Scalers for preprocessing (`5k_scaler.pkl`)

Training outputs (graphs, figures) are also stored in the `models/` directory.

---

## Helpful Commands

* `sudo apt install python3.10-venv` - Install venv for virtual environment
* `python -m venv venv` — Create virtual environment
* `source venv/bin/activate` — Activate virtual environment

---

## Notes / Future Work

* Improve model accuracy with more real-world data consiting of average runners
* Expand to other race distances

---

