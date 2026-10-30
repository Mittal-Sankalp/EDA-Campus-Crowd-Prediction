# Campus Crowd Level Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mittal-Sankalp/EDA-Campus-Crowd-Prediction/blob/main/Smart_Campus_Assistant.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A time-series machine learning pipeline that predicts how crowded a campus gym will be at the next 10-minute interval, using time-of-day patterns, recent crowd history, and environmental and academic-calendar features.

## Dataset

The notebook downloads the **Campus Gym Crowd** dataset from [OpenML](https://www.openml.org/d/43752) (ID `43752`) at runtime, so an internet connection is required. Each record is a gym occupancy reading (`number_people`) taken at roughly 10-minute intervals, along with temperature, holiday, weekend and semester flags.

## Approach

**Validation.** Standard K-Fold would leak future information into training, so the data is split chronologically (first 80% train, last 20% test) and hyperparameters are tuned with an expanding-window `TimeSeriesSplit` (5 folds).

**Engineered features**

| Group | Features |
|---|---|
| Cyclical time | Sine/cosine of time of day and day of week, so 23:50 and 00:00 are treated as neighbours |
| Lags | Crowd size 10 minutes, 1 hour and 24 hours earlier |
| Rolling statistics | Mean and standard deviation of the previous hour (current reading excluded, so no target leakage) |
| Context | Temperature, holiday, weekend, during-semester and start-of-semester flags |

**Model.** `HistGradientBoostingRegressor`, tuned with `GridSearchCV` over learning rate, number of iterations, tree depth and L2 regularisation (16 candidates × 5 folds = 80 fits, scored on MAE). Predictions are clipped at zero since a crowd cannot be negative.

## Results

| Metric | Value |
|---|---|
| Cross-validation MAE (training data) | 4.05 people |
| **Test MAE** (held-out final 20%) | **2.92 people** |
| **Test RMSE** | **4.42 people** |

Best parameters found: `learning_rate=0.05`, `max_iter=200`, `max_depth=None`, `l2_regularization=0.1`.

The notebook also plots predicted vs. actual crowd size for the first 48 hours of the test set.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib

## Getting Started

### Option 1: Google Colab

Click the **Open in Colab** badge above and run all cells.

### Option 2: Run locally

```bash
git clone https://github.com/Mittal-Sankalp/EDA-Campus-Crowd-Prediction.git
cd EDA-Campus-Crowd-Prediction
pip install -r requirements.txt jupyter
jupyter notebook Smart_Campus_Assistant.ipynb
```

Then run all cells from top to bottom.

## Repository Structure

```
├── Smart_Campus_Assistant.ipynb   # Data loading, feature engineering, tuning, evaluation, plots
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md
```

## Limitations and Next Steps

- This is a one-step-ahead forecast: it relies on the most recent observed reading, so it is not a long-horizon forecast.
- Results are from a single chronological split and have not yet been compared against simple baselines (for example, "same as 10 minutes ago").
- Lags are row-based and assume evenly spaced 10-minute readings.

## License

Released under the [MIT License](LICENSE).
