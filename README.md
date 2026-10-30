# Campus Crowd Level Predictor

## Overview
This machine learning pipeline predicts crowd levels at campus facilities (such as libraries and canteens) using temporal, cyclical, and environmental data. Trained on a continuous 10-minute interval dataset spanning a full year (approx. 26,000 records), the model accurately forecasts population surges, lunch rushes, and overnight dips.

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Modeling:** Scikit-Learn (`HistGradientBoostingRegressor`)
* **Validation:** `TimeSeriesSplit`, `GridSearchCV`
* **Visualization:** Matplotlib

## Model Architecture & Feature Engineering
Standard K-Fold validation causes data leakage in time-series data. This project utilizes an expanding chronological window (`TimeSeriesSplit`) alongside a natively optimized Gradient Boosting algorithm.

**Key Features Engineered:**
* **Cyclical Time Encoding:** Sine and cosine transformations applied to hours and days to represent the continuous nature of a 24-hour clock.
* **Lag Features:** Captured crowd states at $t-10$ mins, $t-1$ hour, and $t-24$ hours.
* **Rolling Statistics:** 1-hour moving averages and standard deviations to detect sudden traffic surges.

## Performance Metrics
The model was tuned across 80 fits and evaluated on an unseen chronological test set, yielding highly accurate tracking of both daytime peaks and overnight zeroes.

* **Final Mean Absolute Error (MAE):** 2.92 people
* **Final Root Mean Squared Error (RMSE):** 4.42 people

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/Mittal-Sankalp/campus-crowd-predictor.git](https://github.com/Mittal-Sankalp/campus-crowd-predictor.git)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run all the blocks of `Smart_Campus_Assistant.ipynb`
