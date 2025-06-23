# CatBoost Model

The CatBoost model implementation provides a gradient boosting regression approach optimized for hydrological discharge prediction.

## Overview

- Utilizes `catboost.CatBoostRegressor` with support for GPU acceleration.
- Hyperparameters are tuned using Optuna.
- Handles categorical features natively (e.g., `gauge_id`).

## Data Preparation

1. Static features are loaded and selected using `find_valid_gauges` and `select_uncorrelated_features`.
2. Time series data is created via the `data_creator` function (see `src/readers/hydro_data_reader.py`).
3. Data is split into training, validation, and test sets by date ranges.

## Usage

```bash
python catboost_optuna.py
```

Or interactively in the Jupyter notebook:

```python
from catboost import CatBoostRegressor, Pool

# Load prepared pools
train_pool, valid_pool, test_pool = create_pools(...)

# Define base parameters
base_params = {
    'loss_function': 'RMSE',
    'random_seed': 42,
    'task_type': 'GPU',
    'verbose': 0,
}

# Optimize hyperparameters
best_params = optimize_catboost(base_params, train_pool, valid_pool)

# Train final model
model = CatBoostRegressor(**best_params)
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# Evaluate on test
y_pred = model.predict(test_pool)
```

## Configuration

- Configurable parameters include:
  - `iterations`, `learning_rate`, `depth`
  - `l2_leaf_reg`, `random_strength`, `bagging_temperature`

- Optuna optimization settings are defined in `src/models/catboost/optimizer.py`.

## Performance Metrics

- **Kling-Gupta Efficiency (KGE)** is used as the optimization objective.
- **RMSE** and **MAPE** are available as evaluation metrics.
