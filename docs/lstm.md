# LSTM Model (Coming Soon)

The LSTM model will implement a recurrent neural network architecture for sequence forecasting of hydrological variables.

## Planned Features

- Data preparation using PyTorch's `Dataset` and `DataLoader` abstractions.
- Sequence windowing for temporal dependencies.
- Model implemented with `torch.nn.LSTM` layers, optional attention mechanisms.
- Hyperparameter tuning via Optuna.
- Early stopping and checkpointing.
- Metrics: RMSE, MAPE, and KGE.

## Proposed API

```python
from src.models.lstm import HydrologyLSTM
from src.models.lstm.optimizer import optimize_lstm

# Prepare datasets
train_dataset, valid_dataset, test_dataset = prepare_lstm_data(...)

# Optimize LSTM hyperparameters
best_params = optimize_lstm(train_dataset, valid_dataset)

# Train final model
model = HydrologyLSTM(**best_params)
train_model(model, train_dataset, valid_dataset)

# Evaluate on test data
y_pred = evaluate_model(model, test_dataset)
```
