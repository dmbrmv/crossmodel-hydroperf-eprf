# HBV96 Model (Coming Soon)

The HBV96 bucket-type model will simulate runoff components based on precipitation and temperature inputs.

## Planned Features

- Parsimonious bucket scheme with soil moisture accounting.
- Model equations implemented in NumPy for speed.
- Calibration via Optuna with NSE and KGE objectives.
- Ability to run multiple basins in parallel.

## Proposed API

```python
from src.models.hbv96 import HBV96Model
from src.models.hbv96.optimizer import optimize_hbv96

# Initialize model and optimizer
model = HBV96Model()
best_params = optimize_hbv96(model, precip, temp, obs)

# Simulate streamflow
simulated = model.run(best_params, precip, temp)
```
