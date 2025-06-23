# GR4J Model

The GR4J model is a conceptual rainfall–runoff hydrological model implemented in Python.

## Overview

- Based on the GR4J algorithm for daily streamflow simulation.
- Wrapped for easy integration and parameter calibration via Optuna.
- Handles precipitation and potential evapotranspiration inputs.

## Usage

```bash
python gr4j_optuna.py
```

Or interactively in a Jupyter notebook:

```python
from src.readers.hydro_data_reader import data_creator
from src.models.gr4j.gr4j_wrapper import GR4JModel
from src.models.gr4j.optimizer import optimize_gr4j

# Prepare data arrays: precip, pet, obs
model = GR4JModel()
best_params = optimize_gr4j(model, precip, pet, obs)

# Simulate streamflow
simulated = model.run(best_params, precip, pet)
```

## Configuration

- Calibration parameters include GR4J's four reservoirs and production store components:
  - `x1`, `x2`, `x3`, `x4`
- Optimization settings are defined in `src/models/gr4j/optimizer.py`.

## Performance Metrics

- **Nash–Sutcliffe Efficiency (NSE)** and **Kling-Gupta Efficiency (KGE)** are available.

## References

- Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation.
