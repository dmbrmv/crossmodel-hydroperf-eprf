# ParFlow Model (Coming Soon)

The ParFlow model integration will provide an interface to the ParFlow groundwater and surface water simulator.

## Planned Features

- Python API wrapper around ParFlow executables or library.
- Preprocessing of topography and inputs via `xarray`.
- Parallel execution across distributed HPC environments.
- Postprocessing of results to extract hydrographs.
- Calibration hook with Optuna for model parameters.

## Proposed API

```python
from src.models.parflow import ParFlowModel
from src.models.parflow.optimizer import optimize_parflow

# Prepare domain input files
workflow = prepare_parflow_workflow(watershed_shapefile, dem)

# Optimize parameters
best_params = optimize_parflow(workflow, obs)

# Run simulation
results = run_parflow(best_params)
```
