# HydroEPRF

HydroEPRF (Hydrological Ensemble Prediction Research Framework) is a modular, extensible Python toolkit for hydrological modeling, data preparation, and performance evaluation. It currently supports CatBoost-based machine learning models and the GR4J hydrological model with Optuna hyperparameter optimization. Planned additions include LSTM, HBV96, and ParFlow model implementations.

## Features

- Data loading and preprocessing pipeline for meteorological and hydrological datasets
- Static feature engineering and selection utilities
- CatBoost-based regression model with Optuna hyperparameter tuning
- GR4J model wrapper and optimization scripts
- Interactive Jupyter notebooks for model exploration and visualization
- Logging and reproducibility: fixed random seeds, structured log files
- Modular architecture: easily add new model types (e.g., LSTM, HBV96, ParFlow)

## Repository Structure

```bash
├── README.md
├── LICENSE
├── data/                     # Raw and processed datasets
├── docs/                     # Detailed documentation files
├── images/                   # Figures and plots
├── logs/                     # Log files
├── notebooks/                # Exploratory and tutorial notebooks
├── src/                      # Core library modules
│   ├── models/               # Model implementations and utilities
│   │   ├── catboost/         # CatBoost data loaders and optimizer
│   │   └── ...               # Future models: lstm, hbv96, parflow
│   └── readers/              # Data reading and geospatial utilities
├── catboost_optuna.py        # Script for CatBoost optimization
├── gr4j_optuna.py            # Script for GR4J optimization
└── requirements.txt          # Python dependencies
```

## Installation

See [Installation Guide](docs/installation.md) for detailed setup instructions.


## Quick Start

Option A: Python virtual environment

1. Create and activate a Python 3.10+ environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

Option B: Conda environment (recommended for geospatial dependencies)

```bash
conda env create -f geo_env.yml
conda activate Geo
```

1. Install Python dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

1. Prepare data directories under `data/` as described in the [Installation Guide](docs/installation.md).

1. Run model optimization scripts:

   - **CatBoost**:

     ```bash
     python catboost_optuna.py
     ```

   - **GR4J**:

     ```bash
     python gr4j_optuna.py
     ```

1. (Optional) Execute additional model workflows as they become available (LSTM, HBV96, ParFlow).

1. Explore results and visualizations in Jupyter notebooks under `notebooks/`:

   ```bash
   jupyter lab notebooks/
   ```

## Models

- **CatBoost**: Gradient boosting regressor with Optuna tuning. See [CatBoost Model](docs/catboost.md).
- **GR4J**: Conceptual rainfall–runoff model wrapped in Python. See [GR4J Model](docs/gr4j.md).
- **LSTM**: Recurrent neural network for sequence forecasting. (Coming soon) See [LSTM Model](docs/lstm.md).
- **HBV96**: Bucket-type hydrological model. (Coming soon) See [HBV96 Model](docs/hbv96.md).
- **ParFlow**: Distributed hydrologic flow simulator. (Coming soon) See [ParFlow Model](docs/parflow.md).

## Publication

- **New**: The latest CatBoost model, trained on an expanded dataset of 302 basins, achieves a median Nash-Sutcliffe Efficiency (NSE) of 0.82 on the test set.
- Comprehensive comparison of four meteorological input data sources (E-OBS, ERA5-Land, MSWEP, interpolated station) across five modeling frameworks (CatBoost, LSTM, GR4J, HBV96, ParFlow).
- ERA5-Land yields the highest Nash–Sutcliffe Efficiency (NSE) for machine learning models; MSWEP excels for conceptual models.
- Interpolated station data shows strengths in snow-dominated catchments, highlighting catchment-specific sensitivities.
- Results demonstrate the critical impact of meteorological forcing choice on model performance and computational efficiency.
- View full details and quantitative results in [Comparative Analysis Publication](docs/publication.tex).

## Documentation

The `docs/` folder contains detailed information on installation, usage, and model-specific guidelines. Browse the markdown files or view the rendered site if you host the docs.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or documentation improvements. Follow the contributing guidelines in [Contributing Guide](docs/contributing.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
