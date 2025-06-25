# HydroEPRF

HydroEPRF (Hydrological Ensemble Prediction Research Framework) is a modular, extensible Python toolkit for hydrological modeling, data preparation and performance evaluation. It currently includes CatBoost, GR4J and an LSTM implementation with static catchment attributes. Upcoming modules will add HBV96 and ParFlow models.

## Features

- Data loading and preprocessing pipeline for meteorological and hydrological datasets
- Static feature engineering and selection utilities
- CatBoost-based regression model with Optuna hyperparameter tuning
- GR4J model wrapper and optimization scripts
- LSTM model combining dynamic meteorology with static catchment attributes
- Robust loss options including NSE, KGE, Weighted MSE and Huber
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
│   ├── loaders/              # Data download utilities for meteorological and soil data
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

## Data Download

The framework requires several datasets for full functionality which can be downloaded using the provided utilities:

### Meteorological Data (ERA5-Land)

```bash
# Download ERA5-Land meteorological data for a specific region and period
python -m src.loaders.load_era5_land
```

This will download ERA5-Land data for variables like radiation, temperature, pressure and wind components for the specified area of interest.

### Soil Data (SoilGrids)

```bash
# Download SoilGrids data for model training and inference
python -m src.loaders.load_soil_grids --aoi 70.0 10.0 42.0 45.0
```

Alternatively, use the Jupyter notebook for interactive processing:

```bash
# Execute the staticMaps notebook to fetch and reproject SoilGrids layers
jupyter notebook notebooks/staticMaps.ipynb
```

### Discharge Data

Prepare discharge observation data:

```bash
# Create required directories
mkdir -p data/AISxls/discharge_csv

# Add your discharge observation CSV files here
```

## Sample Scripts

After data preparation, run the following example workflows:

### CatBoost Machine Learning Model

```bash
# Run CatBoost optimization with Optuna 
python catboost_optuna.py
```

### GR4J Hydrological Model

```bash
# Run GR4J parameter optimization 
python gr4j_optuna.py
```

### Interactive Notebooks

Explore pre-built Jupyter notebooks for data analysis and visualization:

```bash
# Launch notebook server
jupyter lab notebooks/

# Key notebooks:
# - staticMaps.ipynb: Download and process static catchment features
# - MeteoPreparation.ipynb: Prepare meteorological data
# - gr4jModel.ipynb: Explore GR4J model setup and results
# - catboostModel.ipynb: Explore machine learning model results
```

## Data Loaders

The `src.loaders` module provides utilities for acquiring and processing input data:

### ERA5-Land Loader

Downloads meteorological data from the ERA5-Land reanalysis dataset:

- Automatic download of variables like radiation, temperature, wind components
- Configurable spatial extent and time period
- Asynchronous processing for efficient data retrieval
- Built-in error handling and logging

Example usage:

```python
from src.meteo.era5_land_loader import download_era

await download_era(
    start_date="2007-01-01", 
    last_date="2022-12-31",
    save_path="data/MeteoData", 
    meteo_variables=["2m_temperature", "total_precipitation"],
    data_extent=[70.0, 20.0, 42.0, 45.0],  # [N, W, S, E]
    max_concurrent_downloads=6
)
```

### SoilGrids Loader

Downloads and reprojects soil property data from SoilGrids:

- Support for standard soil properties (bulk density, clay content, pH, etc.)
- Automatic handling of CRS transformations from Homolosine to WGS84
- Multi-threaded download and processing
- Configurable depth layers and statistics

The SoilGrids data provides crucial static catchment features for hydrological modeling. The loader handles the complex CRS transformations and efficient data retrieval from the global dataset.

```python
# Command line usage
python -m src.loaders.load_soil_grids --aoi 70.0 10.0 42.0 45.0 --res_folder data/SpatialData/SoilGrids
```

## Models

- **CatBoost**: Gradient boosting regressor with Optuna tuning. See [CatBoost Model](docs/catboost.md).
- **GR4J**: Conceptual rainfall–runoff model wrapped in Python. See [GR4J Model](docs/gr4j.md).
- **LSTM**: Recurrent neural network with static and dynamic inputs. See [LSTM Model](docs/lstm.md).
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
