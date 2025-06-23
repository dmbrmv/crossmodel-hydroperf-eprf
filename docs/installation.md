# Installation Guide

This guide walks you through setting up the HydroEPRF toolkit on a Linux environment.

## Prerequisites

- Python 3.10+
- zsh (default shell) or bash
- Git
- [Optional] GPU drivers and CUDA toolkit for CatBoost GPU support

## Clone Repository

```bash
git clone https://github.com/yourusername/HydroEPRF.git
cd HydroEPRF
```

## Create Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

1. Create data directories:

```bash
mkdir -p data/MeteoData/ProcessedGauges/meteo_ru_nc_02/res/
mkdir -p data/HydroFiles
```

1. Download meteorological and hydrological datasets and place them in the corresponding folders under `data/`.
1. Ensure static_data.csv is available in `data/Geometry/static_data.csv`.

## Verify Installation

Run sample notebooks:

```bash
jupyter lab notebooks/catboosModel.ipynb
```

If notebooks open and necessary kernels are available, installation is successful.
