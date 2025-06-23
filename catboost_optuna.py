"""CatBoost Training and Hyperparameter Optimization Script.

This script loads hydrological and meteorological data, selects features, splits data,
performs hyperparameter optimization using Optuna, trains CatBoost models, and saves them.
"""

import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from catboost import CatBoostRegressor

# Add parent directory to Python path to access src modules
sys.path.append(str(Path("..").resolve()))
from src.models.catboost.data_loaders import (
    create_pools,
    get_data_masks,
    get_feature_lists,
    round_static_features,
    save_catboost_model,
)
from src.models.catboost.optimizer import catboost_objective, limit_optuna_logging
from src.readers.geo_char_reader import (
    get_combined_features,
    load_static_data,
)
from src.readers.geom_reader import load_geodata
from src.readers.hydro_data_reader import (
    data_creator,
    find_valid_gauges,
)
from src.utils.logger import setup_logger

# ---- Configuration ----
SEED = 42
N_TRIALS = 60
ES_ROUNDS = 80
MODEL_DIR = Path("data/res/catboost/")
METEO_DATASETS = ["meteo_ru_nc_02", "e_obs", "era5_land", "mswep"]
GPU_OK = torch.cuda.is_available()

# ---- Reproducibility ----
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Logging ----
warnings.filterwarnings("ignore")
limit_optuna_logging()
logger = setup_logger(
    name="CatboostTraining",
    log_file="logs/catboost.log",
    log_level="INFO",
)


def train_catboost_for_dataset(
    meteo_dataset: str,
    train_gauges: list[str],
    test_gauges: list[str],
    combined_feature: list[str],
    combined_features_df: Any,
) -> None:
    """Train and save a CatBoost model for a specific meteorological dataset.

    Args:
        meteo_dataset: Name of the meteorological dataset.
        train_gauges: List of gauge IDs for training.
        test_gauges: List of gauge IDs for testing.
        combined_feature: List of combined static features.
        combined_features_df: DataFrame of combined static features.
    """
    logger.info(f"\nCatBoost training for {meteo_dataset} dataset")
    model_path = MODEL_DIR / f"catboost_model_{meteo_dataset}.cbm"
    if model_path.is_file():
        logger.info(f"Model already exists for {meteo_dataset} at {model_path}. Skipping training.")
        return

    meteo_dir = Path(f"data/MeteoData/ProcessedGauges/{meteo_dataset}/res/")
    hydro_dir = Path("data/HydroFiles/")
    temp_dir = Path("data/MeteoData/ProcessedGauges/era5_land/res/")

    logger.info(f"Creating training dataset with {len(train_gauges)} gauges...")
    data = data_creator(
        full_gauges=train_gauges,
        static_data=combined_features_df,
        meteo_dir=meteo_dir,
        hydro_dir=hydro_dir,
        temp_dir=temp_dir,
    )

    logger.info(f"Creating test dataset with {len(test_gauges)} gauges...")
    test_data = data_creator(
        full_gauges=test_gauges,
        static_data=combined_features_df,
        meteo_dir=meteo_dir,
        hydro_dir=hydro_dir,
        temp_dir=temp_dir,
    )

    data = round_static_features(data, combined_feature)
    test_data = round_static_features(test_data, combined_feature)
    data["day_of_year"] = data["day_of_year"].astype(int)
    test_data["day_of_year"] = test_data["day_of_year"].astype(int)
    train_mask, valid_mask, test_mask = get_data_masks(data)
    logger.info(
        f"Training samples: {train_mask.sum()}, Validation samples: {valid_mask.sum()}, Test samples: {test_mask.sum()}"
    )

    categorical_features, numeric_features = get_feature_lists(data, combined_feature)
    logger.info(
        f"Prepared {len(numeric_features)} features with "
        f"{len(categorical_features)} categorical features"
    )

    if GPU_OK:
        logger.info("GPU is available for training.")
    else:
        logger.warning("GPU is not available. Training will use CPU, which may be slower.")

    train_pool, valid_pool, _ = create_pools(
        data, test_data, train_mask, valid_mask, test_mask, numeric_features, categorical_features
    )

    base_params: dict[str, object] = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": ES_ROUNDS,
        "random_seed": SEED,
        "verbose": 0,
        "task_type": "GPU" if GPU_OK else "CPU",
        "devices": "0",
        "grow_policy": "SymmetricTree",
        "bootstrap_type": "Bayesian",
        "border_count": 256,
        "gpu_ram_part": 0.95 if GPU_OK else None,
    }

    study = optuna.create_study(
        study_name="catboost_mape",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )
    study.optimize(
        lambda trial: catboost_objective(trial, base_params, train_pool, valid_pool),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    best_params: dict[str, Any] = base_params | study.best_params
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(
        train_pool,
        eval_set=valid_pool,
        verbose=500,
        use_best_model=True,
    )

    save_catboost_model(final_model, MODEL_DIR, meteo_dataset)


def main() -> None:
    """Main function to orchestrate CatBoost training and optimization."""
    e_obs_ws, _ = load_geodata(folder_depth=".")
    logger.info("Finding gauges with valid data...")
    full_gauges, partial_gauges = find_valid_gauges(e_obs_ws, Path("data/HydroFiles"))
    static_data = load_static_data(full_gauges + partial_gauges, path_prefix=None)
    combined_feature, combined_features_df = get_combined_features(static_data)

    for meteo_dataset in METEO_DATASETS:
        train_catboost_for_dataset(
            meteo_dataset,
            full_gauges,
            full_gauges,
            combined_feature,
            combined_features_df,
        )


if __name__ == "__main__":
    main()
