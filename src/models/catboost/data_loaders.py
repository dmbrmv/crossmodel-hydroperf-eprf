import sys
import warnings
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# Add parent directory to Python path to access src modules
sys.path.append(str(Path("..").resolve()))
from src.models.catboost.optimizer import limit_optuna_logging
from src.utils.logger import setup_logger

SEED = 42
# ---- Logging ----
warnings.filterwarnings("ignore")
limit_optuna_logging()
logger = setup_logger(
    name="catboost_loader",
    log_file="logs/notebooks/catboost.log",
    log_level="INFO",
)


def split_gauges(full_gauges: list[str], test_size: float = 0.15) -> tuple[list[str], list[str]]:
    """Split gauges into train and test sets."""
    train_gauges, test_gauges = train_test_split(full_gauges, test_size=test_size, random_state=SEED)
    logger.info(
        f"Split {len(full_gauges)} gauges into {len(train_gauges)} train and {len(test_gauges)} test gauges"
    )
    return train_gauges, test_gauges


def round_static_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Round and convert static features to int."""
    for feature in features:
        df[feature] = df[feature].round(0).astype(int)
    return df


def get_data_masks(data: pd.DataFrame):
    """Return boolean masks for train, validation, and test periods."""
    train_mask = ("2008-01-01" <= data["date"]) & (data["date"] <= "2018-12-31")
    valid_mask = ("2019-01-01" <= data["date"]) & (data["date"] <= "2020-12-31")
    test_mask = ("2021-01-01" <= data["date"]) & (data["date"] <= "2022-12-31")
    return train_mask, valid_mask, test_mask


def get_feature_lists(data: pd.DataFrame, combined_feature: list[str]) -> tuple[list[str], list[str]]:
    """Return lists of categorical and numeric features."""
    categorical_features = ["gauge_id", "day_of_year"] + combined_feature
    numeric_features = [col for col in data.columns if col not in ["date", "q"]]
    return categorical_features, numeric_features


def create_pools(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    test_mask: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pool, Pool, Pool]:
    """Create CatBoost Pool objects for train, validation, and test sets."""
    train_pool = Pool(
        data=data.loc[train_mask, numeric_features],
        label=data.loc[train_mask, "q"],
        cat_features=categorical_features,
    )
    valid_pool = Pool(
        data=data.loc[valid_mask, numeric_features],
        label=data.loc[valid_mask, "q"],
        cat_features=categorical_features,
    )
    test_pool = Pool(
        data=test_data.loc[test_mask, numeric_features],
        label=test_data.loc[test_mask, "q"],
        cat_features=categorical_features,
    )
    return train_pool, valid_pool, test_pool


def save_catboost_model(model: CatBoostRegressor, output_dir: Path, meteo_dataset: str) -> None:
    """Save a trained CatBoost model to a file.

    Args:
        model (CatBoostRegressor): Trained CatBoost model.
        output_dir (Path): Directory where the model will be saved.
        meteo_dataset (str): Name of the meteorological dataset, used in the filename.

    Raises:
        Exception: If saving the model fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"catboost_model_{meteo_dataset}.cbm"
    try:
        model.save_model(model_path)
        logger.info(f"CatBoost model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save CatBoost model: {e}")
        raise
