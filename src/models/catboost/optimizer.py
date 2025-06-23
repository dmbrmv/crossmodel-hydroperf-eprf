"""CatBoost model utilities and parameter optimization.

This module provides utilities for CatBoost model training, optimization,
and parameter tuning with Optuna.
"""

import logging

import optuna
from catboost import CatBoostRegressor, Pool
from optuna.trial import FrozenTrial


def limit_optuna_logging() -> None:
    """Configure Optuna to reduce verbose logging output.

    This function sets Optuna's logging level to WARNING to suppress
    the default INFO-level trial result outputs.
    """
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.WARNING)


def log_improved_results(study: optuna.Study, trial: FrozenTrial) -> None:
    """Optuna callback to log results only when a better metric is found.

    Args:
        study: The current Optuna study
        trial: The most recent finished trial

    Returns:
        None
    """
    # Get logger from the calling module
    logger = logging.getLogger("CatboostTraining")

    # Only log when the current trial is the best so far
    if study.best_trial.number == trial.number:
        # Format key parameters for more readable output
        key_params = {
            "learning_rate": f"{trial.params.get('learning_rate', 'N/A'):.5f}",
            "depth": trial.params.get("depth", trial.params.get("max_depth", "N/A")),
            "bootstrap_type": trial.params.get("bootstrap_type", "N/A"),
        }
        logger.info(
            f"Trial {trial.number + 1}: New best RMSE = {trial.value:.3f} (key params: {key_params})"
        )


def catboost_objective(
    trial: optuna.Trial, base_params: dict, train_pool: Pool, valid_pool: Pool
) -> float:
    """Objective function for Optuna hyperparameter optimization of CatBoostRegressor."""
    params = base_params | {
        "iterations": trial.suggest_int("iterations", 420, 6000, step=420),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
        "depth": trial.suggest_int("depth", 6, 15),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 4, 20),
        "random_strength": trial.suggest_float("random_strength", 1e-4, 1.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
    }
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model.best_score_["validation"]["RMSE"]
