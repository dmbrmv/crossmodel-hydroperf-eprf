"""GR4J Optuna multi-objective optimization logic."""

import optuna
import pandas as pd

from src.models.gr4j import model as gr4j
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model

logger = setup_logger("main_gr4j_optuna", log_file="logs/gr4j_optuna.log")


def multi_objective(
    trial: optuna.Trial, data: pd.DataFrame, calibration_period: tuple[str, str]
) -> tuple[float, float, float, float, float]:
    """Multi-objective optimization for GR4J parameters (KGE, NSE, logNSE, -PBIAS, -RMSE_norm).

    Args:
        trial: Optuna trial object.
        data: Hydrometeorological data.
        calibration_period: (start_date, end_date) for calibration.

    Returns:
        Tuple of (KGE, NSE, logNSE, -abs(PBIAS), -RMSE_normalized).
    """
    calib_data = data[calibration_period[0] : calibration_period[1]]
    x1 = trial.suggest_float("x1", 10.0, 3000.0, log=True)
    x2 = trial.suggest_float("x2", -20.0, 10.0)
    x3 = trial.suggest_float("x3", 1.0, 4000.0, log=True)
    x4 = trial.suggest_float("x4", 0.05, 20.0)
    ctg = trial.suggest_float("ctg", 0.0, 3.0)
    kf = trial.suggest_float("kf", 1.0, 10.0)
    tt = trial.suggest_float("tt", -1.5, 3.0)
    params = [x1, x2, x3, x4, ctg, kf, tt]
    q_sim = gr4j.simulation(calib_data, params)
    metrics = evaluate_model(calib_data["q_mm_day"].values, q_sim)
    kge, nse, log_nse, rmse, pbias_abs = (
        metrics["KGE"],
        metrics["NSE"],
        metrics["logNSE"],
        metrics["RMSE"],
        abs(metrics["PBIAS"]),
    )
    obs_mean = calib_data["q_mm_day"].mean()
    rmse_norm = rmse / obs_mean if obs_mean > 0 else rmse
    return kge, nse, log_nse, -pbias_abs, -rmse_norm


def early_stopping_callback(thresholds: dict | None = None):
    """Optuna callback to stop if all metric thresholds are met."""
    if thresholds is None:
        thresholds = {"NSE": 0.8, "KGE": 0.8, "logNSE": 0.8}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        kge, nse, lognse = trial.values[0], trial.values[1], trial.values[2]
        if (
            kge is not None
            and nse is not None
            and lognse is not None
            and kge > thresholds["KGE"]
            and nse > thresholds["NSE"]
            and lognse > thresholds["logNSE"]
        ):
            logger.info(
                f"Early stopping: NSE={nse:.3f}, KGE={kge:.3f}, logNSE={lognse:.3f} exceed thresholds."
            )
            study.stop()

    return callback


def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log progress every 500 trials."""
    if (trial.number + 1) % 500 == 0:
        logger.info(f"Trial {trial.number + 1} finished with values {trial.values}")


def run_optimization(
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    study_name: str | None,
    n_trials: int = 100,
    timeout: int = 600,
    verbose: bool = False,
) -> optuna.Study:
    """Run Optuna multi-objective optimization for GR4J model parameters."""
    if study_name is None:
        study_name = f"GR4J_multiobj_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Creating multi-objective study: {study_name}")
    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        directions=["maximize"] * 5,
        sampler=sampler,
        study_name=study_name,
        load_if_exists=True,
    )
    try:
        start_time = pd.Timestamp.now()
        logger.info(f"Multi-objective optimization started at {start_time}")
        original_verbosity = optuna.logging.get_verbosity()
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda trial: multi_objective(trial, data, calibration_period),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
            callbacks=[
                early_stopping_callback({"NSE": 0.85, "KGE": 0.8, "logNSE": 0.85}),
                progress_callback,
            ],
            gc_after_trial=True,
        )
        optuna.logging.set_verbosity(original_verbosity)
    except Exception as e:
        logger.error(f"Multi-objective optimization failed: {str(e)}")
        raise
    return study
