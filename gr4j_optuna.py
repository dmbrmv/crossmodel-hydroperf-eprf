"""Main script to run GR4J Optuna optimization for multiple gauges."""

import multiprocessing as mp
from functools import partial
from pathlib import Path

from src.models.gr4j.parallel import process_gr4j_gauge, run_parallel_optimization
from src.readers.geom_reader import load_geodata
from src.readers.hydro_data_reader import find_valid_gauges
from src.utils.logger import setup_logger

# Create log directories before importing modules that might use them
Path("logs").mkdir(exist_ok=True)

# Setup logger
logger = setup_logger("main_gr4j_optuna", log_file="logs/gr4j_optuna.log")


def main() -> None:
    # Load gauge and watershed data
    e_obs_ws, e_obs_gauge = load_geodata()
    # Find valid gauges (no missing data)
    logger.info("Finding gauges with valid data...")
    full_gauges, _ = find_valid_gauges(e_obs_ws, Path("data/HydroFiles"))
    logger.info(f"Found {len(full_gauges)} valid gauges")

    selected_gauges = full_gauges[:] if len(full_gauges) > 3 else full_gauges
    logger.info(f"Selected gauges for testing: {selected_gauges}")

    # Set up optimization parameters
    calibration_period = ("2008-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2022-12-31")
    save_storage = Path("data/res/gr4j_optuna/")
    save_storage.mkdir(parents=True, exist_ok=True)

    # Use reduced number of trials (10) for testing
    n_trials = 6000
    timeout = 1200  # Reduced timeout (20 minutes)
    overwrite_existing_results = False  # Set to True to overwrite existing results
    datasets = ["meteo_ru_nc_02", "mswep", "e_obs", "era5_land"]
    # Run optimization in parallel
    logger.info(f"Starting optimization with {n_trials} trials per gauge/dataset")
    # n_processes = min(10, len(full_gauges))  # Don't use more processes than gauges

    run_parallel_optimization(
        gauge_ids=selected_gauges,
        process_gauge_func=partial(
            process_gr4j_gauge,
            datasets=datasets,
            calibration_period=calibration_period,
            validation_period=validation_period,
            save_storage=save_storage,
            e_obs_gauge=e_obs_gauge,
            n_trials=n_trials,
            timeout=timeout,
            overwrite_results=overwrite_existing_results,
        ),
        n_processes=mp.cpu_count() - 2,  # Use all but one CPU core
    )
    logger.info("Optimization completed successfully")


if __name__ == "__main__":
    main()
