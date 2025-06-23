import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_observed_vs_simulated(
    observed: np.ndarray,
    simulated: np.ndarray,
    index: pd.Index,
    title: str = "Observed vs Simulated Discharge",
    ylabel: str = "Discharge (mm/day)",
    figsize: tuple = (14, 6),
) -> None:
    """Plot observed and simulated discharge time series.

    Args:
        observed (np.ndarray): Array of observed discharge values.
        simulated (np.ndarray): Array of simulated discharge values.
        index (pd.Index): Datetime index for the x-axis.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(index, observed, label="Observed", color="black", linewidth=1.5)
    plt.plot(index, simulated, label="Simulated", color="royalblue", linewidth=1.2, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
