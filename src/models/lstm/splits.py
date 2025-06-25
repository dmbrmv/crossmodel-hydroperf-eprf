import warnings

import pandas as pd
import torch
from torch.utils.data import Subset

from src.utils.logger import setup_logger

from .data_builders import RiverDataset  # adjust import to your path

logger = setup_logger(name="LSTM_splits", log_file="logs/lstm_splits.log")


# ------------------------------------------------------------------
#  A. Fraction-based split (quick and reproducible)
# ------------------------------------------------------------------
def rolling_split_fraction(
    dataset: RiverDataset, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> tuple[Subset, Subset, Subset]:
    """
    Deterministically split each gauge’s rolling windows in time order.

    *All* windows whose **end-timestep** falls in the first *train_frac*
    portion of the timeline go to the training set, etc.
    """
    torch.manual_seed(seed)

    n_gauges, T, _ = dataset.dyn.shape
    n_windows = T - dataset.seq_len

    n_train = int(n_windows * train_frac)
    n_val = int(n_windows * val_frac)

    idx_train: list[int] = []
    idx_val: list[int] = []
    idx_test: list[int] = []

    for g in range(n_gauges):
        base = g * n_windows
        idx_train.extend(range(base, base + n_train))
        idx_val.extend(range(base + n_train, base + n_train + n_val))
        idx_test.extend(range(base + n_train + n_val, base + n_windows))

    return Subset(dataset, idx_train), Subset(dataset, idx_val), Subset(dataset, idx_test)


def split_by_ranges(
    dataset,
    dates: pd.DatetimeIndex,
    *,
    train_range: tuple[str, str],
    val_range: tuple[str, str],
    test_range: tuple[str, str],
    expand: bool = True,  # True → shift each start back by seq_len
):
    """
    Assign rolling-window samples to train / val / test according to **END dates**.

    Parameters
    ----------
    dataset      : RiverDataset
        The rolling-window dataset (needs .seq_len).
    dates        : pd.DatetimeIndex
        Timeline returned by the loader; length == dataset.dyn.shape[1].
    train_range, val_range, test_range : (str, str)
        Inclusive END-date ranges for each split, in 'YYYY-MM-DD' format.
    expand       : bool
        If True (default) subtracts `dataset.seq_len` days from the *start*
        of every range so you automatically get the preceding history.

    Returns
    -------
    (train_ds, val_ds, test_ds) : tuple[Subset, Subset, Subset]
        Ready to wrap in DataLoaders.
    """

    # ------------------------------------------------------------------
    # 0. Checks & date parsing
    # ------------------------------------------------------------------
    if not (dates.is_monotonic_increasing and len(dates) == dataset.dyn.shape[1]):
        raise ValueError("dates must be sorted and match dataset time dimension")

    def _parse(pair, label):
        start, end = pd.to_datetime(pair)
        if start > end:
            raise ValueError(f"{label} start date is after end date")
        return start, end

    tr_start, tr_end = _parse(train_range, "train")
    va_start, va_end = _parse(val_range, "val")
    te_start, te_end = _parse(test_range, "test")

    if expand:
        shift = pd.Timedelta(days=dataset.seq_len)
        tr_start -= shift
        va_start -= shift
        te_start -= shift
        warnings.warn(
            f"split_by_ranges(expand=True): start dates shifted "
            f"back by {dataset.seq_len} days to include full context.",
            RuntimeWarning,
        )

    # ------------------------------------------------------------------
    # 1. Pre-compute end-dates of windows
    # ------------------------------------------------------------------
    window_end_dates = dates[dataset.seq_len :]  # len = T - seq_len
    n_gauges = dataset.dyn.shape[0]
    n_windows = len(window_end_dates)

    idx_train, idx_val, idx_test = [], [], []

    # ------------------------------------------------------------------
    # 2. Assign each window
    # ------------------------------------------------------------------
    for g in range(n_gauges):
        base = g * n_windows
        for t_end, d in enumerate(window_end_dates):
            gidx = base + t_end
            if tr_start <= d <= tr_end:
                idx_train.append(gidx)
            elif va_start <= d <= va_end:
                idx_val.append(gidx)
            elif te_start <= d <= te_end:
                idx_test.append(gidx)

    return (
        Subset(dataset, idx_train),
        Subset(dataset, idx_val),
        Subset(dataset, idx_test),
    )
