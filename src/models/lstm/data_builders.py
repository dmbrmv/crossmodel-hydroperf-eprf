import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

sys.path.append(str(Path("../").resolve()))

from src.utils.logger import setup_logger

random.seed(42)
logger = setup_logger(name="LSTM_model", log_file="logs/lstm_model.log")


def build_dataset_from_folder(
    gauge_ids: list[str],
    *,
    meteo_dir: Path,
    hydro_dir: Path,
    temp_dir: Path | None,
    df_static: pd.DataFrame,
    dyn_feature_cols: list[str],
    seq_len: int,
    date_col: str = "date",
    target_col: str = "q",
    start_date: str | None = "2007-01-01",
    n_workers: int = 8,
) -> tuple["RiverDataset", pd.DatetimeIndex]:
    """Load per‑gauge NetCDF/CSV files concurrently and build a ``RiverDataset``.

    Parameters
    ----------
    gauge_ids : list[str]
        Gauges to include (strings to allow leading zeros).
    meteo_dir : Path
        Directory with meteorological NetCDF files named ``<gauge_id>.nc``.
        If ``meteo_dir.parent.name == 'mswep'`` the file supplies *only* PRCP
        and ERA5‑Land files are expected in ``temp_dir``.
    hydro_dir : Path
        Directory with discharge CSVs (columns: date, q_mm_day).
    temp_dir : Path | None
        Needed only for the MSWEP‑plus‑ERA5 case; ignored otherwise.
    df_static : pd.DataFrame
        Static attributes with *index* == gauge_id.
    dyn_feature_cols : list[str]
        Dynamic predictor names in the merged dataframe.
    seq_len : int
        Antecedent window length (e.g. 365).
    date_col, target_col : str
        Names of the timestamp and discharge columns after merge.
    start_date : str | None
        Crop to a common start date (string parsable by ``pd.to_datetime``).
    n_workers : int
        Degree of parallelism for the I/O stage (``ThreadPoolExecutor``).
    """

    # ------------------------------------------------------------------
    #  STEP 1 ‑‑ parallel read & pre‑filter per gauge                    #
    # ------------------------------------------------------------------

    def _load_single_gauge(gid: str):
        """Read meteo + hydro, return (dyn_df, tgt_series, dates)."""

        # ---------- Meteorology ----------
        if meteo_dir.parent.name == "mswep":
            # Two‑file combo: PRCP from MSWEP + temps from ERA5‑Land
            with xr.open_dataset(meteo_dir / f"{gid}.nc") as ds:
                mswep_df = ds.to_dataframe()
            if temp_dir is None:
                raise ValueError("temp_dir must be provided for MSWEP dataset.")
            with xr.open_dataset(temp_dir / f"{gid}.nc") as ds:
                meteo_df = ds.to_dataframe()
            meteo_df.loc[:, "prcp"] = mswep_df["prcp"]
        else:
            with xr.open_dataset(meteo_dir / f"{gid}.nc") as ds:
                meteo_df = ds.to_dataframe()

        # ---------- Discharge ----------
        q_df = pd.read_csv(hydro_dir / f"{gid}.csv", parse_dates=["date"], index_col="date").rename(
            columns={"q_mm_day": target_col}
        )[[target_col]]

        # ---------- Merge ----------
        df = pd.concat([meteo_df, q_df], axis=1, join="inner")
        df[date_col] = df.index
        df["gauge_id"] = gid
        df = df.reset_index(drop=True)

        # ---------- Validate ----------
        required = set([date_col, target_col, *dyn_feature_cols])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Gauge {gid}: missing columns {missing}")

        # ---------- Crop & sort ----------
        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        df = df.sort_values(date_col).reset_index(drop=True)
        return gid, df[dyn_feature_cols].values, df[target_col].values, pd.DatetimeIndex(df[date_col])

    results: dict[str, tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_single_gauge, gid): gid for gid in gauge_ids}
        for fut in as_completed(futures):
            gid, dyn_arr, tgt_arr, dates = fut.result()
            results[gid] = (
                torch.tensor(dyn_arr, dtype=torch.float32),
                torch.tensor(tgt_arr, dtype=torch.float32),
                dates,
            )

    # ------------------------------------------------------------------
    #  STEP 2 ‑‑ compute common timeline & align tensors                 #
    # ------------------------------------------------------------------

    common_dates: pd.DatetimeIndex | None = None
    for _, _, dates in results.values():
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

    if common_dates is None or len(common_dates) == 0:
        raise RuntimeError("No overlapping dates across gauges after cropping.")

    dyn_tensor_list: list[torch.Tensor] = []
    tgt_tensor_list: list[torch.Tensor] = []

    for gid in gauge_ids:
        dyn_arr, tgt_arr, dates = results[gid]
        mask = torch.from_numpy(dates.isin(common_dates))
        dyn_tensor_list.append(dyn_arr[mask])
        tgt_tensor_list.append(tgt_arr[mask])

    dyn_tensor = torch.stack(dyn_tensor_list)  # [n_gauges, T, n_dyn]
    tgt_tensor = torch.stack(tgt_tensor_list)  # [n_gauges, T]

    # Static features aligned to gauge_ids order
    static_tensor = torch.tensor(df_static.loc[gauge_ids].values, dtype=torch.float32)

    return RiverDataset(dyn_tensor, static_tensor, tgt_tensor, seq_len), common_dates


class RiverDataset(Dataset):
    """Rolling‑window dataset: returns (seq, static, target)."""

    def __init__(
        self,
        dyn_data: torch.Tensor,
        static_data: torch.Tensor,
        targets: torch.Tensor,
        seq_len: int,
    ) -> None:
        self.dyn = dyn_data  # [n_gauges, T, n_dyn]
        self.static = static_data
        self.tgt = targets
        self.seq_len = seq_len
        assert self.dyn.dim() == 3 and self.dyn.shape[:2] == self.tgt.shape[:2]

    def __len__(self) -> int:
        n_gauges, T, _ = self.dyn.shape
        return (T - self.seq_len) * n_gauges

    def __getitem__(self, idx: int):
        n_gauges, T, _ = self.dyn.shape
        g = idx // (T - self.seq_len)
        t = idx % (T - self.seq_len)
        seq = self.dyn[g, t : t + self.seq_len, :]
        stat = self.static[g]
        y = self.tgt[g, t + self.seq_len]
        return seq, stat, y
