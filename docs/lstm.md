# LSTM Model

The LSTM component provides a sequence‑to‑one network for daily discharge forecasting. Each gauge has a vector of static physiographic attributes and a matrix of dynamic meteorological features mixed with historical discharge. A rolling window of `N` antecedent days is used to predict the next day.

## Key Features

- Unified dataset builder that aligns meteorology, discharge and static attributes.
- Flexible fusion modes for static features (`repeat`, `init`, `late`).
- Loss functions: NSE, KGE, Weighted MSE and **Huber**.
- Reproducible train/val/test splits by date range.
- Example notebook: [`lstmStaticModel.ipynb`](../notebooks/lstmStaticModel.ipynb).

## Example Usage

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.lstm.data_builders import build_dataset_from_folder
from src.models.lstm.model import HydroLSTM
from src.models.lstm.losses import HuberLoss
from src.models.lstm.splits import split_by_ranges

lstm_ds, dates = build_dataset_from_folder(
    ['0001', '0002'],
    meteo_dir=Path('data/meteo'),
    hydro_dir=Path('data/hydro'),
    temp_dir=None,
    df_static=static_df,
    dyn_feature_cols=['prcp', 't_mean'],
    seq_len=90,
)
train_ds, val_ds, test_ds = split_by_ranges(
    lstm_ds,
    dates,
    train_range=('2008-01-01', '2018-12-31'),
    val_range=('2019-01-01', '2020-12-31'),
    test_range=('2021-01-01', '2022-12-31'),
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

model = HydroLSTM(
    n_dyn=lstm_ds.dyn.shape[-1],
    n_static=lstm_ds.static.shape[-1],
).to('cpu')
loss_fn = HuberLoss()
```
