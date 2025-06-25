"""Module for NetCDF grid processing and meteorological data extraction.

This module provides functions for:
- Extracting maximum longitude from NetCDF files
- Rounding coordinates to nearest grid resolution
- Finding spatial extents for watersheds
- Subsetting NetCDF data by spatial extent
- Calculating intersection weights for polygon-grid overlap
- Computing weighted meteorological aggregations
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import product
from pathlib import Path
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy import dtype, float64
from shapely.geometry import MultiPolygon, Polygon

from src.geometry.watershed import create_gdf, poly_from_multipoly, polygon_area
from src.grids.grid_edit import aggregation_definer, get_square_vertices
from src.utils.logger import setup_logger

logging = setup_logger("nc_grid", log_file="/logs/nc_grid.log")


def get_max_lon_from_netcdf(nc_path: Path) -> float:
    """Extract the maximum longitude from a NetCDF file.

    Args:
        nc_path (Path): Path to the NetCDF file.

    Returns:
        float: The maximum longitude value in the NetCDF file.

    Raises:
        ValueError: If longitude variable is not found in the NetCDF file.

    """
    with xr.open_dataset(nc_path) as ds:
        # Try common longitude variable names
        for lon_name in ["lon", "longitude", "y"]:
            if lon_name in ds.variables:
                max_lon = float(ds[lon_name].max())
                return max_lon
    raise ValueError("longitude variable not found in NetCDF file.")


@lru_cache(maxsize=1024)
def round_nearest(x: float, a: float) -> float:
    """Round x to the nearest multiple of a.

    Args:
        x: Value to round
        a: Multiple to round to

    Returns:
        Rounded value

    Raises:
        ValueError: If a is zero or negative

    """
    if a <= 0:
        raise ValueError("Rounding multiple 'a' must be positive")

    if a == 0:
        return x

    # More efficient calculation of fractional digits
    log_a = math.log10(abs(a))
    frac_digits = max(0, -int(math.floor(log_a)) + 10)

    # Limit fractional digits to prevent overflow
    frac_digits = min(frac_digits, 15)

    return round(round(x / a) * a, frac_digits)


def find_extent(
    ws: Polygon, grid_res: float, dataset: str = ""
) -> np.ndarray[tuple[int, ...], dtype[float64]] | list[float]:
    """Find extent of watershed with given grid resolution.

    Args:
        ws: Watershed polygon
        grid_res: Grid resolution in decimal degrees
        dataset: Dataset name, used for rounding values. Defaults to "".

    Returns:
        List of four floats representing the extent [min_lon, max_lon, min_lat, max_lat]

    Raises:
        ValueError: If dataset is provided but not recognized
        AttributeError: If polygon doesn't have exterior coordinates

    """
    try:
        # Get bounds directly from polygon for better performance
        min_lon, min_lat, max_lon, max_lat = ws.bounds
    except AttributeError:
        raise AttributeError("Invalid polygon geometry - cannot extract bounds") from None
    true_grid_res = grid_res * 2
    # Round values to the nearest grid_res
    min_lon = round_nearest(min_lon, grid_res)
    max_lon = round_nearest(max_lon, grid_res)
    min_lat = round_nearest(min_lat, grid_res)
    max_lat = round_nearest(max_lat, grid_res)

    # Check if the extent is too small and adjust it
    lon_diff = abs(min_lon - max_lon)
    lat_diff = abs(min_lat - max_lat)

    if np.round(lon_diff, 3) <= true_grid_res:
        max_lon = round_nearest(max_lon + grid_res, grid_res)
        min_lon = round_nearest(min_lon - grid_res, grid_res)

    if np.round(lat_diff, 3) <= true_grid_res:
        max_lat = round_nearest(max_lat + grid_res, grid_res)
        min_lat = round_nearest(min_lat - grid_res, grid_res)

    return [
        round_nearest(min_lon - grid_res, grid_res),
        round_nearest(max_lon + grid_res, grid_res),
        round_nearest(min_lat - grid_res, grid_res),
        round_nearest(max_lat + grid_res, grid_res),
    ]


def nc_by_extent(
    nc: xr.Dataset, shape: Polygon | MultiPolygon, grid_res: float, dataset: str = ""
) -> xr.Dataset:
    """Select netCDF data by extent of given shape. Return masked netCDF.

    Args:
        nc: NetCDF dataset
        shape: Shape of the area of interest
        grid_res: Grid resolution in decimal degrees
        dataset: Dataset name. Defaults to "".

    Returns:
        Masked netCDF dataset

    Raises:
        ValueError: If required dimensions are missing from dataset

    """
    # Standardize coordinate names more efficiently
    coord_mapping = {"latitude": "lat", "longitude": "lon"}

    # Only rename if necessary
    rename_dict = {old: new for old, new in coord_mapping.items() if old in nc.dims}
    if rename_dict:
        nc = nc.rename(rename_dict)

    # Validate required coordinates
    if "lat" not in nc.dims or "lon" not in nc.dims:
        raise ValueError("Dataset must contain 'lat' and 'lon' dimensions")

    # Find biggest polygon
    big_shape = poly_from_multipoly(ws_geom=shape)

    # Find extent coordinates
    min_lon, max_lon, min_lat, max_lat = find_extent(ws=big_shape, grid_res=grid_res, dataset=dataset)

    # More efficient subsetting using sel with slice
    try:
        masked_nc = nc.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    except (KeyError, ValueError):
        # Fallback to where method if slice doesn't work
        masked_nc = (
            nc.where(nc.lat >= min_lat, drop=True)
            .where(nc.lat <= max_lat, drop=True)
            .where(nc.lon >= min_lon, drop=True)
            .where(nc.lon <= max_lon, drop=True)
        )

    return masked_nc


def get_weights(
    weight_path: Path,
    mask_nc: xr.Dataset,
    ws_geom: Polygon,
    ws_area: float,
    grid_res: float = 0.05,
    n_workers: int | None = None,
) -> xr.DataArray:
    """Calculate weights for a given watershed and grid resolution.

    Parameters
    ----------
    weight_path : Path
        Path to a file where the weights should be saved
    mask_nc : xr.Dataset
        Initial netCDF file
    ws_geom : Polygon
        Watershed geometry
    ws_area : float
        Watershed area
    grid_res : float, optional
        Grid resolution, by default 0.05

    Returns:
    -------
    xr.DataArray
        DataArray with weights

    """
    if weight_path.is_file():
        # Load weights from file
        weights = xr.open_dataarray(weight_path)
        return weights

    # Watershed geometry
    ws_shape = poly_from_multipoly(ws_geom)

    # Get lat, lon which help define area for intersection
    nc_lat, nc_lon = mask_nc.lat.values, mask_nc.lon.values

    # Pre-allocate arrays for better performance
    grid_shape = (len(nc_lat), len(nc_lon))
    inter_mask = np.zeros(grid_shape, dtype=bool)
    weights_data = np.zeros(grid_shape, dtype=np.float64)

    # Vectorized approach using list comprehension with enumerate for direct indexing
    lat_lon_combinations = [
        (i, j, float(lat), float(lon))
        for i, lat in enumerate(nc_lat)
        for j, lon in enumerate(nc_lon)
    ]

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    def _process_cell(args: tuple[int, int, float, float]) -> tuple[int, int, bool, float]:
        i, j, lat, lon = args
        try:
            cell = Polygon(get_square_vertices(mm=(lon, lat), h=grid_res, phi=0))
            intersection = ws_shape.intersection(cell)
            if not intersection.is_empty:
                weight = polygon_area(poly_from_multipoly(intersection)) / ws_area
                return i, j, True, weight
        except Exception as exc:  # pragma: no cover - logging only
            logging.exception(f"Error processing cell lat={lat}, lon={lon}: {exc}")
        return i, j, False, 0.0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_cell, args) for args in lat_lon_combinations]
        for fut in as_completed(futures):
            i, j, mask, weight = fut.result()
            if mask:
                inter_mask[i, j] = True
                weights_data[i, j] = weight


    # Create DataArrays efficiently
    inter_mask_da = xr.DataArray(data=inter_mask, dims=["lat", "lon"], coords=[nc_lat, nc_lon])
    weights = xr.DataArray(data=weights_data, dims=["lat", "lon"], coords=[nc_lat, nc_lon])
    weights.name = "weights"
    weights = weights.where(inter_mask_da, drop=True).fillna(0)

    # Save weights to file
    weights.to_netcdf(weight_path)

    return weights


def calculate_weighted_meteorology(
    ws_nc: xr.Dataset,
    weights: xr.DataArray,
    magnitude_factor: float = 1e2,
) -> pd.DataFrame:
    """Calculate weighted meteorological data from NetCDF dataset.

    Args:
        ws_nc: NetCDF dataset with meteorological variables
        weights: DataArray with spatial weights for aggregation
        magnitude_factor: Factor to scale the results (default: 1e2)

    Returns:
        DataFrame with weighted meteorological data indexed by date

    """
    time_coord = next(
        (coord for coord in ws_nc.coords if np.issubdtype(ws_nc[coord].dtype, np.datetime64)), "time"
    )

    # Create weighted dataset for all variables at once
    weighted_ds = ws_nc.weighted(weights)

    # Calculate aggregated values for all variables efficiently
    aggregated_data = {}
    for var in ws_nc.data_vars:
        agg_method = aggregation_definer(var)
        if agg_method == "sum":
            aggregated_data[var] = (weighted_ds.sum(dim=["lat", "lon"])[var] * magnitude_factor).values
        else:
            aggregated_data[var] = (weighted_ds.mean(dim=["lat", "lon"])[var]).values

    # Create DataFrame efficiently
    result_df = pd.DataFrame({"date": ws_nc[time_coord].values, **aggregated_data}).set_index("date")

    # Clip precipitation if requested
    result_df["prcp"] = result_df["prcp"].clip(lower=0)

    return result_df
