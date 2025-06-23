import geopandas as gpd

from src.utils.logger import setup_logger

loader_logger = setup_logger("geom_loader", log_file="logs/data_loader.log")


def load_geodata() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load watershed and gauge geometry data."""
    e_obs_ws = gpd.read_file("data/Geometry/EOBSWatersheds2025.gpkg", ignore_geometry=False)
    e_obs_ws.set_index("gauge_id", inplace=True)
    e_obs_gauge = gpd.read_file("data/Geometry/EOBSPoints2025.gpkg", ignore_geometry=False)
    e_obs_gauge.set_index("gauge_id", inplace=True)
    return e_obs_ws, e_obs_gauge
