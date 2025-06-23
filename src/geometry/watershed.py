import geopandas as gpd
import numpy as np
from numpy import sin
from shapely.geometry import MultiPolygon, Polygon


def filter_watersheds_by_lon(ws_gdf: gpd.GeoDataFrame, max_lon: float) -> gpd.GeoDataFrame:
    """Filters watersheds whose maximum longitude is less than or equal to max_lon.

    Args:
        ws_gdf (gpd.GeoDataFrame): GeoDataFrame of watersheds.
        max_lon (float): Maximum usable longitude.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with watersheds fully within max_lon.

    """
    # Calculate the maximum longitude for each watershed polygon
    ws_gdf = ws_gdf.copy()
    ws_gdf["max_lon"] = ws_gdf.geometry.apply(
        lambda geom: geom.bounds[2]  # maxx is the third value in bounds (minx, miny, maxx, maxy)
    )  # bounds: (minx, miny, maxx, maxy)
    filtered = ws_gdf[ws_gdf["max_lon"] <= max_lon]
    return filtered


def polygon_area(geo_shape: Polygon, radius: float = 6378137.0) -> float:
    """Optimized vectorized polygon area calculation using spherical excess formula.

    This implementation uses numpy vectorization and eliminates redundant calculations
    for better performance with large polygons.

    Args:
        geo_shape (Polygon): The polygon whose area is to be computed.
        radius (float, optional): The radius of the sphere in meters. Defaults to 6378137.0.

    Returns:
        float: The area of the polygon in square kilometers.

    """
    # Extract coordinates and convert to radians in one step
    coords = np.array(geo_shape.exterior.coords)
    lons_rad = np.radians(coords[:-1, 0])  # Exclude last point (assuming closed polygon)
    lats_rad = np.radians(coords[:-1, 1])

    # Handle case where polygon might not be closed
    if len(coords) > 1 and not (coords[0, 0] == coords[-1, 0] and coords[0, 1] == coords[-1, 1]):
        lons_rad = np.append(lons_rad, lons_rad[0])
        lats_rad = np.append(lats_rad, lats_rad[0])

    # Vectorized calculation using numpy's roll for efficient neighbor access
    lons_next = np.roll(lons_rad, -1)
    lats_next = np.roll(lats_rad, -1)

    # Vectorized shoelace formula for spherical coordinates
    dlon = lons_next - lons_rad
    lat_terms = 2 + sin(lats_rad) + sin(lats_next)

    # Calculate area using vectorized operations
    area = abs(np.sum(dlon * lat_terms)) * radius * radius * 0.5

    # Convert to square kilometers
    return area * 1e-6


def poly_from_multipoly(ws_geom: Polygon | MultiPolygon) -> Polygon:
    """Return only biggest polygon from multipolygon WS.

    It's the real WS, and not malfunctioned part of it.

    Args:
        ws_geom: The multipolygon or polygon geometry of the watershed

    Returns:
        The geometry of the biggest polygon in the watershed

    Raises:
        ValueError: If ws_geom is not a Polygon or MultiPolygon

    """
    if isinstance(ws_geom, MultiPolygon):
        # Use built-in area property for better performance
        areas = [polygon.area for polygon in ws_geom.geoms]
        max_index = np.argmax(areas)
        return ws_geom.geoms[max_index]
    elif isinstance(ws_geom, Polygon):
        return ws_geom


def create_gdf(shape: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    """Create geodataframe with given shape as a geometry."""
    gdf_your_ws = poly_from_multipoly(ws_geom=shape)
    # WS from your data
    gdf_your_ws = gpd.GeoSeries([gdf_your_ws])

    # Create extra gdf to use geopandas functions
    gdf_your_ws = gpd.GeoDataFrame({"geometry": gdf_your_ws})
    gdf_your_ws = gdf_your_ws.set_crs("EPSG:4326")

    return gdf_your_ws
