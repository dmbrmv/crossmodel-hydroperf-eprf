import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


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


def create_nse_boxplots(df, nse_columns, figsize=(14, 8), color="lightblue", save_path=None, max_cols=3):
    """Create boxplots for NSE columns in a dataframe.

    Args:
        df: DataFrame containing NSE columns
        nse_columns: List of column names containing NSE values
        figsize: Tuple of figure size (width, height)
        color: Color for the boxplots
        save_path: Optional path to save the figure as PNG (e.g., 'output/nse_boxplots.png')
        max_cols: Maximum number of columns in the subplot grid (default: 3)

    Returns:
        matplotlib.figure.Figure: The created figure

    """
    n_cols = len(nse_columns)
    ncols = min(n_cols, max_cols)  # Maximum 3 columns
    nrows = (n_cols + ncols - 1) // ncols  # Calculate required rows

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a list for consistent indexing
    if n_cols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Russian alphabet letters for labeling
    russian_labels = ["а)", "б)", "в)", "г)", "д)", "е)", "ж)", "з)", "и)", "к)"]

    for i, col in enumerate(nse_columns):
        ax = axes[i]

        # Calculate statistics
        median_val = df[col].median()
        total_gauges = len(df[col].dropna())
        satisfactory = (df[col] >= 0.5).sum()
        satisfactory_pct = satisfactory / total_gauges * 100 if total_gauges > 0 else 0

        # Create boxplot
        df[col].plot.box(
            ax=ax,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5),
        )

        # Customize the plot
        title = f"Медиана: {median_val:.2f}, NSE≥0.5: {satisfactory} ({satisfactory_pct:.1f}%)"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel("NSE", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add Russian alphabet label in top right corner
        if i < len(russian_labels):
            ax.text(
                0.95,
                0.95,
                russian_labels[i],
                transform=ax.transAxes,
                fontsize=16,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Limit NSE axis to 0-1.0 for better visibility
        ax.set_ylim(0, 1.0)

        # Add horizontal line at NSE = 0.5
        ax.axhline(
            y=0.5,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
        )

    # Hide unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    return fig


def russia_plots_n(
    gdf_to_plot: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    columns_from_gdf: list[str],
    label_list: list[str],
    nrows: int,
    ncols: int,
    title_text: list[str] = [""],
    hist_name: list[str] = [""],
    rus_extent: list[float] = [50, 140, 32, 90],
    list_of_limits: list[float] = [0.0, 0.5, 0.7, 0.8, 1.0],
    cmap_lims: tuple[float, float] = (0, 1),
    cmap_name: str = "RdYlGn",
    figsize: tuple[float, float] = (4.88189, 3.34646),
    with_histogram: bool = False,
    ugms: bool = False,
    ugms_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    add_cartobase: bool = True,
    cartobase_resolution: str = "110m",
    cartobase_color: str = "#f2f2f2",
    cartobase_edgecolor: str = "#999999",
    dpi: int = 300,
) -> plt.Figure:
    """Plot multiple maps of Russia with data overlays, cartographic background, and optional histograms.

    Args:
        gdf_to_plot (gpd.GeoDataFrame): Data to plot.
        basemap_data (gpd.GeoDataFrame): Basemap for Russia.
        columns_from_gdf (list[str]): Columns to plot.
        label_list (list[str]): Labels for each subplot.
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        title_text (list[str]): Titles for each subplot.
        hist_name (list[str]): Names for histograms.
        rus_extent (list[float]): Map extent [west, east, south, north].
        list_of_limits (list[float]): Color boundaries for classification.
        cmap_lims (tuple[float, float]): Colormap limits.
        cmap_name (str): Colormap name.
        figsize (tuple[float, float]): Figure size in inches.
        with_histogram (bool): If True, add histograms.
        ugms (bool): If True, plot UGMS polygons.
        ugms_gdf (gpd.GeoDataFrame): UGMS polygons.
        add_cartobase (bool): If True, add cartopy Natural Earth background.
        cartobase_resolution (str): Cartopy Natural Earth resolution ("110m", "50m", "10m").
        cartobase_color (str): Land color for cartographic background.
        cartobase_edgecolor (str): Edge color for cartographic background.
        dpi (int): Resolution for the figure in dots per inch.

    Returns:
        plt.Figure: The resulting matplotlib figure.
    """
    # Validate input lengths
    if len(columns_from_gdf) != len(label_list):
        raise ValueError("columns_from_gdf and label_list must have the same length.")

    # Create Albers Equal Area projection centered on Russia
    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    aea_crs_proj4 = aea_crs.proj4_init

    # Create figure and axes with the projection
    fig, axs = plt.subplots(
        figsize=figsize, ncols=ncols, nrows=nrows, subplot_kw={"projection": aea_crs}, dpi=dpi
    )
    axs = np.ravel(axs)

    # Create colormap with specified boundaries
    cmap = cm.get_cmap(cmap_name, len(list_of_limits) - 1)
    norm_cmap = mpl.colors.BoundaryNorm(list_of_limits, len(list_of_limits) - 1)

    # Ensure title_text and hist_name have enough elements
    if len(title_text) < len(columns_from_gdf):
        title_text = (title_text * len(columns_from_gdf))[: len(columns_from_gdf)]
    if len(hist_name) < len(columns_from_gdf):
        hist_name = (hist_name * len(columns_from_gdf))[: len(columns_from_gdf)]

    # Project GeoDataFrames once
    gdf_proj = gdf_to_plot.to_crs(aea_crs_proj4)
    basemap_proj = basemap_data.to_crs(aea_crs_proj4)
    ugms_proj = ugms_gdf.to_crs(aea_crs_proj4) if ugms and not ugms_gdf.empty else None

    for i, ax in enumerate(axs):
        if i >= len(columns_from_gdf):
            ax.set_visible(False)
            continue

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_extent(rus_extent)  # type: ignore

        # Add cartographic background
        if add_cartobase:
            land = cfeature.NaturalEarthFeature(
                "physical",
                "land",
                cartobase_resolution,
                edgecolor=cartobase_edgecolor,
                facecolor=cartobase_color,
                linewidth=0.5,
            )
            borders = cfeature.NaturalEarthFeature(
                "cultural",
                "admin_0_boundary_lines_land",
                cartobase_resolution,
                edgecolor="black",
                facecolor="none",
                linewidth=0.7,
            )
            lakes = cfeature.NaturalEarthFeature(
                "physical",
                "lakes",
                cartobase_resolution,
                edgecolor=cartobase_edgecolor,
                facecolor="#c6ecff",
                linewidth=0.5,
            )
            ax.add_feature(land, zorder=0)
            ax.add_feature(lakes, zorder=1)
            ax.add_feature(borders, zorder=2)

        # Plot basemap polygons if not using UGMS
        if not ugms:
            basemap_proj.plot(
                ax=ax,
                color="grey",
                edgecolor="black",
                legend=False,
                alpha=0.8,
                linewidth=0.6,
            )

        # Plot UGMS polygons if available
        if ugms and ugms_proj is not None:
            ugms_proj.plot(
                ax=ax,
                column=columns_from_gdf[i],
                cmap=cmap,
                norm=norm_cmap,
                legend=False,
                edgecolor="black",
                linewidth=0.6,
                missing_kwds={"color": "#DF60DF00"},
                zorder=3,
            )

        # Plot the point data
        gdf_proj.plot(
            ax=ax,
            column=columns_from_gdf[i],
            cmap=cmap,
            norm=norm_cmap,
            marker="o",
            markersize=16,
            edgecolor="black",
            linewidth=0.5,
            legend=True,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.35,
                "pad": -0.05,
                "anchor": (0.5, 0.5),
                "drawedges": True,
                "label": "",
                "ticks": list_of_limits,
                "format": "%.1f",
            },
            zorder=4,
        )

        # Add label with background
        ax.text(
            0.03,
            0.97,
            label_list[i],
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                alpha=0.9,
                linewidth=0.8,
            ),
            zorder=10,
        )

        # Add histogram if requested
        if with_histogram:
            ax_hist = ax.inset_axes([0.55, 0.05, 0.40, 0.35], facecolor="white")
            ax_hist.set_frame_on(True)
            ax_hist.patch.set_alpha(0.85)
            for spine in ax_hist.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("black")

            hist_data = gdf_proj[columns_from_gdf[i]].dropna()
            counts, edges = np.histogram(hist_data, bins=list_of_limits)
            cmap_vals = cmap(norm_cmap((edges[:-1] + edges[1:]) / 2))

            bars = ax_hist.bar(
                np.arange(len(counts)),
                counts,
                width=0.8,
                color=cmap_vals,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                alpha=0.9,
            )

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax_hist.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + max(counts) * 0.03,
                        f"{int(h)}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        zorder=3,
                    )

            bin_labels = [f"{edges[j]:.1f}-{edges[j + 1]:.1f}" for j in range(len(edges) - 1)]
            ax_hist.set_xticks(np.arange(len(counts)))
            if len(bin_labels) > 4:
                ax_hist.set_xticklabels(bin_labels, rotation=45, fontsize=7, ha="right")
            else:
                ax_hist.set_xticklabels(bin_labels, fontsize=7)
            ax_hist.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
            ax_hist.tick_params(axis="both", which="major", labelsize=7, zorder=3, length=3)
            ax_hist.set_ylim(0, max(counts) * 1.1 if counts.size > 0 else 1)

            if i < len(hist_name) and hist_name[i]:
                ax_hist.set_title(hist_name[i], fontsize=9, pad=4, fontweight="bold")

        # Add subplot title
        if title_text[i]:
            ax.set_title(title_text[i], fontsize=13, fontweight="bold", pad=10)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    return fig


def metric_viewer(gauges_file: gpd.GeoDataFrame, metric_col: str, metric_csv: str):
    model_metric = pd.read_csv(metric_csv)
    model_metric = model_metric.rename(columns={"basin": "gauge_id", "gauge": "gauge_id"})
    model_metric["gauge_id"] = model_metric["gauge_id"].astype("str")
    model_metric = model_metric.set_index("gauge_id")
    if "gauge_id" not in gauges_file.columns:
        res_file = gauges_file.join(model_metric).dropna()
    else:
        res_file = gauges_file.set_index("gauge_id").join(model_metric).dropna()
    nse_median = res_file[metric_col].median()
    # res_file.loc[res_file[metric_col] < 0, metric_col] = 0

    return res_file, nse_median
