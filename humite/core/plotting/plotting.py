import json
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import vector
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from numpy.random import default_rng
from scipy.stats import wasserstein_distance

from ..plotting import utils as plot_utils
from ..plotting.utils import binclip, hist_ratio, hist_w_unc, plot_ratios

logger = logging.getLogger(__name__)


def plot_features(
    ak_array_dict: Dict[str, ak.Array],
    names: Union[List[str], Dict[str, str]] = ["filler"],
    label_prefix: Optional[str] = None,
    flatten: bool = True,
    histkwargs: Optional[dict] = None,
    legend_only_on: Optional[int] = None,
    legend_kwargs: dict = {},
    ax_rows: int = 1,
    decorate_ax_kwargs: dict = {},
    bins_dict: Optional[Mapping[str, Union[int, np.ndarray]]] = None,
    logscale_features: Optional[Union[Sequence[str], str]] = None,
    logscale_xaxis_features: Optional[Union[Sequence[str], str]] = None,
    colors: Optional[Sequence[str]] = None,
    line_styles: Optional[Sequence[str]] = None,
    ax_size: Tuple[float, float] = (3, 2),
    ylabel: Optional[str] = None,
    ratio: bool = False,
    ratio_range: Tuple[float, float] = (0.5, 1.8),
    normed: bool = True,
    ratio_marker_size: float = 5.0,
    fill_flags: Optional[Sequence[bool]] = None,
    fill_alphas: Optional[Sequence[float]] = None,
):
    """Plot the features of the constituents or showers.

    Parameters
    ----------
    ak_array_dict : dict of awkward array
        Dict with {"name": ak.Array, ...} of the constituents or showers to plot.
    names : list of str or dict, optional
        Names of the features to plot. Either a list of names, or a dict of {"name": "label", ...}.
    label_prefix : str, optional
        Prefix for the plot x-axis labels.
    flatten : bool, optional
        Whether to flatten the arrays before plotting. Default is True.
    histkwargs : dict, optional
        Keyword arguments passed to plt.hist.
    legend_only_on : int, optional
        Plot the legend only on the i-th subplot. Default is None.
    legend_kwargs : dict, optional
        Keyword arguments passed to ax.legend.
    ax_rows : int, optional
        Number of rows of the subplot grid. Default is 1.
    decorate_ax_kwargs : dict, optional
        Keyword arguments passed to `decorate_ax`.
    bins_dict : dict, optional
        Dict of {name: bins} for the histograms. `name` has to be the same as the keys in `names`.
    logscale_features : list, optional
        List of features to plot in log scale, of "all" to plot all features in log scale.
    colors : list, optional
        List of colors for the histograms. Has to have the same length as the number of arrays.
        If shorter, the colors will be repeated.
    line_styles : list, optional
        List of line styles for the histograms.
    ax_size : tuple, optional
        Size of the axes. Default is (3, 2).
    ylabel : str, optional
        Label for the y-axis. Default is None, which will not set a label.
    ratio : bool, optional
        Whether to plot the ratio of the histograms. Default is False.
    ratio_range : tuple, optional
        Range of the y-axis for the ratio plot. Default is (0.5, 1.5).
    normed : bool, optional
        Whether to normalize the histograms. Default is True.
    ratio_marker_size : float, optional
        Marker size for ratio overflow/underflow indicators. Default is 5.
    """

    default_hist_kwargs = {"density": False, "histtype": "step", "bins": 100, "linewidth": 1.5}

    # setup colors
    colors_list: Optional[List[str]] = list(colors) if colors is not None else None
    if colors_list is not None:
        if len(colors_list) < len(ak_array_dict):
            print(
                "Warning: colors list is shorter than the number of arrays. "
                "Will use default colors for remaining ones."
            )
            colors_list = colors_list + [
                f"C{i}" for i in range(len(ak_array_dict) - len(colors_list))
            ]

    line_styles_list: Optional[List[str]] = list(line_styles) if line_styles is not None else None
    if line_styles_list is not None:
        if len(line_styles_list) < len(ak_array_dict):
            line_styles_list = line_styles_list + ["-"] * (
                len(ak_array_dict) - len(line_styles_list)
            )

    if histkwargs is None:
        histkwargs = default_hist_kwargs
    else:
        if "density" in histkwargs:
            logger.warning(
                "The 'density' keyword argument is deprecated and will be ignored. "
                "Use 'normed' instead, which is set to True by default."
            )
        histkwargs = default_hist_kwargs | histkwargs

    # if ratio is True, check that all features are specified in the bins_dict
    if ratio:
        if bins_dict is None:
            raise ValueError("bins_dict has to be specified when ratio is True.")
        if not all(
            [name in bins_dict for name in (names if isinstance(names, list) else names.keys())]
        ):
            raise ValueError(
                "All features have to be specified in bins_dict when ratio is True. "
                f"The following features are missing: {set(names if isinstance(names, list) else names.keys()) - set(bins_dict.keys())}"
            )

    # create the bins dict
    if bins_dict is None:
        bins_dict_local: Dict[str, Union[int, np.ndarray]] = {}
    else:
        bins_dict_local = dict(bins_dict)
    # loop over all names - if the name is not in the bins_dict, use the default bins
    for name in names if isinstance(names, list) else list(names.keys()):
        if name not in bins_dict_local:
            bins_dict_local[name] = histkwargs["bins"]

    # remove default bins from histkwargs
    histkwargs.pop("bins")

    if isinstance(names, list):
        names = {name: name for name in names}

    if len(names) == 1 and ax_rows > 1:
        ax_rows = 1
        logger.info("Setting ax_rows to 1 since only one feature is being plotted.")

    ax_cols = len(names) // ax_rows + 1 if len(names) % ax_rows > 0 else len(names) // ax_rows

    # fig, axarr = plt.subplots(
    #     ax_rows, ax_cols, figsize=(ax_size[0] * ax_cols, ax_size[1] * ax_rows)
    # )
    height_factor = 1.3 if not ratio else 1.8
    fig, axarr = plot_utils.get_ax_and_fig(
        rows=ax_rows,
        cols=ax_cols,
        ratio=ratio,
        figsize=(ax_size[0] * ax_cols * 1.3, ax_size[1] * ax_rows * height_factor),
    )
    if ratio:
        axarr_main = cast(np.ndarray, axarr)[:, 0]
        axarr_ratio = cast(np.ndarray, axarr)[:, 1]
    else:
        axarr_main = cast(np.ndarray, axarr)
        axarr_ratio = None

    legend_handles = []
    legend_labels = []

    base_label, base_array = next(iter(ak_array_dict.items()))
    base_values_dict = {}

    for i_label, (label, ak_array) in enumerate(ak_array_dict.items()):
        color = colors_list[i_label] if colors_list is not None else f"C{i_label}"
        line_style = line_styles_list[i_label] if line_styles_list is not None else "-"
        is_filled = (
            bool(fill_flags[i_label])
            if fill_flags is not None and i_label < len(fill_flags)
            else False
        )
        fill_alpha = (
            float(fill_alphas[i_label])
            if fill_alphas is not None and i_label < len(fill_alphas)
            else 0.3
        )
        entry_histkwargs = dict(histkwargs)
        if is_filled:
            entry_histkwargs["histtype"] = "stepfilled"
            entry_histkwargs["alpha"] = fill_alpha
            entry_histkwargs["edgecolor"] = color
        legend_labels.append(label)
        for i, (feat, feat_label) in enumerate(names.items()):
            resolved_feat = feat
            if not hasattr(ak_array, resolved_feat) and resolved_feat.endswith("_logscale"):
                fallback_feat = resolved_feat[: -len("_logscale")]
                if hasattr(ak_array, fallback_feat):
                    resolved_feat = fallback_feat

            if not hasattr(ak_array, resolved_feat):
                logger.info(f"Feature '{feat}' not found in array '{label}', skipping.")
                continue

            if flatten:
                values = ak.flatten(getattr(ak_array, resolved_feat))
            else:
                values = getattr(ak_array, resolved_feat)

            if not isinstance(bins_dict_local[feat], int) and bins_dict_local[feat] is not None:
                values = binclip(values, bins_dict_local[feat])

            # calculate the histogram itself and the corresponding uncertainties
            bin_edges, hist, unc, band = hist_w_unc(
                arr=ak.to_numpy(values) if isinstance(values, ak.Array) else np.asarray(values),
                bins=bins_dict_local[feat],
                normed=normed,
            )

            # draw the histogram on the main axes
            _, bins, patches = axarr_main[i].hist(
                # values, **histkwargs, bins=bins_dict[feat], color=color
                x=bin_edges[:-1],
                bins=bin_edges,
                weights=hist,
                color=color,
                linestyle=line_style,
                **entry_histkwargs,
            )

            # draw the uncertainty band
            band_arr = np.asarray(band)
            bottom_error = np.append([band_arr[0]], band_arr)
            top_arr = band_arr + 2 * np.asarray(unc)
            top_error = np.append([top_arr[0]], top_arr)

            axarr_main[i].fill_between(
                x=bin_edges,
                y1=bottom_error,
                y2=top_error,
                color=color,
                alpha=0.3,
                zorder=1,
                step="pre",
                edgecolor="none",
            )
            axarr_main[i].set_xlabel(
                feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
            )
            if ylabel is not None:
                axarr_main[i].set_ylabel(ylabel)
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=patches[0].get_edgecolor(),
                        lw=patches[0].get_linewidth(),
                        label=label,
                        linestyle=patches[0].get_linestyle(),
                    )
                )

            if ratio:
                assert axarr_ratio is not None
                if label == base_label:
                    base_values_dict[feat] = hist
                elif feat not in base_values_dict:
                    logger.warning(
                        "Skipping ratio for feature '%s' because base '%s' lacks it.",
                        feat,
                        base_label,
                    )
                    continue

                # calculate the ratio of the histogram to the base histogram
                ratio_values, ratio_unc = hist_ratio(
                    numerator=hist,
                    denominator=base_values_dict[feat],
                    numerator_unc=unc,
                    step=False,
                )
                ratio_values = np.asarray(ratio_values)
                ratio_unc = np.asarray(ratio_unc)
                ratio_values = np.nan_to_num(ratio_values, nan=0.0, posinf=0.0, neginf=0.0)
                ratio_unc = np.nan_to_num(ratio_unc, nan=0.0, posinf=0.0, neginf=0.0)

                # draw the ratio
                axarr_ratio[i].step(
                    bins,
                    # repeat the first entry to avoid a gap in the plot (due to step plot)
                    np.append(np.array(ratio_values[0]), ratio_values),
                    where="pre",
                    color=color,
                    linestyle=line_style,
                    linewidth=histkwargs.get("linewidth", 1),
                )

                # calculate the uncertainty band for the ratio
                ratio_bottom_error = ratio_values - ratio_unc
                ratio_top_error = ratio_values + ratio_unc

                # draw the uncertainty band
                axarr_ratio[i].fill_between(
                    bins,
                    # repeat the first entry to avoid a gap in the plot (due to step plot)
                    np.append(np.array(ratio_bottom_error[0]), ratio_bottom_error),
                    np.append(np.array(ratio_top_error[0]), ratio_top_error),
                    color=color,
                    alpha=0.3,
                    zorder=1,
                    step="pre",
                    edgecolor="none",
                )

                axarr_ratio[i].set_xlabel(
                    feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
                )
                axarr_ratio[i].set_ylabel("Ratio")
                axarr_main[i].set_xlabel(None)
                axarr_ratio[i].set_ylim(*ratio_range)
                # find out where the ratio is larger than the upper limit
                ratio_over = ratio_values > ratio_range[1]
                # find out where the ratio is smaller than the lower limit
                ratio_under = ratio_values < ratio_range[0]
                # bin centers
                bin_centers = (bins[:-1] + bins[1:]) / 2
                # plot the over and under limit regions
                ratio_overunderflow_kwargs = dict(
                    clip_on=False,
                    markersize=ratio_marker_size,
                    color=color,
                )
                axarr_ratio[i].plot(
                    bin_centers[ratio_over],
                    ratio_range[1] * np.ones_like(bin_centers)[ratio_over],
                    "^",
                    **ratio_overunderflow_kwargs,
                )
                axarr_ratio[i].plot(
                    bin_centers[ratio_under],
                    ratio_range[0] * np.ones_like(bin_centers)[ratio_under],
                    "v",
                    **ratio_overunderflow_kwargs,
                )

    legend_kwargs["handles"] = legend_handles
    legend_kwargs["labels"] = legend_labels
    legend_kwargs["frameon"] = False
    for i, (_ax, feat_name) in enumerate(zip(axarr_main, names.keys())):
        if legend_only_on is None:
            _ax.legend(**legend_kwargs)
        else:
            if i == legend_only_on:
                _ax.legend(**legend_kwargs)

        if (logscale_features is not None and feat_name in logscale_features) or (
            logscale_features == "all"
        ):
            _ax.set_yscale("log")
        # hide overlapping x ticklabels on main axes when a ratio panel exists
        if ratio:
            _ax.tick_params(labelbottom=False)
        plot_utils.decorate_ax(_ax, **decorate_ax_kwargs)

        if (logscale_xaxis_features is not None and feat_name in logscale_xaxis_features) or (
            logscale_xaxis_features == "all"
        ):
            _ax.set_xscale("log")
            if ratio:
                assert axarr_ratio is not None
                axarr_ratio[i].set_xscale("log")

        # keep main and ratio x-limits identical to the histogram bin range
        if isinstance(bins_dict_local[feat_name], np.ndarray):
            x_min = float(np.min(bins_dict_local[feat_name]))
            x_max = float(np.max(bins_dict_local[feat_name]))
            _ax.set_xlim(x_min, x_max)
            if ratio:
                assert axarr_ratio is not None
                axarr_ratio[i].set_xlim(x_min, x_max)

    # make empty plots invisible
    for i in range(len(names), len(axarr_main)):
        axarr_main[i].axis("off")
        if ratio:
            assert axarr_ratio is not None
            axarr_ratio[i].axis("off")

    # align the ratio and main plot x-ranges
    if ratio:
        fig, axarr = plot_utils.align_main_and_ratio_xrange(fig, axarr)

    # crop the borders
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.00)

    fig.tight_layout()
    return fig, axarr


def plotting_2D_showers(ak_array, inc_en, size_n=8, idx=None, label=""):
    # 2D plots of the showers
    if idx is None:
        n1 = int(ak.num(ak_array.x, axis=0))
        rng = default_rng()
        if size_n > n1:
            print(
                f"Warning: size_n={size_n} is larger than the number of showers, using n1={n1} instead."
            )
            size_n = n1

        idx = rng.choice(n1, size=size_n, replace=False)

    sorted_inc_en = ak.argsort(inc_en[idx])
    x = ak_array.x[idx][sorted_inc_en]
    z = ak_array.z[idx][sorted_inc_en]
    e = ak_array.energy[idx][sorted_inc_en]
    inc_en = inc_en[idx][sorted_inc_en]

    if size_n > 8:
        size_n = 8  # Limit to 8 plots for better visibility

    fig = plt.figure(figsize=(15, 6), facecolor="white")
    fig.suptitle(f"{label}")
    for i in range(size_n):
        if i + 2 >= size_n:
            break
        plt.subplot(2, int(size_n / 2), i + 1)
        plt.scatter(
            x[i],
            z[i],
            s=e[i],
            label=str(np.round(inc_en[i], 1)) + " [GeV]",
        )

        plt.legend()
        plt.ylim(0, 48)
        # plt.xlim(-500, 500)
        plt.xlabel("x [mm]")
        plt.ylabel("z [layer]")
    plt.tight_layout()
    return fig, idx
