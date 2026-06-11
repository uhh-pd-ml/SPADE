from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.transforms import Bbox, ScaledTranslation

from humite.core.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)

rcParams = mpl.rcParams
DEFAULT_ALPHA = 0.95
DEFAULT_LABELS = {}
DEFAULT_COLORS = [
    # list of 10 colors based on table 1 in https://arxiv.org/pdf/2107.02270
    "#3f90da",  # blue
    "#bd1f01",  # red
    "#a4a294",  # grey
    "#ffa90e",  # orange (yellow-ish)
    "#832db6",  # purple
    "#92dadd",  # light blue
    "#a96b59",  # brown
    "#E76300",  # orange (red-ish)
    "#b9ac70",  # olive
    "#817175",  # grey (darker)
]

params_to_update = {
    # --- axes ---
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    "axes.prop_cycle": cycler(
        "color",
        [mpl.colors.ColorConverter().to_rgba(col, DEFAULT_ALPHA) for col in DEFAULT_COLORS],
    ),
    # --- figure ---
    "figure.figsize": (3.5, 2.5),
    # "figure.dpi": 130,
    # --- grid ---
    "grid.color": "black",
    "grid.alpha": 0.1,
    "grid.linestyle": "-",
    "grid.linewidth": 1,
    # --- legend ---
    "legend.fontsize": 10,
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    # --- lines ---
    "lines.linewidth": 1.5,
    "lines.markeredgewidth": 0,
    "lines.markersize": 7,
    # "lines.solid_capstyle": "round",
    # --- patches ---
    "patch.facecolor": "4C72B0",
    "patch.linewidth": 1.7,
    # --- histogram ---
    "hist.bins": 100,
    # --- font ---
    "font.family": "sans-serif",
    "font.sans-serif": "Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif",
    # --- image ---
    "image.cmap": "Greys",
}
params_to_update_dark = {
    # --- axes ---
    "axes.prop_cycle": cycler(
        "color",
        [
            mpl.colors.ColorConverter().to_rgba(col, DEFAULT_ALPHA)
            for col in [
                "dodgerblue",
                "red",
                "mediumseagreen",
                "darkorange",
                "orchid",
                "turquoise",
                "#64B5CD",
            ]
        ],
    ),
    # --- figure ---
    "figure.facecolor": "#1A1D22",
    # --- grid ---
    "grid.color": "lightgray",
    # --- lines ---
    "lines.solid_capstyle": "round",
    # --- font ---
    "font.family": "sans-serif",
    "font.sans-serif": "Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif",
    # --- image ---
    "image.cmap": "Greys",
    # --- legend ---
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    # --- xtick ---
    "xtick.direction": "in",
    "xtick.color": "white",
    # --- ytick ---
    "ytick.direction": "out",
    "ytick.color": "white",
    # --- axes.axisbelow ---
    "axes.axisbelow": True,
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "#1A1D22",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "figure.edgecolor": "#1A1D22",
    "savefig.facecolor": "#1A1D22",
    "savefig.edgecolor": "#1A1D22",
}


def reset_mpl_style():
    """Reset matplotlib rcParams to default."""
    rcParams.update(mpl.rcParamsDefault)


def set_mpl_style(darkmode=False):
    """Set matplotlib rcParams to custom configuration."""
    reset_mpl_style()
    rcParams.update(params_to_update if not darkmode else params_to_update_dark)


def save(fig, saveas, transparent=True):
    """Save a figure both as pdf and as png.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    saveas : str
        The path to save the figure to, expected to end in ".pdf".
    transparent : bool, optional
        Whether to save the figure with a transparent background, by default True
    """
    save_kwargs = dict(transparent=transparent, dpi=300, bbox_inches="tight")
    # create the directory if it does not exist
    if not Path(saveas).parent.exists():
        print(f"Creating directory {Path(saveas).parent}")
        Path(saveas).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {saveas}")
    fig.savefig(saveas, **save_kwargs)
    fig.savefig(str(saveas).replace(".pdf", ".png"), **save_kwargs)


# # default seaborn aesthetic
# # darkgrid + deep palette + notebook context

# # axes.axisbelow: True
# # axes.edgecolor: white
# # axes.facecolor: EAEAF2
# # axes.grid: True
# # axes.labelcolor: .15
# # axes.labelsize: 11
# # axes.linewidth: 0
# # axes.prop_cycle: cycler('color', ['4C72B0', '55A868', 'C44E52', '8172B2', 'CCB974', '64B5CD'])
# axes.prop_cycle: cycler('color', ['55A868', 'C44E52', '4C72B0', '8172B2', 'CCB974', '64B5CD'])
# # axes.prop_cycle: cycler('color', ['55A868', '9E4AC2','4C72B0' , 'C44E52', '8172B2', 'CCB974', '64B5CD'])
# # axes.titlesize: 12

# figure.facecolor: white


# # xtick.color: .15
# # xtick.direction: out
# # xtick.labelsize: 10
# # xtick.major.pad: 7
# # xtick.major.size: 0
# # xtick.major.width: 1
# # xtick.minor.size: 0
# # xtick.minor.width: .5

# # ytick.color: .15
# # ytick.direction: out
# # ytick.labelsize: 10
# # ytick.major.pad: 7
# # ytick.major.size: 0
# # ytick.major.width: 1
# # ytick.minor.size: 0
# # ytick.minor.width: .5

# # figure.facecolor: white
# # text.color: .15
# # axes.labelcolor: .15
# # legend.frameon: False
# # legend.numpoints: 1
# # legend.scatterpoints: 1
# # xtick.direction: in
# # ytick.direction: out
# # xtick.color: .15
# # ytick.color: .15
# # axes.axisbelow: True
# # image.cmap: Greys
# # font.family: sans-serif
# # font.sans-serif: Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif
# # grid.linestyle: -
# # lines.solid_capstyle: round

# # Seaborn dark parameters
# # axes.grid: False
# # axes.facecolor: EAEAF2
# # axes.edgecolor: white
# # axes.linewidth: 0
# # grid.color: white
# # xtick.major.size: 0
# # ytick.major.size: 0
# # xtick.minor.size: 0
# # ytick.minor.size: 0


def decorate_ax(
    ax,
    yscale=1.3,
    text=None,
    text_line_spacing=1.2,
    text_font_size=12,
    draw_legend=False,
    indent=0.7,
    top_distance=1.2,
    hepstyle=False,
    remove_first_ytick=False,
):
    """Helper function to decorate the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to decorate
    yscale : float, optional
        Factor by which the y-axis is scaled, by default 1.3
    text : str, optional
        Text to add to the plot, by default None
    text_line_spacing : float, optional
        Spacing between lines of text, by default 1.2
    text_font_size : int, optional
        Font size of the text, by default 12
    draw_legend : bool, optional
        Draw the legend with `frameon=False`, by default False
    indent : float, optional
        Horizontal indent, by default 0.7
    top_distance : float, optional
        Vertical indent, by default 1.2
    hepstyle : bool, optional
        Use the atlasify function to make the plot look like an ATLAS plot, by default False
    remove_first_ytick : bool, optional
        Remove the first y-tick, by default False.
        Can be useful to avoid overlap with the ratio plot ticks.
    """
    PT = 1 / 72  # 1 point in inches

    # reset the y-axis limits (if they were changed before, it can happen
    # that the y-axis is not scaled correctly. especially it happens that ymin
    # becomes 0 even after setting logscale, which raises an error below as we
    # divide by ymin for logscale)
    if yscale != 1:
        xmin, xmax = ax.get_xlim()
        ax.relim()
        ax.autoscale()
        ax.set_xlim(xmin, xmax)

        # This weird order is necessary to allow for later
        # saving in logscaled y-axis
        if ax.get_yscale() == "log":
            ymin, _ = ax.get_ylim()
            ax.set_yscale("linear")
            _, ymax = ax.get_ylim()
            ax.set_yscale("log")
            yscale = (ymax / ymin) ** (yscale - 0.99)
        else:
            ymin, ymax = ax.get_ylim()

        # scale the y-axis to avoid overlap with text
        ax.set_ylim(top=yscale * (ymax - ymin) + ymin)

    if text is None:
        pass
    elif isinstance(text, str):
        # translation from the left side of the axes (aka indent)
        trans_indent = ScaledTranslation(
            indent * text_line_spacing * PT * text_font_size,
            0,
            ax.figure.dpi_scale_trans,
        )
        # translation from the top of the axes
        trans_top = ScaledTranslation(
            0,
            -top_distance * text_line_spacing * PT * text_font_size,
            ax.figure.dpi_scale_trans,
        )

        # add each line of the tag text to the plot
        for line in text.split("\n"):
            # fmt: off
            ax.text(0, 1, line, transform=ax.transAxes + trans_top + trans_indent, fontsize=text_font_size)  # noqa: E501
            trans_top += ScaledTranslation(0, -text_line_spacing * text_font_size * PT, ax.figure.dpi_scale_trans)  # noqa: E501
            # fmt: on
    else:
        raise TypeError("`text` attribute of the plot has to be of type `str`.")

    if draw_legend:
        ax.legend(frameon=False)

    if remove_first_ytick:
        # remove the first y-tick label of the upper plot to avoid overlap with the ratio plot
        ax.set_yticks(ax.get_yticks()[1:])


def get_good_linestyles(names=None):
    """Returns a list of good linestyles.

    Parameters
    ----------
    names : list or str, optional
        List or string of the name(s) of the linestyle(s) you want to retrieve, e.g.
        "densely dotted" or ["solid", "dashdot", "densely dashed"], by default None

    Returns
    -------
    list
        List of good linestyles. Either the specified selection or the whole list in
        the predefined order.

    Raises
    ------
    ValueError
        If `names` is not a str or list.
    """
    linestyle_tuples = {
        "solid": "solid",
        "densely dashed": (0, (5, 1)),
        "densely dotted": (0, (1, 1)),
        "densely dashdotted": (0, (3, 1, 1, 1)),
        "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
        "dotted": (0, (1, 1)),
        "dashed": (0, (5, 5)),
        "dashdot": "dashdot",
        "loosely dashed": (0, (5, 10)),
        "loosely dotted": (0, (1, 10)),
        "loosely dashdotted": (0, (3, 10, 1, 10)),
        "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
        "dashdotted": (0, (3, 5, 1, 5)),
        "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    }

    default_order = [
        "solid",
        "densely dotted",
        "densely dashed",
        "densely dashdotted",
        "densely dashdotdotted",
        "dotted",
        "dashed",
        "dashdot",
        # "loosely dotted",
        # "loosely dashed",
        # "loosely dashdotted",
        # "loosely dashdotdotted",
        "dashdotted",
        "dashdotdotted",
    ]
    if names is None:
        names = default_order * 3
    elif isinstance(names, str):
        return linestyle_tuples[names]
    elif not isinstance(names, list):
        raise ValueError("Invalid type of `names`, has to be a list of strings or a string.")
    return [linestyle_tuples[name] for name in names]


def get_col(i):
    """Get the i-th color from the default color cycle."""
    return rcParams["axes.prop_cycle"].by_key()["color"][i % len(rcParams["axes.prop_cycle"])]


def get_label(name):
    """Get the label for the given name."""
    return DEFAULT_LABELS[name]


def get_10_subplots(ratio=True, figsize=None, orientation="2x5"):
    """Create a 2x5 or 5x2 grid of axes for plotting with/without ratios. The axes are returned as
    an array of shape (2, 10), where the first row contains the main plot axes and the second row
    contains the ratio plot axes.

    Parameters:
    ----------
    ratio : bool, optional
        Whether to include a ratio plot. Default is True.
    figsize : tuple, optional
        Size of the figure.
    orientation : str, optional
        Orientation of the grid, either '5x2' or '2x5'. Default is '5x2'.
    """
    if orientation == "5x2":
        return get_ax_and_fig_5x2(ratio=ratio, figsize=figsize)
    elif orientation == "2x5":
        return get_ax_and_fig_2x5(ratio=ratio, figsize=figsize)
    else:
        raise ValueError("Invalid orientation, has to be '5x2' or '2x5'.")


def get_ax_and_fig_2x5(ratio=True, figsize=(18, 6)):
    """Create a 2x5 grid of axes for plotting histograms and ratios. The axes are returned as an
    array of shape (2, 10), where the first row contains the main plot axes and the second row
    contains the ratio plot axes.

    Parameters:
    ----------
    ratio : bool, optional
        Whether to include a ratio plot. Default is True.
    figsize : tuple, optional
        Size of the figure.
    """

    if ratio:
        gridspec = dict(hspace=0.0, height_ratios=[1, 0.3, 0.3, 1, 0.3])
        fig, ax = plt.subplots(5, 5, figsize=figsize, gridspec_kw=gridspec)
        # make third row invisible
        for i in range(5):
            ax[2, i].axis("off")
        # remove the dummy axes in the middle
        axes = np.concatenate([ax[:2, :], ax[3:, :]], axis=1)
    else:
        fig, ax = plt.subplots(2, 5, figsize=figsize)
        axes = ax.flatten()
    return fig, axes


def get_ax_and_fig_5x2(ratio=True, figsize=(6, 18)):
    """Create a 5x2 grid of axes for plotting histograms and ratios. The axes are returned as an
    array of shape (2, 10), where the first row contains the main plot axes and the second row
    contains the ratio plot axes.

    Parameters:
    ----------
    ratio : bool, optional
        Whether to include a ratio plot. Default is True.
    figsize : tuple, optional
        Size of the figure.
    """

    if ratio:
        gridspec = dict(hspace=0.0, height_ratios=(5 * [1, 0.3, 0.3]))
        fig, ax = plt.subplots(15, 2, figsize=figsize, gridspec_kw=gridspec)
        # make every third row invisible
        for i in range(5):
            ax[3 * i + 2, 0].axis("off")
            ax[3 * i + 2, 1].axis("off")
        # remove the dummy axes in the middle
        main_plot_axes = np.concatenate([ax[i, :] for i in range(0, 15, 3)], axis=0)
        ratio_axes = np.concatenate([ax[i, :] for i in range(1, 15, 3)], axis=0)
        axes = np.array([main_plot_axes, ratio_axes])
    else:
        fig, ax = plt.subplots(5, 2, figsize=figsize)
        axes = ax.flatten()
    return fig, axes


def get_ax_and_fig_for_ratio_plot(figsize=(4, 3)):
    """Returns fig, axes for a ratio plot. ax[0] is the histogram, ax[1] is the ratio.

    Parameters:
    ----------
    figsize : tuple, optional
        Size of the figure.
    """

    gridspec = dict(hspace=0.0, height_ratios=[1, 0.3])
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw=gridspec)
    # make third second ax invisible
    # ax[1].axis("off")
    # remove the dummy axes in the middle
    # axes = np.concatenate([ax[0], ax[-1]], axis=1)
    return fig, ax


def save_two_subplots(
    fig,
    ax1,
    ax2,
    saveas: str,
    expanded_x=1.05,
    expanded_y=1.05,
):
    """Save two subplots of a figure to a single file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the subplots.
    ax1 : matplotlib.axes.Axes
        The first subplot.
    ax2 : matplotlib.axes.Axes
        The second subplot.
    saveas : str
        The path to save the figure to.
    expanded_x : float, optional
        Factor by which to expand the bounding box in the x-direction, by default 1.05
    expanded_y : float, optional
        Factor by which to expand the bounding box in the y-direction, by default 1.05
    """
    bbox1 = ax1.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    bbox2 = ax2.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    bbox_total = Bbox.union([bbox1, bbox2])
    fig.savefig(saveas, bbox_inches=bbox_total.expanded(expanded_x, expanded_y))


def save_subplot(
    fig,
    ax,
    saveas: str,
    expanded_x=1.05,
    expanded_y=1.05,
):
    """Save a single subplot of a figure to a file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the subplot.
    ax : matplotlib.axes.Axes
        The subplot to save.
    saveas : str
        The path to save the figure to.
    expanded_x : float, optional
        Factor by which to expand the bounding box in the x-direction, by default 1.05
    expanded_y : float, optional
        Factor by which to expand the bounding box in the y-direction, by default 1.05
    """
    bbox = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(saveas, bbox_inches=bbox.expanded(expanded_x, expanded_y))


def get_ax_and_fig(rows=5, cols=2, ratio=True, figsize=(6, 18)):
    """Create a grid of axes for plotting histograms and ratios. The axes are returned as an
    array of shape (2, rows*cols), where the first row contains the main plot axes and the second row
    contains the ratio plot axes.

    Parameters:
    ----------
    rows : int, optional
        Number of rows of plots. Default is 5.
    cols : int, optional
        Number of columns of plots. Default is 2.
    ratio : bool, optional
        Whether to include a ratio plot. Default is True.
    figsize : tuple, optional
        Size of the figure.
    """

    if ratio:
        # figure out needed space for xlabel based on fontsize and then set the
        gridspec = dict(hspace=0.0, wspace=0.4, height_ratios=(rows * [1, 0.1, 0.3, 0.4]))
        fig, ax = plt.subplots(4 * rows, cols, figsize=figsize, gridspec_kw=gridspec)
        if cols == 1:
            ax = ax[:, np.newaxis]
        # make every second and fourth row invisible
        for i in range(rows):
            for j in range(cols):
                ax[4 * i + 1, j].set_visible(False)
                ax[4 * i + 3, j].set_visible(False)
        # remove the dummy axes in the middle
        main_plot_axes = np.concatenate([ax[i, :] for i in range(0, 4 * rows, 4)], axis=0)
        ratio_axes = np.concatenate([ax[i, :] for i in range(2, 4 * rows, 4)], axis=0)
        # remove the x-tick labels from the main plot axes
        for ax in main_plot_axes:
            ax.set_xticklabels([])
        axes = np.array([main_plot_axes, ratio_axes]).transpose()
    else:
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        axes = ax.flatten() if rows * cols > 1 else np.array([ax])
    return fig, axes


def align_main_and_ratio_xrange(fig, axarr):
    """Align the x-axis ranges of the main and ratio plots in a grid of axes.

    Parameters:
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes.
    axarr : numpy.ndarray
        Array of axes.
    """
    # connect each main plot axis with the corresponding ratio axis
    for i in range(axarr.shape[0]):
        main_ax, ratio_ax = axarr[i]
        xlim = main_ax.get_xlim()
        ratio_ax.set_xlim(xlim)
    return fig, axarr


def plot_ratios(ax_twin, features_list, bins, feature, labels, colors, mask, weights=None):
    """Plots ratio plots for each feature compared to the first feature.

    Args:
        ax_twin (matplotlib.axes.Axes): The twin axis on which to plot the ratios.
        features_list (list): A list of dictionaries, each containing a feature's data.
        bins (array-like): The bin edges for the histogram.
        feature (str): Name of the feature to plot.
        labels (list): A list of labels for each feature.
        colors (list): A list of colors for each feature.
        weights (str, optional): Key for weights in the feature dictionaries. Default is None.

    Returns:
        None
    """

    for i in range(1, len(features_list)):  # Compare feature 1 with feature 2 and 3
        if weights:
            counts1, _ = np.histogram(
                features_list[0][feature], bins=bins, weights=features_list[0][weights]
            )
            counts2, _ = np.histogram(
                features_list[i][feature], bins=bins, weights=features_list[i][weights]
            )
        else:
            counts1, _ = np.histogram(features_list[0][feature], bins=bins)
            counts2, _ = np.histogram(features_list[i][feature], bins=bins)

        ratios = []
        for j in range(len(counts1)):
            if counts2[j] == 0:
                ratios.append(np.nan)
            else:
                ratios.append(counts2[j] / counts1[j])
        linestyle = "--" if i > 1 else "-"
        ax_twin.step(
            bins,  # Use all energy bin edges
            ratios + [ratios[-1]],  # Duplicate last ratio for the step
            where="post",  # 'post' creates the step after the data point
            color=colors[i],  # Use color from the current feature
            label=f"ratio to {labels[i]}",  # Use label from the current feature
            linestyle=linestyle,  # Use linestyle from the current feature
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        # Add out-of-bounds markers at the edge of the plot
        mask_below = np.array(ratios) < mask[0]
        mask_above = np.array(ratios) > mask[1]

        ax_twin.plot(
            bin_centers[mask_below],
            np.full_like(bin_centers[mask_below], mask[0]),
            marker="v",
            color=colors[i],
            markersize=6,
            clip_on=False,
            linestyle="None",
        )
        ax_twin.plot(
            bin_centers[mask_above],
            np.full_like(bin_centers[mask_above], mask[1]),
            marker="^",
            color=colors[i],
            markersize=6,
            clip_on=False,
            linestyle="None",
        )


def save_divide(
    numerator,
    denominator,
    default: float = 1.0,
):
    """
    Division using numpy divide function returning default value in cases where
    denominator is 0.

    Parameters
    ----------
    numerator: array_like, int, float
        Numerator in the ratio calculation.
    denominator: array_like, int, float
        Denominator in the ratio calculation.
    default: float
        Default value which is returned if denominator is 0.

    Returns
    -------
    ratio: array_like, float
    """
    if isinstance(numerator, (int, float, np.number)) and isinstance(
        denominator, (int, float, np.number)
    ):
        output_shape = 1
    else:
        try:
            output_shape = denominator.shape
        except AttributeError:
            output_shape = numerator.shape

    ratio = np.divide(
        numerator,
        denominator,
        out=np.ones(
            output_shape,
            dtype=float,
        )
        * default,
        where=(denominator != 0),
    )
    if output_shape == 1:
        return float(ratio)
    return ratio


def hist_w_unc(
    arr,
    bins,
    filled: bool = False,
    bins_range=None,
    normed: bool = True,
    weights: np.ndarray = None,
    bin_edges: np.ndarray = None,
    sum_squared_weights: np.ndarray = None,
    underoverflow: bool = False,
):
    """
    Computes histogram and the associated statistical uncertainty.

    Parameters
    ----------
    arr : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str
        `bins` parameter from `np.histogram`
    bins_range : tuple, optional
        `range` parameter from `np.histogram`. This is ignored if `bins` is array like,
        because then the entries of `bins` are used as bin edges.
    normed : bool, optional
        If True (default) the calculated histogram is normalised to an integral
        of 1.
    weights : np.ndarray, optional
        Weights for the input data. Has to be an array of same length as the input
        data with a weight for each entry. If not specified, weight 1 will be given
        to each entry. The uncertainty of bins with weighted entries is
        sqrt(sum_i{w_i^2}) where w_i are the weights of the entries in this bin.
        By default None.
    underoverflow : bool, optional
        Option to include under- and overflow values in outermost bins.

    Returns
    -------
    bin_edges : array of dtype float
        Return the bin edges (length(hist)+1)
    hist : numpy array
        The values of the histogram. If normed is true (default), returns the
        normed counts per bin
    unc : numpy array
        Statistical uncertainty per bin.
        If normed is true (default), returns the normed values.
    band : numpy array
        lower uncertainty band location: hist - unc
        If normed is true (default), returns the normed values.
    """
    if weights is None:
        weights = np.ones(len(arr))

    # Check if there are nan values in the input values
    nan_mask = np.isnan(arr)
    if np.sum(nan_mask) > 0:
        logger.warning("Histogram values contain %i nan values!", np.sum(nan_mask))
        # Remove nan values
        arr = arr[~nan_mask]
        weights = weights[~nan_mask]

    # Check if there are inf values in the input values
    inf_mask = np.isinf(arr)
    if np.sum(inf_mask) > 0:
        logger.warning("Histogram values contain %i +-inf values!", np.sum(inf_mask))

    # If the histogram is not already filled we need to produce the histogram counts
    # and bin edges
    if not filled:
        # Calculate the counts and the bin edges
        counts, bin_edges = np.histogram(arr, bins=bins, range=bins_range, weights=weights)

        # calculate the uncertainty with sum of squared weights (per bin, so we use
        # np.histogram again here)
        unc = np.sqrt(np.histogram(arr, bins=bins, range=bins_range, weights=weights**2)[0])

        if underoverflow:
            # add two dummy bins (from outermost bins to +-infinity)
            bins_with_overunderflow = np.hstack(
                [
                    np.array([-np.inf]),
                    bin_edges,
                    np.array([np.inf]),
                ]
            )
            # recalculate the histogram with this adjusted binning
            counts, _ = np.histogram(arr, bins=bins_with_overunderflow, weights=weights)
            counts[1] += counts[0]  # add underflow values to underflow bin
            counts[-2] += counts[-1]  # add overflow values to overflow bin
            counts = counts[1:-1]  # remove dummy bins

            # calculate the sum of squared weights
            sum_squared_weights = np.histogram(
                arr, bins=bins_with_overunderflow, weights=weights**2
            )[0]

            # add sum of squared weights from under/overflow values
            # to under/overflow bin
            sum_squared_weights[1] += sum_squared_weights[0]
            sum_squared_weights[-2] += sum_squared_weights[-1]
            # remove dummy bins
            sum_squared_weights = sum_squared_weights[1:-1]

            # uncertainty is sqrt(sum_squared_weights)
            unc = np.sqrt(sum_squared_weights)

        if normed:
            sum_of_weights = float(np.sum(weights))
            bin_widths = np.diff(bin_edges)
            counts = save_divide(counts, sum_of_weights * bin_widths, 0)
            unc = save_divide(unc, sum_of_weights * bin_widths, 0)

    # If the histogram is already filled then the uncertainty is computed
    # differently
    else:
        if sum_squared_weights is not None:
            sum_squared_weights = np.array(sum_squared_weights)[~nan_mask]
            unc = np.sqrt(sum_squared_weights)
        else:
            unc = np.sqrt(arr)  # treat arr as bin heights (counts)

        counts = arr

        if normed:
            if bin_edges is None:
                raise ValueError(
                    "`bin_edges` must be provided when `filled` is True and `normed` is True."
                )
            counts_sum = float(np.sum(counts))
            bin_widths = np.diff(bin_edges)
            counts = save_divide(counts, counts_sum * bin_widths, 0)
            unc = save_divide(unc, counts_sum * bin_widths, 0)

    # regardless of if the histogram is filled
    band = counts - unc
    hist = counts

    return bin_edges, hist, unc, band


def hist_ratio(
    numerator,
    denominator,
    numerator_unc,
    step: bool = True,
    method: str = "divide",
):
    """
    Calculate the ratio of the given bincounts and
    returns the input for a step function that plots the ratio.

    Parameters
    ----------
    numerator : array_like
        Numerator in the ratio calculation.
    denominator : array_like
        Denominator in the ratio calculation.
    numerator_unc : array_like
        Uncertainty of the numerator.
    step : bool
        if True duplicates first bin to match with step plotting function,
        by default True
    method : str
        Selects the method by which the ratio should be calculated.
        "divide" calculates the ratio as the division of the numerator by the denominator.
        "root_square_diff" calculates the Root Square Difference
        between the numerator and the denominator.
        by default "divide"


    Returns
    -------
    step_ratio : array_like
        Ratio returning 1 in case the denominator is 0.
    step_ratio_unc : array_like
        Stat. uncertainty of the step_ratio

    Raises
    ------
    AssertionError
        If inputs don't have the same shape.

    """
    numerator, denominator, numerator_unc = (
        np.array(numerator),
        np.array(denominator),
        np.array(numerator_unc),
    )
    if numerator.shape != denominator.shape:
        raise AssertionError("Numerator and denominator don't have the same length")
    if numerator.shape != numerator_unc.shape:
        raise AssertionError("Numerator and numerator_unc don't have the same length")

    if method == "divide":
        step_ratio = save_divide(numerator, denominator, 1 if step else np.inf)
        # Calculate ratio uncertainty
        step_unc = save_divide(numerator_unc, denominator, default=0 if step else np.inf)
    elif method == "root_square_diff":
        step_ratio = np.multiply(
            np.sign(numerator - denominator), np.sqrt(np.abs(numerator**2 - denominator**2))
        )
        # Calculate ratio uncertainty
        step_unc = np.zeros_like(step_ratio)
        step_unc = np.divide(
            np.multiply(numerator, numerator_unc),
            np.sqrt(np.abs(numerator**2 - denominator**2)),
            where=(numerator - denominator != 0),
        )
    else:
        raise ValueError("'method' can only be 'divide' or 'root_square_diff'.")

    if step:
        # Add an extra bin in the beginning to have the same binning as the input
        # Otherwise, the ratio will not be exactly above each other (due to step)
        step_ratio = np.append(np.array([step_ratio[0]]), step_ratio)
        step_unc = np.append(np.array([step_unc[0]]), step_unc)

    return step_ratio, step_unc


def binclip(x, bins, dropinf=False):
    binfirst_center = bins[0] + (bins[1] - bins[0]) / 2
    binlast_center = bins[-2] + (bins[-1] - bins[-2]) / 2
    if dropinf:
        print("Dropping inf")
        print("len(x) before:", len(x))
        x = x[~np.isinf(x)]
        print("len(x) after:", len(x))
    return np.clip(x, binfirst_center, binlast_center)
