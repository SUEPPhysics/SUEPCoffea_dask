import re
import warnings

import hist
import hist.intervals
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from utils import hist_utils
from utils.style_utils import getColor, getStyles

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.max_open_warning": 0})
matplotlib.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 70
plt.style.use(hep.style.CMS)
hep.style.use("CMS")


def styled_plot_ratio(
    hlist,
    labels,
    stacked_hlist=None,
    stacked_labels=None,
    density=False,
    systs=None,
    xlabel=None,
    xlim=None,
    log=True,
):
    styles = getStyles(labels)
    pretty_labels = [s.get("label", l) for s, l in zip(styles, labels)]
    fig, axs = plot_ratio(
        hlist,
        pretty_labels,
        density=density,
        systs=systs,
        xlabel=xlabel,
        xlim=xlim,
        log=log,
        cmap=[style.get("color", None) for style in styles],
        linewidth=[style.get("linewidth", 1) for style in styles],
        linestyle=[style.get("linestyle", "-") for style in styles],
        fmt=[style.get("fmt", "") for style in styles],
    )
    axs[0].legend(fontsize="x-small", loc=(1.01, 0))

    if stacked_hlist:
        if stacked_labels:
            stacked_styles = getStyles(stacked_labels)
        pretty_stacked_labels = [
            s.get("label", l) for s, l in zip(stacked_styles, stacked_labels)
        ]
        _default_cmap = plt.cm.jet(np.linspace(0, 1, len(stacked_hlist)))
        cmap = []
        for ic in range(len(_default_cmap)):
            cmap.append(stacked_styles[ic].get("color", _default_cmap[ic]))
        hep.histplot(
            stacked_hlist,
            label=pretty_stacked_labels,
            ax=axs[0],
            density=density,
            stack=True,
            histtype="fill",
            color=cmap,
            zorder=0,
        )
        axs[0].set_xlabel("")
        if pretty_stacked_labels:
            leg_handles, leg_labels = axs[0].get_legend_handles_labels()
            # reverse order to follow the stacking
            stacked_leg_labels = [l for l in leg_labels if l in pretty_stacked_labels][
                ::-1
            ]
            stacked_leg_handles = [
                leg_handles[leg_labels.index(l)] for l in stacked_leg_labels
            ]
            # for unstacked histograms, reorder legend labels and handles to follow the parameter 'labels' order
            # this is already done in plot_ratio, but since we are adding the stacked histograms after the fact, we need to do it again
            other_leg_labels = pretty_labels
            other_leg_handles = [
                leg_handles[leg_labels.index(l)] for l in other_leg_labels
            ]
            # put them back together
            leg_handles = other_leg_handles + stacked_leg_handles
            leg_labels = other_leg_labels + stacked_leg_labels
            axs[0].legend(leg_handles, leg_labels, fontsize="x-small", loc=(1.01, 0))

    return fig, axs


def plot_ratio(
    hlist,
    labels=None,
    systs=None,
    density=False,
    cmap=None,
    linewidth=None,
    linestyle=None,
    fmt=None,
    xlabel=None,
    xlim=None,
    log=True,
):
    """
    Plots ratio of a list of Hist histograms, the ratio is wrt to the first one in the list.
    The errors in the ratio are taken to be independent between histograms.
    """

    # pre-processing of histograms before plotting routine
    if density:
        for i, h in enumerate(hlist):
            hlist[i] = h.copy() / (np.sum(np.diff(h.axes[0].edges) * h.values()))

    # Set up variables for the stacked histogram
    fig = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    # Set up default values for the optional draw arguments
    if labels is None:
        labels = [None] * len(hlist)
    _default_cmap = plt.cm.brg(np.linspace(0, 1, len(hlist)))
    if cmap is None:
        cmap = _default_cmap
    for ic in range(len(cmap)):
        if cmap[ic] is None:
            cmap[ic] = _default_cmap[ic]
    if linewidth is None:
        linewidth = [1] * len(hlist)
    if linestyle is None:
        linestyle = ["-"] * len(hlist)
    if fmt is None:
        fmt = [""] * len(hlist)

    # plot the histograms and errorbars
    for ihist, hist in enumerate(hlist):
        y, x = hist.to_numpy()
        x_mid = hist.axes.centers[0]
        y_errs = np.sqrt(hist.variances())
        errorbar = ax1.errorbar(
            x_mid,
            y,
            yerr=y_errs,
            color=cmap[ihist],
            fmt=fmt[ihist],
            elinewidth=linewidth[ihist],
            drawstyle="default",
            linestyle="",
            label=None if linestyle[ihist] != "" else labels[ihist],
        )
        ax1.stairs(
            y,
            x,
            color=errorbar.lines[0].get_color(),
            label=None if linestyle[ihist] == "" else labels[ihist],
            linewidth=linewidth[ihist],
            linestyle=linestyle[ihist],
        )

    # set x and y limits, scales
    if log:
        ax1.set_yscale("log")
        y_max = np.max([max(h.values()) for h in hlist]) * 5
        y_min = (
            np.min([min(h.values()[h.values() > 0]) for h in hlist]) * 0.5
            if density
            else 5e-1
        )
        ax1.set_ylim(y_min, y_max)
    if xlim is not None:
        xmin = xlim[0]
        xmax = xlim[1]
        ax1.set_xlim([xmin, xmax])
    else:
        xmins, xmaxs = [], []
        for h in hlist:
            xvals = h.axes.centers[0]
            yvals = h.values()
            i_xmin = (
                np.min([i for i, x in enumerate(yvals > 0) if x])
                if len(xvals[yvals > 0])
                else 0
            )
            i_xmax = (
                np.max([i for i, x in enumerate(yvals > 0) if x])
                if len(xvals[yvals > 0])
                else -1
            )
            xmins.append(xvals[i_xmin] - h.axes.widths[0][0] / 2)
            xmaxs.append(xvals[i_xmax] + h.axes.widths[0][-1] / 2)
        xmin = min(xmins)
        xmax = max(xmaxs)
        xrange = xmax - xmin
        ax1.set_xlim([xmin - xrange * 0.1, xmax + xrange * 0.1])

    # define the ratio axis
    ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.axhline(1, ls="--", color="gray")

    # calculate the ratio, with error propagation, and plot them
    for ihist, hist in enumerate(hlist):
        if ihist == 0:
            continue
        ratio = np.divide(
            hist.values(),
            hlist[0].values(),
            out=np.ones_like(hist.values()),
            where=hlist[0].values() != 0,
        )
        ratio_err = np.where(
            hlist[0].values() > 0,
            np.sqrt(
                (hlist[0].values() ** -2) * (hist.variances())
                + (hist.values() ** 2 * hlist[0].values() ** -4)
                * (hlist[0].variances())
            ),
            0,
        )
        ax2.errorbar(
            hlist[0].axes.centers[0],
            ratio,
            yerr=ratio_err,
            color=cmap[ihist],
            fmt="o",
            linestyle="none",
            label=None,
        )

    # plot systematics as a gray band in the ratio
    if systs is not None:
        assert len(systs) == len(hlist[0].axes.centers[0])
        widths = hlist[0].axes.widths[0]
        up_height = np.where(systs > 0, systs, 0)
        down_height = np.where(systs > 0, 1 / (1 + systs) - 1, 0)
        ax2.bar(
            hlist[0].axes.centers[0],
            height=up_height,
            bottom=1,
            width=widths,
            alpha=0.3,
            color="gray",
        )
        ax2.bar(
            hlist[0].axes.centers[0],
            height=down_height,
            bottom=1,
            width=widths,
            alpha=0.3,
            color="gray",
        )
        # put legend in a2 in the top right
        from matplotlib import patches as mpatches

        ax2.legend(
            [mpatches.Patch(color="gray", alpha=0.3)],
            ["Syst."],
            loc="upper right",
            fontsize="x-small",
        )

    # set labels, legend
    if density:
        ax1.set_ylabel("Normalized Events", y=1, ha="right")
    else:
        ax1.set_ylabel("Events", y=1, ha="right")
    if labels != [None] * len(hlist):  # manually re-order legend to follow the labels
        leg_handles, leg_labels = ax1.get_legend_handles_labels()
        leg_handles = [leg_handles[leg_labels.index(l)] for l in labels]
        ax1.legend(leg_handles, labels, loc="best", fontsize="x-small")
    if xlabel is None:
        xlabel = hlist[0].axes[0].label
        if xlabel == "Axis 0":
            xlabel = None
    ax2.set_xlabel(xlabel, y=1)
    ax2.set_ylabel(
        f"Ratio to {labels[0]}" if labels is not None else "Ratio",
        y=1,
        ha="right",
        fontsize="small",
    )

    return fig, (ax1, ax2)


def plot_ratio_regions(plots, plot_label, sample1, sample2, regions, density=False):
    fig = plt.figure()
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    _ = plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    offset = 0
    mids = []
    for i, r in enumerate(regions):
        h1 = plots[sample1][plot_label.replace("A_", r + "_")].copy()
        h2 = plots[sample2][plot_label.replace("A_", r + "_")].copy()

        if density:
            h1 = h1 / h1.sum().value
            h2 = h2 / h2.sum().value

        y1, x1 = h1.to_numpy()
        x1 = x1[:-1]
        y2, x2 = h2.to_numpy()
        x2 = x2[:-1]

        xmin1 = np.argwhere(y1 > 0)[0] if any(y1 > 0) else [len(x1)]
        xmin2 = np.argwhere(y2 > 0)[0] if any(y2 > 0) else [len(x2)]
        xmax1 = np.argwhere(y1 > 0)[-1] if any(y1 > 0) else [0]
        xmax2 = np.argwhere(y2 > 0)[-1] if any(y2 > 0) else [0]
        xmin = min(np.concatenate((xmin1, xmin2)))
        xmax = max(np.concatenate((xmax1, xmax2)))
        x1 = x1[xmin : xmax + 1]
        x2 = x2[xmin : xmax + 1]
        y1 = y1[xmin : xmax + 1]
        y2 = y2[xmin : xmax + 1]

        x1 = x1 - x1[0]
        x2 = x2 - x2[0]

        this_offset = x1[-1] - x1[0]
        x1 = x1 + offset
        x2 = x2 + offset
        offset += this_offset

        mids.append((x1[-1] + x1[0]) / 2)

        y1_errs = np.sqrt(h1.variances())
        y1_errs = y1_errs[xmin : xmax + 1]
        if i == 0:
            print(sample1)
            ax1.step(x1, y1, color="midnightblue", label=sample1, where="mid")
        else:
            ax1.step(x1, y1, color="midnightblue", where="mid")
        ax1.errorbar(
            x1,
            y1,
            yerr=y1_errs,
            color="midnightblue".upper(),
            fmt="",
            drawstyle="steps-mid",
        )

        y2_errs = np.sqrt(h2.variances())
        y2_errs = y2_errs[xmin : xmax + 1]
        if i == 0:
            print(sample2)
            ax1.step(x2, y2, color="maroon", label=sample2, where="mid")
        else:
            ax1.step(x2, y2, color="maroon", where="mid")
        ax1.errorbar(
            x2, y2, yerr=y2_errs, color="maroon".upper(), fmt="", drawstyle="steps-mid"
        )

        ax1.axvline(x2[0], ls="--", color="black")
        ax2.axvline(x2[0], ls="--", color="black")

        # calculate the upper and lower errors
        # suppress errors where the denonminator is 0
        y1 = np.where(y1 > 0, y1, -1)
        yerrors_up = np.where(y1 > 0, y2 / y1 - (y2 - y2_errs) / (y1 + y1_errs), np.nan)
        yerrors_low = np.where(
            y1 > 0, (y2 + y2_errs) / (y1 - y1_errs) - y2 / y1, np.nan
        )
        ratio_errs = [yerrors_up, yerrors_low]
        ratios = np.where((y2 > 0) & (y1 > 0), y2 / y1, 1)
        ax2.errorbar(
            x1, ratios, yerr=ratio_errs, color="black", fmt="", drawstyle="steps-mid"
        )

    ax1.set_yscale("log")

    ax1.set_xticks(mids)
    ax1.set_xticklabels(list(regions))

    ax1.set_ylabel("Events", y=1, ha="right")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    ax2.axhline(1, ls="--", color="gray")
    ax2.set_ylim(0.4, 1.6)
    ax2.set_ylabel("Ratio", y=1, ha="right")
    ax2.set_xlabel(h1.axes[0].label, y=1)

    return fig, (ax1, ax2)


def plot_all_regions(
    plots,
    plot_label,
    samples,
    labels,
    regions="ABCDEFGH",
    density=False,
    xlim="default",
    log=True,
):
    fig = plt.figure(figsize=(20, 7))
    ax = fig.subplots()

    offset = 0
    mids = []
    for i, r in enumerate(regions):
        # get (x, y) for each sample in rhig region
        hists, ys, xs = [], [], []
        for sample in samples:
            h = plots[sample][plot_label.replace("A_", r + "_")]
            if density:
                h = h / h.sum().value
            y, x = h.to_numpy()
            x = x[:-1]
            hists.append(h)
            ys.append(y)
            xs.append(x)

        # get args for min and max
        xmins, xmaxs = [], []
        for _x, y in zip(xs, ys):
            xmin = np.argwhere(y > 0)[0] if any(y > 0) else [1e6]
            xmax = np.argwhere(y > 0)[-1] if any(y > 0) else [1e-6]
            xmins.append(xmin)
            xmaxs.append(xmax)
        xmin = min(xmins)[0]
        xmax = max(xmaxs)[0]

        # get only range that matters
        Xs, Ys = [], []
        for x, y in zip(xs, ys):
            x = x[xmin : xmax + 1]
            y = y[xmin : xmax + 1]
            x = x - x[0]
            this_offset = x[-1] - x[0]
            x = x + offset
            Xs.append(x)
            Ys.append(y)

        # total offset
        offset += this_offset

        mids.append((Xs[0][-1] + Xs[0][0]) / 2)

        for h, x, y, sample, label in zip(hists, Xs, Ys, samples, labels):
            y_errs = np.sqrt(h.variances())
            y_errs = y_errs[xmin : xmax + 1]
            if i == 0:
                if label == "I":
                    label == "SR"
                ax.step(x, y, color=getColor(sample), label=label, where="mid")
            else:
                ax.step(x, y, color=getColor(sample), where="mid")

        ax.axvline(Xs[0][0], ls="--", color="black")

    if log:
        ax.set_yscale("log")

    ax.set_xticks(mids)
    ax.set_xticklabels(list(regions))

    ax.set_ylabel("Events", y=1, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    return fig, ax


def plot_sys_variations(plots_sample, plot_label, sys, rebin=1j):
    """
    Plot variations for a systemtaic
    """
    h = plots_sample["_".join([plot_label])][::rebin]
    h_up = plots_sample["_".join([plot_label, sys, "up"])][::rebin]
    h_down = plots_sample["_".join([plot_label, sys, "down"])][::rebin]

    fig, axs = plot_ratio(
        [h, h_up, h_down], [sys + " nominal", sys + " up", sys + " down"]
    )
    axs[0].legend()
    axs[1].set_ylim(0.9, 1.1)
    return fig, axs


def plot_sliced_hist2d(
    hist,
    regions_list,
    stack=False,
    density=False,
    slice_var="y",
    labels=None,
    ratio=False,
):
    """
    Takes a 2d histogram, slices it in different regions, and plots the regions.
    Inputs:
        hist: 2d Hist histogram.
        regions_list: list of regions using Hist slicing. e.g. [[10j,20j],[20j,30j],...]
        bin_var: 'x' or 'y', which dimensions to slice in
        labels: list of strings to use as labels in plot.
    Returns:
        matplotlib fig and ax
    """
    if labels:
        assert len(labels) == len(regions_list)
    hist_list = hist_utils.slice_hist2d(hist, regions_list, slice_var)
    cmap = plt.cm.jet(np.linspace(0, 1, len(hist_list)))

    if stack:
        histtype = "fill"
    else:
        histtype = "step"

    if not ratio:
        fig = plt.figure()
        axs = fig.subplots()
        hep.histplot(
            hist_list,
            yerr=True,
            stack=stack,
            histtype=histtype,
            density=density,
            label=labels,
            color=cmap,
            ax=axs,
        )
        axs.legend(
            fontsize=14,
            framealpha=1,
            facecolor="white",
            shadow=True,
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
        )
        axs.set_yscale("log")

    else:
        if stack:
            print("Stacking not supported in ratio plot")
        fig, axs = plot_ratio(hist_list, density=density, labels=labels, cmap=cmap)

        axs[0].legend(
            fontsize=14,
            framealpha=1,
            facecolor="white",
            shadow=True,
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
        )
        axs[0].set_yscale("log")

    return fig, axs


def make_cutflow_table(
    cutflow_dict, samples, selections, efficiencies=False, relative_efficiencies=False
):
    """
    Create a table with the cutflow for each sample.
    :param cutflow_dict: dictionary of cutflows (dimension: sample x selection)
    :param samples: list of samples
    :param selections: list of selections
    :param efficiencies: if True, add efficiency columns
    :param relative_efficiencies: if True, add relative efficiency columns
    """
    table = []

    if efficiencies and relative_efficiencies:
        raise ValueError(
            "Cannot set both efficiencies and relative_efficiencies to True"
        )

    # add cutflow for each sample if needed
    for i in range(len(selections)):
        if not selections[i].startswith("cutflow_"):
            selections[i] = "cutflow_" + selections[i]

    for sample in samples:
        if efficiencies:  # calculate efficiency wrt total
            tot = cutflow_dict[sample]["cutflow_total"]
            this_sample_values = [
                cutflow_dict[sample][selection] / tot for selection in selections
            ]
        elif (
            relative_efficiencies
        ):  # calculate relative efficiency wrt previous selection
            this_sample_values = [
                (
                    cutflow_dict[sample][selection]
                    / cutflow_dict[sample][selections[i - 1]]
                    if i > 0
                    else 1
                )
                for i, selection in enumerate(selections)
            ]
        else:  # just the cutflow
            this_sample_values = [
                cutflow_dict[sample][selection] for selection in selections
            ]
        table.append(this_sample_values)

    return np.array(table)


def cutflow_table(
    cutflow_dict,
    samples,
    selections,
    selection_labels: str = [],
    sig_figs: int = 2,
    efficiencies: bool = False,
    relative_efficiencies: bool = False,
):
    """
    Create a table with the cutflow for each sample.
    :param cutflow_dict: dictionary of cutflows (dimension: sample x selection)
    :param samples: list of samples
    :param selections: list of selections
    :param selection_labels: labels for the selections
    :param sig_figs: number of significant figures to round to
    :param efficiencies: if True, add efficiency columns
    :param relative_efficiencies: if True, add relative efficiency columns
    """
    from prettytable import PrettyTable

    prettytable = PrettyTable()

    if len(selection_labels) == 0:
        selection_labels = [s.replace("cutflow_", "") for s in selections]
    prettytable.add_column("Selection", selection_labels)

    table = make_cutflow_table(
        cutflow_dict, samples, selections, efficiencies, relative_efficiencies
    )

    # add cutflow for each sample if needed
    for sample, sample_values in zip(samples, table):
        # round if needed
        values = [
            "{:g}".format(float("{:.{p}g}".format(v, p=sig_figs)))
            for v in sample_values
        ]
        prettytable.add_column(sample, values)

    return prettytable


def cutflow_plot(cutflow_dict, samples, selections, selection_labels: str = []):
    """
    Create a plot with the cutflow for each sample.
    :param cutflow_dict: dictionary of cutflows (dimension: sample x selection)
    :param samples: list of samples
    :param selections: selections to plot
    :param selection_labels: labels for the selections
    """
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylabel("Events")

    table = make_cutflow_table(cutflow_dict, samples, selections)

    for sample, cutflow_this_sample in zip(samples, table):
        ax.stairs(cutflow_this_sample, label=sample)

    ax.legend(loc=(1.02, 0.0), fontsize="xx-small")
    hep.cms.label(ax=ax)
    if len(selection_labels) == 0:  # in the case these are not defined
        selection_labels = [s.replace("cutflow_", "") for s in selections]
    ax.set_xticks(
        np.arange(len(selection_labels)) + 0.5,
        selection_labels,
        rotation=90,
        fontsize=10,
    )

    return fig, ax


def make_n1_plots(
    plots: dict,
    tag: str,
    density: bool = False,
    samples: list = [],
    stackedSamples: list = [],
):
    """
    Make n-1 plots (produced by make_hists.py as "<histogram-name>_N-1").
    :param plots: dictionary of histograms (dimension: sample x plot)
    :param cutflows: dictionary of cutflows (dimension: sample x selection)
    :param tag: tag to use for the n-1 plots
    :param density: if True, plot densities
    :param samples: list of samples to plot separately
    :param stackedSamples: list of samples to stack
    :param tag: tag to use for the n-1 plots
    :return: list of figures
    """

    figs = []
    allSamples = samples + stackedSamples
    if len(allSamples) == 0:
        raise ValueError(
            "No samples provided. Provide at least one samples or one stackedSamples."
        )

    n1_plots = [k for k in plots[allSamples[0]].keys() if "_noCut_" in k and tag in k]

    samples_color = plt.cm.rainbow(np.linspace(0, 1, len(samples)))
    for p in n1_plots:

        p_notag = p.replace("_" + tag, "")

        var = p_notag.split("_noCut_")[0]
        cut_bits = p_notag.split("_noCut_")[1].split("_")
        cut_val = float(cut_bits[-1])
        cut_operator = cut_bits[-2]

        fig = plt.figure()
        ax = fig.subplots()
        if len(stackedSamples) > 0:
            hep.histplot(
                [plots[s][p] for s in stackedSamples],
                label=stackedSamples,
                density=density,
                stack=True,
                histtype="fill",
                ax=ax,
            )
        if len(samples) > 0:
            hep.histplot(
                [plots[s][p] for s in samples],
                label=samples,
                density=density,
                stack=False,
                histtype="step",
                linestyle="dashed",
                linewidth=2,
                color=samples_color,
                ax=ax,
            )
        if cut_val:
            ax.vlines(
                cut_val,
                0,
                ax.get_ylim()[1],
                color="black",
                linestyle="--",
                linewidth=4,
                label=f"Cut value: {cut_val}",
            )
        ax.set_yscale("log")
        pretty_sel = p_notag.split("_noCut_")[1].replace("_", " ")
        title = f"N-1 plot for region: {tag}\nSelection omitted: {pretty_sel}"
        ax.legend(
            fontsize="xx-small", loc=(1.05, 0), title=title, title_fontsize="xx-small"
        )
        figs.append(fig)

    return figs
