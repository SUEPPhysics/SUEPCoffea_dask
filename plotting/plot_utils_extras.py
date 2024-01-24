from typing import List, Dict, Optional, Union

import hist
import hist.intervals
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import mplhep as hep
import numpy as np


def bin_midpoints(bins):
    midpoints = []
    for i in range(len(bins) - 1):
        midpoints.append((bins[i] + bins[i + 1]) / 2)
    return np.array(midpoints)


def convert_permuon_to_perevent(h):
    h_new = hist.Hist.new.Reg(
        10, 0, 10, name=h.axes[0].name, label=h.axes[0].label
    ).Weight()
    h_new[0] = h[0]
    for i in range(1, 10):
        h_new[i] = hist.accumulators.WeightedSum(
            h[i].value / i, h[i].variance / (i**2)
        )
    return h_new


def apply_cuts(plots, cuts):
    """
    Applies a list cuts to a list of plots.
    Returns a dictionary of applied cuts to dictionaries of input histograms.

    Should do the following:
        1. loop over the cuts (name and slice)
        2. loop over processes (that's the higher structure in plots: plots[process][plot_name])
        3. loop over the plots in the process
        4. check for each plot if they have an axis matching the cut and which axis it is
        5. apply the cut to the axis
        6. store using the following structure: hists[cut][process][plot_name]


    Parameters
    ----------
    plots : dict
        Dictionary of input histograms
    cuts : list
        List of cuts to apply
    """
    hists = {cut["name"]: {} for cut in cuts}
    for cut in cuts:
        cut_name = cut["name"]
        cut_slice = cut["slice"]
        for process in plots.keys():
            hists[cut_name][process] = {}
            for plot_name in plots[process].keys():
                if cut_name in plots[process][plot_name].axes.name:
                    axis_i = plots[process][plot_name].axes.name.index(cut_name)
                    hist_slice = [slice(None)] * len(
                        plots[process][plot_name].axes.name
                    )
                    hist_slice[axis_i] = cut_slice
                    hists[cut_name][process][plot_name] = plots[process][plot_name][
                        tuple(hist_slice)
                    ].copy()
    return hists


def split_bkgs_per_process(
    plots: Dict[str, Dict[str, hist.Hist]], bkg_list: List[float]
) -> Dict[str, Dict[str, Dict[str, hist.Hist]]]:
    plots_out = {}
    for bkg in bkg_list:
        plots_for_process = {
            "c/b": {},
            "light/other": {},
            "prompt": {},
            "tau": {},
        }
        for plot in plots[bkg].keys():
            if "genPartFlav" not in plot:
                continue
            htemp = plots[bkg][plot]
            slc_cb = [slice(None)] * len(htemp.axes.name)
            slc_pion1 = [slice(None)] * len(htemp.axes.name)
            slc_pion2 = [slice(None)] * len(htemp.axes.name)
            slc_prompt = [slice(None)] * len(htemp.axes.name)
            slc_tau = [slice(None)] * len(htemp.axes.name)
            for i_n, name in enumerate(htemp.axes.name):
                if name == "Muon_genPartFlav":
                    slc_cb[i_n] = slice(4j, 6j, sum)
                    slc_pion1[i_n] = slice(0, 1j, sum)
                    slc_pion2[i_n] = slice(3j, 4j, sum)
                    slc_prompt[i_n] = slice(1j, 3j, sum)
                    slc_tau[i_n] = slice(15j, 16j, sum)
            plots_for_process["c/b"][plot] = htemp[tuple(slc_cb)].copy()
            plots_for_process["light/other"][plot] = (
                htemp[tuple(slc_pion1)].copy() + htemp[tuple(slc_pion2)].copy()
            )
            plots_for_process["prompt"][plot] = htemp[tuple(slc_prompt)].copy()
            plots_for_process["tau"][plot] = htemp[tuple(slc_tau)].copy()
        plots_out[bkg] = plots_for_process
    return plots_out


def split_bkgs(
    plots: Dict[str, Dict[str, hist.Hist]], bkg_list: List[float]
) -> Dict[str, Dict[str, hist.Hist]]:
    plots_out = {
        "c/b": {},
        "light/other": {},
        "prompt": {},
        "tau": {},
    }
    for bkg in bkg_list:
        for plot in plots[bkg].keys():
            if "genPartFlav" not in plot:
                continue
            htemp = plots[bkg][plot]
            slc_cb = [slice(None)] * len(htemp.axes.name)
            slc_pion1 = [slice(None)] * len(htemp.axes.name)
            slc_pion2 = [slice(None)] * len(htemp.axes.name)
            slc_prompt = [slice(None)] * len(htemp.axes.name)
            slc_tau = [slice(None)] * len(htemp.axes.name)
            for i_n, name in enumerate(htemp.axes.name):
                if name == "Muon_genPartFlav":
                    slc_cb[i_n] = slice(4j, 6j, sum)
                    slc_pion1[i_n] = slice(0, 1j, sum)
                    slc_pion2[i_n] = slice(3j, 4j, sum)
                    slc_prompt[i_n] = slice(1j, 3j, sum)
                    slc_tau[i_n] = slice(15j, 16j, sum)
            if plot not in plots_out["c/b"].keys():
                plots_out["c/b"][plot] = htemp[tuple(slc_cb)].copy()
                plots_out["light/other"][plot] = (
                    htemp[tuple(slc_pion1)].copy() + htemp[tuple(slc_pion2)].copy()
                )
                plots_out["prompt"][plot] = htemp[tuple(slc_prompt)].copy()
                plots_out["tau"][plot] = htemp[tuple(slc_tau)].copy()
            else:
                plots_out["c/b"][plot] += htemp[tuple(slc_cb)].copy()
                plots_out["light/other"][plot] += (
                    htemp[tuple(slc_pion1)].copy() + htemp[tuple(slc_pion2)].copy()
                )
                plots_out["prompt"][plot] += htemp[tuple(slc_prompt)].copy()
                plots_out["tau"][plot] += htemp[tuple(slc_tau)].copy()
    return plots_out


def invert_dict(plots, processes, sources):
    """
    Gets dictionary of sources for each process.
    Returns dictionary of processes for each source.
    """
    inverted = {}
    for source in sources:
        inverted[source] = {}
    for process in processes:
        for source in sources:
            if source in plots[process].keys():
                inverted[source][process] = plots[process][source]
    return inverted


def cut_to_sum_slice(
    plot: hist.Hist, cut: Union[List[Dict], Dict]
) -> tuple[slice, ...]:
    """
    Converts a cut to a slice tuple that sums all axes.
    Can be used to project a histogram when one of the elements is modified.
    """
    hist_slice = [slice(None, None, sum)] * len(plot.axes.name)
    if not isinstance(cut, list):
        cut = [cut]
    for c in cut:
        if c["name"] in plot.axes.name:
            axis_i = plot.axes.name.index(c["name"])
            new_slice = slice(c["slice"].start, c["slice"].stop, sum)
            hist_slice[axis_i] = new_slice
    return tuple(hist_slice)


def project(h: hist.Hist, axis: str, flow: Optional[bool] = True) -> hist.Hist:
    """
    Projects a histogram onto a given axis.
    Extends the hist.Hist.project method by allowing to specify if flow is included.

    Parameters
    ----------
    h : hist.Hist
        Histogram to be projected
    axis : str
        Axis to project onto
    flow : bool
        Whether to include overflow and underflow bins
    """
    if flow:
        return h.project(axis)
    axis_i = h.axes.name.index(axis)
    hist_slice = [slice(None, None, sum)] * len(h.axes.name)
    hist_slice[axis_i] = slice(None)
    return h[tuple(hist_slice)].copy()


def plot_overlay(
    plots: Dict[str, Dict[str, hist.Hist]],
    bkg_list: List[str],
    label: tuple[str, str],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cut: Optional[tuple[slice, ...]] = None,
    override_slice: Optional[bool] = False,
    slc: Optional[slice] = None,
    ylim: Optional[tuple[float, float]] = None,
    xlog: Optional[bool] = False,
    ylog: Optional[bool] = False,
    density: Optional[bool] = False,
    per_muon: Optional[bool] = False,
    text_to_show: Optional[str] = None,
    int_lumi: Optional[float] = 1,
) -> None:
    """
    Plots the overlay of a list of Hist histograms

    Parameters
    ----------
    plots : dict
        Dictionary of Hist histograms
    bkg_list : list
        List of background samples to be stacked
    label : tuple of str
        Tuple of two strings: [0]: Label of the plot, [0]: Label of the axis
    slc : tuple
        Tuple of slices to apply to the histograms
    """

    # Set up figure and axes
    no_input_fig = False
    if fig is None or ax is None:
        no_input_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Get the histograms
    lbl_plot, lbl_axis = label
    if slc is None:
        slc = slice(None)
    hists = []
    hist_max = []
    for bkg in bkg_list:
        h_temp = None
        if cut is None:
            h_temp = plots[bkg][lbl_plot].project(lbl_axis)[slc]
        else:
            axis_i = plots[bkg][lbl_plot].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[bkg][lbl_plot][tuple(cut)]
        if density:
            h_temp /= h_temp.sum(flow=False).value
        hist_max.append(h_temp.values().max())
        hists.append(h_temp.copy())

    # calculate desired y-axis range
    if ylim is None:
        ylim = (0, max(hist_max) * 1.5)
        if density:
            ylim = (0, 1)
            if ylog:
                ylim = (0, 2)

    # Plot the stacked histogram
    hep.histplot(
        hists,
        yerr=[np.sqrt(h.variances()) for h in hists],
        label=[b.replace("_2018", "") for b in bkg_list],
        lw=2,
        ax=ax,
    )

    ax.legend(ncol=2, loc="best")
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(hists[0].axes.name[0])
    ax.set_ylabel("Events")
    if per_muon:
        ax.set_ylabel("Muons")
    if density:
        ax.set_ylabel("a.u.")
    hep.cms.label(llabel="Preliminary", data=False, lumi=int_lumi, ax=ax)

    if xlog:
        ax.xaxis.set_minor_locator(LogLocator(numticks=999, subs="all"))

    if text_to_show is not None:
        midpoint = int(len(hists[0].axes[0].edges) / 2)
        x_txt = hists[0].axes[0].edges[midpoint]
        y_txt = max(hist_max) * 1.1
        if density:
            y_txt = 0.4
        ax.text(x_txt, y_txt, text_to_show, horizontalalignment="center")

    if no_input_fig:
        plt.tight_layout()
        plt.plot()
    return


def plot_stack(
    plots: Dict[str, Dict[str, hist.Hist]],
    bkg_list: List[str],
    label: tuple[str, str],
    sig_list: Optional[List[str]] = [],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cut: Optional[tuple[slice, ...]] = None,
    override_slice: Optional[bool] = False,
    slc: Optional[slice] = None,
    ylim: Optional[tuple[float, float]] = None,
    xlog: Optional[bool] = False,
    ylog: Optional[bool] = False,
    per_muon: Optional[bool] = False,
    text_to_show: Optional[str] = None,
    int_lumi: Optional[float] = 1,
) -> None:
    """
    Plots the stack of a list of bkg Hist histograms

    Parameters
    ----------
    plots : dict
        Dictionary of Hist histograms
    bkg_list : list
        List of background samples to be stacked
    label : tuple of str
        Tuple of two strings: [0]: Label of the plot, [0]: Label of the axis
    slc : tuple
        Tuple of slices to apply to the histograms
    """

    # Set up figure and axes
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Get the histograms
    lbl_plot, lbl_axis = label
    if slc is None:
        slc = slice(None)
    hists = []
    hist_bkg_total = None
    for bkg in bkg_list:
        h_temp = plots[bkg][lbl_plot].project(lbl_axis)[slc]
        hists.append(h_temp.copy())
        if hist_bkg_total is None:
            hist_bkg_total = h_temp.copy()
        else:
            hist_bkg_total += h_temp.copy()

    hists_sig = []
    for sig in sig_list:
        h_temp = None
        if cut is None:
            h_temp = plots[sig][lbl_plot].project(lbl_axis)[slc]
        else:
            axis_i = plots[sig][lbl_plot].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[sig][lbl_plot][tuple(cut)]
        hists_sig.append(h_temp.copy())

    # calculate desired y-axis range
    if ylim is None:
        ylim = (0, hist_bkg_total.values().max() * 1.5)

    # Plot the stacked histogram
    hep.histplot(
        hists,
        label=[b.replace("_2018", "") for b in bkg_list],
        stack=True,
        histtype="fill",
        ec="black",
        lw=2,
        ax=ax,
        zorder=1,
    )

    # Overlay an uncertainty hatch
    # NOTE: the fill_between function requires the bin edges and the y, y_err values
    #       to be duplicated in order for the hatch to be drawn correctly.
    x_hatch = np.vstack(
        (hist_bkg_total.axes[0].edges[:-1], hist_bkg_total.axes[0].edges[1:])
    ).reshape((-1,), order="F")
    y_hatch1 = np.vstack((hist_bkg_total.values(), hist_bkg_total.values())).reshape(
        (-1,), order="F"
    )
    y_hatch1_unc = np.vstack(
        (np.sqrt(hist_bkg_total.variances()), np.sqrt(hist_bkg_total.variances()))
    ).reshape((-1,), order="F")
    ax.fill_between(
        x=x_hatch,
        y1=y_hatch1 - y_hatch1_unc,
        y2=y_hatch1 + y_hatch1_unc,
        label="Stat. Unc.",
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
        zorder=2,
    )

    # Plot the signal histograms if they exist
    if len(sig_list) > 0:
        hep.histplot(
            hists_sig,
            label=[sig.replace("_2018", "").replace("SUEP-", "") for sig in sig_list],
            lw=3,
            ax=ax,
        )

    ax.legend(ncol=2, loc="best")
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(hist_bkg_total.axes.name[0])
    ax.set_ylabel("Events")
    if per_muon:
        ax.set_ylabel("Muons")
    hep.cms.label(llabel="Preliminary", data=False, lumi=int_lumi, ax=ax)

    if xlog:
        ax.xaxis.set_minor_locator(LogLocator(numticks=999, subs="all"))

    if text_to_show is not None:
        midpoint = int(len(hist_bkg_total.axes[0].edges) / 2)
        x_txt = hist_bkg_total.axes[0].edges[midpoint]
        y_txt = hist_bkg_total.values().max() * 1.1
        ax.text(x_txt, y_txt, text_to_show, horizontalalignment="center")

    plt.tight_layout()
    plt.plot()
    return


def plot_ratio_overlay(
    plots: Dict[str, Dict[str, hist.Hist]],
    hist_list: List[str],
    label: tuple[str, str],
    cut: Optional[tuple[slice, ...]] = None,
    override_slice: Optional[bool] = False,
    slc: Optional[slice] = None,
    ylim: Optional[tuple[float, float]] = None,
    xlog: Optional[bool] = False,
    ylog: Optional[bool] = False,
    ratio_ylog: Optional[bool] = False,
    fig: Optional[plt.Figure] = None,
    gs: Optional[gridspec.GridSpec] = None,
    density: Optional[bool] = False,
    per_muon: Optional[bool] = False,
    int_lumi: Optional[float] = 1,
) -> None:
    """
    Plots ratio of a list of two Hist histograms.
    The errors in the ratio are taken to be independent between histograms.

    Parameters
    ----------
    plots : dict
        Dictionary of Hist histograms
    bkg_list : list
        List of background samples to be stacked
    label : tuple of str
        Tuple of two strings: [0]: Label of the plot, [0]: Label of the axis
    sig_list : list
        List of signal sample names to be plotted
    cut : tuple
        Tuple of slices to apply to the histograms
    slc : slice
        Slice to apply to the histograms.
    ylim : tuple
        Tuple of y-axis limits
    xlog : bool
        Whether to plot the x-axis in log scale
    ylog : bool
        Whether to plot the y-axis in log scale
    fig : plt.Figure
        Figure to plot on
    gs : gridspec.GridSpec
        Gridspec to plot on
    per_muon : bool
        Whether to plot the y-axis in per muon
    """

    # Set up figure and axes
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    if gs is None:
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
    else:
        ax1 = plt.subplot(gs[0:3, :])
        ax2 = plt.subplot(gs[3, :], sharex=ax1)

    # Get the histograms
    lbl_plot, lbl_axis = label
    if slc is None:
        slc = slice(None)
    hists = []
    for h_i in hist_list:
        h_temp = None
        if cut is None:
            h_temp = plots[h_i][lbl_plot].project(lbl_axis)[slc]
        else:
            axis_i = plots[h_i][lbl_plot].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[h_i][lbl_plot][tuple(cut)]
        if density:
            h_temp /= h_temp.sum(flow=False).value
        hists.append(h_temp.copy())

    # calculate desired y-axis range
    if ylim is None:
        maximum = max([h.values().max() for h in hists])
        ylim = (0, maximum * 1.8)
        if ylog:
            ylim = (1e-1, 1e12)
        if density:
            ylim = (0, 1)

    # Plot the stacked histogram
    hep.histplot(
        hists,
        yerr=[np.sqrt(h.variances()) for h in hists],
        label=[h_i.replace("_2018", "") for h_i in hist_list],
        lw=2,
        ax=ax1,
        zorder=1,
    )

    ax1.legend(ncol=2, loc="best")
    if xlog:
        ax1.set_xscale("log")
    if ylog:
        ax1.set_yscale("log")
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("")
    ax1.set_ylabel("Events")
    if density:
        ax1.set_ylabel("a.u.")
    if per_muon:
        ax1.set_ylabel("Muons")
    hep.cms.label(llabel="Preliminary", data=False, lumi=int_lumi, ax=ax1)

    # Calculate the ratio, with error propagation, and plot it
    plt.setp(ax1.get_xticklabels(), visible=False)
    ratio = np.divide(
        hists[0].values(),
        hists[1].values(),
        out=np.ones_like(hists[0].values()),
        where=hists[1].values() != 0,
    )
    ratio_err = np.where(
        hists[1].values() > 0,
        np.sqrt(
            (hists[1].values() ** -2) * (hists[0].variances())
            + (hists[0].values() ** 2 * hists[1].values() ** -4)
            * (hists[1].variances())
        ),
        0,
    )
    ax2.errorbar(
        hists[0].axes.centers[0],
        ratio,
        yerr=ratio_err,
        color="black",
        fmt="o",
        linestyle="none",
    )

    # Draw a filled hatch area with the relative uncertainty of the MC in the ratio plot.
    mc_rel_unc = np.divide(
        np.sqrt(hists[1].variances()),
        hists[1].values(),
        out=np.zeros_like(hists[1].values()),
        where=hists[1].values() != 0,
    )
    x_hatch = np.vstack(
        (hists[1].axes[0].edges[:-1], hists[1].axes[0].edges[1:])
    ).reshape((-1,), order="F")
    y_hatch2 = np.vstack(
        (np.ones_like(hists[1].values()), np.ones_like(hists[1].values()))
    ).reshape((-1,), order="F")
    y_hatch2_unc = np.vstack((mc_rel_unc, mc_rel_unc)).reshape((-1,), order="F")
    ax2.fill_between(
        x=x_hatch,
        y1=y_hatch2 - y_hatch2_unc,
        y2=y_hatch2 + y_hatch2_unc,
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
    )

    ax2.axhline(1, ls="--", color="gray")
    ax2.set_xlabel(hists[0].axes.name[0])
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0, 2)
    if xlog:
        ax1.xaxis.set_minor_locator(LogLocator(numticks=999, subs="all"))
    if fig is None:
        plt.plot()
    return


def plot_ratio_stack(
    plots: Dict[str, Dict[str, hist.Hist]],
    bkg_list: List[str],
    label: tuple[str, str],
    sig_list: Optional[List[str]] = [],
    cut: Optional[tuple[slice, ...]] = None,
    override_slice: Optional[bool] = False,
    slc: Optional[slice] = None,
    ylim: Optional[tuple[float, float]] = None,
    xlog: Optional[bool] = False,
    ylog: Optional[bool] = False,
    fig: Optional[plt.Figure] = None,
    gs: Optional[gridspec.GridSpec] = None,
    per_muon: Optional[bool] = False,
    int_lumi: Optional[float] = 1,
) -> None:
    """
    Plots ratio of a list of bkg Hist histograms over a data Hist histogram.
    The errors in the ratio are taken to be independent between histograms.

    Parameters
    ----------
    plots : dict
        Dictionary of Hist histograms
    bkg_list : list
        List of background samples to be stacked
    label : tuple of str
        Tuple of two strings: [0]: Label of the plot, [0]: Label of the axis
    sig_list : list
        List of signal sample names to be plotted
    cut : tuple
        Tuple of slices to apply to the histograms
    slc : slice
        Slice to apply to the histograms.
    ylim : tuple
        Tuple of y-axis limits
    xlog : bool
        Whether to plot the x-axis in log scale
    ylog : bool
        Whether to plot the y-axis in log scale
    fig : plt.Figure
        Figure to plot on
    gs : gridspec.GridSpec
        Gridspec to plot on
    per_muon : bool
        Whether to plot the y-axis in per muon
    int_lumi : float
        Integrated luminosity in fb-1
    """

    # Set up figure and axes
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    if gs is None:
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
    else:
        ax1 = plt.subplot(gs[0:3, :])
        ax2 = plt.subplot(gs[3, :], sharex=ax1)

    # Get the histograms
    lbl_plot, lbl_axis = label
    if slc is None:
        slc = slice(None)
    hists = []
    hist_bkg_total = None
    for bkg in bkg_list:
        h_temp = None
        if cut is None:
            h_temp = plots[bkg][lbl_plot].project(lbl_axis)[slc]
        else:
            axis_i = plots[bkg][lbl_plot].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[bkg][lbl_plot][tuple(cut)]
        hists.append(h_temp.copy())
        if hist_bkg_total is None:
            hist_bkg_total = h_temp.copy()
        else:
            hist_bkg_total += h_temp.copy()

    data_name = "DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD_histograms_2018"
    hist_data = None
    if cut is None:
        hist_data = plots[data_name][lbl_plot].project(lbl_axis)[slc]
    else:
        axis_i = plots[data_name][lbl_plot].axes.name.index(lbl_axis)
        cut = list(cut)
        cut[axis_i] = (
            slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
        )
        if len(cut) != len(plots[data_name][lbl_plot].axes.name):
            cut.append(slice(None, None, sum))
        hist_data = plots[data_name][lbl_plot][tuple(cut)]

    hists_sig = []
    for sig in sig_list:
        h_temp = None
        if cut is None:
            h_temp = plots[sig][lbl_plot].project(lbl_axis)[slc]
        else:
            axis_i = plots[sig][lbl_plot].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[sig][lbl_plot][tuple(cut)]
        hists_sig.append(h_temp.copy())

    # calculate desired y-axis range
    if ylim is None:
        ylim = (0, hist_bkg_total.values().max() * 1.8)
        if ylog:
            ylim = (1e-1, 1e10)
        if len(sig_list) > 0:
            ylim = (0, hist_bkg_total.values().max() * 2.2)
            if ylog:
                ylim = (1e-1, 1e9)

    # Plot the stacked histogram
    hep.histplot(
        hists,
        label=[b.replace("_2018", "") for b in bkg_list],
        stack=True,
        histtype="fill",
        ec="black",
        lw=2,
        ax=ax1,
        zorder=1,
    )

    # Overlay an uncertainty hatch
    # NOTE: the fill_between function requires the bin edges and the y, y_err values
    #       to be duplicated in order for the hatch to be drawn correctly.
    x_hatch = np.vstack(
        (hist_bkg_total.axes[0].edges[:-1], hist_bkg_total.axes[0].edges[1:])
    ).reshape((-1,), order="F")
    y_hatch1 = np.vstack((hist_bkg_total.values(), hist_bkg_total.values())).reshape(
        (-1,), order="F"
    )
    y_hatch1_unc = np.vstack(
        (np.sqrt(hist_bkg_total.variances()), np.sqrt(hist_bkg_total.variances()))
    ).reshape((-1,), order="F")
    ax1.fill_between(
        x=x_hatch,
        y1=y_hatch1 - y_hatch1_unc,
        y2=y_hatch1 + y_hatch1_unc,
        label="Stat. Unc.",
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
        zorder=2,
    )

    # Plot the data points
    hep.histplot(
        hist_data,
        label=["Data"],
        histtype="errorbar",
        mec="black",
        mfc="black",
        ecolor="black",
        ax=ax1,
    )

    # Plot the signal histograms if they exist
    if len(sig_list) > 0:
        hep.histplot(
            hists_sig,
            label=[sig.replace("_2018", "").replace("SUEP-", "") for sig in sig_list],
            lw=3,
            ax=ax1,
        )

    ax1.legend(ncol=2, loc="best")
    if xlog:
        ax1.set_xscale("log")
    if ylog:
        ax1.set_yscale("log")
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("")
    ax1.set_ylabel("Events")
    if per_muon:
        ax1.set_ylabel("Muons")
    hep.cms.label(llabel="Preliminary", data=False, lumi=int_lumi, ax=ax1)

    # Calculate the ratio, with error propagation, and plot it
    plt.setp(ax1.get_xticklabels(), visible=False)
    ratio = np.divide(
        hist_data.values(),
        hist_bkg_total.values(),
        out=np.ones_like(hist_data.values()),
        where=hist_bkg_total.values() != 0,
    )
    ratio_err = np.where(
        hist_bkg_total.values() > 0,
        np.sqrt(
            (hist_bkg_total.values() ** -2) * (hist_data.variances())
            + (hist_data.values() ** 2 * hist_bkg_total.values() ** -4)
            * (hist_bkg_total.variances())
        ),
        0,
    )
    ax2.errorbar(
        hist_data.axes.centers[0],
        ratio,
        yerr=ratio_err,
        color="black",
        fmt="o",
        linestyle="none",
    )

    # Draw a filled hatch area with the relative uncertainty of the MC in the ratio plot.
    mc_rel_unc = np.divide(
        np.sqrt(hist_bkg_total.variances()),
        hist_bkg_total.values(),
        out=np.zeros_like(hist_bkg_total.values()),
        where=hist_bkg_total.values() != 0,
    )
    y_hatch2 = np.vstack(
        (np.ones_like(hist_bkg_total.values()), np.ones_like(hist_bkg_total.values()))
    ).reshape((-1,), order="F")
    y_hatch2_unc = np.vstack((mc_rel_unc, mc_rel_unc)).reshape((-1,), order="F")
    ax2.fill_between(
        x=x_hatch,
        y1=y_hatch2 - y_hatch2_unc,
        y2=y_hatch2 + y_hatch2_unc,
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
    )

    ax2.axhline(1, ls="--", color="gray")
    ax2.set_xlabel(hist_data.axes.name[0])
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0, 2)
    if xlog:
        ax1.xaxis.set_minor_locator(LogLocator(numticks=999, subs="all"))
    if fig is None:
        plt.plot()
    return


def plot_ratio_stack_combine(
    plots: Dict[str, Dict[str, hist.Hist]],
    bkg_list: List[str],
    label: tuple[str, str],
    sig_list: Optional[List[str]] = [],
    cut: Optional[tuple[slice, ...]] = None,
    override_slice: Optional[bool] = False,
    slc: Optional[slice] = None,
    ylim: Optional[tuple[float, float]] = None,
    xlog: Optional[bool] = False,
    ylog: Optional[bool] = False,
    fig: Optional[plt.Figure] = None,
    gs: Optional[gridspec.GridSpec] = None,
    text_to_show: Optional[str] = None,
    per_muon: Optional[bool] = False,
    int_lumi: Optional[float] = 1,
) -> None:
    """
    Plots ratio of a list of bkg Hist histograms over a data Hist histogram.
    The errors in the ratio are taken to be independent between histograms.

    Parameters
    ----------
    plots : dict
        Dictionary of Hist histograms
    bkg_list : list
        List of background samples to be stacked
    label : tuple of str
        Tuple of two strings: [0]: Label of the plot, [0]: Label of the axis
    sig_list : list
        List of signal sample names to be plotted
    cut : tuple
        Tuple of slices to apply to the histograms
    slc : slice
        Slice to apply to the histograms.
    ylim : tuple
        Tuple of y-axis limits
    xlog : bool
        Whether to plot the x-axis in log scale
    ylog : bool
        Whether to plot the y-axis in log scale
    fig : plt.Figure
        Figure to plot on
    gs : gridspec.GridSpec
        Gridspec to plot on
    per_muon : bool
        Whether to plot the y-axis in per muon
    int_lumi : float
        Integrated luminosity in fb-1
    """

    # Set up figure and axes
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    if gs is None:
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
    else:
        ax1 = plt.subplot(gs[0:3, :])
        ax2 = plt.subplot(gs[3, :], sharex=ax1)

    # Get the histograms
    lbl_plot, lbl_axis = label
    if slc is None:
        slc = slice(None)
    hists = []
    hist_bkg_total = None
    for bkg in bkg_list:
        h_temp = None
        if cut is None:
            h_temp = plots[bkg][slc]
        else:
            axis_i = plots[bkg].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[bkg][tuple(cut)]
        hists.append(h_temp.copy())
        if hist_bkg_total is None:
            hist_bkg_total = h_temp.copy()
        else:
            hist_bkg_total += h_temp.copy()

    data_name = "data_obs"
    hist_data = None
    if cut is None:
        hist_data = plots[data_name][slc]
    else:
        axis_i = plots[data_name].axes.name.index(lbl_axis)
        cut = list(cut)
        cut[axis_i] = (
            slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
        )
        if len(cut) != len(plots[data_name].axes.name):
            cut.append(slice(None, None, sum))
        hist_data = plots[data_name][tuple(cut)]

    hists_sig = []
    for sig in sig_list:
        h_temp = None
        if cut is None:
            h_temp = plots[sig][slc]
        else:
            axis_i = plots[sig].axes.name.index(lbl_axis)
            cut = list(cut)
            cut[axis_i] = (
                slice(cut[axis_i].start, cut[axis_i].stop) if override_slice else slc
            )
            h_temp = plots[sig][tuple(cut)]
        hists_sig.append(h_temp.copy())

    # calculate desired y-axis range
    if ylim is None:
        ylim = (0, hist_bkg_total.values().max() * 1.8)
        if ylog:
            ylim = (1e-1, 1e10)
        if len(sig_list) > 0:
            ylim = (0, hist_bkg_total.values().max() * 2.2)
            if ylog:
                ylim = (1e-1, 1e9)

    # Plot the stacked histogram
    hep.histplot(
        hists,
        label=[b.replace("_2018", "") for b in bkg_list],
        stack=True,
        histtype="fill",
        ec="black",
        lw=2,
        ax=ax1,
        zorder=1,
    )

    # Overlay an uncertainty hatch
    # NOTE: the fill_between function requires the bin edges and the y, y_err values
    #       to be duplicated in order for the hatch to be drawn correctly.
    x_hatch = np.vstack(
        (hist_bkg_total.axes[0].edges[:-1], hist_bkg_total.axes[0].edges[1:])
    ).reshape((-1,), order="F")
    y_hatch1 = np.vstack((hist_bkg_total.values(), hist_bkg_total.values())).reshape(
        (-1,), order="F"
    )
    y_hatch1_unc = np.vstack(
        (np.sqrt(hist_bkg_total.variances()), np.sqrt(hist_bkg_total.variances()))
    ).reshape((-1,), order="F")
    ax1.fill_between(
        x=x_hatch,
        y1=y_hatch1 - y_hatch1_unc,
        y2=y_hatch1 + y_hatch1_unc,
        label="Stat. Unc.",
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
        zorder=2,
    )

    # Plot the data points
    hep.histplot(
        hist_data,
        label=["Data"],
        histtype="errorbar",
        mec="black",
        mfc="black",
        ecolor="black",
        ax=ax1,
    )

    # Plot the signal histograms if they exist
    if len(sig_list) > 0:
        hep.histplot(
            hists_sig,
            label=[sig.replace("_2018", "").replace("SUEP-", "") for sig in sig_list],
            lw=3,
            ax=ax1,
        )

    ax1.legend(ncol=2, loc="best")
    if xlog:
        ax1.set_xscale("log")
    if ylog:
        ax1.set_yscale("log")
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("")
    ax1.set_ylabel("Events")
    if per_muon:
        ax1.set_ylabel("Muons")
    hep.cms.label(llabel="Preliminary", data=False, lumi=int_lumi, ax=ax1)

    # Calculate the ratio, with error propagation, and plot it
    plt.setp(ax1.get_xticklabels(), visible=False)
    ratio = np.divide(
        hist_data.values(),
        hist_bkg_total.values(),
        out=np.ones_like(hist_data.values()),
        where=hist_bkg_total.values() != 0,
    )
    ratio_err = np.where(
        hist_bkg_total.values() > 0,
        np.sqrt(
            (hist_bkg_total.values() ** -2) * (hist_data.variances())
            + (hist_data.values() ** 2 * hist_bkg_total.values() ** -4)
            * (hist_bkg_total.variances())
        ),
        0,
    )
    ax2.errorbar(
        hist_data.axes.centers[0],
        ratio,
        yerr=ratio_err,
        color="black",
        fmt="o",
        linestyle="none",
    )

    # Draw a filled hatch area with the relative uncertainty of the MC in the ratio plot.
    mc_rel_unc = np.divide(
        np.sqrt(hist_bkg_total.variances()),
        hist_bkg_total.values(),
        out=np.zeros_like(hist_bkg_total.values()),
        where=hist_bkg_total.values() != 0,
    )
    y_hatch2 = np.vstack(
        (np.ones_like(hist_bkg_total.values()), np.ones_like(hist_bkg_total.values()))
    ).reshape((-1,), order="F")
    y_hatch2_unc = np.vstack((mc_rel_unc, mc_rel_unc)).reshape((-1,), order="F")
    ax2.fill_between(
        x=x_hatch,
        y1=y_hatch2 - y_hatch2_unc,
        y2=y_hatch2 + y_hatch2_unc,
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
    )

    ax2.axhline(1, ls="--", color="gray")
    ax2.set_xlabel(hist_data.axes.name[0])
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0, 2)
    if xlog:
        ax1.xaxis.set_minor_locator(LogLocator(numticks=999, subs="all"))

    if text_to_show is not None:
        midpoint = int(len(hist_bkg_total.axes[0].edges) / 2)
        x_txt = hist_bkg_total.axes[0].edges[midpoint]
        y_txt = hist_bkg_total.values().max() * 1.1
        ax1.text(x_txt, y_txt, text_to_show, horizontalalignment="center")

    if fig is None:
        plt.plot()
    return
