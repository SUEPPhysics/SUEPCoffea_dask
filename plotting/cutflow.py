import argparse
import itertools
import json
import math

import fill_utils
import hist
import matplotlib
import mplhep as hep
import numpy as np
import plot_utils
import sympy as sp
from hist import Hist
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from rich.progress import track

plt.style.use(hep.style.CMS)


def find_optimum(histogram):
    """
    NOTE: not done yet
    Find the optimum cut value(s) for a histogram
    Parameters
    ----------
    histogram : hist.Hist
        Histogram to find optimum cut value for
    Returns
    -------
    cut : numpy.ndarray
        Array of optimum cut value(s)
    """
    # Find the maximum significance
    array = histogram.to_numpy()
    values = array[0]
    edges = array[1:]
    return edges[values.argmax()]


def significance_functions(alpha=2, beta=5, mode="punzi_full_smooth"):
    """
    Calculate the significance of a signal given the number of signal and background
    events. The significance is calculated using the following methods:
    - punzi_simple: simplified case where alpha = beta
    - punzi_full: full Punzi formula
    - punzi_full_smooth: full Punzi formula smoothened with a fit
    - s_over_b: S / sqrt(B)
    - s_over_b_and_s: S / sqrt(B + S)
    Parameters
    ----------
    alpha : float
        Punzi parameter
    beta : float
        Punzi parameter
    mode : str
        Significance mode, one of "punzi_simple", "punzi_full", "punzi_full_smooth",
        "s_over_b", "s_over_b_and_s"
    Returns
    -------
    significance, significance uncertainty : tuple(sympy.FunctionClass instance, sympy.FunctionClass instance)
        Significance and uncertainty
    """
    S, S_tot, B, dS, dS_tot, dB = sp.symbols("S S_tot B dS dS_tot dB")
    epsilon = S / S_tot
    punziCommon = alpha * sp.sqrt(B) + (beta / 2) * sp.sqrt(
        beta**2 + 4 * alpha * sp.sqrt(B) + 4 * B
    )
    if mode == "punzi_simple":
        sig = epsilon / ((alpha**2) / 2 + sp.sqrt(B))
    elif mode == "punzi_full":
        sig = epsilon / ((beta**2) / 2 + punziCommon)
    elif mode == "punzi_full_smooth":
        sig = epsilon / ((alpha**2) / 8 + 9 * (beta**2) / 13 + punziCommon)
    elif mode == "s_over_b":
        sig = epsilon / sp.sqrt(B)
    elif mode == "s_over_b_and_s":
        sig = epsilon / sp.sqrt(B + S)
    else:
        raise ValueError("Invalid mode")

    partial_S = sp.diff(sig, S)
    partial_S_tot = sp.diff(sig, S_tot)
    partial_B = sp.diff(sig, B)
    delta_sig = (
        (partial_S * dS) ** 2 + (partial_S_tot * dS_tot) ** 2 + (partial_B * dB) ** 2
    )
    return sp.lambdify([S, S_tot, B], sig), sp.lambdify(
        [S, S_tot, B, dS, dS_tot, dB], delta_sig
    )


def significance_scan(h_bkg, h_sig, columns_list, sig_func):
    """
    Scan the significance of a signal given the histograms of signal and background
    events. The significance is calculated using the Punzi formula.
    Parameters
    ----------
    h_bkg : hist.Hist
        Histogram of background events
    h_sig : hist.Hist
        Histogram of signal events
    columns_list : list
        List of columns to scan significance for
    sig_func : tuple
        Tuple of significance functions
    Returns
    -------
    h_significance : hist.Hist
        Histogram of significance
    """
    h_significance = h_sig.copy()
    h_significance.reset()
    for ax in h_significance.axes:
        ax.label += " >= cutvalue"
    n_dims = len(columns_list)
    n_bins = h_bkg.shape
    S_tot = h_sig.sum(flow=True)
    iterators = [range(n_bins[i]) for i in range(n_dims)]
    for indices in track(
        itertools.product(*iterators),
        total=len(list(itertools.product(*iterators))),
        description="Significance scan",
    ):
        cut = [slice(index, n_bins[count]) for count, index in enumerate(indices)]
        B = h_bkg[tuple(cut)].sum(flow=True)
        S = h_sig[tuple(cut)].sum(flow=True)
        signfificance = (
            sig_func[0](S.value, S_tot.value, B.value),
            sig_func[1](
                S.value,
                S_tot.value,
                B.value,
                math.sqrt(S.variance),
                math.sqrt(S_tot.variance),
                math.sqrt(B.variance),
            ),
        )
        h_significance[indices] = signfificance
    return h_significance


def make_histogram(axes, columns, files, datasets, merge_datasets=False):
    """
    Make a histogram from a list of files and datasets
    Parameters
    ----------
    axes : dict
        Dictionary of axes
    columns : list
        List of columns to fill histogram with
    files : list
        List of files to fill histogram with
    datasets : list
        List of datasets to fill histogram with
    merge_datasets : bool
        Merge datasets into one histogram. Useful when datasets are
        split into multiple bins, e.g. for the QCD Pt or HT bins
    Returns
    -------
    h : hist.Hist
        Histogram
    """
    axes = [axes[c] for c in columns]
    h = Hist(
        *axes,
        storage=hist.storage.Weight(),
    )
    if not merge_datasets:
        h = {dataset: h.copy() for dataset in datasets}
    for file, dataset in zip(files, datasets):
        df, metadata = fill_utils.h5load(file, "vars")

        # check if file is corrupted
        if type(df) == int:
            continue

        # check if file is empty
        if "empty" in list(df.keys()):
            continue
        if df.shape[0] == 0:
            continue

        is_signal = False
        if "SUEP" in dataset:
            is_signal = True

        gensumweight = metadata["gensumweight"]
        xsection = fill_utils.getXSection(dataset, 2018, SUEP=is_signal)
        lumi = plot_utils.findLumi(year=None, auto_lumi=True, infile_name=dataset)
        weight = xsection * lumi / gensumweight
        if merge_datasets:
            axes = h.axes.name
        else:
            axes = h[dataset].axes.name
        df_dict = {}
        df_dict = df[list(axes)].to_dict("list")
        if merge_datasets:
            h.fill(**df_dict, weight=weight)
        else:
            h[dataset].fill(**df_dict, weight=weight)
    return h


def plot_1d(h_bkg, h_sig, h_significance, save_plot_as=None, show_plot=True):
    """
    Plot the histograms of signal and background events and the significance
    when there is only one variable
    Parameters
    ----------
    h_bkg : hist.Hist
        Histogram of background events
    h_sig : hist.Hist
        Histogram of signal events
    h_significance : hist.Hist
        Histogram of significance
    save_plot_as : str
        Save the plot as this file name
    show_plot : bool
        Show the plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.tight_layout()
    h_bkg.plot(ax=ax[0], label="QCD")
    h_sig.plot(ax=ax[0], label="Signal")
    ax[0].set_title("Events")
    ax[0].legend()
    ax[0].set_yscale("log")
    h_significance.plot(ax=ax[1])
    ax[1].set_title("Significance")
    if save_plot_as is not None:
        plt.savefig(save_plot_as + ".pdf")
    if show_plot:
        plt.show()


def plot_1d_merged(h_bkg, h_sig, h_significance, dataset=None, fig=None, ax=None):
    """
    Plot the histograms of signal and background events and the significance
    when there is only one variable and keep all signal points in the same plots
    Parameters
    ----------
    h_bkg : hist.Hist
        Histogram of background events
    h_sig : hist.Hist
        Histogram of signal events
    h_significance : hist.Hist
        Histogram of significance
    dataset : str
        Dataset name
    fig : matplotlib.figure.Figure
        Figure to plot on
    ax : matplotlib.axes.Axes
        Axes to plot on
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure
    ax : matplotlib.axes.Axes
        Axes
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        fig.tight_layout()
        ax[0].set_title("Events")
        ax[0].set_yscale("log")
        ax[1].set_title("Significance")
        hep.histplot(
            h_bkg,
            yerr=np.sqrt(h_bkg.variances()),
            ax=ax[0],
            label="QCD",
            color=plot_utils.default_colors["QCD"],
        )

    label = "Signal"
    if dataset is not None:
        label = dataset.replace("+RunIIAutumn18-private+MINIAODSIM", "").replace(
            "SUEP-", ""
        )

    linestyle = "solid"
    if "darkPhoHad" in label:
        linestyle = "dashed"

    hep.histplot(
        h_sig,
        yerr=np.sqrt(h_sig.variances()),
        ax=ax[0],
        label=label,
        color=plot_utils.default_colors["SUEP-" + label + "_2018"],
        linestyle=linestyle,
    )
    hep.histplot(
        h_significance,
        yerr=np.sqrt(h_significance.variances()),
        ax=ax[1],
        label=label,
        color=plot_utils.default_colors["SUEP-" + label + "_2018"],
        linestyle=linestyle,
    )
    return fig, ax


def plot_2d(h_bkg, h_sig, h_significance, save_plot_as=None, show_plot=True):
    """
    Plot the histograms of signal and background events and the significance
    when there are two variables
    Parameters
    ----------
    h_bkg : hist.Hist
        Histogram of background events
    h_sig : hist.Hist
        Histogram of signal events
    h_significance : hist.Hist
        Histogram of significance
    save_plot_as : str
        Save the plot as this file name
    show_plot : bool
        Show the plot
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    fig.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.94, top=0.92, bottom=0.13, wspace=0.4)
    h_bkg.plot(norm=matplotlib.colors.LogNorm(), ax=ax[0])
    ax[0].set_title("QCD")
    h_sig.plot(norm=matplotlib.colors.LogNorm(), ax=ax[1])
    ax[1].set_title("Signal")
    h_significance.plot(ax=ax[2])
    ax[2].set_title("Significance")
    if save_plot_as is not None:
        plt.savefig(save_plot_as + ".pdf")
    if show_plot:
        plt.show()


def plot_Nd(h_bkg, h_sig, h_significance, save_plot_as=None, show_plot=True):
    """
    Plot the histograms of signal and background events and the significance
    when there are more than two variables
    Parameters
    ----------
    h_bkg : hist.Hist
        Histogram of background events
    h_sig : hist.Hist
        Histogram of signal events
    h_significance : hist.Hist
        Histogram of significance
    save_plot_as : str
        Save the plot as this file name
    show_plot : bool
        Show the plot
    """
    n_axes = len(h_significance.axes)
    label_scale = 3 / n_axes
    for ax in h_significance.axes:
        ax.label = ax.name
    fig, ax = plt.subplots(ncols=n_axes, nrows=n_axes, figsize=(12, 12))
    fig.tight_layout()
    fig.subplots_adjust(
        left=0.05,
        right=0.94,
        top=0.96,
        bottom=0.08,
        wspace=0.32 + 0.04 * n_axes / 3,
        hspace=0.32 + 0.04 * n_axes / 3,
    )
    for i, j in track(
        itertools.product(range(n_axes), range(n_axes)),
        total=n_axes**2,
        description="Plotting",
    ):
        if i == j:
            h_bkg_project = h_bkg.project(i)
            h_sig_project = h_sig.project(i)
            hep.histplot(
                h_bkg_project,
                ax=ax[i, j],
                yerr=np.sqrt(h_bkg_project.variances()),
                label="QCD",
            )
            hep.histplot(
                h_sig_project,
                ax=ax[i, j],
                yerr=np.sqrt(h_sig_project.variances()),
                label="Signal",
            )
            ax[i, j].set_yscale("log")
            ax[i, j].xaxis.label.set_size(20 * label_scale)
            ax[i, j].xaxis.labelpad = 4 * label_scale * 0.7
            ax[i, j].legend(fontsize=20 * label_scale * 0.9)
            continue
        h_significance.project(i, j).plot(ax=ax[i, j])
        ax[i, j].xaxis.label.set_size(20 * label_scale)
        ax[i, j].yaxis.label.set_size(20 * label_scale)
        ax[i, j].xaxis.labelpad = 4 * label_scale * 0.7
        ax[i, j].yaxis.labelpad = 4 * label_scale * 0.7
    if save_plot_as is not None:
        plt.savefig(save_plot_as + ".pdf")
    if show_plot:
        plt.show()


def plot(h_bkg, h_sig, h_significance, save_plots_as=None, show_plot=True):
    """
    Plot the histograms of signal and background events and the significance
    Parameters
    ----------
    h_bkg : hist.Hist
        Histogram of background events
    h_sig : hist.Hist
        Histogram of signal events
    h_significance : hist.Hist
        Histogram of significance
    save_plots_as : str
        Save the plot as this file name
    show_plot : bool
        Show the plot
    """
    if len(h_significance.axes) == 1:
        plot_1d(
            h_bkg,
            h_sig,
            h_significance,
            save_plot_as=save_plots_as,
            show_plot=show_plot,
        )
    elif len(h_significance.axes) == 2:
        plot_2d(
            h_bkg,
            h_sig,
            h_significance,
            save_plot_as=save_plots_as,
            show_plot=show_plot,
        )
    elif len(h_significance.axes) > 2:
        plot_Nd(
            h_bkg,
            h_sig,
            h_significance,
            save_plot_as=save_plots_as,
            show_plot=show_plot,
        )
    else:
        raise ValueError("The number of axes must be >= 1")


# Parse command line arguments
parser = argparse.ArgumentParser(description="Calculate and plot cutflows")
parser.add_argument(
    "-c", "--config", metavar="configuration_file", help="Configuration file"
)
args = parser.parse_args()

# Load configuration
with open(args.config) as f:
    config = json.load(f)
qcd_pt_bins = config["background_datasets"]["QCD_Pt"]
masses_s = config["signal_points"]["masses_s"]
decays = config["signal_points"]["decays"]
columns_list = config["columns_list"]
show_plot = config["show_plot"]
merged_plots = config["merged_plots"]
local_path = config["local_path"]
plots_path = config["plots_path"]


# Define axes
axes_dict = {
    "ntracks": hist.axis.Regular(
        300, 0, 300, name="ntracks", label="nTracks", underflow=False, overflow=True
    ),
    "nMuons": hist.axis.Regular(
        30, 0, 30, name="nMuons", label="nMuons", underflow=False, overflow=True
    ),
    "nMuons_category1": hist.axis.Regular(
        20,
        0,
        20,
        name="nMuons_category1",
        label="nMuons_cat1",
        underflow=False,
        overflow=True,
    ),
    "nMuons_category2": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_category2",
        label="nMuons_cat2",
        underflow=False,
        overflow=True,
    ),
    "nMuons_category3": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_category3",
        label="nMuons_cat3",
        underflow=False,
        overflow=True,
    ),
    "nMuons_highPurity": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_highPurity",
        label="nMuons highPurity",
        underflow=False,
        overflow=True,
    ),
    "nMuons_looseId": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_looseId",
        label="nMuons looseId",
        underflow=False,
        overflow=True,
    ),
    "nMuons_mediumId": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_mediumId",
        label="nMuons mediumId",
        underflow=False,
        overflow=True,
    ),
    "nMuons_tightId": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_tightId",
        label="nMuons tightId",
        underflow=False,
        overflow=True,
    ),
    "nMuons_isTracker": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_isTracker",
        label="nMuons isTracker",
        underflow=False,
        overflow=True,
    ),
    "nMuons_triggerIdLoose": hist.axis.Regular(
        10,
        0,
        10,
        name="nMuons_triggerIdLoose",
        label="nMuons triggerIdLoose",
        underflow=False,
        overflow=True,
    ),
}

# Make list of background files
qcd_files = [
    f"{local_path}condor_test_QCD_Pt_{bin}+RunIISummer20UL18.hdf5"
    for bin in qcd_pt_bins
]
dataset_suffix = "-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM"
qcd_datasets = [
    f"QCD_Pt_{bin}_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2{dataset_suffix}"
    for bin in qcd_pt_bins
]

# Make list of signal files
signal_files = [
    f"{local_path}condor_test_SUEP-m{mass_s}-{decay}+RunIIAutumn18.hdf5"
    for mass_s in masses_s
    for decay in decays
]
signal_datasets = [
    f"SUEP-m{mass_s}-{decay}+RunIIAutumn18-private+MINIAODSIM"
    for mass_s in masses_s
    for decay in decays
]

# Make lists for custom legends
custom_lines_masses = []
for mass_s in masses_s:
    custom_lines_masses.append(
        Line2D(
            [0],
            [0],
            color=plot_utils.default_colors[f"SUEP-m{mass_s}-darkPho_2018"],
            lw=2,
        )
    )
custom_lines_decays = []
for decay in decays:
    if decay == "darkPho":
        linestyle = "solid"
    elif decay == "darkPhoHad":
        linestyle = "dashed"
    custom_lines_decays.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyle,
            lw=2,
        )
    )


if __name__ == "__main__":
    # Make histograms
    h_bkg = make_histogram(
        axes_dict, columns_list, qcd_files, qcd_datasets, merge_datasets=True
    )
    h_sig = make_histogram(
        axes_dict, columns_list, signal_files, signal_datasets, merge_datasets=False
    )

    # Get significance functions
    sig_funcs = significance_functions()

    # Loop over signal points
    fig_merged, ax_merged = None, None
    for signal_dataset in signal_datasets:
        # Perform significance scan
        h_significance = significance_scan(
            h_bkg, h_sig[signal_dataset], columns_list, sig_funcs
        )

        if merged_plots and len(columns_list) == 1:
            # Plot merged histograms
            fig_merged, ax_merged = plot_1d_merged(
                h_bkg,
                h_sig[signal_dataset],
                h_significance,
                dataset=signal_dataset,
                fig=fig_merged,
                ax=ax_merged,
            )
        else:
            # Plot histograms
            plot(
                h_bkg,
                h_sig[signal_dataset],
                h_significance,
                save_plots_as=plots_path + signal_dataset,
                show_plot=show_plot,
            )

    # Save merged plots
    if merged_plots and len(columns_list) == 1:
        legend1 = ax_merged[1].legend(
            custom_lines_masses, masses_s, title=r"$m_S$ (GeV)", loc=2
        )
        legend2 = ax_merged[1].legend(
            custom_lines_decays, decays, title="Decay mode", loc=3
        )
        ax_merged[1].add_artist(legend1)
        fig_merged.savefig(plots_path + columns_list[0] + "_merged.pdf")
