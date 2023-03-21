import itertools
import math

import fill_utils
import hist
import matplotlib
import mplhep as hep
import plot_utils
import sympy as sp
from hist import Hist
from matplotlib import pyplot as plt
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


def significance_scan(h_sig, h_bkg, columns_list, sig_func):
    """
    Scan the significance of a signal given the histograms of signal and background
    events. The significance is calculated using the Punzi formula.
    Parameters
    ----------
    h_sig : hist.Hist
        Histogram of signal events
    h_bkg : hist.Hist
        Histogram of background events
    columns_list : list
        List of columns to scan significance for
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
        axes = h.axes.name
        df_dict = {}
        df_dict = df[list(axes)].to_dict("list")
        if merge_datasets:
            h.fill(**df_dict, weight=weight)
        else:
            h[dataset].fill(**df_dict, weight=weight)
    return h


def plot_1d(h_bkg, h_sig, h_significance, save_plot=False, show_plot=True):
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
    if save_plot:
        plt.savefig("significance.png")
    if show_plot:
        plt.show()


def plot_2d(h_bkg, h_sig, h_significance, save_plot=False, show_plot=True):
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
    plt.show()


def plot_Nd(h_bkg, h_sig, h_significance, save_plot=False, show_plot=True):
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
            h_bkg.project(i).plot(ax=ax[i, j], label="QCD")
            h_sig.project(i).plot(ax=ax[i, j], label="Signal")
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
    plt.show()


def plot(h_bkg, h_sig, h_significance, save_plot=False, show_plot=True):
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
    save_plot : bool
        Save the plot to a file
    show_plot : bool
        Show the plot
    """
    if len(h_significance.axes) == 1:
        plot_1d(h_bkg, h_sig, h_significance, save_plot, show_plot)
    elif len(h_significance.axes) == 2:
        plot_2d(h_bkg, h_sig, h_significance, save_plot, show_plot)
    elif len(h_significance.axes) > 2:
        plot_Nd(h_bkg, h_sig, h_significance, save_plot, show_plot)
    else:
        raise ValueError("The number of axes must be >= 1")


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

# Local configuration
local_path = "../temp_output.nosync/"

# List of background files
qcd_pt_bins = [
    "15to30",
    "30to50",
    "50to80",
    "80to120",
    "120to170",
    "170to300",
    "300to470",
    "470to600",
    "600to800",
    "800to1000",
    "1000to1400",
    "1400to1800",
    "1800to2400",
    "2400to3200",
    "3200toInf",
]
qcd_files = [
    f"{local_path}condor_test_QCD_Pt_{bin}+RunIISummer20UL18.hdf5"
    for bin in qcd_pt_bins
]
dataset_suffix = "-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM"
qcd_datasets = [
    f"QCD_Pt_{bin}_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2{dataset_suffix}"
    for bin in qcd_pt_bins
]

# List of signal files
# masses_s = [125, 400, 750, 1000]
# decays = ["darkPho", "darkPhoHad"]
masses_s = [125]
decays = ["darkPhoHad"]
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

# List of variables to plot
# That's the main input for the significance scan
columns_list = [
    "nMuons_looseId",
    "nMuons_mediumId",
    "nMuons_tightId",
    "nMuons_isTracker",
    "nMuons_triggerIdLoose",
]
save_plots = True
show_plots = False

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
    for signal_dataset in signal_datasets:
        # Perform significance scan
        h_significance = significance_scan(
            h_sig[signal_dataset], h_bkg, columns_list, sig_funcs
        )

        # Plot histograms
        plot(h_bkg, h_sig[signal_dataset], h_significance, save_plots, show_plots)
