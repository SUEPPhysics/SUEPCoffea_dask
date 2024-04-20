import glob
import itertools
import numpy as np
import hist
from hist import Hist
import matplotlib.pyplot as plt
from rich.pretty import pprint
import plot_utils
import fill_utils


def loader(region, verbose=False):
    # input .pkl files
    # Region is either prompt or cb for the moment
    plotDir = f"./{region}_combine_output_histograms/"
    infile_names = glob.glob(plotDir + "*.pkl")

    # generate list of files that you want to merge histograms for
    offline_files_SUEP = [
        f for f in infile_names if ("SUEP" in f) and ("histograms.pkl" in f)
    ]
    offline_files_normalized = [f for f in infile_names if ("normalized.pkl" in f)]
    offline_files_other = [
        f for f in infile_names if ("pythia8" in f) and ("histograms.pkl" in f)
    ]
    offline_files = offline_files_normalized + offline_files_other
    if verbose:
        pprint(offline_files)

    data_files = [
        f for f in infile_names if ("DoubleMuon" in f) and ("histograms.pkl" in f)
    ]
    if verbose:
        pprint(data_files)

    other_bkg_names = {
        "DY0JetsToLL": "DY0JetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_NLO": "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYLowMass_NLO": "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "DYLowMass_LO": "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        "TTJets": "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "ttZJets": "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "WWZ_4F": "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+MINIAODSIM",
        "ZZTo4L": "ZZTo4L_TuneCP5_13TeV_powheg_pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "ZZZ": "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8+"
        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+MINIAODSIM",
        "WJets_inclusive": "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+"
        "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
        "ST_tW": "ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8+"
        "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
    }

    # merge the histograms, apply lumis, exclude low HT bins
    plots_SUEP_2018 = plot_utils.loader(
        offline_files_SUEP, year=2018, custom_lumi=573.120134
    )
    plots_2018 = plot_utils.loader(offline_files, year=2018, custom_lumi=573.120134)
    plots_data = plot_utils.loader(data_files, year=2018, is_data=True)

    # put everything in one dictionary, apply lumi for SUEPs
    plots = {}
    for key in plots_SUEP_2018.keys():
        plots[key + "_2018"] = fill_utils.apply_normalization(
            plots_SUEP_2018[key],
            fill_utils.getXSection(
                key + "+RunIIAutumn18-private+MINIAODSIM", "2018", SUEP=True
            ),
        )
    for key in plots_2018.keys():
        is_binned = False
        binned_samples = [
            "QCD_Pt",
            "WJetsToLNu_HT",
            "WZTo",
            "WZ_all",
            "WWTo",
            "WW_all",
            "ST_t-channel",
            "JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
            "DYNJetsToLL",
        ]
        for binned_sample in binned_samples:
            if binned_sample in key:
                is_binned = True
        if is_binned and ("normalized" not in key) and ("cutflow" in key):
            continue
        if is_binned or ("bkg" in key):
            plots[key + "_2018"] = plots_2018[key]
        else:
            plots[key + "_2018"] = fill_utils.apply_normalization(
                plots_2018[key],
                fill_utils.getXSection(other_bkg_names[key], "2018", SUEP=False),
            )

    for key in plots_data.keys():
        plots[key + "_2018"] = plots_data[key]

    # Combine DYJetsToLL_NLO with DYLowMass_NLO
    dy_nlo_all = {}
    for plt_i in plots["DYLowMass_NLO_2018"].keys():
        dy_nlo_all[plt_i] = (
            plots["DYLowMass_NLO_2018"][plt_i] + plots["DYJetsToLL_NLO_2018"][plt_i]
        )
    plots["DY_NLO_all_2018"] = dy_nlo_all

    # Combine DYNJetsToLL with DYLowMass_LO
    dy_combined = {}
    for plt_i in plots["DYLowMass_LO_2018"].keys():
        dy_combined[plt_i] = (
            plots["DYLowMass_LO_2018"][plt_i] + plots["DYNJetsToLL_2018"][plt_i]
        )
    plots["DYCombined_2018"] = dy_combined

    # Combine ZZZ with WWZ
    vvv_combined = {}
    for plt_i in plots["WWZ_4F_2018"].keys():
        vvv_combined[plt_i] = plots["WWZ_4F_2018"][plt_i] + plots["ZZZ_2018"][plt_i]
    plots["VVV_2018"] = vvv_combined

    # Combine ZZ, WZ, and WW
    vv_combined = {}
    for plt_i in plots["WZ_all_2018"].keys():
        vv_combined[plt_i] = (
            plots["WW_all_2018"][plt_i]
            + plots["WZ_all_2018"][plt_i]
            + plots["ZZTo4L_2018"][plt_i]
        )
    plots["VV_2018"] = vv_combined

    # Combine ST
    st_combined = {}
    for plt_i in plots["ST_t-channel_2018"].keys():
        st_combined[plt_i] = (
            plots["ST_t-channel_2018"][plt_i] + plots["ST_tW_2018"][plt_i]
        )
    plots["ST_2018"] = st_combined

    # Combine WJetsHT and WJets_inclusive
    wjets_combined = {}
    for plt_i in plots["WJetsToLNu_HT_2018"].keys():
        wjets_combined[plt_i] = (
            plots["WJetsToLNu_HT_2018"][plt_i] + plots["WJets_inclusive_2018"][plt_i]
        )
    plots["WJets_all_2018"] = wjets_combined

    # Normalize QCD MuEnriched if it exists
    if "QCD_Pt_MuEnriched_2018" in plots.keys():
        for plot in plots["QCD_Pt_MuEnriched_2018"].keys():
            plots["QCD_Pt_MuEnriched_2018"][plot] = plots["QCD_Pt_MuEnriched_2018"][
                plot
            ]
    return plots


def connect_histograms(*args):
    nbins = [h.axes[0].size for h in args]
    other_axes = args[0].axes[1:]
    h_out = Hist(
        hist.axis.Regular(sum(nbins), 0, sum(nbins), name="h_total"),
        *other_axes,
        storage=hist.storage.Weight(),
    )
    for i, h in enumerate(args):
        for idx in itertools.product(*[range(nbins_i) for nbins_i in h.shape]):
            idx_new = list(idx)
            idx_new[0] = sum(nbins[:i]) + idx_new[0]
            h_out[tuple(idx_new)] = h[idx]
    return h_out


def prepare_histograms(plots, processes, nMuon, count_mode=False, blind=False):
    """
    This function prepares the histograms for the fit. It takes the following inputs:
    - plots: the dictionary of histograms
    - processes: the list of processes to include in the fit
    - nMuon: the number of muons to consider (if not an integer, then will sum over that axis)
    """
    data = "DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD_histograms_2018"
    hist_names = [
        # "ht",
        "muon_pt",
        # "muon_eta",
    ]
    if count_mode:
        hist_names = ["muon_pt"]
        if nMuon == 0:
            hist_names = ["ht"]
    # Indexes for splitting the histograms
    idx_ndf = [1j, 4j, 5j, 15j]
    idx_df = [0, 3j]

    # Initialize the ouput arrays
    n = []
    templates = []

    # Project the nMuon and genPartFlav axes
    slc_none = slice(None)
    slc_sum = slice(None, None, sum)
    if isinstance(nMuon, int):
        slc_nMuon = nMuon * 1j
    else:
        slc_nMuon = slice(None, None, sum)
    if not blind:
        for hn in hist_names:
            slc = slc_none, slc_sum, slc_nMuon
            if "Muon_genPartFlav" in plots[data][hn].axes.name:
                slc = slc_none, slc_sum, slc_sum, slc_nMuon
            n.append(plots[data][hn][slc].copy())
        n = connect_histograms(*n)
    for p in processes:
        hists_ndf = []
        hists_df = []
        for hn in hist_names:
            # separate df & ndf sources
            if "Muon_genPartFlav" in plots[p][hn].axes.name:
                plot_ndf = None
                for idx in idx_ndf:
                    if plot_ndf is None:
                        plot_ndf = plots[p][hn][:, idx, slc_sum, slc_nMuon].copy()
                    else:
                        plot_ndf += plots[p][hn][:, idx, slc_sum, slc_nMuon]
                plot_df = None
                for idx in idx_df:
                    if plot_df is None:
                        plot_df = plots[p][hn][:, idx, :, slc_nMuon].copy()
                    else:
                        plot_df += plots[p][hn][:, idx, :, slc_nMuon]
            else:
                # for non-per-muon plots, put them in the ndf part
                plot_ndf = plots[p][hn][:, slc_sum, slc_nMuon].copy()
                plot_df = plots[p][hn][:, :, slc_nMuon].copy().reset()
            hists_ndf.append(plot_ndf.copy())
            hists_df.append(plot_df.copy())
        templates.append(
            [connect_histograms(*hists_ndf), connect_histograms(*hists_df)]
        )
    return n, templates


def get_data(plots, processes, nMuon):
    data = "DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD_histograms_2018"
    hist_names = [
        "ht",
        "muon_pt",
        "muon_eta",
    ]
    idx_ndf = [1j, 4j, 5j, 15j]
    idx_df = [0, 3j]
    xe = None  # array of bin edges
    n = None  # array of data counts
    t_ndf = None  # array of arrays of MC counts for non-decays-in-flight
    vt_ndf = None  # array of arrays of MC variances for non-decays-in-flight
    t_ndf = None  # array of arrays of MC counts for decays-in-flight
    vt_ndf = None  # array of arrays of MC variances for decays-in-flight
    # Take care of the data
    for hn in hist_names:
        slc = slice(None), slice(nMuon * 1j, (nMuon + 1) * 1j)
        if "Muon_genPartFlav" in plots[data][hn].axes.name:
            slc = (
                slice(None),
                slice(None, None, sum),
                slice(nMuon * 1j, (nMuon + 1) * 1j),
            )
        if n is None:
            n = plots[data][hn][slc].values()
        else:
            n = np.append(n, plots[data][hn][slc].values())
    # bin edges
    xe = np.arange(len(n) + 1)
    # loop over the processes to grab the templates
    for p in processes:
        ti_ndf = None
        vti_ndf = None
        ti_df = None
        vti_df = None
        # loop over the histograms for a specific process
        for hn in hist_names:
            # separate df & ndf sources
            if "Muon_genPartFlav" in plots[p][hn].axes.name:
                plot_ndf = None
                for idx in idx_ndf:
                    if plot_ndf is None:
                        plot_ndf = plots[p][hn][:, idx, nMuon * 1j].copy()
                    else:
                        plot_ndf += plots[p][hn][:, idx, nMuon * 1j]
                plot_df = None
                for idx in idx_df:
                    if plot_df is None:
                        plot_df = plots[p][hn][:, idx, nMuon * 1j].copy()
                    else:
                        plot_df += plots[p][hn][:, idx, nMuon * 1j]
            else:
                # for non-per-muon plots, put them in the ndf part
                plot_ndf = plots[p][hn][:, nMuon * 1j].copy()
                plot_df = plot_ndf.copy().reset()
            # transorm plots to arrays
            if ti_ndf is None:
                ti_ndf = plot_ndf.values()
                vti_ndf = plot_ndf.variances()
                ti_df = plot_df.values()
                vti_df = plot_df.variances()
            else:
                ti_ndf = np.append(ti_ndf, plot_ndf.values())
                vti_ndf = np.append(vti_ndf, plot_ndf.variances())
                ti_df = np.append(ti_df, plot_df.values())
                vti_df = np.append(vti_df, plot_df.variances())
        # Fill final output arrays
        if t_ndf is None:
            t_ndf = np.array([ti_ndf])
            vt_ndf = np.array([vti_ndf])
            t_df = np.array([ti_df])
            vt_df = np.array([vti_df])
        else:
            t_ndf = np.array([*t_ndf, ti_ndf])
            vt_ndf = np.array([*vt_ndf, vti_ndf])
            t_df = np.array([*t_df, ti_df])
            vt_df = np.array([*vt_df, vti_df])
    t_ndf = np.where(t_ndf < 0, 0, t_ndf)
    t_df = np.where(t_df < 0, 0, t_df)
    return xe, n, t_ndf, vt_ndf, t_df, vt_df


def print_details(n, t, vt, processes):
    np.set_printoptions(suppress=True, precision=1)
    print("Data values:")
    pprint(n)
    print("MC values:")
    pprint(t)
    print("MC variances:")
    pprint(vt)
    np.set_printoptions(suppress=True, precision=1)
    print("Sum of MC (per process):")
    for i, p in enumerate(processes):
        print(f"{p[:-5]}: {np.sum(t[i]):.1f} + {np.sqrt(np.sum(vt[i])):.1f}")


def plot_comparisons(n, t, vt, processes):
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))
    x_vals = range(len(n))
    ax1.errorbar(x_vals, n, yerr=np.sqrt(n), drawstyle="steps-mid", label="data")
    ax1.errorbar(
        x_vals,
        np.sum(t, axis=0),
        yerr=np.sqrt(np.sum(vt, axis=0)),
        drawstyle="steps-mid",
        label="MC sum",
    )
    ax1.set_xlabel("bin")
    ax1.set_ylabel("events")
    ax1.legend()
    ax2.errorbar(x_vals, n / np.sum(n), drawstyle="steps-mid", label="data")
    for i, ti in enumerate(t):
        ax2.errorbar(
            x_vals, ti / np.sum(ti), drawstyle="steps-mid", label=f"{processes[i][:-5]}"
        )
    ax2.set_xlabel("bin")
    ax2.set_ylabel("density")
    ax2.legend()


def plot_compare_df_vs_ndf(n, t_ndf, vt_ndf, t_df, vt_df, process):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    x_vals = range(len(t_ndf))
    # ax.errorbar(x_vals, n, yerr=np.sqrt(n), drawstyle="steps-mid", label="data")
    ax.errorbar(
        x_vals, t_ndf, yerr=np.sqrt(vt_ndf), drawstyle="steps-mid", label="MC ndf"
    )
    ax.errorbar(x_vals, t_df, yerr=np.sqrt(vt_df), drawstyle="steps-mid", label="MC df")
    ax.set_xlabel("bin")
    ax.set_ylabel("events")
    ax.set_title(process)
    ax.legend()


def get_nuissances(n, mu, mu_var):
    beta_var = np.where(mu > 0, mu_var / mu**2, 0)
    p = 0.5 - 0.5 * mu * beta_var
    beta = p + np.sqrt(p**2 + n * beta_var)
    return beta, beta_var
