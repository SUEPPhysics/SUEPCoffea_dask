import logging
import os
import pickle
import shutil
import subprocess
from collections import defaultdict

import boost_histogram as bh
import hist
import hist.intervals
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sympy import diff, sqrt, symbols

default_colors = {
    "bkg": "black",
    "bkg_2018": "black",
    "QCD": "midnightblue",
    "QCD_HT": "lightblue",
    "QCD_HT_2018": "lightblue",
    "QCD_HT_2017": "lightblue",
    "QCD_HT_2016": "lightblue",
    "QCD_HT_allyears": "lightblue",
    "QCD_Pt": "midnightblue",
    "QCD_Pt_2018": "midnightblue",
    "QCD_Pt_2017": "midnightblue",
    "QCD_Pt_2016": "midnightblue",
    "QCD_Pt_allyears": "midnightblue",
    "QCD_Pt_MuEnriched": "darkred",
    "QCD_Pt_MuEnriched_2018": "darkred",
    "DYJetsToLL": "olive",
    "DYJetsToLL_2018": "olive",
    "DYJetsToLL_HT": "orange",
    "DYJetsToLL_HT_2018": "orange",
    "DYJetsToLL_M-4to50_HT": "green",
    "DYJetsToLL_M-4to50_HT_2018": "green",
    "DYJetsToLL_M-50_HT": "cyan",
    "DYJetsToLL_M-50_HT_2018": "cyan",
    "DYNJetsToLL": "maroon",
    "DYNJetsToLL_2018": "maroon",
    "DYJetsToMuMu": "red",
    "DYJetsToMuMu_2018": "red",
    "TTJets": "gray",
    "TTJets_2018": "gray",
    "TTTo2L2Nu": "pink",
    "TTTo2L2Nu_2018": "pink",
    "ttZJets": "darkgreen",
    "ttZJets_2018": "darkgreen",
    "WWZ_4F": "darkcyan",
    "WWZ_4F_2018": "darkcyan",
    "WWZJetsTo4L2Nu_4F": "darkmagenta",
    "WWZJetsTo4L2Nu_4F_2018": "darkmagenta",
    "ZZTo4L": "darkorange",
    "ZZTo4L_2018": "darkorange",
    "ZZZ": "yellow",
    "ZZZ_2018": "yellow",
    "ZToMuMu": "sienna",
    "ZToMuMu_2018": "sienna",
    "data": "maroon",
    "data": "maroon",
    "data_2018": "maroon",
    "data_2017": "maroon",
    "data_2016": "maroon",
    "data_allyears": "maroon",
    "SUEP-m1000-darkPho_2018": "red",
    "SUEP-m1000-darkPhoHad_2018": "red",
    "SUEP-m1000-generic_2018": "red",
    "SUEP-m750-darkPho_2018": "orange",
    "SUEP-m750-darkPhoHad_2018": "orange",
    "SUEP-m750-generic_2018": "orange",
    "SUEP-m400-darkPho_2018": "green",
    "SUEP-m400-darkPhoHad_2018": "green",
    "SUEP-m400-generic_2018": "green",
    "SUEP-m125-darkPho_2018": "cyan",
    "SUEP-m125-darkPhoHad_2018": "cyan",
    "SUEP-m125-generic_2018": "cyan",
    "SUEP-m125-generic-htcut_2018": "magenta",
    "SUEP-m1000-darkPho_2017": "red",
    "SUEP-m1000-darkPhoHad_2017": "red",
    "SUEP-m1000-generic_2017": "red",
    "SUEP-m750-darkPho_2017": "orange",
    "SUEP-m750-darkPhoHad_2017": "orange",
    "SUEP-m750-generic_2017": "orange",
    "SUEP-m400-darkPho_2017": "green",
    "SUEP-m400-darkPhoHad_2017": "green",
    "SUEP-m400-generic_2017": "green",
    "SUEP-m125-darkPho_2017": "cyan",
    "SUEP-m125-darkPhoHad_2017": "cyan",
    "SUEP-m125-generic_2017": "cyan",
    "SUEP-m125-generic-htcut_2017": "magenta",
    "SUEP-m1000-darkPho_2016": "red",
    "SUEP-m1000-darkPhoHad_2016": "red",
    "SUEP-m1000-generic_2016": "red",
    "SUEP-m750-darkPho_2016": "orange",
    "SUEP-m750-darkPhoHad_2016": "orange",
    "SUEP-m750-generic_2016": "orange",
    "SUEP-m400-darkPho_2016": "green",
    "SUEP-m400-darkPhoHad_2016": "green",
    "SUEP-m400-generic_2016": "green",
    "SUEP-m125-darkPho_2016": "cyan",
    "SUEP-m125-darkPhoHad_2016": "cyan",
    "SUEP-m125-generic_2016": "cyan",
    "SUEP-m125-generic-htcut_2016": "magenta",
    "M125_2018": "cyan",
    "M200_2018": "blue",
    "M300_2018": "lightseagreen",
    "M400_2018": "green",
    "M500_2018": "darkgreen",
    "M600_2018": "lawngreen",
    "M700_2018": "goldenrod",
    "M800_2018": "orange",
    "M900_2018": "sienna",
    "M1000_2018": "red",
}

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016_apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    "2018": 59817.406,
}

sample_names = {
    "QCD_Pt_": "QCD_Pt",
    "QCD_HT": "QCD_HT",
    "MuEnriched": "QCD_Pt_MuEnriched",
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX": "DYJetsToLL",
    "DYJetsToMuMu": "DYJetsToMuMu",
    "DY1JetsToLL": "DYNJetsToLL",
    "DY2JetsToLL": "DYNJetsToLL",
    "DY3JetsToLL": "DYNJetsToLL",
    "DY4JetsToLL": "DYNJetsToLL",
    "DYJetsToLL_M-50_HT": "DYJetsToLL_HT",
    "DYJetsToLL_M-4to50_HT": "DYJetsToLL_HT",
    "TTJets": "TTJets",
    "TTTo2L2Nu": "TTTo2L2Nu",
    "ttZJets": "ttZJets",
    "WWZ_4F": "WWZ_4F",
    "WWZJetsTo4L2Nu_4F": "WWZJetsTo4L2Nu_4F",
    "ZZTo4L": "ZZTo4L",
    "ZZZ": "ZZZ",
    "ZToMuMu": "ZToMuMu",
    "JetHT+Run": "data",
    "ScoutingPFHT": "data",
}


def lumiLabel(year):
    if year in ["2017", "2018"]:
        return round(lumis[year] / 1000, 1)
    elif year == "2016":
        return round((lumis[year] + lumis[year + "_apv"]) / 1000, 1)


def findLumi(year, auto_lumi, infile_name):
    if auto_lumi:
        if "20UL16MiniAODv2" in infile_name:
            lumi = lumis["2016"]
        if "20UL17MiniAODv2" in infile_name:
            lumi = lumis["2017"]
        if "20UL16MiniAODAPVv2" in infile_name:
            lumi = lumis["2016_apv"]
        if "20UL18" in infile_name:
            lumi = lumis["2018"]
        if "SUEP-m" in infile_name:
            lumi = lumis["2018"]
        if "JetHT+Run" in infile_name:
            lumi = 1
    if year and not auto_lumi:
        lumi = lumis[str(year)]
    if year and auto_lumi:
        raise Exception("Apply lumis automatically or based on year")
    return lumi


def fill_background(infile_name, plots, lumi):
    is_background = False
    backgrounds = [
        "QCD_Pt_",
        "DY1JetsToLL",
        "DY2JetsToLL",
        "DY3JetsToLL",
        "DY4JetsToLL",
        "TTJets",
        "ttZJets",
        "WWZ_4F",
        "ZZTo4L",
        "ZZZ",
    ]
    for background in backgrounds:
        if background in infile_name:
            is_background = True

    if not is_background:
        return plots

    if "bkg" not in list(plots.keys()):
        plots["bkg"] = openpkl(infile_name)
        for plot in list(plots["bkg"].keys()):
            plots["bkg"][plot] = plots["bkg"][plot] * lumi
    else:
        plotsToAdd = openpkl(infile_name)
        for plot in list(plotsToAdd.keys()):
            plots["bkg"][plot] = plots["bkg"][plot] + plotsToAdd[plot] * lumi
    return plots


def fillSample(infile_name, plots, lumi):
    found_name = False
    sample = None
    for name in sample_names.keys():
        if name in infile_name:
            if found_name:
                raise Exception(f"Found multiple sample names in file name: {name}")
            sample = sample_names[name]
            found_name = True

    is_binned = False
    binned_samples = [
        "QCD_Pt_",
        "QCD_HT",
        "MuEnriched",
        "ZToMuMu",
        "DY1JetsToLL",
        "DY2JetsToLL",
        "DY3JetsToLL",
        "DY4JetsToLL",
        "DYJetsToLL_M-50_HT",
        "DYJetsToLL_M-4to50_HT",
    ]
    for binned_sample in binned_samples:
        if binned_sample in infile_name:
            is_binned = True

    if is_binned:
        # include this block to import the bins individually
        temp_sample = infile_name.split("/")[-1].split(".pkl")[0]
        plots[temp_sample] = openpkl(infile_name)
        for plot in list(plots[temp_sample].keys()):
            plots[temp_sample][plot] = plots[temp_sample][plot] * lumi
    elif "SUEP" in infile_name:
        if "+" in infile_name:
            sample = infile_name.split("/")[-1].split("+")[0]
        elif "new_generic" in infile_name:
            sample = infile_name.split("/")[-1].split("_")[
                1
            ]  # hack for Carlos naming convention
        else:
            sample = infile_name.split("/")
    elif sample is None:
        sample = infile_name
    return sample, plots


# load file(s)
def loader(infile_names, year=None, auto_lumi=False, exclude_low_bins=False):
    plots = {}
    for infile_name in infile_names:
        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            continue
        elif ".pkl" not in infile_name:
            continue
        elif (
            "QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-pilot"
            in infile_name
        ):
            continue

        # sets the lumi based on year
        lumi = findLumi(year, auto_lumi, infile_name)

        # exclude low bins
        if exclude_low_bins:
            if "50to100" in infile_name:
                continue
            if "100to200" in infile_name:
                continue

        # plots[sample] sample is filled here
        sample, plots = fillSample(infile_name, plots, lumi)

        # plots["bkg"] is filled here
        plots = fill_background(infile_name, plots, lumi)

        if sample not in list(plots.keys()):
            plots[sample] = openpkl(infile_name)
            for plot in list(plots[sample].keys()):
                plots[sample][plot] = plots[sample][plot] * lumi
        else:
            plotsToAdd = openpkl(infile_name)
            for plot in list(plotsToAdd.keys()):
                plots[sample][plot] = plots[sample][plot] + plotsToAdd[plot] * lumi

    return plots


def combineYears(inplots, tag="QCD_HT", years=None):
    """
    Combines all samples in plots with a certain tag and with certain
    years. Returns combined plots.
    """
    if not years:
        years = ["2018", "2017", "2016"]
    outPlots = {}
    yearsAdded = []
    initialize = True
    for sample in inplots.keys():
        if tag not in sample:
            continue
        if not any([y in sample for y in years]):
            continue

        # keep track of which years we've added already
        for year in years:
            if year in sample:
                if year in yearsAdded:
                    raise Exception("Already loaded this year: " + year)
                yearsAdded.append(year)

        # combine samples
        if initialize:
            outPlots = inplots[sample].copy()
        else:
            for plot in list(inplots[sample].keys()):
                outPlots[plot] = outPlots[plot] + inplots[sample][plot].copy()

        initialize = False

    return outPlots


def check_proxy(time_min=100):
    """
    Checks for existence of proxy with at least time_min
    left on it.
    If it's inactive or below time_min, it will regenerate
    it with 140 hours.
    """
    home_base = os.environ["HOME"]
    proxy_base = f"x509up_u{os.getuid()}"
    proxy_copy = os.path.join(home_base, proxy_base)
    regenerate_proxy = False
    if not os.path.isfile(proxy_copy):
        logging.warning("--- proxy file does not exist")
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ["voms-proxy-info", "--file", proxy_copy, "--timeleft"]
        )
        lifetime = float(lifetime)
        lifetime = lifetime / (60 * 60)
        if lifetime < time_min:
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system("voms-proxy-init -voms cms --hours=140")
            lifetime = 140
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile("/tmp/" + proxy_base, proxy_copy)

    return lifetime


def apply_binwise_scaling(h_in, bins, scales, dim="x"):
    """
    Apply scales to bins of a particular histogram.
    """
    h_npy = h_in.to_numpy()
    if len(h_npy) == 2:
        h_fragments = []
        for iBin in range(len(bins) - 1):
            h_fragments.append(h_in[bins[iBin] : bins[iBin + 1]] * scales[iBin])
        h = hist.Hist(hist.axis.Variable(h_npy[1]), storage=bh.storage.Weight())
        new_z = np.concatenate([f.to_numpy()[0] for f in h_fragments])
        h[:] = np.stack([new_z, new_z], axis=-1)
    elif len(h_npy) == 3:
        h_fragments = []
        for iBin in range(len(bins) - 1):
            if dim == "x":
                h_fragments.append(h_in[bins[iBin] : bins[iBin + 1], :] * scales[iBin])
            elif dim == "y":
                h_fragments.append(h_in[:, bins[iBin] : bins[iBin + 1]] * scales[iBin])
        h = hist.Hist(
            hist.axis.Variable(h_npy[1]),
            hist.axis.Variable(h_npy[2]),
            storage=bh.storage.Weight(),
        )
        if dim == "x":
            new_z = np.concatenate([f.to_numpy()[0] for f in h_fragments])
        if dim == "y":
            new_z = np.hstack([f.to_numpy()[0] for f in h_fragments])
        h[:, :] = np.stack([new_z, new_z], axis=-1)
    return h


# function to load files from pickle
def openpkl(infile_name):
    plots = {}
    with open(infile_name, "rb") as openfile:
        while True:
            try:
                plots.update(pickle.load(openfile))
            except EOFError:
                break
    return plots


def plot1d(h, ax, label, color="default", lw=1):
    if color == "default":
        color = default_colors[label]
    if label == "QCD" and lw == 1:
        lw = 3

    y, x = h.to_numpy()
    e = np.sqrt(h.variances())
    x = x[:-1]

    # ax.step(x[:-1],values, label=label, color=color, lw=lw)
    ax.errorbar(
        x, y, yerr=e, label=label, lw=lw, color=color, fmt="", drawstyle="steps-mid"
    )
    ax.set_xlabel(h.axes[0].label)
    ax.set_ylabel("Events")


def plot1d_stacked(hlist, ax, labels, color="midnightblue", lw=1):
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

    ylist = []
    for lbl, h, c in zip(labels, hlist, cmap):
        y, x = h.to_numpy()
        e = np.sqrt(h.variances())
        x = x[:-1]

        if len(ylist) > 0:
            y = y + ylist[len(ylist) - 1]
        ylist.append(y)

        # ax.step(x[:-1],values, label=label, color=color, lw=lw)
        ax.errorbar(
            x, y, yerr=e, label=lbl, lw=lw, color=c, fmt="", drawstyle="steps-mid"
        )
    ax.set_xlabel(hlist[0].axes[0].label)
    ax.set_ylabel("Events")


def plot2d(h, fig, ax, log=False, cmap="RdYlBu"):
    w, x, y = h.to_numpy()
    if log:
        mesh = ax.pcolormesh(x, y, w.T, cmap=cmap, norm=colors.LogNorm())
    else:
        mesh = ax.pcolormesh(x, y, w.T, cmap=cmap)
    ax.set_xlabel(h.axes[0].label)
    ax.set_ylabel(h.axes[1].label)
    fig.colorbar(mesh)


def bin_midpoints(bins):
    midpoints = []
    for i in range(len(bins) - 1):
        midpoints.append((bins[i] + bins[i + 1]) / 2)
    return np.array(midpoints)


def plot_ratio(hlist, labels=None, plot_label=None, xlim="default", log=True):
    # Set up variables for the stacked histogram
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

    if labels is None:
        labels = [None] * len(hlist)

    cmap = plt.cm.jet(np.linspace(0, 1, len(hlist)))
    for c, h, l in zip(cmap, hlist, labels):
        y, x = h.to_numpy()
        x_mid = h.axes.centers[0]
        y_errs = np.sqrt(h.variances())
        ax1.stairs(h.values(), x, color=c, label=l)
        ax1.errorbar(
            x_mid,
            y,
            yerr=y_errs,
            color=c,
            fmt="",
            drawstyle="default",
            linestyle="",
        )

    # Set parameters that will be used to make the plots prettier
    if log:
        ax1.set_yscale("log")
    if type(xlim) is not str:
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

    ax1.set_ylabel("Events", y=1, ha="right")

    ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # calculate the ratio, with poisson errors, and plot them
    for i, (c, h) in enumerate(zip(cmap, hlist)):
        if i == 0:
            continue
        ratio = np.divide(
            h.values(),
            hlist[0].values(),
            out=np.ones_like(h.values()),
            where=hlist[0].values() != 0,
        )
        ratio_err = hist.intervals.ratio_uncertainty(h.values(), hlist[0].values())
        ax2.errorbar(
            hlist[0].axes.centers[0],
            ratio,
            yerr=ratio_err,
            color=c,
            fmt="o",
            linestyle="none",
        )

    ax2.axhline(1, ls="--", color="gray")
    ax2.set_ylabel("Ratio", y=1, ha="right")

    if plot_label is None:
        plot_label = hlist[0].axes[0].label
        if plot_label == "Axis 0":
            plot_label = None
    ax1.legend(loc="best")
    ax2.set_xlabel(plot_label, y=1)

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
        h1 = plots[sample1][plot_label.replace("A_", r + "_")]
        h2 = plots[sample2][plot_label.replace("A_", r + "_")]

        if density:
            h1 = h1 / h1.sum().value
            h2 = h2 / h2.sum().value

        y1, x1 = h1.to_numpy()
        x1 = x1[:-1]
        y2, x2 = h2.to_numpy()
        x2 = x2[:-1]

        xmin1 = np.argwhere(y1 > 0)[0] if any(y1 > 0) else [len(x1)]
        xmin2 = np.argwhere(y2 > 0)[0] if any(y2 > 0) else [len(x1)]
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
            ax1.step(x1, y1, color="midnightblue", label=sample1, where="mid")
        else:
            ax1.step(x1, y1, color="midnightblue", where="mid")
        ax1.errorbar(
            x1, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle="steps-mid"
        )

        y2_errs = np.sqrt(h2.variances())
        y2_errs = y2_errs[xmin : xmax + 1]
        if i == 0:
            ax1.step(x2, y2, color="maroon", label=sample2, where="mid")
        else:
            ax1.step(x2, y2, color="maroon", where="mid")
        ax1.errorbar(
            x2, y2, yerr=y2_errs, color="blue".upper(), fmt="", drawstyle="steps-mid"
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
                ax.step(
                    x, y, color=default_colors.get(sample), label=label, where="mid"
                )
            else:
                ax.step(x, y, color=default_colors.get(sample), where="mid")

        ax.axvline(Xs[0][0], ls="--", color="black")

    if log:
        ax.set_yscale("log")

    ax.set_xticks(mids)
    ax.set_xticklabels(list(regions))

    ax.set_ylabel("Events", y=1, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    return fig, ax


def slice_hist2d(hist, regions_list, slice_var="y"):
    """
    Inputs:
        hist: 2d Hist histogram.
        regions_list: list of regions using Hist slicing. e.g. [[10j,20j],[20j,30j],...]
        slice_var: 'x' or 'y', which dimensions to slice in
    Returns:
        A list of Hist histograms.
    """
    hist_list = []
    for regions in regions_list:
        if slice_var == "y":
            h = hist[:, regions[0] : regions[1] : sum]
        elif slice_var == "x":
            h = hist[regions[0] : regions[1] : sum, :]
        hist_list.append(h)
    return hist_list


def plot_sliced_hist2d(hist, regions_list, slice_var="y", labels=None):
    """
    Takes a 2d histogram, slices it in different regions, and plots the
    regions stacked.
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
    hist_list = slice_hist2d(hist, regions_list, slice_var)
    hist_list_err = [np.sqrt(h.variances()) for h in hist_list]
    cmap = plt.cm.jet(np.linspace(0, 1, len(hist_list)))

    fig = plt.figure()
    ax = fig.subplots()
    hep.histplot(
        hist_list,
        yerr=hist_list_err,
        stack=True,
        histtype="fill",
        label=labels,
        color=cmap,
        ax=ax,
    )
    ax.legend(
        fontsize=14,
        framealpha=1,
        facecolor="white",
        shadow=True,
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )
    ax.set_yscale("log")

    return fig, ax


def ABCD_4regions(
    hist_abcd,
    xregions,
    yregions,
    sum_var="x",
):
    if sum_var == "x":
        A = hist_abcd[xregions[0] : xregions[1] : sum, yregions[0] : yregions[1]]
        B = hist_abcd[xregions[0] : xregions[1] : sum, yregions[1] : yregions[2]]
        C = hist_abcd[xregions[1] : xregions[2] : sum, yregions[0] : yregions[1]]
        SR = hist_abcd[xregions[1] : xregions[2] : sum, yregions[1] : yregions[2]]
        SR_exp = B * C.sum().value / A.sum().value
    elif sum_var == "y":
        A = hist_abcd[xregions[0] : xregions[1], yregions[0] : yregions[1] : sum]
        B = hist_abcd[xregions[0] : xregions[1], yregions[1] : yregions[2] : sum]
        C = hist_abcd[xregions[1] : xregions[2], yregions[0] : yregions[1] : sum]
        SR = hist_abcd[xregions[1] : xregions[2], yregions[1] : yregions[2] : sum]
        SR_exp = C * B.sum().value / A.sum().value

    return SR, SR_exp


def ABCD_6regions(hist_abcd, xregions, yregions, sum_var="x"):
    if sum_var == "x":
        if len(xregions) == 3:
            A = hist_abcd[xregions[0] : xregions[1] : sum, yregions[0] : yregions[1]]
            B = hist_abcd[xregions[0] : xregions[1] : sum, yregions[1] : yregions[2]]
            C = hist_abcd[xregions[0] : xregions[1] : sum, yregions[2] : yregions[3]]
            D = hist_abcd[xregions[1] : xregions[2] : sum, yregions[0] : yregions[1]]
            E = hist_abcd[xregions[1] : xregions[2] : sum, yregions[1] : yregions[2]]
            SR = hist_abcd[xregions[1] : xregions[2] : sum, yregions[2] : yregions[3]]
        elif len(xregions) == 4:
            A = hist_abcd[xregions[0] : xregions[1] : sum, yregions[0] : yregions[1]]
            B = hist_abcd[xregions[1] : xregions[2] : sum, yregions[0] : yregions[1]]
            C = hist_abcd[xregions[2] : xregions[3] : sum, yregions[0] : yregions[1]]
            D = hist_abcd[xregions[0] : xregions[1] : sum, yregions[1] : yregions[2]]
            E = hist_abcd[xregions[1] : xregions[2] : sum, yregions[1] : yregions[2]]
            SR = hist_abcd[xregions[2] : xregions[3] : sum, yregions[1] : yregions[2]]
        SR_exp = (
            E
            * E.sum().value
            * C.sum().value
            * A.sum().value
            / (B.sum().value ** 2 * D.sum().value)
        )
    elif sum_var == "y":
        if len(xregions) == 3:
            A = hist_abcd[xregions[0] : xregions[1], yregions[0] : yregions[1] : sum]
            B = hist_abcd[xregions[0] : xregions[1], yregions[1] : yregions[2] : sum]
            C = hist_abcd[xregions[0] : xregions[1], yregions[2] : yregions[3] : sum]
            D = hist_abcd[xregions[1] : xregions[2], yregions[0] : yregions[1] : sum]
            E = hist_abcd[xregions[1] : xregions[2], yregions[1] : yregions[2] : sum]
            SR = hist_abcd[xregions[1] : xregions[2], yregions[2] : yregions[3] : sum]
        elif len(xregions) == 4:
            A = hist_abcd[xregions[0] : xregions[1], yregions[0] : yregions[1] : sum]
            B = hist_abcd[xregions[1] : xregions[2], yregions[0] : yregions[1] : sum]
            C = hist_abcd[xregions[2] : xregions[3], yregions[0] : yregions[1] : sum]
            D = hist_abcd[xregions[0] : xregions[1], yregions[1] : yregions[2] : sum]
            E = hist_abcd[xregions[1] : xregions[2], yregions[1] : yregions[2] : sum]
            SR = hist_abcd[xregions[2] : xregions[3], yregions[1] : yregions[2] : sum]
        SR_exp = (
            C
            * E.sum().value ** 2
            * A.sum().value
            / (B.sum().value ** 2 * D.sum().value)
        )

    return SR, SR_exp


def ABCD_9regions(hist_abcd, xregions, yregions, sum_var="x", return_all=False):
    if sum_var == "x":
        A = hist_abcd[xregions[0] : xregions[1] : sum, yregions[0] : yregions[1]]
        B = hist_abcd[xregions[0] : xregions[1] : sum, yregions[1] : yregions[2]]
        C = hist_abcd[xregions[0] : xregions[1] : sum, yregions[2] : yregions[3]]
        D = hist_abcd[xregions[1] : xregions[2] : sum, yregions[0] : yregions[1]]
        E = hist_abcd[xregions[1] : xregions[2] : sum, yregions[1] : yregions[2]]
        F = hist_abcd[xregions[1] : xregions[2] : sum, yregions[2] : yregions[3]]
        G = hist_abcd[xregions[2] : xregions[3] : sum, yregions[0] : yregions[1]]
        H = hist_abcd[xregions[2] : xregions[3] : sum, yregions[1] : yregions[2]]
        SR = hist_abcd[xregions[2] : xregions[3] : sum, yregions[2] : yregions[3]]
        SR_exp = (
            F
            * F.sum().value ** 3
            * (G.sum().value * C.sum().value / A.sum().value)
            * ((H.sum().value / E.sum().value) ** 4)
            * (G.sum().value * F.sum().value / D.sum().value) ** -2
            * (H.sum().value * C.sum().value / B.sum().value) ** -2
        )
    elif sum_var == "y":
        A = hist_abcd[xregions[0] : xregions[1], yregions[0] : yregions[1] : sum]
        B = hist_abcd[xregions[0] : xregions[1], yregions[1] : yregions[2] : sum]
        C = hist_abcd[xregions[0] : xregions[1], yregions[2] : yregions[3] : sum]
        D = hist_abcd[xregions[1] : xregions[2], yregions[0] : yregions[1] : sum]
        E = hist_abcd[xregions[1] : xregions[2], yregions[1] : yregions[2] : sum]
        F = hist_abcd[xregions[1] : xregions[2], yregions[2] : yregions[3] : sum]
        G = hist_abcd[xregions[2] : xregions[3], yregions[0] : yregions[1] : sum]
        H = hist_abcd[xregions[2] : xregions[3], yregions[1] : yregions[2] : sum]
        SR = hist_abcd[xregions[2] : xregions[3], yregions[2] : yregions[3] : sum]
        SR_exp = (
            H
            * H.sum().value ** 3
            * (G.sum().value * C.sum().value / A.sum().value)
            * ((F.sum().value / E.sum().value) ** 4)
            * (G.sum().value * F.sum().value / D.sum().value) ** -2
            * (H.sum().value * C.sum().value / B.sum().value) ** -2
        )

    if return_all:
        return A, B, C, D, E, F, G, H, SR, SR_exp
    else:
        return SR, SR_exp


def ABCD_9regions_errorProp(abcd, xregions, yregions, sum_var="x"):
    """
    Does 9 region ABCD using error propagation of the statistical
    uncerantities of the regions. We need to scale histogram F or H
    by some factor 'alpha' (defined in exp). For example, for F,
    the new variance is:
    variance = F_value**2 * sigma_alpha**2 + alpha**2 * F_variance**2
    """

    A, B, C, D, E, F, G, H, SR, SR_exp = ABCD_9regions(
        abcd, xregions, yregions, sum_var=sum_var, return_all=True
    )

    # define the scaling factor function
    a, b, c, d, e, f, g, h = symbols("A B C D E F G H")
    if sum_var == "x":
        exp = f * h**2 * d**2 * b**2 * g**-1 * c**-1 * a**-1 * e**-4
    elif sum_var == "y":
        exp = h * d**2 * b**2 * f**2 * g**-1 * c**-1 * a**-1 * e**-4

    # defines lists of variables (sympy symbols) and accumulators (hist.sum())
    variables = [a, b, c, d, e, f, g, h]
    accs = [A.sum(), B.sum(), C.sum(), D.sum(), E.sum(), F.sum(), G.sum(), H.sum()]

    # calculate scaling factor by substituting values of the histograms' sums for the sympy symbols
    alpha = exp.copy()
    for var, acc in zip(variables, accs):
        alpha = alpha.subs(var, acc.value)

    # calculate the error on the scaling factor
    variance = 0
    for var, acc in zip(variables, accs):
        der = diff(exp, var)
        var = abs(acc.variance)
        variance += der**2 * var
    for var, acc in zip(variables, accs):
        variance = variance.subs(var, acc.value)
    sigma_alpha = sqrt(variance)

    # define SR_exp and propagate the error on the scaling factor to the bin variances
    if sum_var == "x":
        SR_exp = F.copy()
    elif sum_var == "y":
        SR_exp = H.copy()
    new_var = SR_exp.values() ** 2 * float(sigma_alpha) ** 2 + float(alpha) ** 2 * abs(
        SR_exp.variances()
    )
    SR_exp.view().variance = new_var
    SR_exp.view().value = SR_exp.view().value * float(alpha)

    return SR, SR_exp, alpha, sigma_alpha


def integrate(h, lower, upper):
    i = h[lower:upper].sum()
    return i.value, np.sqrt(i.variance)


def find_nth(string, substring, n):
    if n == 1:
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)


def rebin_piecewise(h_in, bins, histtype="hist"):
    """
    Inputs:
        h : histogram
        bins: list of bins as real numbers
        histtype: one of allowed_histtypes to return

    Returns:
        h_out: a histogram of type 'histtype', rebinned according to desired bins
    """

    # only 1D hists supported for now
    if len(h_in.shape) != 1:
        raise Exception("Only 1D hists supported for now")

    # only hist and bh supported
    allowed_histtypes = ["hist", "bh"]
    if histtype not in allowed_histtypes:
        raise Exception("histtype in not in allowed_histtypes")

    # check that the bins are real numbers
    if any([x.imag != 0 for x in bins]):
        raise Exception("Only pass real-valued bins")

    # split the histogram by the bins
    # and for each bin, calculate total amount of events and variance
    z_vals, z_vars = [], []
    for iBin in range(len(bins) - 1):
        if histtype == "hist":
            bin_lo = bins[iBin] * 1.0j
            bin_hi = bins[iBin + 1] * 1.0j
        elif histtype == "bh":
            bin_lo = bh.loc(bins[iBin])
            bin_hi = bh.loc(bins[iBin + 1])

        h_fragment = h_in[bin_lo:bin_hi]
        z_vals.append(h_fragment.sum().value)
        z_vars.append(h_fragment.sum().variance)

    # fill the histograms
    if histtype == "hist":
        h_out = hist.Hist(
            hist.axis.Variable(bins, label=h_in.axes[0].label, name=h_in.axes[0].name),
            storage=hist.storage.Weight(),
            label=h_in.axes[0].label,
        )
        h_out[:] = np.stack([z_vals, z_vars], axis=-1)

    elif histtype == "bh":
        h_out = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
        h_out[:] = np.stack([z_vals, z_vars], axis=-1)

    return h_out


def linearFit2DHist(h):
    z_values = h.values().flatten()
    x_centers = h.axes[0].centers
    y_centers = h.axes[1].centers
    x_values = np.array([])
    y_values = np.array([])
    for i in range(len(x_centers)):
        x_values = np.concatenate((x_values, np.ones_like(y_centers) * x_centers[i]))
    for _i in range(len(x_centers)):
        y_values = np.concatenate((y_values, y_centers))
    p = np.poly1d(np.polyfit(x_values, y_values, 1, w=z_values, cov=False))
    logging.info("Linear fit result:", p)
    return p


def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, type))
