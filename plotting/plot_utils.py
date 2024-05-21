import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
from collections import defaultdict
import time
import boost_histogram as bh
import hist
import hist.intervals
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sympy
import uproot
from sympy import diff, sqrt, symbols

sys.path.append("..")
from histmaker import fill_utils

default_colors = {
    "mS125": "cyan",
    "mS200": "blue",
    "mS300": "lightseagreen",
    "mS400": "green",
    "mS500": "darkgreen",
    "mS600": "lawngreen",
    "mS700": "goldenrod",
    "mS800": "orange",
    "mS900": "sienna",
    "mS1000": "red",
    "data": "maroon",
    "QCD": "slateblue",
    "MC": "slateblue",
    "TTJets": "midnightblue",
    "VVV": "green",
    "VG": "slategray",
    "VV": "brown",
    "VH": "pink",
    "WJets": "blue",
    "DYJetsToLL": "darktorquoise",
    "ttX": "orange",
    "tt": "red",
    "ST": "gold",
}


def getColor(sample):

    if sample in default_colors.keys():
        return default_colors[sample]

    if "GluGluToSUEP" and "mS" in sample:
        sample = sample[sample.find("mS") + 2 :]
        sample = sample.split("_")[0]
        return default_colors["mS" + sample]

    elif "QCD" in sample:
        return default_colors["QCD"]

    elif "data" in sample.lower():
        return default_colors["data"]

    elif "ttjets" in sample.lower():
        return default_colors["TTJets"]

    elif "VVV" in sample:
        return default_colors["VVV"]

    elif "VG" in sample:
        return default_colors["VG"]

    elif "VV" in sample:
        return default_colors["VV"]

    elif "ST" in sample:
        return default_colors["ST"]

    elif "ttX" in sample:
        return default_colors["ttX"]

    elif "WJetsToLNu" in sample:
        return default_colors["WJets"]

    elif "DYJetsToLL" in sample:
        return default_colors["DYJetsToLL"]

    elif "tt" in sample:
        return default_colors["tt"]

    elif "VH" in sample:
        return default_colors["VH"]

    elif "MC" in sample:
        return default_colors["MC"]

    else:
        return None


def getColors(samples):
    colors = []
    for sample in samples:
        colors.append(getColor(sample))
    return colors


# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    "2018": 59817.406,
    "all": 19497.914 + 16810.813 + 41471.589 + 59817.406,
}

lumis_scouting = {
    "2016apv": 18843.384721292190552,
    "2016": 16705.324242775104523,
    "2017": 35718.640387367889404,
    "2018": 58965.346247952011108,
    "all": 18843.384721292190552
    + 16705.324242775104523
    + 35718.640387367889404
    + 58965.346247952011108,
}


def lumiLabel(year, scouting=False):
    if scouting:
        lumidir = lumis_scouting
    else:
        lumidir = lumis
    if year in ["2017", "2018"]:
        return round(lumidir[year] / 1000, 1)
    elif year == "2016":
        return round((lumidir[year] + lumidir[year + "apv"]) / 1000, 1)
    elif year == "all":
        return round(lumidir[year] / 1000, 1)


def findLumiAndEra(year, auto_lumi, infile_name, scouting):
    if scouting:
        lumidir = lumis_scouting
    else:
        lumidir = lumis

    if auto_lumi and not year:
        # try to figure it out from sample name
        if "20UL16MiniAODv2" in infile_name:
            lumi = lumidir["2016"]
            era = "2016"
        elif "20UL17MiniAODv2" in infile_name:
            lumi = lumidir["2017"]
            era = "2017"
        elif "20UL16MiniAODAPVv2" in infile_name:
            lumi = lumidir["2016apv"]
            era = "2016apv"
        elif "20UL18" in infile_name:
            lumi = lumidir["2018"]
            era = "2018"
        elif any([s in infile_name for s in ["JetHT+Run", "ScoutingPFHT+Run"]]):
            lumi = 1
            era = infile_name.split("Run")[1][0:4]
        else:
            raise Exception(
                "I cannot find luminosity matched to file name: " + infile_name
            )
    elif year and not auto_lumi:
        lumi = lumidir[str(year)]
        era = str(year)
    else:
        raise Exception(
            "Apply lumis automatically OR based on a specific year you pass in. One and only one of those should be passed."
        )

    print(f"Found lumi {lumi} and era {era}")

    return lumi, era


def getHistLists(plotDir, tag, filename, filters=None, file_ext=".root"):
    hists = []
    with open(filename) as file:
        for line in file:
            sample_name = line.strip().split("/")[-1]
            sample_name = sample_name.replace(".root", "")
            result_path = f"{plotDir}{sample_name}_{tag}{file_ext}"
            if filters:
                if not all([filt in sample_name for filt in filters]):
                    continue
            hists.append(result_path)
    return hists


def formatGluGluToSUEPNaming(file):
    file = file.split("/")[-1]
    tokens = file.split("_")

    temp = tokens[2]
    mS = tokens[3]
    mPhi = tokens[4]
    decay = tokens[6]

    if "p" in temp:
        temp = temp.replace("p", ".")
        temp = "T" + str(float(temp[1:]))

    if "." in mS:
        mS = mS[: mS.find(".")]

    if "." in mPhi:
        mPhi = "mPhi" + str(float(mPhi[4:]))

    if "mode" in decay:
        decay = decay[4:]

    name = "_".join([mS, temp, mPhi, decay])
    return name


def formatWHToSUEPNaming(file):
    file = file.split("/")[-1]
    tokens = file.split("_")

    mS = tokens[1]
    mPhi = tokens[2]
    T = tokens[3]
    decay = tokens[4]

    if "." in mS:
        mS = mS[: mS.find(".")]

    T = "T" + str(float(T[1:]))
    mPhi = "mPhi" + str(float(mPhi[4:]))

    if "mode" in decay:
        decay = decay[4:]

    name = "SUEP-WH-" + "_".join([mS, T, mPhi, decay])
    return name


def formatTTHToSUEPNaming(file):
    file = file.split("/")[-1]
    tokens = file.split("_")

    decay = tokens[1]
    mS = tokens[2]
    mPhi = tokens[3]
    T = tokens[4]

    if "." in mS:
        mS = mS[: mS.find(".")]
    mS = mS.replace("MS", "mS")
    mS = mS.replace("M", "mS")

    T = "T" + str(float(T[1:]))

    if "MD" in mPhi:
        mPhi = "mPhi" + str(float(mPhi[2:]))

    if "mode" in decay:
        decay = decay[4:]

    name = "SUEP-ttH-" + "_".join([mS, T, mPhi, decay])
    return name


def getSampleNameAndBin(sample_name):
    """
    From input sample, return a cleaned up sample name (var: bin),
    as well as the bigger sample it might belong to (var: sample) (e.g. data, QCD, TTBkg, STBkg).
    The loader() will use merge the samples with the same name,
    and if by_bin=True, will also load the bins indepndently.
    """

    # if needed, remove the preceding path
    if "/" in sample_name:
        path_name = sample_name
        sample_name = sample_name.split("/")[-1]

    if "QCD_Pt" in sample_name:
        sample = "QCD_Pt"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif "QCD_HT" in sample_name:
        sample = "QCD_HT"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif any([s in sample_name for s in ["TTTo2L2Nu", "TTToSemiLeptonic"]]):
        sample = "tt"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif sample_name.startswith("TTJets_HT"):
        sample = "TTJets_HT"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif sample_name.startswith("TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8"):
        sample = "TTJets_incl"
        bin = "TTJets_incl"

    elif any(
        [
            s in sample_name
            for s in [
                "ttHTobb",
                "ttHToNonbb",
                "TTWJetsToLNu",
                "TTZToQQ",
                "TTWJetsToQQ",
                "TTZToLLNuNu",
                "ttZJets",
            ]
        ]
    ):
        sample = "ttX"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif any([s in sample_name for s in ["ST_t", "ST_tW", "ST_s"]]):
        sample = "ST"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif "WJetsToLNu_HT" in sample_name:
        sample = "WJetsToLNu_HT"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif (
        "WJetsToLNu_Pt" in sample_name
        or "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8" in sample_name
    ):
        # merging of inclusive sample and pT-binned. This is done by LHE_Vpt > 100 selection in ntuplemaker on the inclusive sample,
        # and is accounted for in its normalization via its kr factor.
        sample = "WJetsToLNu_Pt"
        bin = sample_name.split(".root")[0].split("_MatchEWPDG20")[0]
        if "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8" in sample_name:
            bin = "WJetsToLNu_incl"

    elif "DYJetsToLL_LHEFilterPtZ-" in sample_name:
        sample = "DYJetsToLL_LHEFilterPtZ"
        bin = sample_name.split(".root")[0].split("_MatchEWPDG20")[0]

    elif "DYJetsToLL_M" in sample_name:
        sample = "DYJetsToLL_M"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif any(
        [
            s in sample_name
            for s in [
                "WWTo1L1Nu2Q_4f",
                "WWTo2L2Nu",
                "WZTo1L1Nu2Q_4f",
                "WZTo1L3Nu_4f",
                "WZTo2Q2L_mllmin4p0",
                "WZTo3LNu_mllmin4p0",
                "ZZTo2L2Nu",
                "ZZTo2Q2L_mllmin4p0",
                "ZZTo4L_TuneCP5",
            ]
        ]
    ):
        sample = "VV"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif any([s in sample_name for s in ["ZZZ_TuneCP5_13TeV", "WWZ_4F_TuneCP5_13TeV"]]):
        sample = "VVV"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif any([s in sample_name for s in ["WGToLNuG", "ZGToLLG_01J_5f"]]):
        sample = "VG"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif any(
        [
            s in sample_name
            for s in [
                "WminusH_HToBB_WToLNu_M-125",
                "WplusH_HToBB_WToLNu_M-125",
                "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8",
            ]
        ]
    ):
        sample = "VH"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

    elif "JetHT+Run" in sample_name or "ScoutingPFHT" in sample_name:
        sample = "data"
        bin = sample_name.split("-")[0]

    elif sample_name.startswith("ttHpythia"):  # private ttH samples
        sample = formatTTHToSUEPNaming(sample_name)
        bin = None

    elif "GluGluToSUEP" in sample_name:  # ggF samples
        sample = formatGluGluToSUEPNaming(sample_name)
        bin = None

    elif sample_name.startswith("SUEP_mS125.000"):  # this is bad naming
        sample = formatWHToSUEPNaming(sample_name)
        bin = None

    else:
        sample = sample_name.split(".root")[0]
        bin = None

    return sample, bin


def fillSample(this_hists: dict, sample: str, plots: dict, norm: int = 1) -> dict:
    """
    Fill the plots dictionary with the histograms from the current sample.
    plots is expected to have dimensions of {sample: {plot: hist}}.
    this_hists is the dictionary of histograms from the current sample, and
    is expected to have dimensions of {plot: hist}.
    """
    plotsToAdd = this_hists.copy()
    if norm != 1:
        plotsToAdd = fill_utils.apply_normalization(plotsToAdd, norm)

    if sample not in list(plots.keys()):
        plots[sample] = plotsToAdd
    else:
        try:
            for plot in list(plotsToAdd.keys()):
                plots[sample][plot] = plots[sample][plot] + plotsToAdd[plot]
        except KeyError:
            print("WARNING: " + sample + " has a different set of plots")

    return plots


def fillCutflows(
    this_metadata: dict, sample: str, cutflows: dict, norm: int = 1
) -> dict:
    """
    Fill the cutflows dictionary with the cutflows from the current sample.
    cutflows is expected to have dimensions of {sample: {selection: value}}.
    this_metadata is the dictionary of metadata from the current sample, and
    is expected to have dimensions of {selection: value}.
    """
    metaToAdd = {}
    for key in this_metadata.keys():
        if "cutflow" in key:
            metaToAdd[key] = float(this_metadata[key])

    if norm != 1:
        metaToAdd = fill_utils.apply_normalization(metaToAdd, norm)

    if sample not in list(cutflows.keys()):
        cutflows[sample] = metaToAdd
    else:
        try:
            for key in list(metaToAdd.keys()):
                cutflows[sample][key] += metaToAdd[key]
        except KeyError:
            print("WARNING: " + sample + " has a different set of cutflows.")

    return cutflows


def getLumi(era: str, scouting: bool) -> float:
    if scouting:
        lumidir = lumis_scouting
    else:
        lumidir = lumis
    return lumidir[era]


def loader(
    infile_names,
    year=None,  # once everyone starts making histograms with metadata, these can be dropped
    auto_lumi=True,  # once everyone starts making histograms with metadata, these can be dropped
    scouting=False,  # once everyone starts making histograms with metadata, these can be dropped
    by_bin=False,
    by_year=False,
    load_cutflows=False,
    only_cutflows=False,
    verbose=False,
):
    """
    Load histograms (or cutflows) from input files and perform various operations such as normalization, and grouping by sample, sample bin, and sample year.

    Parameters:
    - infile_names (list): List of input file names.
    - year (int, optional): Year of the data. Default is None.
    - auto_lumi (bool, optional): Flag to automatically determine the luminosity based on the year and sample name. Default is True.
    - scouting (bool, optional): Flag to indicate whether the data is from scouting, used for the lumi. Default is False.
    - by_bin (bool, optional): Flag to group histograms by bin. Default is False.
    - by_year (bool, optional): Flag to group histograms by year. Default is True.
    - load_cutflows (bool, optional): Flag to load cutflows along side histograms. Default is False.
    - only_cutflows (bool, optional): Flag to load cutflows instead of histograms. Default is False.

    Returns:
    - output (dict): Dictionary containing the loaded histograms (or cutflows) grouped by sample, bin, and year.
    """
    output = {}
    for infile_name in infile_names:
        if verbose:
            print("Loading", infile_name)

        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            continue
        elif ".root" not in infile_name and ".pkl" not in infile_name:
            continue

        file_hists, file_metadata = openHistFile(infile_name)
        norm = 1

        # finds era
        if (
            file_metadata
            and ("era" in file_metadata.keys())
            and ("lumi" in file_metadata.keys())
        ):
            era = file_metadata["era"]
            lumi = float(file_metadata["lumi"])
        else:
            # for older histograms, we need to scale by lumi, and find era via the filename
            # once everyone starts making histograms with metadata, this can be dropped
            lumi, era = findLumiAndEra(year, auto_lumi, infile_name, scouting)
            norm *= lumi
            if verbose:
                print("Applying lumi", lumi)
        if verbose:
            print("Found era", era)

        # get the normalization factor for SUEP samples
        # xsec is already apply in make_hists.py for non SUEP samples
        if "signal" in file_metadata.keys():
            if int(file_metadata["signal"]):
                xsec = float(file_metadata["xsec"])
                if verbose:
                    print("Applying xsec", xsec)
                    print("Applying lumi", lumi)
                norm *= xsec * lumi
        elif "SUEP" in infile_name.split("/")[-1]:
            # for older histograms, we didn't have metadata, so we need to find xsec via string manipulation
            sample_name = infile_name.split("/")[-1].split("13TeV")[0] + "13TeV-pythia8"
            xsec = fill_utils.getXSection(sample_name, year=era)
            if verbose:
                print("Applying xsec", xsec)
            norm *= xsec

        # get the sample name and the bin name
        # e.g. for QCD_Pt_15to30_.. the sample is QCD_Pt and the bin is QCD_Pt_15to30
        sample, bin = getSampleNameAndBin(infile_name)
        if verbose:
            print("Found sample", sample)
            if by_bin:
                print("and bin", bin)

        samplesToAdd = [sample]
        if by_bin and (bin is not None) and (bin != sample):
            samplesToAdd.append(bin)
        if by_year:
            samplesToAdd.append("_".join([sample, era]))
            if by_bin and (bin is not None):
                samplesToAdd.append("_".join([bin, era]))

        for s in samplesToAdd:
            if only_cutflows or load_cutflows:
                output = fillCutflows(file_metadata, s, output, norm)
            if not only_cutflows:
                output = fillSample(file_hists, s, output, norm)

    return output


def openHistFile(infile_name):
    if infile_name.endswith(".root"):
        hists, metadata = openroot(infile_name)
    elif infile_name.endswith(".pkl"):
        hists, metadata = openpickle(infile_name)
    return hists, metadata


def combineSamples(plots: dict, samples: list, new_tag: str) -> dict:
    plots[new_tag] = {}
    for key in plots[samples[0]].keys():
        for i, sample in enumerate(samples):
            if i == 0:
                plots[new_tag][key] = plots[sample][key].copy()
            else:
                try:
                    plots[new_tag][key] += plots[sample][key].copy()
                except KeyError:
                    print(
                        f"WARNING: couldn't merge histrogram {key} for sample {sample}. Skipping."
                    )
    return plots


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
        shutil.copyfile(os.environ['X509_USER_PROXY'] + proxy_base, proxy_copy)

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
def openpickle(infile_name):
    _plots = {}
    _metadata = {}
    with open(infile_name, "rb") as openfile:
        while True:
            try:
                input = pickle.load(openfile)
                _plots.update(input["hists"])
                _metadata.update(input["metadata"])
            except EOFError:
                break
    return _plots, _metadata


def openroot(infile_name):
    _plots = {}
    _metadata = {}
    _infile = uproot.open(infile_name)
    for k in _infile.keys():
        if "metadata" == k.split(";")[0]:
            for kk in _infile[k].keys():
                _metadata[kk.split(";")[0]] = _infile[k][kk].title()
        elif "metadata" not in k:
            _plots[k.split(";")[0]] = _infile[k].to_hist()
    return _plots, _metadata


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


def plot_ratio(
    hlist,
    labels=None,
    systs=None,
    density=False,
    cmap=None,
    plot_label=None,
    xlim="default",
    log=True,
):
    """
    Plots ratio of a list of Hist histograms, the ratio is wrt to the first one in the list.
    The errors in the ratio are taken to be independent between histograms.
    """

    # Set up variables for the stacked histogram
    fig = plt.figure(figsize=(14, 12))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

    if density:
        for i, h in enumerate(hlist):
            hlist[i] = h.copy() / h.sum().value

    if labels is None:
        labels = [None] * len(hlist)
    if cmap is None:
        cmap = plt.cm.jet(np.linspace(0, 1, len(hlist)))
    for c, h, l in zip(cmap, hlist, labels):
        y, x = h.to_numpy()
        x_mid = h.axes.centers[0]
        y_errs = np.sqrt(h.variances())
        ax1.stairs(y, x, color=c, label=l)
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

    # calculate the ratio, with error propagation, and plot them
    for i, (c, h) in enumerate(zip(cmap, hlist)):
        if i == 0:
            continue
        ratio = np.divide(
            h.values(),
            hlist[0].values(),
            out=np.ones_like(h.values()),
            where=hlist[0].values() != 0,
        )
        ratio_err = np.where(
            hlist[0].values() > 0,
            np.sqrt(
                (hlist[0].values() ** -2) * (h.variances())
                + (h.values() ** 2 * hlist[0].values() ** -4) * (hlist[0].variances())
            ),
            0,
        )
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
        # add to legend
        ax1.plot([0, 0], color="gray", label="Systematics")

    if plot_label is None:
        plot_label = hlist[0].axes[0].label
        if plot_label == "Axis 0":
            plot_label = None
    ax1.legend(loc="best", fontsize="xx-small")
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


def plot_sliced_hist2d(
    hist, regions_list, stack=False, density=False, slice_var="y", labels=None
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
    hist_list = slice_hist2d(hist, regions_list, slice_var)
    cmap = plt.cm.jet(np.linspace(0, 1, len(hist_list)))

    if stack:
        histtype = "fill"
    else:
        histtype = "step"

    fig = plt.figure()
    ax = fig.subplots()
    hep.histplot(
        hist_list,
        yerr=True,
        stack=stack,
        histtype=histtype,
        density=density,
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


def make_ABCD_9regions(hist_abcd, xregions, yregions, sum_var="X"):
    if sum_var == "x":
        A = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        B = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        C = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[2][0] : yregions[2][1]
        ]
        D = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        E = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        F = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[2][0] : yregions[2][1]
        ]
        G = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        H = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        SR = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[2][0] : yregions[2][1]
        ]

    elif sum_var == "y":
        A = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        B = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        C = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[2][0] : yregions[2][1]
        ]
        D = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        E = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        F = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[2][0] : yregions[2][1]
        ]
        G = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        H = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        SR = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[2][0] : yregions[2][1]
        ]

    return A, B, C, D, E, F, G, H, SR


def ABCD_9regions_errorProp(
    abcd, xregions, yregions, sum_var="x", approx=True, new_bins=None
):
    """
    Does 9 region ABCD using error propagation of the statistical uncertanties of the regions.
    """

    if sum_var == "y":
        raise Exception("sum_var='y' not implemented yet")

    A, B, C, D, E, F, G, H, SR = make_ABCD_9regions(
        abcd, xregions, yregions, sum_var=sum_var
    )
    SR_exp = SR.copy()

    # we need to rebin here in the case appox=True
    if new_bins:
        if sum_var == "x":
            F = rebin_piecewise(F, new_bins)
            C = rebin_piecewise(C, new_bins)
            SR = rebin_piecewise(SR, new_bins)
            SR_exp = rebin_piecewise(SR_exp, new_bins)

    preds, preds_err = [], []
    for i in range(len(F.values())):
        # this is needed in order to do error propagation correctly
        F_bin = F[i]
        C_bin = C[i]
        F_other = F.copy()
        F_other[i] = hist.accumulators.WeightedSum()

        # define the scaling factor function
        a, b, c, c_bin, d, e, f_bin, f_other, g, h = symbols(
            "A B C C_bin D E F_bin F_other G H"
        )
        if sum_var == "x" and approx:
            exp = (
                f_bin
                * (f_other + f_bin)
                * h**2
                * d**2
                * b**2
                * g**-1
                * c**-1
                * a**-1
                * e**-4
            )
        elif sum_var == "x" and not approx:
            exp = f_bin**2 * h**2 * d**2 * b**2 * g**-1 * c_bin**-1 * a**-1 * e**-4
        elif sum_var == "y":
            pass

        # defines lists of variables (sympy symbols) and accumulators (hist.sum())
        variables = [a, b, c, c_bin, d, e, f_bin, f_other, g, h]
        accs = [
            A.sum(),
            B.sum(),
            C.sum(),
            C_bin,
            D.sum(),
            E.sum(),
            F_bin,
            F_other.sum(),
            G.sum(),
            H.sum(),
        ]

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
        sigma_alpha = variance

        if type(alpha) != sympy.core.numbers.Float or alpha <= 0:
            alpha = 0

        preds.append(alpha)
        preds_err.append(sigma_alpha)

    SR_exp.view().variance = preds_err
    SR_exp.view().value = preds

    return SR, SR_exp


def ABCD_9regions_yield(abcd, xregions, yregions):
    A, B, C, D, E, F, G, H, SR, _ = ABCD_9regions(
        abcd, xregions, yregions, sum_var="x", return_all=True
    )

    A = A.sum().value
    B = B.sum().value
    C = C.sum().value
    D = D.sum().value
    E = E.sum().value
    F = F.sum().value
    G = G.sum().value
    H = H.sum().value
    SR = SR.sum().value
    tot = A + B + C + D + E + F + G + H + SR

    SR_exp = (F**2) * (H**2) * (D**2) * (B**2) * (G**-1) * (C**-1) * (A**-1) * (E**-4)
    sigma_SR_exp = (
        np.sqrt(
            4 * (F**-1)
            + 4 * (H**-1)
            + 4 * (D**-1)
            + 4 * (B**-1)
            + (G**-1)
            + (C**-1)
            + (A**-1)
            + 16 * (E**-1)
        )
        * SR_exp
    )

    return SR, SR_exp, sigma_SR_exp


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


def poly_fit_hist2d(h, deg=1):
    z_values = h.values().flatten()
    x_centers = h.axes[0].centers
    y_centers = h.axes[1].centers
    x_values = np.array([])
    y_values = np.array([])
    for i in range(len(x_centers)):
        x_values = np.concatenate((x_values, np.ones_like(y_centers) * x_centers[i]))
    for _i in range(len(x_centers)):
        y_values = np.concatenate((y_values, y_centers))
    p = np.poly1d(np.polyfit(x_values, y_values, deg, w=z_values, cov=False))
    logging.info("Linear fit result:", p)
    return p    


def hist_mean(hist):
    """
    Calculates the mean of a 1-dimensional Hist histogram.
    """
    bin_values = hist.values()
    bin_centers = hist.axes[0].centers
    mean = np.average(bin_centers, weights=bin_values)
    return mean


def hist_std_dev(hist, axis=0):
    """
    Calculates the standard deviation of a 1-dimensional Hist histogram.
    """
    bin_values = hist.values()
    bin_centers = hist.axes[0].centers
    mean = hist_mean(hist)

    # Calculate the sum of squared differences from the mean
    squared_diff_sum = np.sum(bin_values * (bin_centers - mean) ** 2)

    # Calculate the standard deviation
    standard_deviation = np.sqrt(squared_diff_sum / np.sum(bin_values))

    return standard_deviation


def hist2d_correlation(h):
    """
    Calculates Pearson Coefficient from a 2-dimensional Hist histogram.
    """

    coeff = 0

    assert len(h.axes) == 2

    xvals = h.axes[0].centers
    yvals = h.axes[1].centers
    zvals = h.values()

    xmean = hist_mean(h[:, sum])
    ymean = hist_mean(h[sum, :])
    xdev = hist_std_dev(h[:, sum])
    ydev = hist_std_dev(h[sum, :])

    if xdev == 0 or ydev == 0:
        return

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            coeff += (xvals[i] - xmean) * (yvals[j] - ymean) * zvals[i, j]

    coeff /= xdev * ydev * h.sum().value
    error = np.sqrt((1 - coeff**2) / (h.sum().value - 2))

    return coeff, error


def sf(value, error):
    # Calculate the number of significant figures based on the error
    significant_figures = round(-math.log10(error)) + 1

    # Round the value and error to the determined significant figures
    rounded_value = round(value, significant_figures)
    rounded_error = round(error, significant_figures)

    return rounded_value, rounded_error


def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, type))


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
        rotation=45,
        fontsize=10,
    )

    return fig, ax


def make_n1_plots(
    plots: dict,
    cutflows: dict,
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

    n1_plots = [
        k for k in plots[allSamples[0]].keys() if k[:-1].endswith(tag + "_N-1")
    ]

    cuts = [k for k in cutflows[allSamples[0]].keys() if k.endswith(tag)]
    for cut in cuts:
        cut_bits = cut.split("_")
        cut_val = float(cut_bits[-2])

    samples_color = plt.cm.rainbow(np.linspace(0, 1, len(samples)))
    for p in n1_plots:

        var = p[:-1].replace("_" + tag + "_N-1", "")
        cut_val = None
        for cut in cuts:
            if var in cut:
                cut_bits = cut.split("_")
                cut_val = float(cut_bits[-2])
                cut_op = str(cut_bits[-3])
                break

        if cut_val is None:
            print("Could not find a cutflow for", var)

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
                linewidth=3,
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
        ax.legend(fontsize="xx-small", loc=(1.05, 0))
        figs.append(fig)

    return figs
