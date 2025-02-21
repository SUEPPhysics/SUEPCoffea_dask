import os
import gc
import uproot
import pickle
import sys
import hist

sys.path.append("../")
from data import data_utils

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
    file = file.replace(".root", "")
    file = file.replace(".pkl", "")
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

    elif "WJetsToLNu_Pt" in sample_name:
        sample = "WJetsToLNu"
        bin = sample_name.split(".root")[0].split("_MatchEWPDG20")[0]

    elif "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8" in sample_name:
        sample = "WJetsToLNu"
        bin = None

    elif "DYJetsToLL_LHEFilterPtZ-" in sample_name:
        sample = "DYJetsToLL"
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

    elif "EGamma+Run" in sample_name:
        sample = "EGamma"
        bin = sample_name.split("-")[0]

    elif "SingleElectron+Run" in sample_name:
        sample = "SingleElectron"
        bin = sample_name.split("-")[0]

    elif "SinglePhoton+Run" in sample_name:
        sample = "SinglePhoton"
        bin = sample_name.split("-")[0]

    elif "SingleMuon+Run" in sample_name:
        sample = "SingleMuon"
        bin = sample_name.split("-")[0]

    elif any(
        [
            s in sample_name
            for s in ["WGammaToJJGamma_TuneCP5_13TeV", "ZGammaToJJGamma_TuneCP5_13TeV"]
        ]
    ):
        sample = "VGammaToJJGamma"
        bin = None

    elif sample_name.startswith("ttHpythia"):  # private ttH samples
        sample = formatTTHToSUEPNaming(sample_name)
        bin = None

    elif "GluGluToSUEP" in sample_name:  # ggF samples
        sample = formatGluGluToSUEPNaming(sample_name)
        bin = None

    elif sample_name.startswith("SUEP_mS125.000"):  # this is bad naming
        sample = formatWHToSUEPNaming(sample_name)
        bin = None

    elif sample_name.startswith("GJets_HT-"):
        sample = "GJets"
        bin = sample_name.split(".root")[0].split("_Tune")[0]

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
        plotsToAdd = data_utils.apply_normalization(plotsToAdd, norm)

    if sample not in list(plots.keys()):
        plots[sample] = plotsToAdd
    else:
        for plot in list(plotsToAdd.keys()):
            try:
                plots[sample][plot] = plots[sample][plot] + plotsToAdd[plot]
            except ValueError:
                print(f"WARNING: could not merge histogram {plot} in sample {sample}.")
            except KeyError:
                print(f"WARNING: could not find histogram {plot} in sample {sample}.")

    del plotsToAdd
    gc.collect()

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
        metaToAdd = data_utils.apply_normalization(metaToAdd, norm)

    if sample not in list(cutflows.keys()):
        cutflows[sample] = metaToAdd
    else:
        for key in list(metaToAdd.keys()):
            try:
                cutflows[sample][key] += metaToAdd[key]
            except KeyError:
                print(f"WARNING: could not find cutflow {key} in sample {sample}.")

    return cutflows

def loader(
    infile_names,
    by_bin=True,
    by_year=True,
    load_cutflows=True,
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
    hists, cutflows = {}, {}
    nFailed = 0
    for infile_name in infile_names:
        if verbose:
            print("Loading", infile_name)

        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            nFailed += 1
            continue
        elif ".root" not in infile_name and ".pkl" not in infile_name:
            nFailed += 1
            continue

        file_hists, file_metadata = openHistFile(infile_name)
        norm = 1

        # finds era
        era = file_metadata["era"]
        lumi = float(file_metadata["lumi"])
        if verbose:
            print("\tFound era", era)

        # get the normalization factor for SUEP samples
        # xsec is already apply in make_hists.py for non SUEP samples
        if "signal" in file_metadata.keys():
            if int(bool(file_metadata["signal"])):
                xsec = float(file_metadata["xsec"])
                if verbose:
                    print("\tApplying xsec", xsec)
                    print("\tApplying lumi", lumi)
                norm *= xsec * lumi

        # get the sample name and the bin name
        # e.g. for QCD_Pt_15to30_.. the sample is QCD_Pt and the bin is QCD_Pt_15to30
        sample, bin = getSampleNameAndBin(infile_name)
        if verbose:
            print("\tFound sample", sample)
            if by_bin:
                print("\tFound bin", bin)

        samplesToAdd = [sample]
        if by_bin and (bin is not None) and (bin != sample):
            samplesToAdd.append(bin)
        if by_year:
            samplesToAdd.append("_".join([sample, era]))
            if by_bin and (bin is not None):
                samplesToAdd.append("_".join([bin, era]))

        for s in samplesToAdd:
            output[s] = {}
            if only_cutflows or load_cutflows:
                cutflows = fillCutflows(file_metadata, s, cutflows, norm)
            if not only_cutflows:
                hists = fillSample(file_hists, s, hists, norm)
            output[s].update(hists.get(s, {}))
            output[s].update(cutflows.get(s, {}))

        del file_hists, file_metadata, samplesToAdd
        gc.collect()

        if verbose:
            print("\tFinished loading sample")

    if nFailed:
        print(f"WARNING: {nFailed} files failed to load")
    print("Finished loading all files")
    return output


def openHistFile(infile_name):
    if infile_name.endswith(".root"):
        hists, metadata = openroot(infile_name)
    elif infile_name.endswith(".pkl"):
        hists, metadata = openpickle(infile_name)
    return hists, metadata

def openpickle(infile_name):
    _plots = {}
    _metadata = {}
    with open(infile_name, "rb") as openfile:
        while True:
            try:
                input = pickle.load(openfile)
                _plots.update(input["hists"].copy())
                _metadata.update(input["metadata"].copy())
            except EOFError:
                break
    del input
    gc.collect()
    return _plots, _metadata

def openroot(infile_name):
    _plots = {}
    _metadata = {}
    with uproot.open(infile_name) as _infile:
        for k in _infile.keys():
            if "metadata" == k.split(";")[0]:
                for kk in _infile[k].keys():
                    _metadata[kk.split(";")[0]] = _infile[k][kk].title()
            elif "metadata" not in k:
                _plots[k.split(";")[0]] = _infile[k].to_hist()
    gc.collect()
    return _plots, _metadata

def getHistList(plotDir, tag, filename, filters=None, file_ext=".root"):
    hists = []
    with open(filename) as file:
        for line in file:
            sample_name = line.strip().split("/")[-1]
            sample_name = sample_name.replace(".root", "")
            result_path = f"{plotDir}/{tag}/{sample_name}{file_ext}"
            if filters:
                if not all([filt in sample_name for filt in filters]):
                    continue
            hists.append(result_path)
    return hists

def combineSamples(plots: dict, samples: list) -> dict:
    out = {}
    missing = [s not in plots.keys() for s in samples]  
    missing_samples = [s for s, m in zip(samples, missing) if m]
    if missing_samples:
        print("WARNING: not all samples are in the plots dictionary: " + ", ".join(missing_samples))

        return out
    for key in plots[samples[0]].keys():
        for i, sample in enumerate(samples):
            h = plots[sample].get(key, None)
            htype = type(h)
            if htype == hist.hist.Hist:  # histograms
                if i == 0:
                    out[key] = h.copy()
                else:
                    try:
                        out[key] += h.copy()
                    except (ValueError, KeyError) as e:
                        print(
                            f"WARNING: couldn't merge histrogram {key} for sample {sample}. Skipping. (Error: {e})"
                        )
            elif htype == float or htype == int:  # cutflows
                if i == 0:
                    out[key] = h
                else:
                    try:
                        out[key] += h
                    except (ValueError, KeyError) as e:
                        print(
                            f"WARNING: couldn't merge cutflow {key} for sample {sample}. Skipping. (Error: {e})"
                        )
            else:
                print(f"WARNING: unknown type for {key} in sample {sample}: {htype}")

    return out