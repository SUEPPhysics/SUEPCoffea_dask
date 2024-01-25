import argparse
import glob
from rich.pretty import pprint
from typing import Dict, List, Optional
from hist import Hist
import uproot
import ROOT
import plot_utils
import fill_utils


def combine_histograms(
    h_list: List[ROOT.TH1D], blind: Optional[bool] = True
) -> ROOT.TH1D:
    n_bins_tot = sum([h.GetNbinsX() for h in h_list])
    entries_tot = sum([h.GetEntries() for h in h_list])
    h_total = ROOT.TH1D("h_total", "h_total", n_bins_tot, 0, n_bins_tot)
    offset = 0
    for h in h_list:
        for j_bin in range(h.GetNbinsX()):
            h_total.SetBinContent(offset + j_bin + 1, h.GetBinContent(j_bin + 1))
            h_total.SetBinError(offset + j_bin + 1, h.GetBinError(j_bin + 1))
            if h_total.GetBinContent(offset + j_bin + 1) <= 0:
                h_total.SetBinContent(offset + j_bin + 1, 1e-8)
                h_total.SetBinError(offset + j_bin + 1, 1e-8)
        offset += h.GetNbinsX()
    h_total.SetEntries(entries_tot)
    return h_total


def loader(lumi: float, tag: str, verbosity: int) -> Dict[str, Dict[str, Hist]]:
    print(f"Loading histograms for {tag}")
    # input .pkl files and select the interesting ones
    offline_names = []
    file_dir = f"./{tag}_output_histograms/"
    search_string = "*.pkl"
    offline_names = glob.glob(file_dir + search_string)

    # select signal files
    signal_files = [
        f for f in offline_names if ("SUEP" in f) and ("histograms.pkl" in f)
    ]
    signal_files.sort()
    if verbosity > 0:
        print("Input signal files:")
        pprint(signal_files)
    # select bkg files
    offline_files_normalized = [f for f in offline_names if ("normalized.pkl" in f)]
    offline_files_other = [
        f for f in offline_names if ("pythia8" in f) and ("histograms.pkl" in f)
    ]
    bkg_files = offline_files_normalized + offline_files_other
    bkg_files.sort()
    if verbosity > 0:
        print("Input bkg files:")
        pprint(bkg_files)
    # select data files
    data_files = [
        f for f in offline_names if ("DoubleMuon" in f) and ("histograms.pkl" in f)
    ]
    data_files.sort()
    if verbosity > 0:
        print("Input data files:")
        pprint(data_files)

    # Define a set of strings that should be present in the names of each dataset
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

    # Load the files
    plots_SUEP_2018 = plot_utils.loader(signal_files, year=2018, custom_lumi=lumi)
    plots_2018 = plot_utils.loader(bkg_files, year=2018, custom_lumi=lumi)
    plots_data = plot_utils.loader(data_files, year=2018, is_data=True)
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

    # Combine DYNJetsToLL with DYLowMass_LO
    dy_lo_all = {}
    for plt_i in plots["DYLowMass_LO_2018"].keys():
        dy_lo_all[plt_i] = (
            plots["DYLowMass_LO_2018"][plt_i]
            + plots["DYNJetsToLL_2018"][plt_i]
            + plots["DY0JetsToLL_2018"][plt_i]
        )
    plots["DY_LO_all_2018"] = dy_lo_all

    # Combine DYJetsToLL_NLO with DYLowMass_NLO
    dy_nlo_all = {}
    for plt_i in plots["DYLowMass_NLO_2018"].keys():
        dy_nlo_all[plt_i] = (
            plots["DYLowMass_NLO_2018"][plt_i] + plots["DYJetsToLL_NLO_2018"][plt_i]
        )
    plots["DY_NLO_all_2018"] = dy_nlo_all

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

    return plots


def dump_plots(plots: Dict[str, Dict[str, Hist]], tag: str, verbosity: int) -> None:
    print(f"Converting and saving histograms for {tag}")
    processes = [
        "SUEP-m125-darkPhoHad_2018",
        "DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD_histograms_2018",
        "VVV_2018",
        "ttZJets_2018",
        "WJets_all_2018",
        "ST_2018",
        "VV_2018",
        "TTJets_2018",
        "DY_LO_all_2018",
        "DY_NLO_all_2018",
        "QCD_Pt_MuEnriched_2018",
    ]
    histograms = {
        "nMuon": [
            "nMuon",
            slice(None),
        ],
        "muon_pt": [
            "muon_pt",
            slice(None),
        ],
        "muon_eta": [
            "muon_eta",
            slice(None),
        ],
        "muon_phi": [
            "muon_phi",
            slice(None),
        ],
    }
    with uproot.recreate(f"temp.root") as file:
        for p in processes:
            if p not in plots.keys():
                continue
            name = "data_obs" if "DoubleMuon" in p else p[:-5]
            for h in histograms.keys():
                file[f"{name}/{h}"] = plots[p][histograms[h][0]][histograms[h][1]]
    h_new = {}
    with uproot.open(f"temp.root") as file:
        for p in processes:
            if p not in plots.keys():
                continue
            name = "data_obs" if "DoubleMuon" in p else p[:-5]
            h_new[name] = []
            for h in histograms.keys():
                h_new[name].append(file[f"{name}/{h}"].to_pyroot().Clone())
    with uproot.recreate(f"{tag}.root") as file:
        h_sum = None
        for p in processes:
            if p not in plots.keys():
                continue
            if "SR" in tag and "DoubleMuon" in p:
                continue
            name = "data_obs" if "DoubleMuon" in p else p[:-5]
            h_combined = combine_histograms(h_new[name])
            if "SUEP" not in p:
                if h_sum is None:
                    h_sum = h_combined.Clone()
                else:
                    h_sum.Add(h_combined)
            file[f"{tag}/{name}"] = uproot.from_pyroot(h_combined.Clone())
            del h_combined
        if "SR" in tag:
            file[f"{tag}/data_obs"] = uproot.from_pyroot(h_sum.Clone())
        del h_sum


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-l", "--lumi", type=float, default=573.120134, help="luminosity in pb^-1"
)
parser.add_argument(
    "-e", "--extra", type=str, default="", help="extra string for input files"
)
parser.add_argument(
    "-v", "--verbosity", type=int, default=0, help="verbosity level (0, 1)"
)
args = parser.parse_args()


tags = ["SR", "CR_prompt", "CR_cb"]

if __name__ == "__main__":
    for tag in tags:
        plots = loader(
            lumi=args.lumi, tag=f"{tag}_{args.extra}", verbosity=args.verbosity
        )
        dump_plots(plots, tag, verbosity=args.verbosity)
