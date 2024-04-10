import json
import subprocess

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from hist import Hist


# Function to read xsection from json file
def read_xsection(sample_name):
    with open("../../data/xsections_2018.json") as f:
        xsections = json.load(f)
    try:
        return xsections[sample_name]["xsec"]
    except KeyError:
        print(f"Sample {sample_name} not found in xsections_2018.json")
        return 1


# Function to select gen level W boson pT
def get_vars(events, sample):
    weight = events["genWeight"].array()
    totweight = np.sum(weight)

    # for signal samples, we don't have LHE information
    if "WSUEP" in sample:
        status = events["GenPart_status"].array()
        pdgId = events["GenPart_pdgId"].array()
        pt = events["GenPart_pt"].array()
        Ws = (status == 22) & (abs(pdgId) == 24)
        pt = pt[Ws]
        pt = ak.fill_none(ak.pad_none(pt, 0 + 1, axis=1, clip=True), -999)[:, 0]
        eta = events["GenPart_eta"].array()
        eta = eta[Ws]
        eta = ak.fill_none(ak.pad_none(eta, 0 + 1, axis=1, clip=True), -999)[:, 0]
        ht = [np.nan] * len(pt)
    else:
        pt = events["LHE_Vpt"].array()
        ht = events["LHE_HT"].array()
        eta = [np.nan] * len(pt)

    # cuts = (ht > 70) & (pt > 100)
    # pt = pt[cuts]
    # weight = weight[cuts]
    # ht = ht[cuts]
    # eta = eta[cuts]

    return totweight, weight, pt, ht, eta


# Main function
def main():
    # Sample names
    samples = {
        "WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        "WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        "WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        "WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        #     "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        #     "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        #     "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        #     "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        #     "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        #     "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        #     "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        #     "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM": "/store/user/paus/nanosu/A02/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
    }
    # samples = {
    #     "VHToNonbb_M125_TuneCP5_13TeV": "/store/mc/RunIISummer20UL18NanoAODv9/VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/",
    #     "WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
    #     "WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM": "/store/user/paus/nanosu/A02/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
    #     "WSUEP_WToLNu_T3p00_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM": "/store/user/paus/nanosu/A02//WSUEP_WToLNu_T3p00_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
    # }

    N_FILES = 3

    hists = {}
    for sample in samples:
        sampleshort = sample.split("+")[0]
        hists["HT_" + sampleshort] = Hist.new.Reg(
            20000, 0, 20000, name=f"HT", label="HT"
        ).Weight()
        hists["PT_" + sampleshort] = Hist.new.Reg(
            5000, 0, 5000, name=f"PT", label="PT"
        ).Weight()
        hists["ETA_" + sampleshort] = Hist.new.Reg(
            10000, -5, 5, name=f"ETA", label="ETA"
        ).Weight()

    # Loop over each sample
    for sample_name, sample_dir in samples.items():
        print(sample_name)
        sampleshort = sample_name.split("+")[0]

        # Use xrootd to list contents of the directory
        if "VHToNonbb_M125_TuneCP5_13TeV" in sample_name:
            redirector = "root://xrootd-vanderbilt.sites.opensciencegrid.org/"
        else:
            redirector = "root://xrootd.cmsaf.mit.edu/"
        command = f"xrdfs {redirector} ls {sample_dir}"
        files = (
            subprocess.check_output(command, shell=True, stderr=subprocess.PIPE)
            .decode()
            .splitlines()
        )

        # Get xsec for the sample
        xsec = read_xsection(sample_name)

        # Loop over N_FILES (you need to define N_FILES)
        weights, pts, hts, etas = [], [], [], []
        totweights = 0
        for file_index in range(min(N_FILES, len(files))):
            print(file_index)

            file_path = f"{redirector}" + files[file_index]

            print(file_path)

            # Read file using uproot
            try:
                tree = uproot.open(file_path)["Events"]
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            # Select gen level W boson pT
            totweight, genWeight, gen_W_pt, ht, eta = get_vars(tree, sample_name)

            pts.extend(gen_W_pt)
            weights.extend(genWeight)
            hts.extend(ht)
            etas.extend(eta)
            totweights += totweight

        assert len(pts) == len(weights)
        assert len(pts) == len(hts)

        weight = np.array(weights) * xsec / totweights

        # Fill histograms
        hists["HT_" + sampleshort].fill(hts, weight=weight)
        hists["ETA_" + sampleshort].fill(etas, weight=weight)
        hists["PT_" + sampleshort].fill(pts, weight=weight)

    with uproot.recreate("gen_study_2.root") as froot:
        for histname, hist in hists.items():
            froot[histname] = hist


if __name__ == "__main__":
    main()
