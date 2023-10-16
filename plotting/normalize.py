import glob
from rich.pretty import pprint
import pickle
import plot_utils
import fill_utils


def normalizer(mode, samples):
    print(f"Processing: {mode}")
    # input .pkl files
    file_dir = f"./temp_output_{mode}/"
    offline_files = []
    for sample in samples:
        search_string = ""
        for s in sample:
            search_string = f"{search_string}*{s}"
        search_string = f"{search_string}*_{mode}.pkl"
        offline_files += glob.glob(file_dir + search_string)

    print("Input files:")
    offline_files.sort()
    pprint(offline_files)

    # merge the histograms, apply lumis, exclude low HT bins
    print("Output files:")
    for f in offline_files:
        name = (f.split("/")[-1])[:-4]
        plot = plot_utils.openpkl(f)
        xsection = fill_utils.getXSection(name.replace(f"_{mode}", ""), "2018")
        plot_normalized = fill_utils.apply_normalization(plot, xsection)
        print(
            f"    xs={xsection} pb, ",
            end="",
        )
        pickle.dump(plot_normalized, open(file_dir + name + "_normalized.pkl", "wb"))
        pprint(name + "_normalized.pkl")


# Define a set of strings that should be present in the names of each dataset
samples = [
    ["QCD_Pt_", "20UL18", "MINIAODSIM"],
    [
        "DY",
        "JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
        "MINIAODSIM",
    ],
]

modes = ["histograms", "cutflow"]

if __name__ == "__main__":
    for mode in modes:
        normalizer(mode, samples)
