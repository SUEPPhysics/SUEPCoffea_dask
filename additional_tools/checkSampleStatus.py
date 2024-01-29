"""
Find all samples that have less than min_events in them from the histograms.

WORK IN PRORGESS:
This isn't a perfect check, but it's a good first pass. We only check how many
bins in the histogram have more than 0 events in them, since can't calculate the
number of events because they're normalized by the gensumweight. We should consider
fixing this, e.g. by storing the gensumwright so we can rederive the number of events.

Author: Luca Lavezzo
Date: August 2023
"""

import argparse
import os

import uproot


def check_sample(filename, min_events=100):
    if os.path.isfile(filename):
        try:
            file = uproot.open(filename)
            yvals = file["SUEP_nconst_Cluster70"].to_numpy()[0]
            num_events = len(yvals[yvals > 0])
            if num_events < min_events:
                return "low_stats"
            else:
                return "ok"
        except:
            return "corrupted"
    else:
        return "missing"


def main(era, min_events=100):
    input_file = "filelist/list_full_signal_offline.txt"
    if era == "2016apv":
        tag = "July2023_2016apv"
    elif era == "2016":
        tag = "July2023_2016"
    elif era == "2017":
        tag = "July2023_2017"
    elif era == "2018":
        tag = "March2023"
    else:
        raise ValueError("Era not supported.")

    plotsDir = "/data/submit/lavezzo/SUEP/outputs/"

    missing_samples = []
    low_stats_samples = []
    with open(input_file) as file:
        for line in file:
            sample_name = line.strip()
            file_path = f"{plotsDir}{sample_name}_{tag}.root"
            status = check_sample(file_path, min_events)

            # print(sample_name, status)

            if status == "missing" or status == "corrupted":
                missing_samples.append(sample_name)
            elif status == "low_stats":
                low_stats_samples.append(sample_name)

    # print()
    # print("Missing samples:")
    for s in missing_samples:
        print(s)
    # print()
    # print("Low stats samples:")
    for s in low_stats_samples:
        print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check sample status based on era.")
    parser.add_argument(
        "--era",
        choices=["2016apv", "2016", "2017", "2018"],
        required=True,
        help="Specify the era.",
    )
    args = parser.parse_args()
    main(args.era)
