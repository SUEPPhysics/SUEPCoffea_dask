"""
Script to check the completion of the histogramming step.

Author: Luca Lavezzo
Date: January 2024
"""

import argparse
import os

import uproot
from tqdm import tqdm


def check_samples(filelist, tag, path, countCheck=False):
    missing_samples = []
    completed_samples = 0

    with open(filelist) as f:
        samples = f.read().splitlines()

    for sample in tqdm(samples):
        root_file = f"{path}/{sample}_{tag}.root"
        if not os.path.exists(root_file):
            missing_samples.append(sample)
        elif not countCheck:
            completed_samples += 1
        else:
            try:
                f = uproot.open(root_file)
                histName = "SUEP_nconst_Cluster70"
                eventCount = f[histName].to_hist().sum().value
                if eventCount == 0:
                    missing_samples.append(sample)
                else:
                    completed_samples += 1
            except Exception as e:
                print(e)
                missing_samples.append(sample)

    print(f"Completion: {completed_samples}/{len(samples)}")
    if len(missing_samples) > 0:
        print(f"Missing samples:")
        for sample in missing_samples:
            print(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to the .txt filelist of samples")
    parser.add_argument("--tag", help="Tag to append to the sample names")
    parser.add_argument(
        "--path",
        help="Path to the output directory",
        default="/data/submit/{user}/SUEP/outputs/".format(user=os.environ["USER"]),
    )
    parser.add_argument(
        "-c",
        "--countCheck",
        action="store_true",
        help="Check that the samples have non-zero counts by counting the number of events in an histogram. Check the hardcoded histogram that's being used to compute this.",
    )
    args = parser.parse_args()

    check_samples(args.input, args.tag, args.path, countCheck=args.countCheck)
