"""
This script runs the histmaker on a few files to make sure it works.
It can be configured to run for different channels, different datasets, and with different options.
It does not check the output histograms for accuracy.

To run this script, do:
    python test_histmaker.py
It will error out if any of the tests fail, and let you know if it's successful.

Author: Luca Lavezzo
Date: January 2025
"""

import os
import subprocess
import sys

sys.path.append("..")
from time import time

from termcolor import colored

"""
Define here the runs you want to test. Parameters:
    singularity (optional, str): can define a singularity image to run the command with
    script (str): the script to run
    options (list of str): the options to pass to the script, can define multiple to execute the command multiple times
    out_file (str): the name of the output file that should be produced
"""
runs = {
    "WH": {
        "commands": [
            "cd histmaker",
            "python run_WH_histmaker.py --sample SingleMuon+Run2018A-UL2018_MiniAODv2_GT36-v2+MINIAOD --client local --isMC 0 --channel WH --era 2018 -n 10 --tag WH_1_14_data_GT36_@018 --doSyst 0 --output debug --dataDirLocal $CMSDIR/SUEP/{}/{}/ --maxFiles 100 --blind 0 --saveDir ./"
        ],
        "out_file": os.environ['PWD']+"/histmaker/debug/SingleMuon+Run2018A-UL2018_MiniAODv2_GT36-v2+MINIAOD.pkl"
    }
}


def test_histmaker(run, config):
    commands = config["commands"]
    output_file = config["out_file"]

    os.system("rm " + output_file)

    start = time()
    print(colored(f"Running test for run {run} ...", "blue"))

    # join commands in a one liner
    command = " && ".join(commands)

    # run the command
    command = f"{command} > {run}.out 2> {run}.err"

    print(f"Executing commands:", command)
    subprocess.run(command, shell=True)

    print(output_file)
    print(os.path.exists(output_file))

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(
            colored(
                f"PASSED! Output file for run {run} exists and is not empty.",
                "green",
            )
        )
    else:
        print(
            colored(
                f"FAILED! Output file for run {run} does not exist or is empty.",
                "red",
            )
        )
        sys.exit()

    # delete output and log files if everything was successful
    #os.system("rm -rf " + output_file)
    os.system("rm " + run + ".out")
    os.system("rm " + run + ".err")

    end = time()
    print(colored(f"All tests passed for run {run}", "green"))
    print(
        colored(
            f"The test for run {run} was a SUCCESS. Time: {end - start} seconds",
            "green",
        )
    )


def main():
    # run each test defined in the config dictionary
    startTot = time()
    for run, config in runs.items():
        test_histmaker(run, config)
    endTot = time()
    print()
    print(colored("All tests were SUCCESSFUL!", "green"))
    print(colored(f"Total time: {endTot - startTot} seconds", "green"))


if __name__ == "__main__":
    main()
