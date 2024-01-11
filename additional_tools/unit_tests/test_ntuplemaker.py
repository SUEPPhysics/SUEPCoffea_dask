"""
This script runs the ntuplemaker on a few files to make sure it works.
It can be configured to run for different channels, different datasets, and with different options.
It does not check the output ntuples for accuracy.

To run this script, do:
    python test_ntuplemaker.py
It will error out if any of the tests fail, and let you know if it's successful.

Author: Luca Lavezzo
Date: November 2023
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
    root_files (list of str): the root files to run over, can define multiple to execute the command multiple times
    out_file (str): the name of the output file that should be produced
"""
runs = {
    "ggF-Offline": {
        "singularity": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest",
        "script": "condor_SUEP_ggF.py",
        "options": [
            "--isMC 1 --era 2016 --doSyst 1",
            "--isMC 0 --era 2018 --doSyst 0",
        ],
        "root_files": [
            "root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8+RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1+MINIAODSIM//FAFF5EE2-F08F-0D4F-A43A-8990712DF75B.root",
            "root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/JetHT+Run2018C-UL2018_MiniAODv2-v1+MINIAOD//FFE35F7A-1786-BE4F-8AF4-66DEA58012F3.root",
        ],
        "out_file": "out.hdf5",
    },
    "ggF-Scouting": {
        "script": "condor_Scouting.py",
        "options": [
            "--isMC 1 --era 2018 --doSyst 1",
            "--isMC 0 --era 2018 --doSyst 0",
        ],
        "root_files": [
            "root://xrootd.cmsaf.mit.edu//store/user/paus/nanosc/E07/QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18RECO-106X_upgrade2018_realistic_v11_L1v1-v2+AODSIM//FFEBDF0B-33D5-F84A-B899-0F8EF89FA734.root",
            "root://xrootd.cmsaf.mit.edu//store/user/paus/nanosc/E08/ScoutingPFCommissioning+Run2016B-v2+RAW//FAF17B72-201D-E611-BA11-02163E012571.root",
        ],
        "out_file": "out.hdf5",
    },
    "WH": {
        "singularity": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest",
        "script": "condor_SUEP_WH.py",
        "options": [
            "--isMC 1 --era 2018 --doSyst 1 --maxChunks 2",
            "--isMC 0 --era 2018 --doSyst 0",
        ],
        "root_files": [
            "root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM//FFAB73E1-5B63-0B43-867E-B35DB3C75E60.root",
            "root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/JetHT+Run2018C-UL2018_MiniAODv2-v1+MINIAOD//FFE35F7A-1786-BE4F-8AF4-66DEA58012F3.root",
        ],
        "out_file": "out.hdf5",
    },
}


def test_ntuplemaker(run, config):
    script = config["script"]
    options = config["options"]
    root_files = config["root_files"]
    output_file = config["out_file"]

    start = time()
    print(colored(f"Running test for run {run} ...", "blue"))
    print(colored(f"Found {len(options)} options to test for run {run}", "blue"))

    for option, root_file in zip(options, root_files):
        command = (
            f"python {script} --infile {root_file} {option} > {run}.out 2> {run}.err"
        )

        if "singularity" in config:
            command = f"singularity exec {config['singularity']} {command}"

        print("Executing command:", command)
        subprocess.run(command, shell=True)

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
        os.system("rm " + output_file)
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
        test_ntuplemaker(run, config)
    endTot = time()
    print()
    print(colored("All tests were SUCCESSFUL!", "green"))
    print(colored(f"Total time: {endTot - startTot} seconds", "green"))


if __name__ == "__main__":
    main()
