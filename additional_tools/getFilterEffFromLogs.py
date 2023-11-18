"""
Used to calculate the filter efficiency for each sample from the logs
from the production.

Author: Luca Lavezzo
Date: August 2023
"""


import json
import os
import re

import numpy as np


def extract_sample_info(out_file_path, err_file_path):
    with open(out_file_path) as out_file:
        out_content = out_file.read()

    with open(err_file_path) as err_file:
        err_content = err_file.read()

    # Extract sample name from out file using regular expression
    sample_name_match = re.search(r"###########\n(.+)\n###########", out_content)
    sample_name = sample_name_match.group(1) if sample_name_match else None

    # Extract efficiency from err file using regular expression
    efficiency_match = re.search(
        r"Filter efficiency \(event-level\)= (\S+) / (\S+) = (\S+) \+- (\S+)",
        err_content,
    )

    num = float(efficiency_match.group(1)[1:-1]) if efficiency_match else None
    denom = float(efficiency_match.group(2)[1:-1]) if efficiency_match else None

    return sample_name, num, denom


def process_files(directory):
    sample_data = {}

    iFile = 0
    for filename in os.listdir(directory):
        if filename.endswith(".out"):
            iFile += 1
            sample_name = filename.split(".")[0]
            out_file_path = os.path.join(directory, filename)
            err_file_path = os.path.join(directory, f"{sample_name}.err")

            if os.path.isfile(err_file_path):
                sample_name, nom, denom = extract_sample_info(
                    out_file_path, err_file_path
                )

                if sample_name and nom is not None and denom is not None:
                    if sample_name in sample_data:
                        old_nom, old_denom = sample_data[sample_name]
                        sample_data[sample_name] = (old_nom + nom, old_denom + denom)
                    else:
                        sample_data[sample_name] = (nom, denom)

    for sample in sample_data.keys():
        num, denom = sample_data[sample]
        eff = num / denom
        eff_err = np.sqrt(eff * (1 - eff) / denom)
        sample_data[sample] = (num, denom, eff, eff_err)

    return sample_data


def average_efficiency_by_ms(sample_data):
    averaged_efficiency_by_ms = {}

    for sample_name, (num, denom, efficiency, efficiency_error) in sample_data.items():
        ms_value = re.search(r"mS(\d+\.\d+)", sample_name)
        if ms_value:
            ms_value = float(ms_value.group(1))
            if ms_value in averaged_efficiency_by_ms:
                # If the mS value already exists, update the average efficiency
                old_num, old_denom = averaged_efficiency_by_ms[ms_value]
                averaged_efficiency_by_ms[ms_value] = (old_num + num, old_denom + denom)
            else:
                averaged_efficiency_by_ms[ms_value] = (num, denom)

    for sample in averaged_efficiency_by_ms.keys():
        num, denom = averaged_efficiency_by_ms[sample]
        eff = num / denom
        eff_err = np.sqrt(eff * (1 - eff) / denom)
        averaged_efficiency_by_ms[sample] = (num, denom, eff, eff_err)

    return averaged_efficiency_by_ms


if __name__ == "__main__":
    directory_path = "/work/submit/freerc/suep/official_private/2016/logs/"
    process = True

    # either process the files or load with json
    if process:
        sample_data = process_files(directory_path)
    else:
        with open("sampleEffs.json") as json_file:
            sample_data = json.load(json_file)

    # Calculate the average efficiency by mS value
    averaged_efficiency_by_ms = average_efficiency_by_ms(sample_data)

    if process:
        # save the sample_data to json
        with open("sampleEffs.json", "w") as outfile:
            json.dump(sample_data, outfile)

    # save the averaged_efficiency_by_ms to json
    with open("averagedEffs.json", "w") as outfile:
        json.dump(averaged_efficiency_by_ms, outfile)

    # read in 'kr' values from data/xsections_2018_SUEP.json
    # for all mS values that we have calculated the efficiency for and print them
    with open("../data/xsections_SUEP.json") as json_file:
        xsections = json.load(json_file)

    truth_values = {}
    for sample_name in xsections.keys():
        # get mS from sample_name
        ms = re.search(r"mS(\d+\.\d+)", sample_name)
        if ms is None:
            continue
        ms = float(ms.group(1))
        if ms in truth_values:
            continue
        truth_values[ms] = xsections[sample_name]["kr"]

    for ms_value, (
        nom,
        denom,
        efficiency,
        efficiency_error,
    ) in averaged_efficiency_by_ms.items():
        print(f"mS: {ms_value:.3f}")
        print(f"Average Efficiency: {efficiency:.5e} +- {efficiency_error:.5e}")
        print(f"Truth value: {truth_values.get(ms_value, 0):.5e}")
        print()
