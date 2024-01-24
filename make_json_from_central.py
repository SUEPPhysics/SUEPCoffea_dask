import argparse
import subprocess
import json

from rich.progress import track

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tag", help="Tag")
args = parser.parse_args()

xrootd_redirector = "root://cmsxrootd.fnal.gov/"

with open(f"{args.tag}.txt", "r") as f:
    datasets = f.read().splitlines()

file_dict = {}

for dataset in track(datasets, description="Processing..."):
    key = dataset.replace("/", "+")[1:]
    file_dict[key] = []
    command = f'dasgoclient -query="file dataset={dataset}"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    command_str = " ".join(command)
    files = result.stdout.splitlines()
    for _file in files:
        if _file:  # ignore empty lines
            file_dict[key].append(xrootd_redirector + _file)

# Write the dictionary to a JSON file
with open(f"filelist/{args.tag}.json", "w") as f:
    json.dump(file_dict, f, indent=4)
