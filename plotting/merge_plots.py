import argparse
import getpass
import multiprocessing
import os
import subprocess
import sys

import fill_utils
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Famous Submitter")
parser.add_argument(
    "-dataset", "--dataset", type=str, default="QCD", help="dataset name", required=True
)
parser.add_argument(
    "-t", "--tag", type=str, default="IronMan", help="production tag", required=False
)
parser.add_argument("--isMC", type=int, default=1, help="Is this MC or data")
options = parser.parse_args()

# script parameters
username = getpass.getuser()
tag = options.tag
dataset = options.dataset
redirector = "root://submit50.mit.edu/"
localDataDir = "/data/submit/cms/store/user/{}/SUEP/{}/{}/".format(
    username, tag, dataset
)
dataDir = f"/store/user/{username}/SUEP/{tag}/{dataset}/"
localOutDir = localDataDir + "/merged/"
outDir = dataDir + "/merged/"

# create output dir
if not os.path.isdir(localOutDir):
    os.mkdir(localOutDir)

# list files in dir using xrootd
result = subprocess.check_output(["xrdfs", redirector, "ls", dataDir])
result = result.decode("utf-8")
files = result.split("\n")
files = [f for f in files if (".hdf5" in f) and ("merged" not in f)]


# SAVE OUTPUTS
def save_dfs(df_tot, output):
    if type(df_tot) == int:
        print("No events in df_tot.")
        df_tot = pd.DataFrame(["empty"], columns=["empty"])
    store = pd.HDFStore(output)
    store.put("vars", df_tot)
    store.get_storer("vars").attrs.metadata = metadata_tot
    store.close()


def move(q, infile, outfile):
    returncode = q.get()
    result = subprocess.run(["xrdcp", "-s", infile, outfile, "-f"])
    q.put(result.returncode)
    # return result


def time_limited_move(infile, outfile, time_limit=60, max_attempts=5):
    # Make max_attempts to move a file, each attempt within a time_limit
    attempt = 0
    while attempt < max_attempts:
        returncode = -1
        q = multiprocessing.Manager().Queue()
        q.put(returncode)
        p = multiprocessing.Process(
            target=move, name="move_" + infile, args=(q, infile, outfile)
        )
        p.start()
        # Wait a maximum of time_limit seconds for foo
        # Usage: join([timeout in seconds])
        p.join(time_limit)
        # If thread is active
        if p.is_alive():
            print("TIME ERROR", file, "taking too long to be transferred")
            p.terminate()
            p.join()
            attempt += 1
        # thread finished
        else:
            returncode = q.get()
            # something failed in the transfer
            if returncode != 0:
                print("XRootD ERROR:", returncode, "for file", infile, "to", outfile)
                attempt += 1
            # thread finished and everything worked
            else:
                return 1
    return 0


df_tot = 0
metadata_tot = 0
i_out = 0
for ifile, file in enumerate(tqdm(files)):
    if os.path.exists(dataset + ".hdf5"):
        subprocess.run(["rm", dataset + ".hdf5"])
    xrd_file = redirector + file

    # If this script is running for a while, some xrdcp start to hang for too long,
    # so we re-attempt it a couple times before quitting
    result = time_limited_move(
        xrd_file, dataset + ".hdf5", time_limit=120, max_attempts=3
    )
    if result == 0:
        sys.exit("Result of xrootd transfer was 0: " + xrd_file)

    df, metadata = fill_utils.h5load(dataset + ".hdf5", "vars")

    # corrupted
    if type(df) == int:
        continue

    ### MERGE METADATA
    if options.isMC:
        if type(metadata_tot) == int:
            metadata_tot = metadata
        else:
            metadata_tot["gensumweight"] += metadata["gensumweight"]

    # don't need to add empty ones
    if "empty" in list(df.keys()):
        subprocess.run(["rm", dataset + ".hdf5"])
        continue

    if df.shape[0] == 0:
        subprocess.run(["rm", dataset + ".hdf5"])
        continue

    ### MERGE DF VARS
    if type(df_tot) == int:
        df_tot = df
    else:
        df_tot = pd.concat((df_tot, df))

    subprocess.run(["rm", dataset + ".hdf5"])

    # save every N events
    if df_tot.shape[0] > 5000000:
        output_file = dataset + "_merged_" + str(i_out) + ".hdf5"
        save_dfs(df_tot, output_file)
        print(f"xrdcp {output_file} {redirector + outDir}")

        # Allow a couple resubmissions in case of xrootd failures
        result = time_limited_move(
            output_file, redirector + outDir, time_limit=600, max_attempts=3
        )
        if result == 0:
            print("Something messed up with file " + xrd_file)
        else:
            subprocess.run(["rm", output_file])
        i_out += 1
        df_tot = 0
        metadata_tot = 0

# save last file as well
output_file = dataset + "_merged_" + str(i_out) + ".hdf5"
save_dfs(df_tot, output_file)
print(f"xrdcp {output_file} {redirector + outDir}")
# Allow a couple resubmissions in case of xrootd failures
result = time_limited_move(
    output_file, redirector + outDir, time_limit=600, max_attempts=3
)
if result == 0:
    print("Result of xrootd transfer was 0: " + xrd_file)
else:
    subprocess.run(["rm", output_file])
