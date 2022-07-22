import os
import subprocess
import shlex
import argparse
import getpass
import multiprocessing
from multiprocessing.pool import ThreadPool
import time
import numpy as np
from plot_utils import check_proxy

parser = argparse.ArgumentParser(description="Famous Submitter")
parser.add_argument(
    "-i",
    "--inputList",
    type=str,
    default="data.txt",
    help="input datasets",
    required=True,
)
parser.add_argument(
    "-c",
    "--code",
    type=str,
    default="None",
    help="Which code to multithread",
    required=True,
)
parser.add_argument(
    "-t", "--tag", type=str, default="IronMan", help="Production tag", required=True
)
parser.add_argument("-isMC", "--isMC", type=int, default=1, help="")
parser.add_argument("-e", "--era", type=str, default="2018", help="")
parser.add_argument("-sc", "--scout", type=int, default=0, help="")
parser.add_argument(
    "-o", "--output", type=str, default="IronMan", help="Output tag", required=False
)
parser.add_argument(
    "--weights",
    type=str,
    default="None",
    help="Pass the filename of the weights, e.g. --weights weights.npy",
)
parser.add_argument(
    "--xrootd",
    type=int,
    default=0,
    help="Local data or xrdcp from hadoop (default=False)",
)
options = parser.parse_args()


working_directory = "/work/submit/{}/dummy_directory{}".format(
    getpass.getuser(), np.random.randint(0, 10000)
)
os.system("mkdir {}".format(working_directory))
os.system("cp -R ../* {}/.".format(working_directory))


def call_process(cmd):
    """This runs in a separate thread."""
    print("----[%] :", cmd)
    p = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=working_directory + "/plotting/",
    )
    out, err = p.communicate()
    return (out, err)


pool = ThreadPool(multiprocessing.cpu_count())

with open(options.inputList, "r") as f:
    input_list = f.readlines()
    input_list = [l.split("/")[-1].strip("\n") for l in input_list]

# if you want to limit what you run over modify the following:
# input_list = [f for f in read_filelist('../filelist/list_2018_MC_A01.txt') if 'SUEP' in f]

results = []
start = time.time()

# Making sure that the proxy is good
if options.xrootd:
    lifetime = check_proxy(time_min=10)
    print("--- proxy lifetime is {} hours".format(round(lifetime, 1)))

for sample in input_list:

    if options.code == "merge":
        cmd = "python3 merge_plots.py --tag={} --dataset={} --isMC={}".format(
            options.tag, sample, options.isMC
        )
        results.append(pool.apply_async(call_process, (cmd,)))

    elif options.code == "plot":
        cmd = "python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --weights={} --isMC={} --era={} --scouting={} ".format(
            options.tag,
            options.output,
            sample,
            options.xrootd,
            options.weights,
            options.isMC,
            options.era,
            options.scout,
        )
        results.append(pool.apply_async(call_process, (cmd,)))

    else:
        print(
            "Please specify which code to multithread using the -c option. Choose from 'merge', 'plot'"
        )


# Close the pool and wait for each running task to complete
pool.close()
pool.join()
for result in results:
    out, err = result.get()
    if "No such file or directory" in str(err):
        print(str(err))
        print(" ----------------- ")
        print()
end = time.time()

print("All done! multithread.py took", round(end - start), "seconds to run.")
os.system("rm -rf {}".format(working_directory))
