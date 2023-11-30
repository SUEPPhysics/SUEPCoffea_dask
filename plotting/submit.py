"""
A submitter for processing many samples parallelly, using either Slurm or multithread.
Pass to this script the same options you would pass make_plots.py or merge_plots.py,
specify whether you want to plot or merge (--code),
and specify if you want multithread or Slurm (--method).

e.g.
python submit.py -i ../filelist/list.txt --isMC 1 --era 2016 -t July2023_2016 -o July2023_2016 --doABCD 1 --code plot --method multithread

If you're not running on submit, you might need to modify the slurm script template
and some of the work and log paths.

Authors: Luca Lavezzo and Chad Freer.
Date: July 2023
"""


import argparse
import getpass
import multiprocessing
import os
import shlex
import subprocess
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
from plot_utils import check_proxy

# SLURM script template
slurm_script_template = """#!/bin/bash
#SBATCH --job-name={sample}
#SBATCH --output={log_dir}{sample}.out
#SBATCH --error={log_dir}{sample}.err
#SBATCH --time=02:00:00
#SBATCH --mem=2GB
#SBATCH --partition=submit

source ~/.bashrc
export X509_USER_PROXY=/home/submit/{user}/{proxy}
conda activate SUEP # Change to your own environment setup
cd {work_dir}
cd ..
cd plotting/
{cmd}
"""


def call_process(cmd, work_dir):
    """This runs in a separate thread."""
    print("----[%] :", cmd)
    p = subprocess.Popen(
        ["bash", "-c", " ".join(shlex.split(cmd))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=work_dir,
        shell=True,
    )
    out, err = p.communicate()
    return (out, err)


parser = argparse.ArgumentParser(description="Famous Submitter")
# Specific to this script
parser.add_argument(
    "-f", "--force", action="store_true", help="overwrite", required=False
)
parser.add_argument(
    "-c",
    "--code",
    type=str,
    help="Which code to multithread (supported: 'plot' or 'merge')",
    required=True,
)
parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="multithread",
    choices=["multithread", "slurm"],
    help="Which system to use to run the script.",
)
# These are the same as make_plots.py, and are passed straight through it
parser.add_argument("-o", "--output", type=str, help="output tag", required=False)
parser.add_argument("-t", "--tag", type=str, help="production tag", required=True)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Use specific input file (.txt) of samples.",
)
parser.add_argument(
    "-s",
    "--save",
    type=str,
    help="Use specific output directory. Overrides MIT-specific paths.",
    required=False,
    default=None,
)
parser.add_argument(
    "--xrootd",
    type=int,
    default=0,
    help="Local data or xrdcp from hadoop (default=False)",
)
# optional: call it with --merged = 1 to append a /merged/ to the paths in options 2 and 3
parser.add_argument("--merged", type=int, default=0, help="Use merged files")
# some info about the files, highly encouraged to specify every time
parser.add_argument("-e", "--era", type=str, help="era", required=False)
parser.add_argument("--isMC", type=int, help="Is this MC or data", required=True)
parser.add_argument("--scouting", type=int, default=0, help="Is this scouting or no")
# some parameters you can toggle freely
parser.add_argument("--doInf", type=int, default=0, help="make GNN plots")
parser.add_argument("--doSyst", type=int, default=0, help="make systematic plots")
parser.add_argument(
    "--doABCD", type=int, default=0, help="make plots for each ABCD+ region"
)
parser.add_argument(
    "--predictSR", type=int, default=0, help="Predict SR using ABCD method."
)
parser.add_argument(
    "--blind", type=int, default=1, help="Blind the data (default=True)"
)
parser.add_argument(
    "--weights",
    default="None",
    help="Pass the filename of the weights, e.g. --weights weights.npy",
)
parser.add_argument(
    "--channel", type=str, help="Analysis channel: ggF, WH", required=True
)
parser.add_argument(
    "--cores",
    type=int,
    help="Maximum number of cores to run multithread on.",
    default=50,
)
options = parser.parse_args()


# Set up where you're gonna work
if options.method == "slurm":
    # Found it necessary to run on a space with enough memory
    work_dir = "/work/submit/{}/dummy_directory{}".format(
        getpass.getuser(), np.random.randint(0, 10000)
    )
    os.system(f"mkdir {work_dir}")
    os.system(f"cp -R ../* {work_dir}/.")
    print("Working in", work_dir)
    work_dir += "/plotting/"
    log_dir = "/work/submit/{}/SUEP/logs/slurm_{}_{}/".format(
        os.environ["USER"],
        options.code,
        options.output if options.code == "plot" else options.tag,
    )
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
elif options.method == "multithread":
    # Found it necessary to run on a space with enough memory
    work_dir = "/work/submit/{}/dummy_directory{}".format(
        getpass.getuser(), np.random.randint(0, 10000)
    )
    os.system(f"mkdir {work_dir}")
    os.system(f"cp -R ../* {work_dir}/.")
    print("Working in", work_dir)
    work_dir += "/plotting/"

    pool = Pool(
        min([multiprocessing.cpu_count(), options.cores]), maxtasksperchild=1000
    )

    results = []

# Read samples from input file
with open(options.input) as f:
    samples = f.read().splitlines()

# Making sure that the proxy is good
if options.xrootd:
    lifetime = check_proxy(time_min=10)
    print(f"--- proxy lifetime is {round(lifetime, 1)} hours")

# Loop over samples
for i, sample in enumerate(samples):
    if "/" in sample:
        sample = sample.split("/")[-1]
    if (
        os.path.isfile(
            f"/data/submit/{getpass.getuser()}/SUEP/outputs/{sample}_{options.output}.root"
        )
        and not options.force
    ):
        print(sample, "finished, skipping")
        continue

    # Code to execute
    if options.code == "merge":
        cmd = (
            "python merge_plots.py --tag={tag} --dataset={sample} --isMC={isMC}".format(
                tag=options.tag, sample=sample, isMC=options.isMC
            )
        )

    elif options.code == "plot":
        cmd = "python make_plots.py --dataset={sample} --tag={tag} --output={output_tag} --xrootd={xrootd} --weights={weights} --isMC={isMC} --era={era} --scouting={scouting} --merged={merged} --doInf={doInf} --doABCD={doABCD} --doSyst={doSyst} --blind={blind} --predictSR={predictSR} --save={save} --channel={channel}".format(
            sample=sample,
            tag=options.tag,
            output_tag=options.output,
            xrootd=options.xrootd,
            weights=options.weights,
            isMC=options.isMC,
            era=options.era,
            scouting=options.scouting,
            merged=options.merged,
            doInf=options.doInf,
            doABCD=options.doABCD,
            doSyst=options.doSyst,
            blind=options.blind,
            predictSR=options.predictSR,
            save=options.save,
            channel=options.channel,
            id=os.getuid(),
        )

    # execute the command with singularity
    singularity_prefix = "singularity run --bind /work/,/data/ /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest "
    cmd = singularity_prefix + cmd

    # Method to execute the code with
    if options.method == "multithread":
        results.append(pool.apply_async(call_process, (cmd, work_dir)))

    elif options.method == "slurm":
        # Generate the SLURM script content
        slurm_script_content = slurm_script_template.format(
            log_dir=log_dir,
            work_dir=work_dir,
            cmd=cmd,
            sample=sample,
            user=getpass.getuser(),
            proxy=f"x509up_u{os.getuid()}",
        )

        # Write the SLURM script to a file
        slurm_script_file = f"{log_dir}submit_{sample}.sh"
        with open(slurm_script_file, "w") as f:
            f.write(slurm_script_content)

        # Submit the SLURM job
        subprocess.run(["sbatch", slurm_script_file])

# Close the pool and wait for each running task to complete
if options.method == "multithread":
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if "error" in str(err).lower():
            print(str(err))
            print(" ----------------- ")
            print()

    # clean up
    os.system(f"rm -rf {work_dir}")
