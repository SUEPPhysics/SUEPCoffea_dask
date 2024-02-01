"""
A submitter for processing many samples parallelly, using either Slurm or multithread.
Pass to this script the same options you would pass make_hists.py or merge_ntuples.py,
specify whether you want to plot or merge (--code),
and specify if you want multithread or Slurm (--method).

e.g.
python submit.py -i ../filelist/list.txt --isMC 1 --era 2016 -t July2023_2016 -o July2023_2016 --doABCD 1 --code plot --method multithread --channel WH

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
import sys
from multiprocessing.pool import Pool, ThreadPool

import numpy as np

sys.path.append("..")
from make_hists import makeParser as makeHistsParser

from plotting.plot_utils import check_proxy

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
cd {work_dir}
cd ..
cd histmaker/
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
    )
    out, err = p.communicate()
    return (out, err)


parser = argparse.ArgumentParser(description="Famous Submitter")
# Specific to this script
parser.add_argument(
    "-i",
    "--input",
    type=str,
    help="File containing list of samples to process.",
    required=True,
)
parser.add_argument(
    "-c",
    "--code",
    type=str,
    help="Which code to run in parallel.",
    choices=["plot", "merge"],
    required=True,
)
parser.add_argument(
    "-f", "--force", action="store_true", help="overwrite", required=False
)
parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="multithread",
    choices=["multithread", "slurm"],
    help="Which system to use to run the script.",
    required=False,
)
parser.add_argument(
    "--cores",
    type=int,
    help="Maximum number of cores to run multithread on.",
    default=10,
    required=False,
)
# parser from make_hists.py, works also for merge_ntuples.py
parser = makeHistsParser(parser)

options = parser.parse_args()

# Set up where you're gonna work
if options.method == "slurm":
    # Found it necessary to run on a space with enough memory
    work_dir = "/work/submit/{}/dummy_directory{}".format(
        getpass.getuser(), np.random.randint(0, 10000)
    )
    os.system(f"mkdir {work_dir}")
    os.system(f"cp -a ../../SUEPCoffea_dask {work_dir}/.")
    print("Working in", work_dir)
    work_dir += "/SUEPCoffea_dask/histmaker/"
    log_dir = "/work/submit/{}/SUEP/logs/slurm_{}_{}/".format(
        os.environ["USER"],
        options.code,
        options.output if options.code == "plot" else options.tag,
    )
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
elif options.method == "multithread":
    # Found it necessary to run on a space with enough memory
    work_dir = "/work/submit/{}/dummy_directory{}/".format(
        getpass.getuser(), np.random.randint(0, 10000)
    )
    os.system(f"mkdir {work_dir}")
    os.system(f"cp -a ../SUEPCoffea_dask {work_dir}/.")
    print("Working in", work_dir)
    work_dir += "/SUEPCoffea_dask/histmaker/"
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
    if options.code == "plot" and (
        os.path.isfile(
            f"/data/submit/{getpass.getuser()}/SUEP/outputs/{sample}_{options.output}.root"
        )
        and not options.force
    ):
        print(sample, "finished, skipping")
        continue

    # Code to execute
    if options.code == "merge":
        cmd = "python merge_ntuples.py --tag={tag} --sample={sample} --isMC={isMC}".format(
            tag=options.tag, sample=sample, isMC=options.isMC
        )

    elif options.code == "plot":
        cmd = "python make_hists.py --sample={sample} --tag={tag} --redirector={redirector} --dataDirLocal={dataDirLocal} --dataDirXRootD={dataDirXRootD} --output={output_tag} --xrootd={xrootd} --weights={weights} --isMC={isMC} --era={era} --scouting={scouting} --merged={merged} --doInf={doInf} --doABCD={doABCD} --doSyst={doSyst} --blind={blind} --predictSR={predictSR} --saveDir={saveDir} --channel={channel} --maxFiles={maxFiles}".format(
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
            saveDir=options.saveDir,
            channel=options.channel,
            maxFiles=options.maxFiles,
            dataDirLocal=options.dataDirLocal,
            dataDirXRootD=options.dataDirXRootD,
            redirector=options.redirector,
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
