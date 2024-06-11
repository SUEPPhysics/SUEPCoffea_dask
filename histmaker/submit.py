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
import logging
import multiprocessing
import os
import re
import shlex
import subprocess
import sys
from multiprocessing.pool import Pool, ThreadPool

import numpy as np

sys.path.append("..")
from make_hists import makeParser as makeHistsParser
from merge_ntuples import makeParser as makeMergeParser

from plotting.plot_utils import check_proxy

# SLURM script template
slurm_script_template = """#!/bin/bash
#SBATCH --job-name={sample}
#SBATCH --output={log_dir}{sample}.out
#SBATCH --error={log_dir}{sample}.err
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH --partition=submit,submit-gpu

source ~/.bashrc
export X509_USER_PROXY=/home/submit/{user}/{proxy}
cd {work_dir}
cd ..
cd histmaker/
echo hostname
hostname
{cmd}
"""


def submit_slurm_job(slurm_script_file):
    result = subprocess.run(
        ["sbatch", slurm_script_file], capture_output=True, text=True
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        logging.info("Submitted batch job " + str(job_id))
        return job_id
    else:
        return None


def call_process(cmd, work_dir):
    """This runs in a separate thread."""
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
general_options, _ = parser.parse_known_args()
if general_options.code == "plot":
    parser = makeHistsParser(parser)
elif general_options.code == "merge":
    parser = makeMergeParser(parser)

options = parser.parse_args()

# logging
logging.basicConfig(level=logging.DEBUG)

# Found it necessary to run on a space with enough disk space
work_dir_base = "/work/submit/{}/dummy_directory{}".format(
    getpass.getuser(), np.random.randint(0, 10000)
)
logging.info("Copying ../../SUEPCoffea_dask to " + work_dir_base)
os.system(f"mkdir {work_dir_base}")
os.system(f"cp -a ../../SUEPCoffea_dask {work_dir_base}/.")
logging.info("Working in " + work_dir_base)
work_dir = work_dir_base + "/SUEPCoffea_dask/histmaker/"

# Set up processing-specific options
if options.method == "slurm":
    log_dir = "/work/submit/{}/SUEP/logs/slurm_{}_{}/".format(
        os.environ["USER"],
        options.code,
        options.output if options.code == "plot" else options.tag,
    )
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if options.code == "plot":
        memory = "12GB"
        time = "02:00:00"
    elif options.code == "merge":
        memory = "32GB"
        time = "12:00:00"
elif options.method == "multithread":
    pool = Pool(
        min([multiprocessing.cpu_count(), options.cores]), maxtasksperchild=1000
    )
    results = []

# Read samples from input file
with open(options.input) as f:
    samples = f.read().splitlines()

# Making sure that the proxy is good
if options.code == "merge" or options.xrootd:
    lifetime = check_proxy(time_min=10)
    logging.info(f"--- proxy lifetime is {round(lifetime, 1)} hours")

# Loop over samples
job_ids = []
for i, sample in enumerate(samples):

    if "/" in sample:
        sample = sample.split("/")[-1]
    if sample.endswith(".root"):
        sample = sample.replace(".root", "")

    logging.info(f"Processing sample {i+1}/{len(samples)}: {sample}")

    # skip if the output file already exists, unless you --force
    if options.code == "plot" and (
        os.path.isfile(
            f"/data/submit/{getpass.getuser()}/SUEP/outputs/{sample}_{options.output}.root"
        )
        and not options.force
    ):
        logging.info("Finished, skipping")
        continue

    # Code to execute
    if options.code == "merge":
        cmd = "python merge_ntuples.py --sample={sample} --tag={tag}  --isMC={isMC} --redirector={redirector} --path={path}".format(
            sample=sample,
            tag=options.tag,
            isMC=options.isMC,
            redirector=options.redirector,
            path=options.path,
        )

    elif options.code == "plot":
        cmd = "python make_hists.py --sample={sample} --tag={tag} --redirector={redirector} --dataDirLocal={dataDirLocal} --dataDirXRootD={dataDirXRootD} --output={output_tag} --xrootd={xrootd} --weights={weights} --isMC={isMC} --era={era} --scouting={scouting} --merged={merged} --doInf={doInf} --doABCD={doABCD} --doSyst={doSyst} --blind={blind} --saveDir={saveDir} --channel={channel} --maxFiles={maxFiles} --pkl={pkl}".format(
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
            saveDir=options.saveDir,
            channel=options.channel,
            maxFiles=options.maxFiles,
            dataDirLocal=options.dataDirLocal,
            dataDirXRootD=options.dataDirXRootD,
            redirector=options.redirector,
            pkl=options.pkl,
        )

    # execute the command with singularity
    singularity_prefix = "singularity run --bind /work/,/data/ /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest "
    cmd = singularity_prefix + cmd
    logging.debug(f"Command to run: {cmd}")

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
            memory=memory,
            time=time,
        )

        # Write the SLURM script to a file
        slurm_script_file = f"{log_dir}submit_{sample}.sh"
        with open(slurm_script_file, "w") as f:
            f.write(slurm_script_content)

        # Submit the SLURM job
        job_id = submit_slurm_job(slurm_script_file)
        job_ids.append(job_id)

# Close the pool and wait for each running task to complete
if options.method == "multithread":
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if "error" in str(err).lower():
            logging.info(str(err))
            logging.info(" ----------------- ")

    # clean up
    os.system(f"rm -rf {work_dir}")

# if submitted slurm jobs, send one final job to clean up the dummy directory
if options.method == "slurm" and len(job_ids) > 0:
    cleanup_script = """#!/bin/bash
#SBATCH --job-name=cleanup_job
#SBATCH --output={log_dir}cleanup_job.out
#SBATCH --error={log_dir}cleanup_job.err
#SBATCH --time=02:00:00
#SBATCH --mem=100MB
#SBATCH --partition=submit,submit-gpu

while true; do
    all_finished=true
    for job_id in {job_ids}; do
        echo "checking $job_id"
        job_state=$(sacct -j $job_id -X -n -o state)
        job_state=$(echo $job_state | xargs)
        echo "job state: $job_state"
        if [[ "$job_state" != "COMPLETED" && "$job_state" != "FAILED" ]]; then
            all_finished=false
            break
        fi
    done
    if $all_finished; then
        echo "All jobs finished, cleaning up"
        echo "rm -rf {work_dir_base}"
        rm -rf {work_dir_base}
        break
    fi
    echo "Sleeping for 1 minute"
    sleep 60
done
""".format(
        log_dir=log_dir, job_ids=" ".join(job_ids), work_dir_base=work_dir_base
    )

    with open(f"{log_dir}cleanup.sh", "w") as f:
        f.write(cleanup_script)

    logging.info("Submitting cleanup job")
    cleanup_id = submit_slurm_job(f"{log_dir}cleanup.sh")
