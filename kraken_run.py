import argparse
import datetime
import getpass
import logging
import os
import shutil
import subprocess
import sys
import time

from histmaker.fill_utils import get_git_info
from plotting.plot_utils import check_proxy

script_TEMPLATE = """#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export X509_USER_PROXY={proxy}
export PATH=$USER_PATH:$PATH
export PATH=$PATH:/opt/conda/bin
export PYTHONWARNINGS="ignore"

export SCRAM_ARCH=slc7_amd64_gcc820
export HOME=.

echo "PATH"
echo $PATH

echo "hostname"
hostname

sleep $[ ( $RANDOM % 1000 )  + 1 ]s

pip install h5py

echo "----- xrdcp the input file over"
echo "xrdcp $2 $3.root"
xrdcp $2 $3.root

echo "----- Found Proxy in: $X509_USER_PROXY"
echo "python3 {condor_file} --jobNum=$1 --isMC={ismc} --era={era} --doInf={doInf} --doSyst={doSyst} --dataset={dataset} --infile=$3.root"
python3 {condor_file} --jobNum=$1 --isMC={ismc} --era={era} --doInf={doInf} --doSyst={doSyst} --dataset={dataset} --infile=$3.root

#echo "----- transferring output to scratch :"
echo "xrdcp {outfile}.{file_ext} {redirector}/{outdir}/$3.{file_ext}"
xrdcp {outfile}.{file_ext} {redirector}/{outdir}/$3.{file_ext}

{extras}

echo "rm *.{file_ext}"
rm *.{file_ext}

echo "rm $3.root"
rm $3.root

echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 2GB
request_memory        = 5GB
#request_cpus          = 1
executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid) $(fileid)
should_transfer_files = YES
transfer_input_files  = {transfer_file}
MAX_TRANSFER_INPUT_MB = 400
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log
initialdir            = {jobdir}
when_to_transfer_output = ON_EXIT
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 3
use_x509userproxy     = True
x509userproxy         = /home/submit/{user}/{proxy}
+AccountingGroup      = "analysis.{user}"
Requirements          = ( BOSCOCluster =!= "t3serv008.mit.edu" && BOSCOCluster =!= "ce03.cmsaf.mit.edu" && BOSCOCluster =!= "eofe8.mit.edu")
+DESIRED_Sites        = "T2_AT_Vienna,T2_BE_IIHE,T2_BE_UCL,T2_BR_SPRACE,T2_BR_UERJ,T2_CH_CERN,T2_CH_CERN_AI,T2_CH_CERN_HLT,T2_CH_CERN_Wigner,T2_CH_CSCS,T2_CH_CSCS_HPC,T2_CN_Beijing,T2_DE_DESY,T2_DE_RWTH,T2_EE_Estonia,T2_ES_CIEMAT,T2_ES_IFCA,T2_FI_HIP,T2_FR_CCIN2P3,T2_FR_GRIF_IRFU,T2_FR_GRIF_LLR,T2_FR_IPHC,T2_GR_Ioannina,T2_HU_Budapest,T2_IN_TIFR,T2_IT_Bari,T2_IT_Legnaro,T2_IT_Pisa,T2_IT_Rome,T2_KR_KISTI,T2_MY_SIFIR,T2_MY_UPM_BIRUNI,T2_PK_NCP,T2_PL_Swierk,T2_PL_Warsaw,T2_PT_NCG_Lisbon,T2_RU_IHEP,T2_RU_INR,T2_RU_ITEP,T2_RU_JINR,T2_RU_PNPI,T2_RU_SINP,T2_TH_CUNSTDA,T2_TR_METU,T2_TW_NCHC,T2_UA_KIPT,T2_UK_London_IC,T2_UK_SGrid_Bristol,T2_UK_SGrid_RALPP,T2_US_Caltech,T2_US_Florida,T2_US_Nebraska,T2_US_Purdue,T2_US_UCSD,T2_US_Vanderbilt,T2_US_Wisconsin,T3_CH_CERN_CAF,T3_CH_CERN_DOMA,T3_CH_CERN_HelixNebula,T3_CH_CERN_HelixNebula_REHA,T3_CH_CMSAtHome,T3_CH_Volunteer,T3_US_HEPCloud,T3_US_NERSC,T3_US_OSG,T3_US_PSC,T3_US_SDSC,T3_US_MIT"
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+JobFlavour           = "{queue}"

queue jobid, fileid from {jobdir}/inputfiles.dat
"""


def main():
    parser = argparse.ArgumentParser(description="Famous Submitter")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data.txt",
        help="input datasets",
        required=True,
    )
    parser.add_argument(
        "-t", "--tag", type=str, default="IronMan", help="production tag", required=True
    )
    parser.add_argument(
        "-isMC", "--isMC", type=int, default=1, help="Is Monte Carlo or data."
    )
    parser.add_argument(
        "-doInf", "--doInf", type=int, default=0, help="Do inference or not."
    )
    parser.add_argument(
        "-doSyst", "--doSyst", type=int, default=1, help="Apply systematics."
    )
    parser.add_argument(
        "-p", "--private", type=int, default=0, help="Private SUEP samples."
    )
    parser.add_argument(
        "-cutflow", "--cutflow", type=int, default=0, help="Cutflow analyzer."
    )
    parser.add_argument("-q", "--queue", type=str, default="espresso", help="")
    parser.add_argument("-e", "--era", type=str, default="2018", help="")
    parser.add_argument(
        "-f", "--force", action="store_true", help="recreate files and jobs"
    )
    parser.add_argument(
        "-dry", "--dryrun", action="store_true", help="running without submission"
    )
    parser.add_argument(
        "-m", "--maxFiles", type=int, default=-1, help="maximum number of files"
    )
    parser.add_argument("--redo-proxy", action="store_true", help="redo the voms proxy")
    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Channel to run.",
        choices=["ggF", "WH"],
    )
    parser.add_argument("-sc", "--scout", type=int, default=0, help="Scouting data.")
    parser.add_argument(
        "-ML", "--ML", type=int, default=0, help="ML samples production."
    )
    parser.add_argument(
        "-w",
        "--wait",
        type=float,
        default=1,
        help="Wait time before submitting the next sample in hours (default = 1 hour). This is needed to avoid overloading the MIT T2 with xrootd requests.",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    options = parser.parse_args()

    # set up logging
    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # script parameters
    username = getpass.getuser()
    outdir = "/data/submit/" + username + "/SUEP/{tag}/{sample}/"
    if os.path.isdir("/data/submit/cms/store/user/" + username):
        outdir = "/data/submit/cms/store/user/" + username + "/SUEP/{tag}/{sample}/"
        outdir_condor = "/cms/store/user/" + username + "/SUEP/{tag}/{sample}/"
    elif os.path.isdir("/data/submit/" + username):
        outdir = "/data/submit/" + username + "/SUEP/{tag}/{sample}/"
        outdir_condor = "/" + username + "/SUEP/{tag}/{sample}/"
    else:
        logging.error(
            "Cannot access /data/submit/$USER or /data/submit/cms/store/user/$USER!"
        )
        sys.exit()
    workdir = os.getcwd()
    logdir = "/work/submit/" + username + "/SUEP/logs/"
    default_input_redirector = "root://xrootd.cmsaf.mit.edu/"
    output_redirector = "root://submit50.mit.edu/"
    proxy_base = f"x509up_u{os.getuid()}"
    home_base = os.environ["HOME"]

    # define which file you want to run, the output file name and extension that it produces
    # these will be transferred back to outdir/outdir_condor
    if options.channel == "ggF":
        if options.scout == 1:
            condor_file = "condor_Scouting.py"
            outfile = "out"
            file_ext = "hdf5"
        elif options.ML == 1:
            condor_file = "condor_ML.py"
            outfile = "out"
            file_ext = "hdf5"
        elif options.cutflow == 1:
            condor_file = "condor_SUEP_cutflow.py"
            outfile = "cutflow"
            file_ext = "coffea"
        else:
            condor_file = "condor_SUEP_ggF.py"
            outfile = "out"
            file_ext = "hdf5"
    elif options.channel == "WH":
        condor_file = "condor_SUEP_WH.py"
        outfile = "out"
        file_ext = "hdf5"

    # Making sure that the proxy is good
    lifetime = check_proxy(time_min=100)
    logging.info(f"--- proxy lifetime is {round(lifetime, 1)} hours")
    proxy_copy = os.path.join(home_base, proxy_base)

    missing_samples = []

    with open(options.input) as stream:
        for iSample, sample_path in enumerate(stream.read().split("\n")):
            # skip commented out or incorrect sample paths
            if len(sample_path) < 1:
                continue
            if "#" in sample_path or (
                "/" in sample_path and len(sample_path.split("/")) <= 1
            ):
                continue

            # extract sample name from each sample path
            if "/" in sample_path:
                sample_name = sample_path.split("/")[-1]
            else:
                sample_name = sample_path
            if sample_name.endswith(
                ".root"
            ):  # case where 1 file is given as input, treated as a separate sample
                sample_name = sample_name.replace(".root", "")

            # if the redirector is specified, take it, and strip it from the sample path, if not use the default
            if sample_path.startswith("root://"):
                sample_input_redirector = "root://" + sample_path.split("//")[1] + "/"
                sample_path = sample_path.replace(sample_input_redirector, "")
            else:
                sample_input_redirector = default_input_redirector

            logging.info("-- sample : " + sample_name)

            # set up the jobs directory
            jobs_dir = "/".join([logdir, options.tag, sample_name])
            if os.path.isdir(jobs_dir):
                if not options.force:
                    logging.error(" " + jobs_dir + " already exists !")
                    continue
                else:
                    logging.warning(
                        " " + jobs_dir + " already exists, forcing its deletion!"
                    )
                    shutil.rmtree(jobs_dir)
                    os.makedirs(jobs_dir)
            else:
                os.makedirs(jobs_dir)

            # get the filelist with xrootd
            Raw_list = []
            comm = subprocess.Popen(
                ["xrdfs", sample_input_redirector, "ls", sample_path],
                stdout=subprocess.PIPE,
            )
            Raw_list = comm.communicate()[0].decode("utf-8").split("\n")
            Raw_list = [sample_input_redirector + f for f in Raw_list if ".root" in f]
            if len(Raw_list) == 0:
                missing_samples.append(sample_name)

            # limit to max number of files, if specified
            if options.maxFiles > 0:
                Raw_list = Raw_list[: options.maxFiles]

            # write list of files to inputfiles.dat
            nfiles = 0
            with open(os.path.join(jobs_dir, "inputfiles.dat"), "w") as infiles:
                for full_file in Raw_list:
                    just_file = full_file.split("/")[-1]
                    infiles.write(full_file + "\t" + just_file.split(".root")[0] + "\n")
                    nfiles += 1
                infiles.close()
            fin_outdir = outdir.format(tag=options.tag, sample=sample_name)
            fin_outdir_condor = outdir_condor.format(
                tag=options.tag, sample=sample_name
            )
            os.system(f"mkdir -p {fin_outdir}")

            # write the executable we give to condor
            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                extras = ""
                script = script_TEMPLATE.format(
                    proxy=proxy_base,
                    ismc=options.isMC,
                    era=options.era,
                    doSyst=options.doSyst,
                    doInf=options.doInf,
                    outdir=fin_outdir_condor,
                    dataset=sample_name,
                    condor_file=condor_file,
                    outfile=outfile,
                    file_ext=file_ext,
                    redirector=output_redirector,
                    extras=extras,
                )
                scriptfile.write(script)
                scriptfile.close()

            # write condor submission script
            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file=",".join(
                        [
                            workdir + "/" + condor_file,
                            workdir + "/workflows",
                            workdir + "/data",
                            proxy_copy,
                        ]
                    ),
                    # just_file=just_file,
                    jobdir=jobs_dir,
                    proxy=proxy_base,
                    queue=options.queue,
                    user=username,
                )
                condorfile.write(condor)
                condorfile.close()

            # write the git info to a file in the output directory where the ntuples will be stored
            commit, diff = get_git_info()
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            with open(
                os.path.join(fin_outdir, f"gitinfo_{formatted_datetime}.txt"), "w"
            ) as gitinfo:
                gitinfo.write("Commit: \n" + commit + "\n")
                gitinfo.write("Diff: \n" + diff + "\n")
                gitinfo.close()

            # don't submit if it's a dryrun
            if options.dryrun:
                continue

            # wait before submitting the next sample
            if iSample != 0 and options.wait > 0:
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%H:%M")
                logging.info(
                    "Waiting {} hours ({:g} minutes) before submitting this sample... (current time: {})".format(
                        options.wait,
                        float("{:.{p}g}".format(options.wait * 60, p=2)),
                        formatted_time,
                    )
                )
                time.sleep(options.wait * 3600)

            # submit!
            htc = subprocess.Popen(
                "condor_submit " + os.path.join(jobs_dir, "condor.sub"),
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            out, err = htc.communicate()
            exit_status = htc.returncode
            logging.info(f"condor submission status : {exit_status}")

    if len(missing_samples) > 0:
        logging.info(r"\Samples with no input files:")
        for s in missing_samples:
            logging.info(s)

    logging.info("All done!")


if __name__ == "__main__":
    main()
