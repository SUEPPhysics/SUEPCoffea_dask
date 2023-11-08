import argparse
import getpass
import logging
import os
import shutil
import subprocess

from plotting.plot_utils import check_proxy

logging.basicConfig(level=logging.DEBUG)

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

echo "----- Found Proxy in: $X509_USER_PROXY"
echo "python3 {condor_file} --jobNum=$1 --isMC={ismc} --era={era} --doInf={doInf} --doSyst={doSyst} --dataset={dataset} --infile=$2"
python3 {condor_file} --jobNum=$1 --isMC={ismc} --era={era} --doInf={doInf} --doSyst={doSyst} --dataset={dataset} --infile=$2

#echo "----- transferring output to scratch :"
echo "xrdcp {outfile}.{file_ext} {redirector}/{outdir}/$3.{file_ext}"
xrdcp {outfile}.{file_ext} {redirector}/{outdir}/$3.{file_ext}

{extras}

echo "rm *.{file_ext}"
rm *.{file_ext}

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
#requirements          = (target.MACHINE == t3btch115.mit.edu)
#requirements          = ( ((BOSCOCluster == "t3serv008.mit.edu") || (BOSCOGroup == "bosco_cms" && BOSCOCluster == "ce03.cmsaf.mit.edu")) && HAS_CVMFS_cms_cern_ch )
#requirements          = (BOSCOGroup == "bosco_cms" && BOSCOCluster == "ce03.cmsaf.mit.edu"  && Machine =!= LastRemoteHost && HAS_CVMFS_cms_cern_ch)
#requirements          = (BOSCOCluster == "t3serv008.mit.edu" && Machine =!= LastRemoteHost && HAS_CVMFS_cms_cern_ch )
# requirements          = ( BOSCOCluster =!= "t3serv008.mit.edu" && BOSCOCluster =!= "ce03.cmsaf.mit.edu")
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
    parser.add_argument("-sc", "--scout", type=int, default=0, help="Scouting data.")
    parser.add_argument(
        "-p", "--private", type=int, default=0, help="Private SUEP samples."
    )
    parser.add_argument(
        "-ML", "--ML", type=int, default=0, help="ML samples production."
    )
    parser.add_argument(
        "-WH", "--WH", type=int, default=0, help="WH ntuples production."
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

    options = parser.parse_args()

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
        print("Cannot access /data/submit/$USER or /data/submit/cms/store/user/$USER!")
        exit()
    workdir = os.getcwd()
    logdir = "/work/submit/" + username + "/SUEPCoffea_dask/logs/"
    redirector = "root://submit50.mit.edu/"
    proxy_base = f"x509up_u{os.getuid()}"
    home_base = os.environ["HOME"]

    # define which file you want to run, the output file name and extension that it produces
    # these will be transferred back to outdir/outdir_condor
    if options.scout == 1:
        condor_file = "condor_Scouting.py"
        outfile = "out"
        file_ext = "hdf5"
    elif options.ML == 1:
        condor_file = "condor_ML.py"
        outfile = "out"
        file_ext = "hdf5"
    elif options.WH == 1:
        condor_file = "condor_SUEP_WH.py"
        outfile = "out"
        file_ext = "hdf5"
    else:
        condor_file = "condor_SUEP_WS.py"
        outfile = "out"
        file_ext = "hdf5"

    # Making sure that the proxy is good
    lifetime = check_proxy(time_min=100)
    logging.info(f"--- proxy lifetime is {round(lifetime, 1)} hours")
    proxy_copy = os.path.join(home_base, proxy_base)

    missing_samples = []

    with open(options.input) as stream:
        for sample in stream.read().split("\n"):
            if len(sample) < 1:
                continue
            if "#" in sample or ("/" in sample and len(sample.split("/")) <= 1):
                continue
            if "/" in sample:
                sample_name = sample.split("/")[-1]
            else:
                sample_name = sample
            jobs_dir = "_".join([logdir + "jobs", options.tag, sample_name])
            logging.info("-- sample_name : " + sample)

            # set up the jobs directory
            if os.path.isdir(jobs_dir):
                if not options.force:
                    logging.error(" " + jobs_dir + " already exist !")
                    continue
                else:
                    logging.warning(
                        " " + jobs_dir + " already exists, forcing its deletion!"
                    )
                    shutil.rmtree(jobs_dir)
                    os.mkdir(jobs_dir)
            else:
                os.mkdir(jobs_dir)

            # ---- getting the list of file for the dataset by xrdfs ls

            if (options.era == "2018" or options.era == "2017") and options.private:
                userOwner = "bmaier/suep"
                sample_path = "/store/user/{}/official_private/{}/{}".format(
                    userOwner, options.era, sample_name
                )
            elif (
                options.era == "2016" or options.era == "2016apv"
            ) and options.private:
                userOwner = "cfreer/suep_correct"
                sample_path = "/store/user/{}/official_private/{}/{}".format(
                    userOwner, options.era, sample_name
                )
            elif not options.private and "/" in sample:
                sample_path = sample
            else:
                sys.exit("Double check this.")

            # get the filelist with xrootd (use same door to take advantage of caching and speed up the process)
            comm = subprocess.Popen(
                ["xrdfs", "root://xrootd5.cmsaf.mit.edu/", "ls", sample_path],
                stdout=subprocess.PIPE,
            )
            raw_input_list = comm.communicate()[0].decode("utf-8").split("\n")

            if raw_input_list == [""]:
                missing_samples.append(sample_name)

            # limit to max number of files, if specified
            if options.maxFiles > 0:
                raw_input_list = raw_input_list[: options.maxFiles]

            Raw_list = []
            for f in raw_input_list:
                if len(f) == 0:
                    continue
                new_f = f"root://xrootd.cmsaf.mit.edu/{f} 0 0 1 1 1 1"
                Raw_list.append(new_f)

            # write list of files to inputfiles.dat
            nfiles = 0
            with open(os.path.join(jobs_dir, "inputfiles.dat"), "w") as infiles:
                for i in Raw_list:
                    full_file = i.split(" ")[0]
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
                if options.cutflow:
                    extras += """
                    echo "xrdcp {outCutflow}.coffea {redirector}/{outdir}/$3_cutflow.coffea"
                    xrdcp {outCutflow}.coffea {redirector}/{outdir}/$3_cutflow.coffea
                    """.format(
                        outdir=fin_outdir_condor,
                        outCutflow="cutflow",
                        redirector=redirector,
                    )
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
                    redirector=redirector,
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

            # don't submit if it's a dryrun
            if options.dryrun:
                continue

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
        logging.info("\nMissing samples:")
        for s in missing_samples:
            logging.info(s)


if __name__ == "__main__":
    main()
