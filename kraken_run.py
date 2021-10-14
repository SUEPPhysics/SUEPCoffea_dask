import os, sys
import argparse
import logging
import pwd
import subprocess
import shutil
import time
#from termcolor import colored

logging.basicConfig(level=logging.DEBUG)

script_TEMPLATE = """#!/bin/bash

export X509_USER_PROXY={proxy}
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820
export HOME=.

echo "hostname:"
hostname

echo $_CONDOR_SCRATCH_DIR 
cd   $_CONDOR_SCRATCH_DIR 

python3 --version
export PYTHONPATH

python3 --version
echo "----- Found Proxy in: $X509_USER_PROXY"
echo "python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=$2"
python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=$2

ls

echo "----- transferring output to scratch :"
mv tree_$1_coffea.root {final_outdir}
mv *.parquet {final_outdir}
echo "----- directory after running :"
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 1024
executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid)
transfer_input_files  = {transfer_file}, $Fp(/home/freerc/.local/lib/python3.8/site-packages/)
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log
initialdir            = {jobdir}
transfer_output_files = ""
#requirements          = ((Arch == "X86_64") && ((GLIDEIN_Site =!= "MIT_CampusFactory") || (GLIDEIN_Site == "MIT_CampusFactory" && BOSCOCluster == "ce03.cmsaf.mit.edu" && BOSCOGroup == "bosco_cms" && HAS_CVMFS_cms_cern_ch)))
Environment           = "PYTHONPATH=/home/freerc/.local/lib/python3.8/site-packages/:/usr/lib64/python3.8/site-packages"
DIRNAME               = "/home/freerc/.local/lib/python3.8/site-packages/"
PYTHONPATH            = $ENV(/home/freerc/.local/lib/python3.8/site-packages/:/usr/lib64/python3.8/site-packages)
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+JobFlavour           = "{queue}"

queue jobid from {jobdir}/inputfiles.dat
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="")
    parser.add_argument("-q"   , "--queue" , type=str, default="testmatch", help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2017"     , help="")
    parser.add_argument("-f"   , "--force" , action="store_true"          , help="recreate files and jobs")
    parser.add_argument("-s"   , "--submit", action="store_true"          , help="submit only")
    parser.add_argument("-dry" , "--dryrun", action="store_true"          , help="running without submission")
    parser.add_argument("--redo-proxy"     , action="store_true"          , help="redo the voms proxy")

    options = parser.parse_args()

    # Making sure that the proxy is good
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    proxy_copy = os.path.join(home_base,proxy_base)
    outdir = '/data/t3home000/freerc/SUEP/{tag}/{sample}/'

    regenerate_proxy = False
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0: # we want at least 10 hours
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms')
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)

    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[-1]
            jobs_dir = '_'.join(['jobs', options.tag, sample_name])
            logging.info("-- sample_name : " + sample)
            print(sample_name)
            if os.path.isdir(jobs_dir):
                if not options.force:
                    logging.error(" " + jobs_dir + " already exist !")
                    continue
                else:
                    logging.warning(" " + jobs_dir + " already exists, forcing its deletion!")
                    shutil.rmtree(jobs_dir)
                    os.mkdir(jobs_dir)
            else:
                os.mkdir(jobs_dir)
            
            if not options.submit:
                # ---- getting the list of file for the dataset (For Kraken these are stored in catalogues on T2)
                input_list = "/home/cmsprod/catalog/t2mit/nanosu/A00/{}/RawFiles.00".format(sample_name)
                Raw_list = open(input_list, "r")
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                     for i in Raw_list:
                         infiles.write(i.split(" ")[0]+"\n")
                     infiles.close()
            fin_outdir =  outdir.format(tag=options.tag,sample=sample_name)
            os.system("mkdir -p {}".format(fin_outdir))
  
            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    home_base=home_base,
                    proxy=proxy_copy,
                    ismc=options.isMC,
                    era=options.era,
                    final_outdir=fin_outdir,          
                    dataset=sample_name
                )
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        "../condor_SUEP_WS.py",
                        "../workflows/SUEP_coffea.py",
                        "../workflows/SumWeights.py",
                        "../xsections_2018.yaml"
                    ]),
                    jobdir=jobs_dir,
                    queue=options.queue
                )
                condorfile.write(condor)
                condorfile.close()
            if options.dryrun:
                continue
 
            htc = subprocess.Popen(
                "condor_submit " + os.path.join(jobs_dir, "condor.sub"),
                shell  = True,
                stdin  = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                close_fds=True
            )
            out, err = htc.communicate()
            exit_status = htc.returncode
            logging.info("condor submission status : {}".format(exit_status))

if __name__ == "__main__":
    main()
