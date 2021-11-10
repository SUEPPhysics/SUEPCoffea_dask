import os, sys
import argparse
import logging
import pwd
import subprocess
import shutil
import getpass

logging.basicConfig(level=logging.DEBUG)

script_TEMPLATE = """#!/bin/bash


export X509_USER_PROXY={proxy}
export PATH=$USER_PATH

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820
export HOME=.

echo "hostname:"
hostname

echo "----- Found Proxy in: $X509_USER_PROXY"
echo "python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=$2"
echo "xrdcp $2 temp.root"
xrdcp $2 temp.root
python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=temp.root
rm temp.root

echo "----- transferring output to scratch :"
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 1024
executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid)
should_transfer_files = YES
transfer_input_files  = {transfer_file}
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log
initialdir            = {jobdir}
when_to_transfer_output = ON_EXIT
requirements          = (BOSCOCluster == "ce03.cmsaf.mit.edu")
transfer_output_remaps = "condor_out.hdf5 = {final_outdir}/$(ProcId).hdf5"
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest"
+JobFlavour           = "{queue}"

queue jobid from {jobdir}/inputfiles.dat
"""

# requirements          = (BOSCOCluster == "t3serv008.mit.edu" || (BOSCOGroup == "bosco_cms" && BOSCOCluster == "ce03.cmsaf.mit.edu") )


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
    username = getpass.getuser()
    outdir = '/work/submit/'+username+'/SUEP/{tag}/{sample}/' 

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
        if lifetime < 3.0: # we want at least 3 hours
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
                input_list = "/home/submit/" + username + "/temp_RawFiles/A01/{}/RawFiles.00".format(sample_name)
                Raw_list = open(input_list, "r")
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                     for i in Raw_list:
                         #i=i.split(" ")[0].replace('root://xrootd.cmsaf.mit.edu/','/mnt/hadoop/cms')
                         #infiles.write(i+"\n")
                         infiles.write(i.split(" ")[0]+"\n")
                     infiles.close()
            fin_outdir =  outdir.format(tag=options.tag,sample=sample_name)
            os.system("mkdir -p {}".format(fin_outdir))
  
            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    #home_base=home_base,
                    proxy=proxy_base,
                    ismc=options.isMC,
                    era=options.era,
                    #final_outdir=fin_outdir,          
                    dataset=sample_name
                )
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        "../condor_SUEP_WS.py",
                        "../workflows",
                        #"../workflows/SUEP_coffea.py",
                        #"../workflows/SumWeights.py",
                        "../data",
                        proxy_copy
                    ]),
                    jobdir=jobs_dir,
                    final_outdir = fin_outdir,
                    proxy=proxy_base,
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
