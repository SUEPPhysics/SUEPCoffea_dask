import os, sys
import argparse
import logging
import pwd
import subprocess
import shutil
import getpass

logging.basicConfig(level=logging.DEBUG)

script_TEMPLATE = """#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820
#export PATH=$USER_PATH
export X509_USER_PROXY={proxy}
export X509_CERT_DIR=/etc/grid-security/certificates

echo "hostname:"
hostname

echo "----- Found Proxy in: $X509_USER_PROXY"

voms-proxy-info --all

echo $_CONDOR_SCRATCH_DIR
cd   $_CONDOR_SCRATCH_DIR

export X509_USER_PROXY=/afs/cern.ch/user/c/cfreer/{proxy}

echo "xrdcp $2 $1.root"
xrdcp $2 $1.root
ls

echo "python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=$2"
python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=$1.root
rm $1.root

#echo "----- transferring output to scratch :"
#echo "mv condor_out.hdf5 {outdir}/$3.hdf5"
mv condor_out.hdf5 {outdir}/$3.hdf5

echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
request_disk          = 1024
executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid) $(fileid)
transfer_input_files  = {transfer_file}
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log
initialdir            = {jobdir}
#when_to_transfer_output = ON_EXIT
#transfer_output_remaps = "condor_out.hdf5 = {outdir}$(fileid).hdf5"
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+JobFlavour           = "{queue}"

queue jobid, fileid from {jobdir}/inputfiles.dat
"""


def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="")
    parser.add_argument("-q"   , "--queue" , type=str, default="espresso", help="")
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
    userletter = username[0]
    outdir = '/eos/user/' + userletter + '/' + username + '/SUEP/{tag}/{sample}/'


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
        if lifetime < 139.00: # we want at least 3 hours
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms --hours=140')
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
                input_list = "/afs/cern.ch/user/c/cfreer/Rawfiles_A01/{}/RawFiles.00".format(sample_name)
                Raw_list = open(input_list, "r")
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                     for i in Raw_list:
                         full_file = i.split(" ")[0]
                         just_file = full_file.split("/")[-1]
                         infiles.write(full_file+"\t"+just_file.split(".root")[0]+"\n")
                     infiles.close()
 
            eosoutdir =  outdir.format(tag=options.tag,sample=sample_name)
             
            os.system("mkdir -p {}".format(eosoutdir))

            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    proxy=proxy_base,
                    ismc=options.isMC,
                    era=options.era,
                    outdir=eosoutdir,          
                    dataset=sample_name
                )
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        "../condor_SUEP_WS.py",
                        "../workflows",
                        "../data",
                    ]),
                    just_file=just_file,
                    jobdir=jobs_dir,
                    outdir = eosoutdir,
                    proxy=proxy_copy,
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
