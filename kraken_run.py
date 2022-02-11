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

#export PATH=$USER_PATH
#export X509_CERT_DIR=/etc/grid-security/certificates
export SCRAM_ARCH=slc7_amd64_gcc820
export HOME=.

export X509_USER_PROXY={proxy}

echo "hostname:"
hostname

voms-proxy-info --all

echo "----- Found Proxy in: $X509_USER_PROXY"
echo "gfal-copy davs://xrootd.cmsaf.mit.edu:1094/$2 temp.root"
#xrdcp $2 temp.root
gfal-copy davs://xrootd.cmsaf.mit.edu:1094/$2 temp.root

ls

echo "python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=$2"
python3 condor_SUEP_WS.py --jobNum=$1 --isMC={ismc} --era={era} --dataset={dataset} --infile=temp.root
rm temp.root

#echo "----- transferring output to scratch :"
echo "gfal-copy condor_out.hdf5  davs://t3serv017.mit.edu:1094/{outdir}/$3.hdf5"
gfal-copy condor_out.hdf5  davs://t3serv017.mit.edu:1094/{outdir}/$3.hdf5


echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 1024
executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid) $(fileid)
should_transfer_files = YES
transfer_input_files  = {transfer_file}
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log
initialdir            = {jobdir}
when_to_transfer_output = ON_EXIT
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 3
#use_x509userproxy     = True
#x509userproxy         = /home/submit/freerc/x509up_u206148
#+AccountingGroup = "analysis.freerc"
#requirements          = ( BOSCOCluster == "t3serv008.mit.edu" || BOSCOCluster == "ce03.cmsaf.mit.edu")
requirements          = ( BOSCOCluster == "ce03.cmsaf.mit.edu")
#requirements          = ( BOSCOCluster == "t3serv008.mit.edu" )
#+DESIRED_Sites        = "T2_AT_Vienna,T2_BE_IIHE,T2_BE_UCL,T2_BR_SPRACE,T2_BR_UERJ,T2_CH_CERN,T2_CH_CERN_AI,T2_CH_CERN_HLT,T2_CH_CERN_Wigner,T2_CH_CSCS,T2_CH_CSCS_HPC,T2_CN_Beijing,T2_DE_DESY,T2_DE_RWTH,T2_EE_Estonia,T2_ES_CIEMAT,T2_ES_IFCA,T2_FI_HIP,T2_FR_CCIN2P3,T2_FR_GRIF_IRFU,T2_FR_GRIF_LLR,T2_FR_IPHC,T2_GR_Ioannina,T2_HU_Budapest,T2_IN_TIFR,T2_IT_Bari,T2_IT_Legnaro,T2_IT_Pisa,T2_IT_Rome,T2_KR_KISTI,T2_MY_SIFIR,T2_MY_UPM_BIRUNI,T2_PK_NCP,T2_PL_Swierk,T2_PL_Warsaw,T2_PT_NCG_Lisbon,T2_RU_IHEP,T2_RU_INR,T2_RU_ITEP,T2_RU_JINR,T2_RU_PNPI,T2_RU_SINP,T2_TH_CUNSTDA,T2_TR_METU,T2_TW_NCHC,T2_UA_KIPT,T2_UK_London_IC,T2_UK_SGrid_Bristol,T2_UK_SGrid_RALPP,T2_US_Caltech,T2_US_Florida,T2_US_MIT,T2_US_Nebraska,T2_US_Purdue,T2_US_UCSD,T2_US_Vanderbilt,T2_US_Wisconsin,T3_CH_CERN_CAF,T3_CH_CERN_DOMA,T3_CH_CERN_HelixNebula,T3_CH_CERN_HelixNebula_REHA,T3_CH_CMSAtHome,T3_CH_Volunteer,T3_US_HEPCloud,T3_US_NERSC,T3_US_OSG,T3_US_PSC,T3_US_SDSC"
#+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest"
+SingularityImage     = "/cvmfs/cvmfs.cmsaf.mit.edu/submit/work/submit/freerc/cvmfs/singularity_coffea/covfefe_gfal.sif"
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
    outdir = '/mnt/T3_US_MIT/hadoop/scratch/'+ username + '/SUEP/{tag}/{sample}/'
    outdir_condor = '/scratch/'+username+'/SUEP/{tag}/{sample}/'

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
                input_list = "/home/tier3/cmsprod/catalog/t2mit/nanosu/A01/{}/RawFiles.00".format(sample_name)
                Raw_list = open(input_list, "r")
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                     for i in Raw_list:
                         #i=i.split(" ")[0].replace('root://xrootd.cmsaf.mit.edu/','/mnt/hadoop/cms')
                         #infiles.write(i+"\n")
                         full_file = i.split(" ")[0]
                         full_file = full_file.replace('root://xrootd.cmsaf.mit.edu/','')
                         just_file = full_file.split("/")[-1]
                         infiles.write(full_file+"\t"+just_file.split(".root")[0]+"\n")
                         #infiles.write(i.split(" ")[0]+"\n")
                     infiles.close()
            fin_outdir =  outdir.format(tag=options.tag,sample=sample_name)
            fin_outdir_condor =  outdir_condor.format(tag=options.tag,sample=sample_name)
            os.system("mkdir -p {}".format(fin_outdir))
  
            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    #home_base=home_base,
                    proxy=proxy_base,
                    ismc=options.isMC,
                    era=options.era,
                    outdir=fin_outdir_condor,          
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
                    just_file=just_file,
                    jobdir=jobs_dir,
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
