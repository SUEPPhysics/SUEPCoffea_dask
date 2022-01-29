import sys, os, subprocess
import argparse
import logging
import time

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-r"   , "--resubmits" , type=int, default=10     , help="Number of resubmissions.", required=False)
parser.add_argument("-n"   , "--hours"     , type=float, default=1.0  , help="Number of hours per resubmission.", required=False)
parser.add_argument("-t"   , "--tag"       , type=str, help="Dataset tag.")

options = parser.parse_args()
nResubmits = options.resubmits
nHours = options.hours
tag = options.tag

# Making sure that the proxy is good
proxy_base = 'x509up_u{}'.format(os.getuid())
home_base  = os.environ['HOME']
proxy_copy = os.path.join(home_base,proxy_base)
username = os.environ['USER']
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
    logging.info("--- proxy lifetime is {} hours".format(round(lifetime,1)))
    if lifetime < nHours * nResubmits * 1.5:
        logging.warning("--- proxy has expired !")
        regenerate_proxy = True

if regenerate_proxy:
    redone_proxy = False
    while not redone_proxy:
        status = os.system('voms-proxy-init -voms cms --hours=' + str(nHours * nResubmits * 1.5))
        if os.WEXITSTATUS(status) == 0:
            redone_proxy = True
    shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)


sleepTime = 60*60*nHours
for i in range(nResubmits):
    logging.info("Resubmission "+str(i))
    logging.info("Removing all jobs...")
    os.system('condor_rm {}'.format(os.environ['USER']))
    logging.info("Checking for corrupted files and removing them...")
    
    # delete files that are corrupted
    dataDir = "/mnt/T3_US_MIT/hadoop/scratch/{}/SUEP/{}/".format(username, tag)
    subdirs = os.listdir(dataDir)
    for subdir in subdirs:
        for file in os.listdir(dataDir + subdir):
            size = os.path.getsize(dataDir + subdir + "/" + file)
            if size == 0: 
                #print(dataDir + subdir + "/" + file)
                subprocess.run(['rm',dataDir + subdir + "/" + file])
        
    logging.info("Executing monitor.py for data...")
    os.system("python3 monitor.py --tag={} --input={} -r=1".format(tag, 'filelist/list_test.txt'))
    logging.info("Executing monitor.py for MC...")
    os.system("python3 monitor.py --tag={} --input={} -r=1".format(tag, 'filelist/list_2018_MC_A01.txt'))
    
    # don't wait if it's the last submission
    if i == nResubmits - 1: 
        logging.info("All done")
        break
        
    logging.info("Sleeping for "+str(nHours)+" hours...")
    time.sleep(sleepTime)