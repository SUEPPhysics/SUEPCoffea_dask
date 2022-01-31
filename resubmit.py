import sys, os, subprocess
import argparse
import logging
import time

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"       , type=str, help="Dataset tag.")
parser.add_argument("-r"   , "--resubmits" , type=int, default=10     , help="Number of resubmissions.", required=False)
parser.add_argument("-hours"   , "--hours"     , type=float, default=1.0  , help="Number of hours per resubmission.", required=False)
parser.add_argument("-m"   , "--move"      , type=int, default=0, help="Move.")
parser.add_argument("-dry" , "--dryrun"    , type=int, default=0, help="running without submission")

options = parser.parse_args()
nResubmits = options.resubmits
nHours = options.hours
tag = options.tag

username = os.environ['USER']
dataDir = "/mnt/T3_US_MIT/hadoop/scratch/{}/SUEP/{}/".format(username, tag)
moveDir = "/work/submit/{}/SUEP/{}/".format(username, tag)

# Making sure that the proxy is good
proxy_base = 'x509up_u{}'.format(os.getuid())
home_base  = os.environ['HOME']
proxy_copy = os.path.join(home_base,proxy_base)
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
    subDirs = os.listdir(dataDir)
    for subDir in subDirs:
        for file in os.listdir(dataDir + subDir):
            size = os.path.getsize(dataDir + subDir + "/" + file)
            if size == 0: subprocess.run(['rm',dataDir + subDir + "/" + file])
        
    if not options.dryrun:
        logging.info("Executing monitor.py for data...")
        os.system("python3 monitor.py --tag={} --input={} -r=1".format(tag, 'filelist/list_2018_data_A01.txt'))
        logging.info("Executing monitor.py for MC...")
        os.system("python3 monitor.py --tag={} --input={} -r=1".format(tag, 'filelist/list_2018_MC_A01.txt'))
    
    if options.move:
        
        t_startToMove = time.time()
        for subDir in subDirs:
            
            # get list of files already in /work
            movedFiles = os.listdir(moveDir + subDir)

            # get list of files in T3
            allFiles = os.listdir(dataDir + subDir)

            # get list of files missing from /work that are in T3
            filesToMove = list(set(allFiles) - set(movedFiles))

            # move those files
            logging.info("Moving " + str(len(filesToMove)) + " files to " + moveDir + subDir)
            if not os.path.isdir(moveDir + subDir): os.system("mkdir " + moveDir + subDir) 
            for file in filesToMove:
                subprocess.run(['xrdcp', 
                           "root://t3serv017.mit.edu/" + dataDir.split('hadoop')[-1] + subDir + "/" + file,
                           moveDir + subDir + "/"])
            
        t_endToMove = time.time()
    
    # don't wait if it's the last submission
    if i == nResubmits - 1: 
        logging.info("All done")
        break
        
    # take into account the time used to move files
    mod = 0
    if options.move:
        mod = (t_endToMove - t_startToMove) 
        logging.info("Moving files took " + str(round(mod)) + " seconds")
        
    logging.info("Sleeping for "+str(round(sleepTime - mod))+" seconds")
    logging.info("("+str(round(nHours - mod/(60*60), 2))+" hours)...")
    time.sleep(sleepTime - mod)