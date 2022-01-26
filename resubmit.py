import sys, os, subprocess
import logging
import time

logging.basicConfig(level=logging.DEBUG)

nResubmits = 1
nHours = 1
tag = 'noJetId'
filelist = 'filelist/list_test.txt'

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
    logging.info("Resubmission"+str(i))
    logging.info("Removing all jobs...")
    os.system('condor_rm {}'.format(os.environ['USER']))
    logging.info("Executing monitor.py...")
    os.system("/home/submit/lavezzo/miniconda3/condabin/conda run -n SUEP python monitor.py --tag={} --input={} -r=1".format(tag, filelist))
    # don't wait if it's the last submission
    if i == nResubmits - 1: break
    logging.info("Sleeping for",nHours,"hours...")
    time.sleep(sleepTime)