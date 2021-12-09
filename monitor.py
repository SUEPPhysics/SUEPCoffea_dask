import os, sys
import argparse
import glob
import logging
from termcolor import colored
from shutil import copyfile
import pandas as pd
import subprocess

logging.basicConfig(level=logging.DEBUG)

def check_file(fname, label='ch'):
    with pd.HDFStore(fname) as store:
        data = store[label][0]
        
    # try:
    #     with pd.HDFStore(fname) as store: data = store[label] 
    #     return 0
    # except:
    #     return 1
    
def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="input"  , required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan", required=True)
    parser.add_argument("-r", "--resubmit"  , type=int, default=0          , help="")
    parser.add_argument("-c", "--corrupted"  , type=int, default=0          , help="")
    options = parser.parse_args()

    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    username = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    out_dir = "/mnt/T3_US_MIT/hadoop/scratch/" + username  + "/SUEP/" + options.tag + "/{}/"
    out_dir_c = "/work/submit/" + username + "/SUEP/" + options.tag + "/{}/"

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
        if lifetime < 10.0: # we want at least 100 hours
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms')
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        copyfile('/tmp/'+proxy_base,  proxy_copy)


    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[-1]
            jobs_dir = '_'.join(['jobs', options.tag, sample_name])

            #We write the original list. inputfiles.dat will now contain missing files. Compare with original list
            if os.path.isfile(jobs_dir + "/" + "original_inputfiles.dat") != True:
                copyfile(jobs_dir + "/" + "inputfiles.dat", jobs_dir + "/" + "original_inputfiles.dat")

            #Find out the jobs that run vs the ones that failed
            jobs = [line.rstrip() for line in open(jobs_dir + "/" + "original_inputfiles.dat")]
            njobs = len(jobs)
            complete_list = os.listdir(out_dir.format(sample_name)) 
            #complete_list = os.listdir("/work/submit/" + username  + "/SUEP/{}/{}/".format(options.tag, sample_name)) 
            nfile = len(complete_list)
            
            #Print out the results
            logging.info(
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name) +
                (
                    colored(" --> completed", "green") if njobs==nfile else colored(
                        " --> ({}/{}) running".format(nfile,njobs), 'red'
                    )
                )
            )

            #If files are missing we resubmit with the same condor.sub 
            if options.resubmit:
                print("resubmitting files for {}".format(sample))
                file_names = []
                for item in complete_list:
                    if 'hdf5' not in item: continue
                    file_names.append(item.split('.hdf5')[0])
                    
                jobs_resubmit = [item for item in jobs if item.split("\t")[-1] not in file_names]
                resubmit_file = open(jobs_dir + "/" + "inputfiles.dat","w")
                for redo_file in jobs_resubmit:
                    resubmit_file.write(redo_file+"\n")
                resubmit_file.close()
              
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
                
            #If files are corrupted we resubmit with the same condor.sub 
            if options.corrupted:
                print("resubmitting corrupted files for {}".format(sample))
                file_names = []
                
                # check which files are corrupted
                for item in complete_list:
                    if 'hdf5' not in item: continue
                    
                    # files need to be a in dir you can read from (i.e. not /scratch, but /work)
                    if check_file(out_dir_c.format(sample_name) + item, 'ch'): continue
                    file_names.append(item.split('.hdf5')[0])
                                   
                jobs_resubmit = [item for item in jobs if item.split("\t")[-1] in file_names]
                resubmit_file = open(jobs_dir + "/" + "inputfiles.dat","w")
                for redo_file in jobs_resubmit:
                    resubmit_file.write(redo_file+"\n")
                resubmit_file.close()
              
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
