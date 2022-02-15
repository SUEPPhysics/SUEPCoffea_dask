import os, sys
import time
import argparse
import glob
import logging
from termcolor import colored
from shutil import copyfile
import pandas as pd
import subprocess

logging.basicConfig(level=logging.DEBUG)

def isFileGood(fname, label='ch'):
    try:
        with pd.HDFStore(fname) as store: data = store[label] 
        return 1
    except:
        return 0
    
def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="input"  , required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan", required=True)
    parser.add_argument("-r", "--resubmit"  , type=int, default=0          , help="")
    parser.add_argument("-m", "--move"  , type=int, default=0          , help="Move files to move_dir (/work/submit/) while you check if they are corrupted.")
    options = parser.parse_args()

    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    username = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    out_dir = "/mnt/T3_US_MIT/hadoop/scratch/" + username  + "/SUEP/" + options.tag + "/{}/"
    out_dir_xrd = "/scratch/" + username  + "/SUEP/" + options.tag + "/{}/"
    move_dir = "/work/submit/" + username + "/SUEP/" + options.tag + "/{}/"
    
    if options.move:
        if not os.path.isdir("/work/submit/" + username + "/SUEP/" + options.tag): 
            subprocess.run(['mkdir', "/work/submit/" + username + "/SUEP/" + options.tag])

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
            
            t_start = time.time()
            
            sample_name = sample.split("/")[-1]
            jobs_dir =  '_'.join(['jobs', options.tag, sample_name])
            
            # delete files that are corrupted (i.e., empty)
            for file in os.listdir(out_dir.format(sample_name)):
                size = os.path.getsize(out_dir.format(sample_name) + "/" + file)
                if size == 0: subprocess.run(['rm',out_dir.format(sample_name) + "/" + file])
            
            #We write the original list. inputfiles.dat will now contain missing files. Compare with original list
            if os.path.isfile(jobs_dir + "/" + "original_inputfiles.dat") != True:
                copyfile(jobs_dir + "/" + "inputfiles.dat", jobs_dir + "/" + "original_inputfiles.dat")

            #Find out the jobs that run vs the ones that failed
            jobs = [line.rstrip() for line in open(jobs_dir + "/" + "original_inputfiles.dat")]
            
            njobs = len(jobs)
            complete_list = os.listdir(out_dir.format(sample_name)) 
            nfile = len(complete_list)
            
            #Print out the results
            logging.info(
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name)
            )
            
            #Print out the results
            percent = nfile / njobs  * 100
            logging.info(
                colored("\t\t --> completed", "green") if njobs==nfile else colored(
                    "\t\t --> ({}/{}) finished. {:.1f}% complete".format(nfile,njobs,percent), 'red'
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
                
                
            if options.move:
                
                if not os.path.isdir(move_dir.format(sample_name)): os.system("mkdir " + move_dir.format(sample_name)) 
            
                # delete files that are corrupted (i.e., empty)
                for file in os.listdir(move_dir.format(sample_name)):
                    size = os.path.getsize(move_dir.format(sample_name) + "/" + file)
                    if size == 0: subprocess.run(['rm',move_dir.format(sample_name) + "/" + file])
                    
                # get list of files already in /work
                movedFiles = os.listdir(move_dir.format(sample_name))

                # get list of files in T3
                allFiles = os.listdir(out_dir.format(sample_name))

                # get list of files missing from /work that are in T3
                filesToMove = list(set(allFiles) - set(movedFiles))

                # move those files
                logging.info("Moving " + str(len(filesToMove)) + " files to " + move_dir.format(sample_name))
                for file in filesToMove:
                    subprocess.run(['xrdcp', 
                               "root://t3serv017.mit.edu/" + out_dir.split('hadoop')[-1].format(sample_name) + "/" + file,
                               move_dir.format(sample_name) + "/"])
                
            # give time to xrootd on T2 to process the jobs
            # for now, this is fixed such that it waits 15mins for 1000 files
            # if options.resubmit:
            #     sleepTime = len(jobs_resubmit) * 15.0*60.0/1000.0
            #     t_end = time.time()
            #     mod = (t_end - t_start) 
            #     logging.info("Deleting, resubmitting, and moving files took " + str(round(mod)) + " seconds")
            #     logging.info("Sleeping for "+str(round(sleepTime - mod))+" seconds")
            #     time.sleep(sleepTime - mod)
                

if __name__ == "__main__":
    main()
