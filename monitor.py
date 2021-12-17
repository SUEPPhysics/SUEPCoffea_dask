import os, sys
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
    parser.add_argument("-c", "--corrupted"  , type=int, default=0          , help="")
    parser.add_argument("-m", "--move"  , type=int, default=0          , help="Move files to out_dir_c (/work/submit/) while you check if they are corrupted. Only useful if -c=1 is called.")
    options = parser.parse_args()

    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    username = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    out_dir = "/mnt/T3_US_MIT/hadoop/scratch/" + username  + "/SUEP/" + options.tag + "/{}/"
    out_dir_xrd = "/scratch/" + username  + "/SUEP/" + options.tag + "/{}/"
    out_dir_c = "/work/submit/" + username + "/SUEP/" + options.tag + "/{}/"
    
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
            sample_name = sample.split("/")[-1]
            jobs_dir = 'jobs/' + '_'.join(['jobs', options.tag, sample_name])
            
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
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name)
            )
            
            #If files are corrupted we resubmit with the same condor.sub 
            if options.corrupted:
                file_names = []
                nCorrupted = 0
                
                if options.move:
                    if not os.path.isdir(out_dir_c.format(sample_name)): 
                        subprocess.run(['mkdir', out_dir_c.format(sample_name)])
                        
                # instead of deleting on hadoop, create a trash bin
                if not os.path.isdir(out_dir.format(sample_name) + '/badfiles'):
                    subprocess.run(['mkdir',out_dir.format(sample_name) + '/badfiles'])
                
                # check which files are corrupted
                for item in complete_list:
                    if 'hdf5' not in item: continue
                    
                    # if file exists in out_dir_c (usually /work/submit) check there
                    if os.path.exists(out_dir_c.format(sample_name) + item):
                                   
                        if isFileGood(out_dir_c.format(sample_name) + item, 'vars'): continue
                        nCorrupted+=1
                        
                        # delete file from out_dir_c
                        subprocess.run(['rm', out_dir_c.format(sample_name) + item])
                    
                    # if not, xrdcp file to local dir, and check it
                    else:
                        
                        # either copy it locally temporarily
                        if not options.move:
                            subprocess.run(['xrdcp',
                                            "root://t3serv017.mit.edu/" + out_dir_xrd.format(sample_name) + item,
                                            "."])
                            good = isFileGood(item, 'vars')
                            
                            # delete file from local dir once the check is finished
                            subprocess.run(['rm', item])
                            
                            if good: continue
                            else: 
                                nCorrupted+=1
                            
                        # or in out_dir_c directly
                        else:
                            subprocess.run(['xrdcp',
                                            "root://t3serv017.mit.edu/" + out_dir_xrd.format(sample_name) + item,
                                            out_dir_c.format(sample_name)])
                            good = isFileGood(out_dir_c.format(sample_name) + item, 'vars')
                            
                            if good: continue
                            else: 
                                subprocess.run(['rm', out_dir_c.format(sample_name) + item])
                                nCorrupted+=1
                        
                    # mv file on /hadoop to a temporary badfiles/ dir
                    subprocess.run(['mv', out_dir.format(sample_name) + item, out_dir.format(sample_name) + '/badfiles'])
                    
                    # append file to the ones that need to be re run
                    file_names.append(item.split('.hdf5')[0])
                                   
                #Print out the results
                logging.info(
                    colored("\t\t --> No corrupted files", "green") if nCorrupted==0 else colored(
                        "\t\t --> ({}/{}) corrupted files".format(nCorrupted,nfile), 'red'
                    )
                )
                
                
            # count again, after we move the corrupted files to the bin
            complete_list = os.listdir(out_dir.format(sample_name)) 
            nfile = len(complete_list)
            
            #Print out the results
            logging.info(
                colored("\t\t --> completed", "green") if njobs==nfile else colored(
                    "\t\t --> ({}/{}) finished".format(nfile,njobs), 'red'
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
                

if __name__ == "__main__":
    main()
