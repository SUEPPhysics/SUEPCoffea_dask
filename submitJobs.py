#!/usr/bin/env python
import os
import sys

print()
print('START')
print()
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
NumberOfJobs= sys.argv[1] # number of jobs to be submitted
FileFolder  = sys.argv[2] # File folder with all the files
OutputDir   = sys.argv[3] # Where to put the stuff
queue       = sys.argv[4] # Which queue to use: expresso (20min), microcentury (1h), longlunch (2h), workday (8h), tomorrow (1d), testmatch (3d), nextweek (1w)
doSubmit    = sys.argv[5] # Whether to submit or not
files = [FileFolder + "/" + f for f in os.listdir(FileFolder)] # list with all the files  
tag=OutputDir

if NumberOfJobs == -1: NumberOfJobs = len(files)

########   customization end   #########

path = os.getcwd()
print()
print('do not worry about folder creation:')
os.system("mkdir %s" %OutputDir)
os.system("rm -rf %s/tmp"%tag)
os.system("rm -rf %s/exec"%tag)
os.system("rm -rf %s/batchlogs"%tag)
os.system("mkdir %s/tmp"%tag)
os.system("mkdir %s/exec"%tag)
print()

##### loop for creating and sending jobs #####
for x in range(int(NumberOfJobs)):
    ##### creates jobs #######
    with open('%s/exec/job_'%tag+str(x)+'.sh', 'w') as fout:
        fout.write("#!/bin/bash\n")
        fout.write("echo 'START---------------'\n")
        fout.write("echo 'WORKDIR ' ${PWD}\n")
        fout.write("cd "+str(path)+"\n")
        fout.write("source /afs/cern.ch/user/c/cericeci/miniconda3/etc/profile.d/conda.sh\n")
        fout.write("conda activate coffea\n")
        fout.write("python condor_SUEP_WS.py  --isMC=1 --era=2018 --dataset=DY --analyzer=ZH_simple --infile=%s --outputdir=%s\n"%(files[x], OutputDir)) 
        fout.write("echo 'STOP---------------'\n")
        fout.write("echo\n")
        fout.write("echo\n")
    os.system("chmod 755 %s/exec/subjob_"%tag+str(x)+".sh")
   
###### create submit.sub file ####
    
os.mkdir("%s/batchlogs"%tag)
with open('submit.sub', 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(Proxy_path) $(ClusterId)$(ProcId)\n")
    fout.write("output                  = %s/batchlogs/$(ClusterId).$(ProcId).out\n"%tag)
    fout.write("error                   = %s/batchlogs/$(ClusterId).$(ProcId).err\n"%tag)
    fout.write("log                     = %s/batchlogs/$(ClusterId).log\n"%tag)
    fout.write("Proxy_path              = /afs/cern.ch/user/c/cericeci/private/x509up_u88688\n")
    fout.write('+JobFlavour = "%s"\n' %(queue))
    fout.write("\n")
    fout.write("queue filename matching (%s/exec/job_*sh)\n"%tag)
    
###### sends bjobs ######
if bool(doSubmit):
  os.system("echo submit.sub")
  os.system("condor_submit submit.sub")
   
print()
print("your jobs:")
os.system("condor_q")
print()
print('END')
print()
