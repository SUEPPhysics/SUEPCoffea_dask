import os
import json
import argparse
import time
import numpy

#Import coffea specific features
from coffea.processor import run_uproot_job, futures_executor

#SUEP Repo Specific
from workflows.SUEP_coffea import *

#Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")
parser.add_argument('--wait', type=float, default=0.0, help="Wait [0,<wait>] seconds, needed so that MIT T2 XRootD doesn't crash.") 
options = parser.parse_args()

if options.wait != 0:
    random_time = np.random.randint(0,int(options.wait))
    print("They can't handle my power. I must sleep", random_time, "seconds.")
    time.sleep(random_time)

out_dir = os.getcwd()
modules_era = []
#Run the SUEP code. Note the xsection as input. For Data the xsection = 1.0 from above
modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, do_inf=True, output_location=out_dir))

for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 2,
                       'schema': processor.NanoAODSchema,
                       'xrootdtimeout': 10,
        },
        chunksize=1000,
    )
