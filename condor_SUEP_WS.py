import os
import json
import argparse

#Import coffea specific features
from coffea.processor import run_uproot_job, futures_executor


#Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")
parser.add_argument('--analyzer', type=str, default="GluGlu", help="")
parser.add_argument('--outputdir', type=str, default=None, help="")
options = parser.parse_args()

out_dir = options.outputdir if options.outputdir else os.getcwd()

## Select analyzer

if options.analyzer == "GluGlu":
  from workflows.SUEP_coffea import *
elif options.analyzer == "ZH_simple":
  from workflows.SUEP_coffea_ZH_simple import *


modules_era = []
#Run the SUEP code. Note the xsection as input. For Data the xsection = 1.0 from above
modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, output_location=out_dir))

for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 1,
                       'schema': processor.NanoAODSchema,
                       'xrootdtimeout': 10,
        },
        chunksize=250000
    )
