import os, sys, glob
import pandas as pd
import json
import argparse
import time
import numpy
from tqdm import tqdm

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
options = parser.parse_args()

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
        executor_args={'workers': 1,
                       'schema': processor.NanoAODSchema,
                       'xrootdtimeout': 10,
        },
        chunksize=100000,
    )
    
#############################################################################
# MERGE
#############################################################################
# Merge each of the chunks' .hdf5 files together
# This won't be triggered unless the SUEP processor runs smoothly through
# all the chunks, thus assuring we processed all the events
# N.B.: Only merging df named 'vars' in the HDF5 object


def h5load(ifile, label):
    try:
        with pd.HDFStore(ifile, 'r') as store:
            try:
                data = store[label] 
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata
        
            except KeyError:
                print("No key",label,ifile)
                return 0, 0
    except:
        print("Some error occurred", ifile)
        return 0, 0
    
files = glob.glob("condor_*.hdf5")
if len(files) == 0: 
    print("No .hdf5 files found")
    sys.exit()

df_tot = None
metadata_tot = None
for ifile, file in enumerate(tqdm(files, desc="Merging")):
    
    df, metadata = h5load(file, 'vars') 
    
    ### Error out here
    if type(df) == int: 
        print("Something screwed up.")
        sys.exit()
        
    # no need to add empty ones
    if 'empty' in list(df.keys()): continue
    
    ### MERGE DF
    if df_tot is None: df_tot = df
    else: df_tot = pd.concat((df_tot, df))
    
    ### MERGE METADATA
    if metadata_tot is None: metadata_tot = metadata
    elif options.isMC: metadata_tot['gensumweight'] += metadata['gensumweight']
    
# SAVE OUTPUTS
if df_tot is None: 
    print("No events in df_tot.")
    df_tot = pd.DataFrame(['empty'], columns=['empty'])
store = pd.HDFStore("out.hdf5")
store.put('vars', df_tot)
store.get_storer('vars').attrs.metadata = metadata_tot
store.close()

# clean up the chunk files that we have already merged together
for file in files:
    os.system("rm " + str(file))
