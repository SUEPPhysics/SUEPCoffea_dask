#############################################################################
# MERGE
#############################################################################
# Merge each of the chunks' .hdf5 files together
# This won't be triggered unless the SUEP processor runs smoothly through
# all the chunks, thus assuring we processed all the events
# N.B.: Only merging df named 'vars' in the HDF5 object

import pandas as pd
import os, sys
import glob
#import tqdm

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

def merge(options): 
    files = glob.glob("condor_*.hdf5")
    if len(files) == 0: 
        print("No .hdf5 files found")
        sys.exit()
    
    df_tot = None
    metadata_tot = None
    for ifile, file in enumerate(files):
        
        df, metadata = h5load(file, 'vars') 
        
        ### Error out here
        if type(df) == int: 
            print("Something screwed up.")
            sys.exit()
            
        ### MERGE METADATA
        if metadata_tot is None: metadata_tot = metadata
        elif options.isMC: metadata_tot['gensumweight'] += metadata['gensumweight']
            
        # no need to add empty ones
        if 'empty' in list(df.keys()): continue
        
        ### MERGE DF
        if df_tot is None: df_tot = df
        else: df_tot = pd.concat((df_tot, df))   
        
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
    return
