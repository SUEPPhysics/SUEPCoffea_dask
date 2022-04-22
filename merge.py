import pandas as pd
import sys, os, glob
import argparse

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
    
parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-isMC"   , "--isMC"   , type=int, default=0, help="MC or real data", required=True)
options = parser.parse_args()

print("Running with isMC set to", options.isMC)

files = glob.glob("*.hdf5")
df_tot = 0
df_tot = 0
metadata_tot = 0
for ifile, file in enumerate(files):
    
    df, metadata = h5load(file, 'vars') 
    
    ### Error out here
    if type(df) == int: 
        print("Something screwed up.")
        sys.exit()
    
    if df.shape[0] == 0: continue
    
    print(file)
    
    # no need to add empty ones
    if 'empty' in list(df.keys()): continue
    if df.shape[0] == 0: continue
    
    ### MERGE DF VARS
    if type(df_tot) == int: df_tot = df
    else: df_tot = pd.concat((df_tot, df))
    
    ### MERGE METADATA
    if type(metadata_tot) == int: metadata_tot = metadata
    elif options.isMC: metadata_tot['gensumweight'] += metadata['gensumweight']
    
# SAVE OUTPUTS
if type(df_tot) == int: 
    print("No events in df_tot.")
    df_tot = pd.DataFrame(['empty'], columns=['empty'])
store = pd.HDFStore("condor_out.hdf5")
store.put('vars', df_tot)
store.get_storer('vars').attrs.metadata = metadata_tot
store.close()