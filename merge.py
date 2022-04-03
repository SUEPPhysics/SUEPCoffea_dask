import pandas as pd
import sys, os, glob

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
    
files = glob.glob("*.hdf5")
df_vars_tot = 0
df_tot = 0
metadata_tot = 0
for ifile, file in enumerate(files):
    
    df_vars, metadata = h5load(file, 'vars') 
    df, _ = h5load(file, 'ch')
    
    ### Error out here
    if type(df_vars) == int: 
        print("Something screwed up.")
        sys.exit()
    
    if df_vars.shape[0] == 0: continue
    
    print(file)
    
    ### MERGE DF VARS
    if type(df_vars_tot) == int: df_vars_tot = df_vars
    else: df_vars_tot = pd.concat((df_vars_tot, df_vars))
    
    ### MERGE METADATA
    if type(metadata_tot) == int: metadata_tot = metadata
    else: metadata_tot['gensumweight'] += metadata['gensumweight']
    
    # no need to add empty ones
    if 'empty' in list(df.keys()): continue
    if df.shape[0] == 0: continue
    
    ### MERGE DF_CH
    if type(df_tot) == int: df_tot = df
    else: df_tot = pd.concat((df_tot, df))
    
# SAVE OUTPUTS
if type(df_vars_tot) == int: 
    print("Something screwed up. No events in df_vars_tot.")
    sys.exit()
if type(df_tot) == int: 
    print("No events in df_tot.")
    df_tot = pd.DataFrame(['empty'], columns=['empty'])
store = pd.HDFStore("condor_out.hdf5")
store.put('vars', df_vars_tot)
store.put('ch', df_tot)
store.get_storer('vars').attrs.metadata = metadata_tot
store.get_storer('ch').attrs.metadata = metadata_tot
store.close()