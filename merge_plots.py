import pandas as pd
import sys, os, glob
import subprocess
import getpass

def h5load(ifile, label):
    #try:
    with pd.HDFStore(ifile, 'r') as store:
        try:
            data = store[label] 
            metadata = store.get_storer(label).attrs.metadata
            return data, metadata

        except KeyError:
            print("No key",label,ifile)
            return 0, 0
    #except:
    #    print("Some error occurred", ifile)
    #    return 0, 0
    
username = getpass.getuser()
tag = 'jetIDv2'
dataset = 'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM'
redirector = "root://t3serv017.mit.edu/"

dataDir = "/scratch/{}/SUEP/{}/{}/".format(username,tag,dataset)
result = subprocess.check_output(["xrdfs",redirector,"ls",dataDir])
result = result.decode("utf-8")
files = result.split("\n")
files = [f for f in files if '.hdf5' in f]


# SAVE OUTPUTS
def save_dfs(df_tot, df_vars_tot, output):
    if type(df_vars_tot) == int: 
        print("Something screwed up. No events in df_vars_tot.")
        sys.exit()
    if type(df_tot) == int: 
        print("No events in df_tot.")
        df_tot = pd.DataFrame(['empty'], columns=['empty'])
    store = pd.HDFStore(output)
    store.put('vars', df_vars_tot)
    store.put('ch', df_tot)
    store.get_storer('vars').attrs.metadata = metadata_tot
    store.get_storer('ch').attrs.metadata = metadata_tot
    store.close()
    

df_vars_tot = 0
df_tot = 0
metadata_tot = 0
i_out = 0

for ifile, file in enumerate(files):
        
    print(file)
    
    if os.path.exists(dataset+'.hdf5'): subprocess.run(['rm',dataset+'.hdf5'])
    xrd_file = redirector + file
    #print("xrdcp {} {}.hdf5".format(xrd_file, dataset))
    subprocess.run(["xrdcp",xrd_file,dataset+".hdf5"])
    
    df_vars, metadata = h5load(dataset+'.hdf5', 'vars') 
    df, _ = h5load(dataset+'.hdf5', 'ch')
            
    ### Error out here
    if type(df_vars) == int: 
        print("Something screwed up.")
        sys.exit()
    
    if df_vars.shape[0] == 0: 
        subprocess.run(['rm',dataset +'.hdf5'])  
        continue
        
    ### MERGE DF VARS
    if type(df_vars_tot) == int: df_vars_tot = df_vars
    else: df_vars_tot = pd.concat((df_vars_tot, df_vars))
    
    ### MERGE METADATA
    if type(metadata_tot) == int: metadata_tot = metadata
    else: metadata_tot['gensumweight'] += metadata['gensumweight']
    
    # no need to add empty ones
    if 'empty' in list(df.keys()): 
        subprocess.run(['rm',dataset+'.hdf5'])    
        continue
    if df.shape[0] == 0: 
        subprocess.run(['rm',dataset+'.hdf5'])    
        continue
    
    ### MERGE DF_CH
    if type(df_tot) == int: df_tot = df
    else: df_tot = pd.concat((df_tot, df))
    
    subprocess.run(['rm',dataset+'.hdf5'])   
    
    if df_tot.shape[0] > 5000000 or ifile == len(files)-1:
        output_file = "condor_out_" + str(i_out) + ".hdf5"
        save_dfs(df_tot, df_vars_tot, output_file)
        print("xrdcp {} {}".format(output_file, redirector+dataDir))
        os.system("xrdcp {} {}".format(output_file, redirector+dataDir))
        subprocess.run(['rm',output_file])
        i_out += 1
        df_vars_tot = 0
        df_tot = 0
        metadata_tot = 0