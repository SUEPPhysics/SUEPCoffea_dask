import os, sys, subprocess
import time
import pandas as pd 
import numpy as np
from hist import Hist
import argparse
import os
import awkward as ak
import uproot
import getpass
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=True)
parser.add_argument("-b"   , "--blind"   , dest='blind',  action='store_true', help="blind", required=False)
parser.add_argument("-u"   , "--unblind"  , dest='blind',  action='store_false', help="unblind", required=False)
parser.set_defaults(blind=True)
options = parser.parse_args()

# parameters for ABCD method
var1_label = 'spher'
var2_label = 'nconst'
var1_val = 0.50
var2_val = 25
nbins = 100
labels = ['mult', 'ch']
output_label = 'V3'

# input parameters
username = getpass.getuser()
dataDir = "/work/submit/{}/SUEP/{}/".format(username,options.tag)

# blinding warning
if not options.blind: 
    print("WARNING: YOU ARE RUNNING THIS SCRIPT WITHOUT BLINDING.")
    answer = input("Do you understand? Type 'I want to unblind' if you want to unblind: ")
    if answer != 'I want to unblind': sys.exit("Safely exited program before unblinding the analysis. Nice!")
    if answer == 'I want to unblind': print("Unblinding...")

# merge all QCD HT bins together, or just import all files from a directory
files = []
for subdir in list(os.listdir(dataDir)):
    if 'JetHT' not in subdir: continue
    files += [subdir+"/"+file for file in os.listdir(dataDir + subdir)]
    
# output histos
def create_output_file(label):
    output = {
            # ABCD hists
            "A_"+label: Hist.new.Reg(nbins, 0, 1, name="A_"+label).Weight(),
            "B_"+label: Hist.new.Reg(nbins, 0, 1, name="B_"+label).Weight(),
            "C_"+label: Hist.new.Reg(nbins, 0, 1, name="C_"+label).Weight(),
            "D_exp_"+label: Hist.new.Reg(nbins, 0, 1, name="D_exp_"+label).Weight(),
        }
    if not options.blind: 
        output.update({"D_obs_"+label: Hist.new.Reg(nbins, 0, 1, name="D_obs_"+label).Weight()})

    return output

# load hdf5 with pandas
def h5load(fname, label):
    try:
        with pd.HDFStore(fname) as store:
            data = store[label]
            return data
    except:
            print(fname)
            return 0

# fill ABCD hists with dfs from hdf5 files
frames = {"mult":[],"ch":[]}
nfailed = 0
for ifile in tqdm(files):
    ifile = dataDir+"/"+ifile
    for label in labels:
        df = h5load(ifile, label)
        if type(df) == int: 
            nfailed += 1
            continue
        frames[label].append(df)
print("nfailed", nfailed)

#fout = uproot.recreate(options.dataset+'_ABCD_plot.root')
fpickle =  open("outputs/JetHT_" + output_label + '.pkl', "wb")
for label in labels:

    # variables for ABCD plots
    var1 = 'SUEP_'+label+'_' + var1_label
    var2 = 'SUEP_'+label+'_' + var2_label

    output = create_output_file(label)
    sizeA, sizeC = 0,0

    # combine the dataframes
    df = pd.concat(frames[label])
    
    #if var2_label == 'nconst': df = df.loc[df['SUEP_'+label+'_nconst'] >= 10]

    # divide the dfs by region and select the variable we want to plot
    A = df[var1].loc[(df[var1] < var1_val) & (df[var2] < var2_val) & (df[var1] > 0.25)].to_numpy()
    B = df[var1].loc[(df[var1] >= var1_val) & (df[var2] < var2_val)].to_numpy()
    C = df[var1].loc[(df[var1] < var1_val) & (df[var2] >= var2_val) & (df[var1] > 0.25)].to_numpy()
    #----------------------------------
    # DO NOT EVEN FILL ANY D observed!
    #----------------------------------    
    
    sizeC += ak.size(C)
    sizeA += ak.size(A)

    # fill the ABCD histograms
    output["A_"+label].fill(A)
    output["B_"+label].fill(B)
    output["D_exp_"+label].fill(B)
    output["C_"+label].fill(C)
    if not options.blind: 
        output["D_obs_"+label].fill(D_obs)
  
    # ABCD method to obtain D expected
    if sizeA>0.0:
    	CoverA =  sizeC / sizeA
    else:
    	CoverA = 0.0
    	print("A region has no occupancy")
    output["D_exp_"+label] = output["D_exp_"+label]*(CoverA)

    #Save to root and to pickle
    #for key in output.keys(): fout[key] = output[key]
    pickle.dump(output, fpickle)
#fout.close()
