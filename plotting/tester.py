#this is to test a bug with make_plots

import pandas as pd 
import numpy as np
from hist import Hist
import argparse
import os, sys, subprocess
import awkward as ak
import uproot
import getpass
import pickle
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-dataset", "--dataset"  , type=str, default="QCD", help="dataset name", required=True)
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=False)
parser.add_argument("-o"   , "--output"   , type=str, default="IronMan"  , help="output tag", required=False)
parser.add_argument("-e"   , "--era"   , type=int, default=2018  , help="era", required=False)
parser.add_argument('--doSyst', type=int, default=0, help="make systematic plots")
parser.add_argument('--isMC', type=int, default=1, help="Is this MC or data")
parser.add_argument('--blind', type=int, default=1, help="Blind the data (default=True)")
parser.add_argument('--xrootd', type=int, default=1, help="Local data or xrdcp from hadoop (xrootd=True)")
options = parser.parse_args()


redirector = "root://t3serv017.mit.edu:/"
username = getpass.getuser()
if options.xrootd:
    dataDir = "/mnt/T3_US_MIT/hadoop/scratch/{}/SUEP/{}/{}/".format(username,options.tag,options.dataset)
else:
    dataDir = "/work/submit/{}/SUEP/{}/{}/".format(username, options.tag, options.dataset)
files = [file for file in os.listdir(dataDir)]



# load hdf5 with pandas
def h5load(ifile, label):
    print("ifile: "+ifile)#FIXME
    print("label: "+label)#FIXME
    try:
        print("try")#FIXME
        with pd.HDFStore(ifile, 'r') as store:
            print("with")#FIXME
            try:
                data = store[label] 
                metadata = store.get_storer(label).attrs.metadata
                #print(" Metadata: "+metadata)#FIXME
                return data, metadata
        
            except KeyError:
                print("No key",label,ifile)
                return 0, 0
    except:
        print("Some error occurred", ifile)
        return 0, 0

#data, metadata = h5load('0075E755-4D7F-C245-A6FA-2D8CA9F50EEF.hdf5', 'vars')

#print(data)
#print(metadata)


#ifile = dataDir+files[0]

for ifile in tqdm(files):
    ifile = dataDir+ifile

    if options.xrootd:
        if os.path.exists(options.dataset+'.hdf5'): os.system('rm ' + options.dataset+'.hdf5')
        xrd_file = redirector + ifile.split('hadoop')[1]
        os.system("xrdcp {} {}.hdf5".format(xrd_file, options.dataset))
        print("--"+xrd_file+"--")#FIXME
        print("_____a_____")#FIXME
        print("options.dataset: "+options.dataset)#FIXME
        df_vars, metadata = h5load(options.dataset+'.hdf5', 'vars')   
    else:
        print("_____b_____")#FIXME
        df_vars, metadata = h5load(ifile, 'vars')

    print(df_vars)
    print(metadata)
