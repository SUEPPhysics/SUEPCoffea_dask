#for testing a bug in make_plots

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

ifile = '0075E755-4D7F-C245-A6FA-2D8CA9F50EEF.hdf5'
label = 'vars'

try:
    with pd.HDFStore(ifile, 'r') as store:
        try:
            data = store[label] 
            metadata = store.get_storer(label).attrs.metadata

        except KeyError:
            print("No key",label,ifile)
except:
    print("Some error occurred", ifile)

print(data)
print(metadata)