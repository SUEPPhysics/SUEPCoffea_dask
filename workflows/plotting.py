import pandas as pd 
import numpy as np
import hist
from hist import Hist
import os, sys
import json
import awkward as ak
import uproot
import vector
vector.register_awkward()


# parameters for input files
dataDir = "/home/lavezzo/SUEP/SUEPCoffea_dask/"
files = [file for file in os.listdir(dataDir) if file.endswith("42211.hdf5")]
label = 'out_vars'

# output histos
def create_output_file(label):
	output = {
		"dphi_SUEPtracks_ISR": Hist.new.Reg(100, 0, 4, name="dphi_SUEPtracks_ISR").Weight(),
		"dphi_ISRtracks_ISR":Hist.new.Reg(100, 0, 4, name="dphi_ISRtracks_ISR").Weight(),
		"dphi_SUEP_ISR":Hist.new.Reg(100, 0, 4, name="dphi_SUEP_ISR").Weight(),
		"uncleaned_tracks": Hist.new.Reg(10000, 0, 10000, name="uncleaned_tracks").Weight(),
		"nCleaned_Cands":Hist.new.Reg(200, 0, 200, name="nCleaned_Cands").Weight(),
		"ngood_fastjets":Hist.new.Reg(15, 0, 15, name="ngood_fastjets").Weight(),
	}
	return output

def h5load(ifile, label):
	with pd.HDFStore(ifile) as store:
		data = store[label]
		metadata = store.get_storer(label).attrs.metadata
	return data, metadata

# fill ABCD hists with dfs
output = create_output_file(label)
sizeA, sizeC = 0,0
for ifile in files:

	df, metadata = h5load(ifile, label)
	df.dropna()

	for var in list(df.columns):
		output[var].fill(df[var], weight = metadata['xsec'])

# save to file
fout = uproot.recreate(dataDir+label+'.root')
for key in output.keys(): fout[key] = output[key]
fout.close()