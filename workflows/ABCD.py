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


# parameters
dataDir = "/home/lavezzo/SUEPCoffea_dask/"
files = [file for file in os.listdir(dataDir) if file.endswith("ch.parquet")]
var1 = 'SUEP_ch_spher'
var2 = 'SUEP_ch_nconst'
var1_val = 0.60
var2_val = 150
nbins = 100

# output histos
output = {
	"A": Hist.new.Reg(nbins, 0, 1, name="A").Weight(),
	"B": Hist.new.Reg(nbins, 0, 1, name="B").Weight(),
	"C": Hist.new.Reg(nbins, 0, 1, name="C").Weight(),
	"D_exp": Hist.new.Reg(nbins, 0, 1, name="D_exp").Weight(),
	"D_obs": Hist.new.Reg(nbins, 0, 1, name="D_obs").Weight(),
	"2D" : Hist(
			hist.axis.Regular(100, 0, 1, name=var1),
			hist.axis.Regular(100, 0, 200, name=var2),
		)
}

def h5load(store, label='ch'):
    data = store[label]
    metadata = store.get_storer(label).attrs.metadata
    return data, metadata


# fill ABCD hists with dfs
sizeA, sizeC = 0,0
for ifile, infile in enumerate(files):

	with pd.HDFStore(infile) as store:
    	data, metadata = h5load(store, 'ch')

	# divide the dfs by region and select the variable we want to plot
	A = df[var1].loc[(df[var1] < var1_val) & (df[var2] < var2_val)].to_numpy()
	B = df[var1].loc[(df[var1] >= var1_val) & (df[var2] < var2_val)].to_numpy()
	C = df[var1].loc[(df[var1] < var1_val) & (df[var2] >= var2_val)].to_numpy()
	D_obs = df[var1].loc[(df[var1] >= var1_val) & (df[var2] >= var2_val)].to_numpy()

	sizeC += ak.size(C) * restored_meta["xsec"]
	sizeA += ak.size(A) * restored_meta["xsec"]

	# fill the histograms
	output["A"].fill(A, weight = restored_meta["xsec"])
	output["B"].fill(B, weight = restored_meta["xsec"])
	output["C"].fill(C, weight = restored_meta["xsec"])
	output["D_obs"].fill(D_obs, weight = restored_meta["xsec"])
	output["2D"].fill(df[var1], df[var2], weight = restored_meta["xsec"])

# ABCD method to obtain D expected
if sizeA>0.0:
	CoverA =  sizeC / sizeA
else:
	CoverA = 0.0
	print("A region has no occupancy")
output["D_exp"] = output["B"]
output["D_exp"] *= (CoverA)

# save to file
fout = uproot.recreate(dataDir+'ABCD.root')
for key in output.keys(): fout[key] = output[key]
fout.close()