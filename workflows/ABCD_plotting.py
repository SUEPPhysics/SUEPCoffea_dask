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


# input files parameters
dataDir = "/home/lavezzo/SUEP/SUEPCoffea_dask/"
files = [file for file in os.listdir(dataDir) if file.endswith("42211.hdf5")]
label = 'pt'

# parameters for ABCD plots
var1 = 'SUEP_pt_spher'
var2 = 'SUEP_pt_nconst'
var1_val = 0.60
var2_val = 150
nbins = 100							


###FIXME: add file label to histos titles/dict?
# output histos
def create_output_file(label):
	output = {
		"A": Hist.new.Reg(nbins, 0, 1, name="A").Weight(),
		"B": Hist.new.Reg(nbins, 0, 1, name="B").Weight(),
		"C": Hist.new.Reg(nbins, 0, 1, name="C").Weight(),
		"D_exp": Hist.new.Reg(nbins, 0, 1, name="D_exp").Weight(),
		"D_obs": Hist.new.Reg(nbins, 0, 1, name="D_obs").Weight(),
		"2D" : Hist(
				hist.axis.Regular(100, 0, 1, name=var1),
				hist.axis.Regular(100, 0, 200, name=var2),
			),
		"nconst" : Hist(hist.axis.Regular(250, 0, 250, name="nconst", label="# Tracks")),
		"pt" : Hist(hist.axis.Regular(100, 0, 2000, name="pt", label="pT")),
		"eta" : Hist(hist.axis.Regular(100, -5, 5, name="eta", label="eta")),
		"phi" : Hist(hist.axis.Regular(100, 0, 6.5, name="phi", label="phi")),
		"mass" : Hist(hist.axis.Regular(150, 0, 4000, name="mass", label="mass")),
		"spher" : Hist(hist.axis.Regular(100, 0, 1, name="spher", label="sphericity")),
		"aplan" : Hist(hist.axis.Regular(100, 0, 1, name="aplan", label="Aplanarity")),
		"FW2M" : Hist(hist.axis.Regular(100, 0, 1, name="FW2M", label="2nd Fox Wolfram Moment")),
		"D" : Hist(hist.axis.Regular(100, 0, 1, name="D", label="D")),
	}
	return output

def h5load(store, label):
	data = store[label]
	metadata = store.get_storer(label).attrs.metadata
	return data, metadata

# fill ABCD hists with dfs
output = create_output_file(label)
sizeA, sizeC = 0,0
for ifile in files:

	with pd.HDFStore(ifile) as store:
		df, metadata = h5load(store, label)

	# divide the dfs by region and select the variable we want to plot
	A = df[var1].loc[(df[var1] < var1_val) & (df[var2] < var2_val)].to_numpy()
	B = df[var1].loc[(df[var1] >= var1_val) & (df[var2] < var2_val)].to_numpy()
	C = df[var1].loc[(df[var1] < var1_val) & (df[var2] >= var2_val)].to_numpy()
	D_obs = df[var1].loc[(df[var1] >= var1_val) & (df[var2] >= var2_val)].to_numpy()

	sizeC += ak.size(C) * metadata["xsec"]
	sizeA += ak.size(A) * metadata["xsec"]

	# fill the ABCD histograms
	output["A"].fill(A, weight = metadata["xsec"])
	output["B"].fill(B, weight = metadata["xsec"])
	output["C"].fill(C, weight = metadata["xsec"])
	output["D_obs"].fill(D_obs, weight = metadata["xsec"])
	output["2D"].fill(df[var1], df[var2], weight = metadata["xsec"])

	# fill the other histos
	plot_labels = ["nconst", "pt", "eta", "phi",
			"mass", "spher", "aplan", "FW2M", "D"]
	for plot in plot_labels: output[plot].fill(df["SUEP_"+label+"_"+plot], weight = metadata["xsec"])

# ABCD method to obtain D expected
if sizeA>0.0:
	CoverA =  sizeC / sizeA
else:
	CoverA = 0.0
	print("A region has no occupancy")
output["D_exp"] = output["B"]
output["D_exp"] *= (CoverA)

# save to file
fout = uproot.recreate(dataDir+label+'_ABCD_plot.root')
for key in output.keys(): fout[key] = output[key]
fout.close()