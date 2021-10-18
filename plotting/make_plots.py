import pandas as pd 
import numpy as np
import hist
from hist import Hist
import os, sys
import json
import awkward as ak
import pickle
import uproot
import vector
vector.register_awkward()


# parameters for input files
dataDir = "/home/lavezzo/SUEPCoffea_dask/"
files = [file for file in os.listdir(dataDir) if file.endswith("42211.hdf5")]
files = ['/mnt/hadoop/scratch/lavezzo/SUEP/SUEP/QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM/395.hdf5']
label = 'ch'

# parameters for ABCD plots
var1 = 'SUEP_'+label+'_spher'
var2 = 'SUEP_'+label+'_nconst'
var1_val = 0.60
var2_val = 20
nbins = 100							


# output histos
def create_output_file(label):
	output = {
		"A": Hist.new.Reg(nbins, 0, 1, name="A").Weight(),
		"B": Hist.new.Reg(nbins, 0, 1, name="B").Weight(),
		"C": Hist.new.Reg(nbins, 0, 1, name="C").Weight(),
		"D_exp": Hist.new.Reg(nbins, 0, 1, name="D_exp").Weight(),
		"D_obs": Hist.new.Reg(nbins, 0, 1, name="D_obs").Weight(),
		"ABCDvars_2D" : Hist.new.Reg(100, 0, 1, name=var1).Reg(100, 0, 200, name=var2).Weight(),
		"nconst" : Hist.new.Reg(800, 0, 800, name="nconst", label="# Tracks").Weight(),
		"pt" : Hist.new.Reg(100, 0, 2000, name="pt", label="pT").Weight(),
		"pt_avg" : Hist.new.Reg(100, 0, 100, name="pt_avg", label="Components pT avg").Weight(),
		"pt_avg_b" : Hist.new.Reg(100, 0, 100, name="pt_avg_b", label="Components pT avg (boosted frame)").Weight(),
		"eta" : Hist.new.Reg(100, -5, 5, name="eta", label="eta").Weight(),
		"phi" : Hist.new.Reg(100, 0, 6.5, name="phi", label="phi").Weight(),
		"mass" : Hist.new.Reg(150, 0, 4000, name="mass", label="mass").Weight(),
		"spher" : Hist.new.Reg(100, 0, 1, name="spher", label="sphericity").Weight(),
		"aplan" : Hist.new.Reg(100, 0, 1, name="aplan", label="Aplanarity").Weight(),
		"FW2M" : Hist.new.Reg(100, 0, 1, name="FW2M", label="2nd Fox Wolfram Moment").Weight(),
		"D" : Hist.new.Reg(100, 0, 1, name="D", label="D").Weight(),
		"girth_pt": Hist.new.Reg(30, 0, 3, name="girth_pt").Weight(),

		# Christos only
		"dphi_chcands_ISR":Hist.new.Reg(100, 0, 4, name="dphi_chcands_ISR").Weight(),
		"dphi_SUEPtracks_ISR": Hist.new.Reg(100, 0, 4, name="dphi_SUEPtracks_ISR").Weight(),
		"dphi_ISRtracks_ISR":Hist.new.Reg(100, 0, 4, name="dphi_ISRtracks_ISR").Weight(),
		"dphi_SUEP_ISR":Hist.new.Reg(100, 0, 4, name="dphi_SUEP_ISR").Weight(),
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
	output["D_exp"].fill(B, weight = metadata["xsec"])
	output["C"].fill(C, weight = metadata["xsec"])
	output["D_obs"].fill(D_obs, weight = metadata["xsec"])
	output["ABCDvars_2D"].fill(df[var1], df[var2], weight = metadata["xsec"])
	
	# fill the other histos
	plot_labels = [key for key in df.keys() if key[key.find(label) + len(label) + 1:] in list(output.keys())]
	for plot in plot_labels: 
		output[plot[plot.find(label) + len(label) + 1:]].fill(df[plot], weight = metadata["xsec"])

# ABCD method to obtain D expected
if sizeA>0.0:
	CoverA =  sizeC / sizeA
else:
	CoverA = 0.0
	print("A region has no occupancy")
output["D_exp"] = output["D_exp"]*(CoverA)

# FIXME: save to root file, remove it?
fout = uproot.recreate(dataDir+label+'_ABCD_plot.root')
for key in output.keys(): fout[key] = output[key]
fout.close()

# save plots to pickle
with open(dataDir+'/plotting/'+label+'_ABCD_plot.pkl', "wb") as f:
    pickle.dump(output, f)
