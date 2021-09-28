import pandas as pd 
import numpy as np
import pyarrow
import pyarrow.parquet as pq
from coffea import hist, processor
import os, sys
import json
import awkward as ak
import uproot3
import vector
vector.register_awkward()


# parameters
dataDir = "/home/lavezzo/SUEPCoffea_dask/"
files = [file for file in os.listdir(dataDir) if file.endswith("ch.parquet")]
custom_meta_key = "SUEP.iot"
var1 = 'SUEP_ch_spher'
var2 = 'SUEP_ch_nconst'
var1_val = 0.60
var2_val = 150
nbins = 100

# output histos
output = {
	"A": hist.Hist(
	    "Events", hist.Bin("A", "A", nbins, 0, 1)),
	"B": hist.Hist(
	    "Events", hist.Bin("B", "B", nbins, 0, 1)),
	"C": hist.Hist(
	    "Events", hist.Bin("C", "C", nbins, 0, 1)),
	"D_exp": hist.Hist(
	    "Events", hist.Bin("D_exp", "D_exp", nbins, 0, 1)),
	"D_obs": hist.Hist(
	    "Events", hist.Bin("D_obs", "D_obs", nbins, 0, 1)),
}

def load_parquet(infile = '', custom_meta_key = 'SUEP.iot'):
	intable = pq.read_table(infile)
	df = intable.to_pandas()
	restored_meta_json = intable.schema.metadata[custom_meta_key.encode()]
	restored_meta = json.loads(restored_meta_json)

	return df, restored_meta

for ifile, infile in enumerate(files):

	# load in dataframe, metadata, and scale by xsec
	df, restored_meta = load_parquet(infile, custom_meta_key)
	df = df * restored_meta['xsec']
	if ifile == 0: full_df = df
	else: full_df = pd.concat([full_df, df])

# divide the dfs by region and select the variable we want to plot
A = full_df[var1].loc[(full_df[var1] < var1_val) & (full_df[var2] < var2_val)].to_numpy()
B = full_df[var1].loc[(full_df[var1] >= var1_val) & (full_df[var2] < var2_val)].to_numpy()
C = full_df[var1].loc[(full_df[var1] < var1_val) & (full_df[var2] >= var2_val)].to_numpy()
D_obs = full_df[var1].loc[(full_df[var1] >= var1_val) & (full_df[var2] >= var2_val)].to_numpy()

# fill the histograms
output["A"].fill(A = A)
output["B"].fill(B = B)
output["C"].fill(C = C)
output["D_obs"].fill(D_obs = D_obs)

# ABCD method to obtain D expected
if ak.size(A)>0.0:
    CoverA =  ak.size(C) /  ak.size(A)
else:
    CoverA = 0.0
    print("A region has no occupancy")
D_expected = B
output["D_exp"].fill(D_exp = D_expected)
output["D_exp"].scale(CoverA)

fout = uproot3.recreate(dataDir+'ABCD.root')
fout['A'] = hist.export1d(output["A"])
fout['B'] = hist.export1d(output["B"])
fout['C'] = hist.export1d(output["C"])
fout['D_obs'] = hist.export1d(output["D_obs"])
fout['D_exp'] = hist.export1d(output["D_exp"])
fout.close()