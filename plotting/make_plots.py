import pandas as pd 
from hist import Hist
import argparse
import os
import awkward as ak
import uproot
import getpass
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-dataset", "--dataset"  , type=str, default="QCD", help="dataset name", required=True)
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=True)
parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="if is MC")
options = parser.parse_args()


# parameters for input files
username = getpass.getuser()
dataDir = "/work/{}/SUEP/{}/{}/".format(username,options.tag,options.dataset)
files = [file for file in os.listdir(dataDir)]
labels = ['mult','ch']

# output histos
def create_output_file(label):
    output = {
            "A_"+label: Hist.new.Reg(nbins, 0, 1, name="A_"+label).Weight(),
            "B_"+label: Hist.new.Reg(nbins, 0, 1, name="B_"+label).Weight(),
            "C_"+label: Hist.new.Reg(nbins, 0, 1, name="C_"+label).Weight(),
            "D_exp_"+label: Hist.new.Reg(nbins, 0, 1, name="D_exp_"+label).Weight(),
            "D_obs_"+label: Hist.new.Reg(nbins, 0, 1, name="D_obs_"+label).Weight(),
            "ABCDvars_2D_"+label : Hist.new.Reg(100, 0, 1, name=var1+label).Reg(100, 0, 200, name=var2).Weight(),
            "nconst_"+label : Hist.new.Reg(800, 0, 800, name="nconst_"+label, label="# Tracks").Weight(),
            "pt_"+label : Hist.new.Reg(100, 0, 2000, name="pt", label="pT_"+label).Weight(),
            "pt_avg_"+label : Hist.new.Reg(100, 0, 100, name="pt_avg_"+label, label="Components pT avg").Weight(),
            "pt_avg_b_"+label : Hist.new.Reg(100, 0, 100, name="pt_avg_b_"+label, label="Components pT avg (boosted frame)").Weight(),
            "eta_"+label : Hist.new.Reg(100, -5, 5, name="eta_"+label, label="eta").Weight(),
            "phi_"+label : Hist.new.Reg(100, 0, 6.5, name="phi_"+label, label="phi").Weight(),
            "mass_"+label : Hist.new.Reg(150, 0, 4000, name="mass_"+label, label="mass").Weight(),
            "spher_"+label : Hist.new.Reg(100, 0, 1, name="spher_"+label, label="sphericity").Weight(),
            "aplan_"+label : Hist.new.Reg(100, 0, 1, name="aplan_"+label, label="Aplanarity").Weight(),
            "FW2M_"+label : Hist.new.Reg(100, 0, 1, name="FW2M_"+label, label="2nd Fox Wolfram Moment").Weight(),
            "D_"+label : Hist.new.Reg(100, 0, 1, name="D_"+label, label="D").Weight(),
            "girth_pt_"+label: Hist.new.Reg(30, 0, 3, name="grith_pt_"+label).Weight(),
    }
    if label == 'ch':# Christos only
        output2 = {
            "dphi_chcands_ISR":Hist.new.Reg(100, 0, 4, name="dphi_chcands_ISR").Weight(),
            "dphi_SUEPtracks_ISR": Hist.new.Reg(100, 0, 4, name="dphi_SUEPtracks_ISR").Weight(),
            "dphi_ISRtracks_ISR":Hist.new.Reg(100, 0, 4, name="dphi_ISRtracks_ISR").Weight(),
            "dphi_SUEP_ISR":Hist.new.Reg(100, 0, 4, name="dphi_SUEP_ISR").Weight(),
        }
        output.update(output2)
    return output

def h5load(ifile, label):
	with pd.HDFStore(ifile) as store:
                data = store[label]
                if options.isMC:
                    metadata = store.get_storer(label).attrs.metadata
                    return data, metadata
                else:
                    return data

# fill ABCD hists with dfs
frames = {"mult":[],"ch":[]}
xsec = -1.0
for ifile in tqdm(files):
        ifile = dataDir+"/"+ifile
        for label in labels:
            df, metadata = h5load(ifile, label)
            if xsec == -1.0 and options.isMC:
                xsec = metadata["xsec"]
            frames[label].append(df)

for label in labels:
    # parameters for ABCD plots
    var1 = 'SUEP_'+label+'_spher'
    var2 = 'SUEP_'+label+'_nconst'
    var1_val = 0.60
    var2_val = 150
    nbins = 100
    output = create_output_file(label)
    sizeA, sizeC = 0,0

    df = pd.concat(frames[label])
    # divide the dfs by region and select the variable we want to plot
    A = df[var1].loc[(df[var1] < var1_val) & (df[var2] < var2_val)].to_numpy()
    B = df[var1].loc[(df[var1] >= var1_val) & (df[var2] < var2_val)].to_numpy()
    C = df[var1].loc[(df[var1] < var1_val) & (df[var2] >= var2_val)].to_numpy()
    D_obs = df[var1].loc[(df[var1] >= var1_val) & (df[var2] >= var2_val)].to_numpy()
    
    sizeC += ak.size(C) * xsec
    sizeA += ak.size(A) * xsec
    
    # fill the ABCD histograms
    output["A"].fill(A, weight = xsec)
    output["B"].fill(B, weight = xsec)
    output["D_exp"].fill(B, weight = xsec)
    output["C"].fill(C, weight = xsec)
    output["D_obs"].fill(D_obs, weight = xsec)
    output["ABCDvars_2D"].fill(df[var1], df[var2], weight = xsec)
    
    # fill the other histos
    plot_labels = [key for key in df.keys() if key[key.find(label) + len(label) + 1:] in list(output.keys())]
    for plot in plot_labels: output[plot[plot.find(label) + len(label) + 1:]].fill(df[plot], weight = xsec)
    
    # ABCD method to obtain D expected
    if sizeA>0.0:
    	CoverA =  sizeC / sizeA
    else:
    	CoverA = 0.0
    	print("A region has no occupancy")
    output["D_exp"] = output["D_exp"]*(CoverA)

# save to file
fout = uproot.recreate(options.dataset+'ABCD_plot.root')
for key in output.keys(): fout[key] = output[key]
fout.close()

# save plots to pickle
with open(dataDir+'/plotting/'+label+'_ABCD_plot.pkl', "wb") as f:
    pickle.dump(output, f)