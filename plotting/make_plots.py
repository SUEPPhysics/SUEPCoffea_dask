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
dataDir = "/work/submit/{}/SUEP/{}/{}/".format(username,options.tag,options.dataset)
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
            "SUEP_"+label+"_nconst" : Hist.new.Reg(800, 0, 800, name="nconst_"+label, label="# Tracks").Weight(),
            "SUEP_"+label+"_pt" : Hist.new.Reg(100, 0, 2000, name="pt_"+label, label=r"$p_T$").Weight(),
            "SUEP_"+label+"_pt_avg" : Hist.new.Reg(100, 0, 100, name="pt_avg_"+label, label=r"Components $p_T$ Avg.").Weight(),
            "SUEP_"+label+"_pt_avg_b" : Hist.new.Reg(100, 0, 100, name="pt_avg_b_"+label, label=r"Components $p_T$ avg (boosted frame)").Weight(),
            "SUEP_"+label+"_eta" : Hist.new.Reg(100, -5, 5, name="eta_"+label, label=r"$\eta$").Weight(),
            "SUEP_"+label+"_phi" : Hist.new.Reg(100, 0, 6.5, name="phi_"+label, label=r"$\phi$").Weight(),
            "SUEP_"+label+"_mass" : Hist.new.Reg(150, 0, 4000, name="mass_"+label, label="Mass").Weight(),
            "SUEP_"+label+"_spher" : Hist.new.Reg(100, 0, 1, name="spher_"+label, label="Sphericity").Weight(),
            "SUEP_"+label+"_aplan" : Hist.new.Reg(100, 0, 1, name="aplan_"+label, label="Aplanarity").Weight(),
            "SUEP_"+label+"_FW2M" : Hist.new.Reg(100, 0, 1, name="FW2M_"+label, label="2nd Fox Wolfram Moment").Weight(),
            "SUEP_"+label+"_D" : Hist.new.Reg(100, 0, 1, name="D_"+label, label="D").Weight(),
            "SUEP_"+label+"_girth_pt": Hist.new.Reg(30, 0, 3, name="girth_pt_"+label, label=r"Girth $p_T$").Weight(),
    }
    if label == 'ch':# Christos only
        output2 = {
            "SUEP_"+label+"_dphi_chcands_ISR":Hist.new.Reg(100, 0, 4, name="dphi_chcands_ISR").Weight(),
            "SUEP_"+label+"_dphi_SUEPtracks_ISR": Hist.new.Reg(100, 0, 4, name="dphi_SUEPtracks_ISR").Weight(),
            "SUEP_"+label+"_dphi_ISRtracks_ISR":Hist.new.Reg(100, 0, 4, name="dphi_ISRtracks_ISR").Weight(),
            "SUEP_"+label+"_dphi_SUEP_ISR":Hist.new.Reg(100, 0, 4, name="dphi_SUEP_ISR").Weight(),
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

fout = uproot.recreate(options.dataset+'_ABCD_plot.root')
fpickle =  open(options.dataset+'_ABCD_plot.pkl', "wb")
for label in labels:
    # parameters for ABCD plots
    var1 = 'SUEP_'+label+'_spher'
    var2 = 'SUEP_'+label+'_nconst'
    var1_val = 0.60
    var2_val = 75
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
    output["A_"+label].fill(A, weight = xsec)
    output["B_"+label].fill(B, weight = xsec)
    output["D_exp_"+label].fill(B, weight = xsec)
    output["C_"+label].fill(C, weight = xsec)
    output["D_obs_"+label].fill(D_obs, weight = xsec)
    output["ABCDvars_2D_"+label].fill(df[var1], df[var2], weight = xsec)    
    
    # fill the other histos
    plot_labels = [key for key in df.keys() if key in list(output.keys())]
    for plot in plot_labels: output[plot].fill(df[plot], weight = xsec)    

    # ABCD method to obtain D expected
    if sizeA>0.0:
    	CoverA =  sizeC / sizeA
    else:
    	CoverA = 0.0
    	print("A region has no occupancy")
    output["D_exp_"+label] = output["D_exp_"+label]*(CoverA)

    #Save to root and to pickle
    for key in output.keys(): fout[key] = output[key]
    pickle.dump(output, fpickle)
fout.close()
