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
parser.add_argument("-dataset", "--dataset"  , type=str, default="QCD", help="dataset name", required=True)
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=True)
parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="if is MC")
options = parser.parse_args()

# parameters for ABCD method
var1_label = 'spher'
var2_label = 'nconst'
var1_val = 0.50
var2_val = 25
nbins = 100
labels = ['ch']
output_label = 'V5_nconst10'

# merge all QCD HT bins together, or just import all files from a specific directory
username = getpass.getuser()
if options.dataset == 'QCD':
    dataDir = "/work/submit/{}/SUEP/{}/".format(username,options.tag)
    files = []
    for subdir in list(os.listdir(dataDir)):
        if 'QCD' not in subdir: continue
        files += [subdir+"/"+file for file in os.listdir(dataDir + subdir)]
else:
    dataDir = "/work/submit/{}/SUEP/{}/{}/".format(username,options.tag,options.dataset)
    files = [file for file in os.listdir(dataDir)]

# output histos
def create_output_file(label):
    output = {
            # variables from the dataframe
            "SUEP_"+label+"_nconst" : Hist.new.Reg(499, 0, 500, name="nconst_"+label, label="# Tracks in SUEP").Weight(),
            "SUEP_"+label+"_ntracks" : Hist.new.Reg(499, 0, 500, name="ntracks_"+label, label="# Tracks in event").Weight(),
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
            "SUEP_"+label+"_girth": Hist.new.Reg(50, 0, 1.0, name="girth_"+label, label=r"Girth").Weight(),
            "SUEP_"+label+"_rho0" : Hist.new.Reg(100, 0, 20, name="rho0_"+label, label=r"$\rho_0$").Weight(),
            "SUEP_"+label+"_rho1" : Hist.new.Reg(100, 0, 20, name="rho1_"+label, label=r"$\rho_1$").Weight(),

            # new hists
            "A_"+label: Hist.new.Reg(nbins, 0, 1, name="A_"+label).Weight(),
            "B_"+label: Hist.new.Reg(nbins, 0, 1, name="B_"+label).Weight(),
            "C_"+label: Hist.new.Reg(nbins, 0, 1, name="C_"+label).Weight(),
            "D_exp_"+label: Hist.new.Reg(nbins, 0, 1, name="D_exp_"+label).Weight(),
            "D_obs_"+label: Hist.new.Reg(nbins, 0, 1, name="D_obs_"+label).Weight(),
            "ABCDvars_2D_"+label : Hist.new.Reg(100, 0, 1, name=var1+label).Reg(99, 0, 200, name=var2).Weight(),
            "2D_girth_nconst_"+label : Hist.new.Reg(50, 0, 1.0, name="girth_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "2D_rho0_nconst_"+label : Hist.new.Reg(100, 0, 20, name="rho0_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "2D_rho1_nconst_"+label : Hist.new.Reg(100, 0, 20, name="rho1_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "2D_spher_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="spher_"+label).Reg(200, 0, 500, name="ntracks_"+label).Weight(),
            "2D_spher_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="spher_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
        
            # AB and AC combined, kinematic variables
            "SUEP_" + label + "_AB_pt_" : Hist.new.Reg(100, 0, 2000, name="AB pt_"+label, label=r"$p_T$").Weight(),
            "SUEP_" + label + "_AB_eta_" : Hist.new.Reg(100, -5, 5, name="AB eta_"+label, label=r"$\eta$").Weight(),
            "SUEP_" + label + "_AB_phi_" : Hist.new.Reg(100, 0, 6.5, name="AB phi_"+label, label=r"$\phi$").Weight(),
            "SUEP_" + label + "_AC_pt_" : Hist.new.Reg(100, 0, 2000, name="AC pt_"+label, label=r"$p_T$").Weight(),
            "SUEP_" + label + "_AC_eta_" : Hist.new.Reg(100, -5, 5, name="AC eta_"+label, label=r"$\eta$").Weight(),
            "SUEP_" + label + "_AC_phi_" : Hist.new.Reg(100, 0, 6.5, name="AC phi_"+label, label=r"$\phi$").Weight(),
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
        try:
            data = store[label] 
            if options.isMC:
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata
            else:
                return data
        except ValueError: 
            print("Empty file!", ifile)
            return 0, 0
        except KeyError:
            print("No key",label,ifile)
            return 0, 0
        

# fill ABCD hists with dfs from hdf5 files
frames = {"mult":[],"ch":[]}
nfailed = 0
for ifile in tqdm(files):
    ifile = dataDir+"/"+ifile
    
    # check if file is corrupted
    try:
        for label in labels:
            df, metadata = h5load(ifile, label)
            if type(df) == int: 
                continue
            if options.isMC:
                df["xsec"] = metadata["xsec"]
            frames[label].append(df)
            
    except:
        print("Corrupted file", ifile)
        nfailed+=1

print("nfailed", nfailed)
            
#fout = uproot.recreate(options.dataset+'_ABCD_plot.root')
fpickle =  open("outputs/" + options.dataset+ "_" + output_label + '.pkl', "wb")
for label in labels:

    # parameters for ABCD plots
    var1 = 'SUEP_'+label+'_' + var1_label
    var2 = 'SUEP_'+label+'_' + var2_label

    output = create_output_file(label)
    sizeA, sizeC = 0,0

    # combine the dataframes
    df = pd.concat(frames[label])

    if var2_label == 'nconst': df = df.loc[df['SUEP_'+label+'_nconst'] >= 10]
    if var1_label == 'spher': df = df.loc[df['SUEP_'+label+'_spher'] >= 0.25]

    # divide the dfs by region
    df_A = df.loc[(df[var1] < var1_val) & (df[var2] < var2_val)]
    df_B = df.loc[(df[var1] >= var1_val) & (df[var2] < var2_val)]
    df_C = df.loc[(df[var1] < var1_val) & (df[var2] >= var2_val)]
    df_D_obs = df.loc[(df[var1] >= var1_val) & (df[var2] >= var2_val)]
    
    #sizeC += ak.size(C) * xsec
    #sizeA += ak.size(A) * xsec
    sizeC += np.sum(df_C['xsec'].to_numpy())
    sizeA += np.sum(df_A['xsec'].to_numpy())

    # fill the ABCD histograms
    output["A_"+label].fill(df_A[var1], weight =  df_A['xsec'])
    output["B_"+label].fill(df_B[var1], weight =  df_B['xsec'])
    output["D_exp_"+label].fill(df_B[var1], weight =  df_B['xsec'])
    output["C_"+label].fill(df_C[var1], weight =  df_C['xsec'])
    output["D_obs_"+label].fill(df_D_obs[var1], weight =  df_D_obs['xsec'])
    output["ABCDvars_2D_"+label].fill(df[var1], df[var2], weight =  df['xsec'])  
  
    # fill the distributions as they are saved in the dataframes
    plot_labels = [key for key in df.keys() if key in list(output.keys())]
    for plot in plot_labels: output[plot].fill(df[plot], weight =  df['xsec'])  

    # fill some new distributions  
    output["2D_girth_nconst_"+label].fill(df["SUEP_"+label+"_girth"], df["SUEP_"+label+"_nconst"], weight= df['xsec'])
    output["2D_rho0_nconst_"+label].fill(df["SUEP_"+label+"_rho0"], df["SUEP_"+label+"_nconst"], weight= df['xsec'])
    output["2D_rho1_nconst_"+label].fill(df["SUEP_"+label+"_rho1"], df["SUEP_"+label+"_nconst"], weight= df['xsec'])
    output["2D_spher_nconst_"+label].fill(df["SUEP_"+label+"_spher"], df["SUEP_"+label+"_nconst"], weight= df['xsec'])
    output["2D_spher_ntracks_"+label].fill(df["SUEP_"+label+"_spher"], df["SUEP_"+label+"_ntracks"], weight= df['xsec'])
    output["SUEP_" + label + "_AB_phi_"].fill(df['SUEP_' + label + '_phi'].loc[(df[var2] < var2_val)].to_numpy(), weight =  df['xsec'].loc[(df[var2] < var2_val)])
    output["SUEP_" + label + "_AB_eta_"].fill(df['SUEP_' + label + '_eta'].loc[(df[var2] < var2_val)].to_numpy(), weight =  df['xsec'].loc[(df[var2] < var2_val)])
    output["SUEP_" + label + "_AB_pt_"].fill(df['SUEP_' + label + '_pt'].loc[(df[var2] < var2_val)].to_numpy(), weight =  df['xsec'].loc[(df[var2] < var2_val)])
    output["SUEP_" + label + "_AC_phi_"].fill(df['SUEP_' + label + '_phi'].loc[(df[var1] < var1_val)].to_numpy(), weight =  df['xsec'].loc[(df[var1] < var1_val)])
    output["SUEP_" + label + "_AC_eta_"].fill(df['SUEP_' + label + '_eta'].loc[(df[var1] < var1_val)].to_numpy(), weight =  df['xsec'].loc[(df[var1] < var1_val)])
    output["SUEP_" + label + "_AC_pt_"].fill(df['SUEP_' + label + '_pt'].loc[(df[var1] < var1_val)].to_numpy(), weight =  df['xsec'].loc[(df[var1] < var1_val)])

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
