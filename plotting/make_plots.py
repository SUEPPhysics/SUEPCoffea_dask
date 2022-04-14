import os, sys
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
parser.add_argument('--xrootd', type=int, default=0, help="Local data or xrdcp from hadoop (default=False)")
options = parser.parse_args()


# parameters for script
var1_label = 'SUEP_spher'
var2_label = 'ntracks'
var1_val = 0.50
var2_val = 100
labels = ['ch']             # which selection to make plots for
output_label = options.output
redirector = "root://t3serv017.mit.edu/"

# selections
default_ABCD = [ 
    ["SUEP_spher", ">=", 0.25],
    ["SUEP_nconst", ">=", 10]
]
nPVs_l35_study = default_ABCD + [['PV_npvs','<',35]]
nPVs_l35_njets2_study = default_ABCD + [['PV_npvs','<',35], ['ngood_fastjets','==',2]]
nPVs_l40_study = default_ABCD + [['PV_npvs','<',40]]
selections = nPVs_l35_njets2_study
    
def apply_selection(df, variable, operator, value):
    """
    Apply a selection on DataFrame df based on on the df column'variable'
    using the 'operator' and 'value' passed as arguments to the function.
    Returns the resulting DataFrame after the operation is applied.
    """
    if operator in ["greater than","gt",">"]:
        return df.loc[df[variable] > value]
    if operator in ["greater than or equal to", ">="]:
        return df.loc[df[variable] >= value]
    elif operator in ["less than", "lt", "<"]:
        return df.loc[df[variable] < value]
    elif operator in ["less than or equal to", "<="]:
        return df.loc[df[variable] <= value]
    elif operator in ["equal to", "eq", "=="]:
        return df.loc[df[variable] == value]
    else:
        sys.exit("Couldn't find operator requested")

# get list of files
username = getpass.getuser()
# username = 'freerc'
if options.xrootd:
    dataDir = "/scratch/{}/SUEP/{}/{}/".format(username,options.tag,options.dataset)
    result = subprocess.check_output(["xrdfs",redirector,"ls",dataDir])
    result = result.decode("utf-8")
    files = result.split("\n")
    files = [f for f in files if 'condor_out' not in f]
else:
    dataDir = "/work/submit/{}/SUEP/{}/{}/".format(username, options.tag, options.dataset)
    files = [dataDir + f for f in os.listdir(dataDir)]

# cross section
xsection = 1.0
with open('../data/xsections_{}.json'.format(options.era)) as file:
    MC_xsecs = json.load(file)
    try:
        xsection *= MC_xsecs[options.dataset]["xsec"]
        xsection *= MC_xsecs[options.dataset]["kr"]
        xsection *= MC_xsecs[options.dataset]["br"]
    except:
        print("WARNING: I did not find the xsection for that MC sample. Check the dataset name and the relevant yaml file")

# output histos
def create_output_file(label):
    output = {
            # ABCD hists
            "A_"+label: Hist.new.Reg(100, 0, 1, name="A_"+label).Weight(),
            "B_"+label: Hist.new.Reg(100, 0, 1, name="B_"+label).Weight(),
            "C_"+label: Hist.new.Reg(100, 0, 1, name="C_"+label).Weight(),
            "D_exp_"+label: Hist.new.Reg(100, 0, 1, name="D_exp_"+label).Weight(),
            "D_obs_"+label: Hist.new.Reg(100, 0, 1, name="D_obs_"+label).Weight(),
            "A_var2_"+label: Hist.new.Integer(0, 500, name="A_var2_"+label).Weight(),
            "B_var2_"+label: Hist.new.Integer(0, 500, name="B_var2_"+label).Weight(),
            "C_var2_"+label: Hist.new.Integer(0, 500, name="C_var2_"+label).Weight(),
            "D_exp_var2_"+label: Hist.new.Integer(0, 500, name="D_exp_var2_"+label).Weight(),
            "D_obs_var2_"+label: Hist.new.Integer(0, 500, name="D_obs_var2_"+label).Weight(),
            "ABCDvars_2D_"+label : Hist.new.Reg(100, 0, 1, name= var1_label+"_"+label).Integer(0, 500, name=var2_label+"_"+label).Weight(),
        
            # 2D histograms
            "2D_SUEP_girth_SUEP_nconst_"+label : Hist.new.Reg(50, 0, 1.0, name="SUEP_girth_"+label).Integer(0, 200, name="SUEP_nconst_"+label).Weight(),
            "2D_SUEP_rho0_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 20, name="SUEP_rho0_"+label).Integer(0, 200, name="nconst_"+label).Weight(),
            "2D_SUEP_rho1_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 20, name="SUEP_rho1_"+label).Integer(0, 200, name="nconst_"+label).Weight(),
            "2D_SUEP_spher_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_spher_"+label).Integer(0, 500, name="ntracks_"+label).Weight(),
            "2D_SUEP_spher_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_spher_"+label).Integer(0, 500, name="nconst_"+label).Weight(),
            "2D_nJets_SUEP_pT_"+label : Hist.new.Integer(0, 20, name="nJets_"+label).Reg(100, 0, 3000, name="pt_"+label).Weight(),
        
    }
    
    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in ["", "A_", "B_", "C_"]:
        output.update({
            r+"SUEP_nconst_"+label : Hist.new.Reg(199, 0, 200, name=r+"SUEP_nconst_"+label, label="# Tracks in SUEP").Weight(),
            r+"SUEP_ntracks_"+label : Hist.new.Reg(499, 0, 500, name=r+"ntracks_"+label, label="# Tracks in event").Weight(),
            r+"SUEP_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r+"SUEP_pt_"+label, label=r"SUEP $p_T$ [GeV]").Weight(),
            r+"SUEP_pt_avg_"+label : Hist.new.Reg(100, 0, 100, name=r+"SUEP_pt_avg_"+label, label=r"SUEP Components $p_T$ Avg.").Weight(),
            r+"SUEP_pt_avg_b_"+label : Hist.new.Reg(100, 0, 100, name=r+"SUEP_pt_avg_b_"+label, label=r"SUEP Components $p_T$ avg (boosted frame)").Weight(),
            r+"SUEP_nLostTracks_"+label : Hist.new.Reg(199,0, 200, name=r+"SUEP_nLostTracks_"+label, label="# Lost Tracks in SUEP").Weight(),
            r+"SUEP_eta_"+label : Hist.new.Reg(100,-5,5, name=r+"SUEP_eta_"+label, label=r"SUEP $\eta$").Weight(),
            r+"SUEP_phi_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"SUEP_phi_"+label, label=r"SUEP $\phi$").Weight(),
            r+"SUEP_mass_"+label : Hist.new.Reg(150, 0, 4000, name=r+"SUEP_mass_"+label, label="SUEP Mass [GeV]").Weight(),
            r+"SUEP_spher_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_spher_"+label, label="SUEP Sphericity").Weight(),
            r+"SUEP_aplan_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_aplan_"+label, label="Aplanarity").Weight(),
            r+"SUEP_FW2M_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_FW2M_"+label, label="SUEP 2nd Fox Wolfram Moment").Weight(),
            r+"SUEP_D_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_D_"+label, label="SUEP_D").Weight(),
            r+"SUEP_girth": Hist.new.Reg(50, 0, 1.0, name=r+"SUEP_girth_"+label, label=r"SUEP Girth").Weight(),
            r+"SUEP_rho0_"+label : Hist.new.Reg(100, 0, 20, name=r+"SUEP_rho0_"+label, label=r"SUEP $\rho_0$").Weight(),
            r+"SUEP_rho1_"+label : Hist.new.Reg(100, 0, 20, name=r+"SUEP_rho1_"+label, label=r"SUEP $\rho_1$").Weight(),
            
            r+"ht_" + label : Hist.new.Reg(1000, 0, 10000, name=r+"ht_"+label, label='HT').Weight(),
            r+"ngood_fastjets_" + label : Hist.new.Reg(49,0, 50, name=r+"ngood_fastjets_"+label, label='# Jets in Event').Weight(),
            r+"nLostTracks_"+label : Hist.new.Reg(199,0, 200, name=r+"nLostTracks_"+label, label="# Lost Tracks in Event ").Weight(),
            r+"PV_npvs_"+label : Hist.new.Reg(199,0, 200, name=r+"PV_npvs_"+label, label="# PVs in Event ").Weight(),
            r+"ngood_ak4jets_" + label : Hist.new.Reg(19,0, 20, name=r+"ngood_ak4jets_"+label, label= '# ak4jets in Event').Weight()
        })
    
    # histograms for A, B, C regions only
    for r in ["A_", "B_", "C_"]:
        output.update({     
            r+"2D_SUEP_pt_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_"+label).Red(199,0, 200, name=r+" nconst_"+label).Weight(),
            r+"2D_SUEP_nconst_ntracks_"+label : Hist.new.Reg(199, 0, 200, name=r+"SUEP_nconst_"+label).Reg(499,0, 500, name=r+"ntracks_"+label).Weight(),
            r+"2D_SUEP_spher_ntracks_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_spher_"+label).Reg(499, 0, 500, name=r+"ntracks_"+label).Weight(),
        })
        
    # ISR removal method only
    if label == 'ch':
        output2 = {
            "SUEP_dphi_chcands_ISR_"+label:Hist.new.Reg(100, 0, 4, name="dphi_chcands_ISR").Weight(),
            "SUEP_dphi_SUEPtracks_ISR_"+label: Hist.new.Reg(100, 0, 4, name="dphi_SUEPtracks_ISR").Weight(),
            "SUEP_dphi_ISRtracks_ISR_"+label:Hist.new.Reg(100, 0, 4, name="dphi_ISRtracks_ISR").Weight(),
            "SUEP_dphi_SUEP_ISR_"+label:Hist.new.Reg(100, 0, 4, name="dphi_SUEP_ISR").Weight(),
        }
        output.update(output2)
        
    # systematic plots
    if options.isMC and options.doSyst:
        output3 = {
            #"SUEP_"+label+"_variable_"+sys:Hist.new.Reg(100, 0, 4, name="variable").Weight(),
        }
        output.update(output3)
        
    return output

# load hdf5 with pandas
def h5load(ifile, label):
    try:
        with pd.HDFStore(ifile, 'r') as store:
            try:
                data = store[label] 
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata
        
            except KeyError:
                print("No key",label,ifile)
                return 0, 0
    except:
        print("Some error occurred", ifile)
        return 0, 0
        
# fill ABCD hists with dfs from hdf5 files
nfailed = 0
weight = 0
fpickle =  open("outputs/" + options.dataset+ "_" + output_label + '.pkl', "wb")
output, sizeA, sizeB, sizeC = {}, {}, {}, {}
for label in labels: 
    output.update(create_output_file(label))
    sizeA.update({label:0})
    sizeB.update({label:0})
    sizeC.update({label:0})
    
for ifile in tqdm(files):

    if options.xrootd:
        if os.path.exists(options.dataset+'.hdf5'): os.system('rm ' + options.dataset+'.hdf5')
        xrd_file = redirector + ifile
        os.system("xrdcp {} {}.hdf5".format(xrd_file, options.dataset))
        df_vars, metadata = h5load(options.dataset+'.hdf5', 'vars')   
    else:
        df_vars, metadata = h5load(ifile, 'vars')   
        
    # check if file is corrupted
    if type(df_vars) == int: 
        nfailed += 1
        continue
            
    # update the gensumweight
    if options.isMC: weight += metadata['gensumweight']
    
    # check if file is empty
    if df_vars.shape[0] == 0: continue    
    
    for label in labels:
        if options.xrootd: df, metadata = h5load(options.dataset+'.hdf5', label) 
        else: df, metadata = h5load(ifile, label)
        if 'empty' in list(df.keys()): continue
        if df.shape[0] == 0: continue
        
        # parameters for ABCD plots
        var1 = var1_label + '_' + label
        var2 = var2_label + '_' + label
                
        # index the event wide variables by the indices of the events passing the 'label'
        df_vars_label = df_vars.loc[df['event_index_'+label]]
        for key in list(df_vars_label.keys()):
            df[key+"_"+label] = df_vars_label[key].to_numpy()
                            
        # apply selections
        for sel in selections: 
            df = apply_selection(df, sel[0] + "_" + label, sel[1], sel[2])
        
        # blind
        if options.blind and not options.isMC:
             df = df.loc[((df[var1] < var1_val) & (df[var2] < var2_val)) | ((df[var1] >= var1_val) & (df[var2] < var2_val)) | ((df[var1] < var1_val) & (df[var2] >= var2_val))]

        # divide the dfs by region
        df_A = df.loc[(df[var1] < var1_val) & (df[var2] < var2_val)]
        df_B = df.loc[(df[var1] >= var1_val) & (df[var2] < var2_val)]
        df_C = df.loc[(df[var1] < var1_val) & (df[var2] >= var2_val)]
        df_D_obs = df.loc[(df[var1] >= var1_val) & (df[var2] >= var2_val)]
                
        # keep track of number of events per region, used to measure D_exp
        sizeC[label] += df_C.shape[0]
        sizeB[label] += df_B.shape[0]
        sizeA[label] += df_A.shape[0]
                
        # fill the ABCD histograms for both variables
        output["A_"+label].fill(df_A[var1])
        output["B_"+label].fill(df_B[var1])
        output["D_exp_"+label].fill(df_B[var1])
        output["C_"+label].fill(df_C[var1])
        output["D_obs_"+label].fill(df_D_obs[var1])
        output["A_var2_"+label].fill(df_A[var2])
        output["B_var2_"+label].fill(df_B[var2])
        output["D_exp_var2_"+label].fill(df_C[var2])
        output["C_var2_"+label].fill(df_C[var2])
        output["D_obs_var2_"+label].fill(df_D_obs[var2])
        output["ABCDvars_2D_"+label].fill(df[var1], df[var2])
        
        # fill the distributions as they are saved in the dataframes
        plot_labels = [key for key in df.keys() if key in list(output.keys())]
        for plot in plot_labels: output[plot].fill(df[plot])  

        # fill some new distributions  
        output["2D_SUEP_girth_SUEP_nconst_"+label].fill(df["SUEP_girth_"+label], df["SUEP_nconst_"+label])
        output["2D_SUEP_rho0_SUEP_nconst_"+label].fill(df["SUEP_rho0_"+label], df["SUEP_nconst_"+label])
        output["2D_SUEP_rho1_SUEP_nconst_"+label].fill(df["SUEP_rho1_"+label], df["SUEP_nconst_"+label])
        output["2D_SUEP_spher_SUEP_nconst_"+label].fill(df["SUEP_spher_"+label], df["SUEP_nconst_"+label])
        output["2D_SUEP_spher_ntracks_"+label].fill(df["SUEP_spher_"+label], df["SUEP_ntracks_"+label])
        
        # per region
        for r, df_r in zip(["A_", "B_", "C_"], [df_A, df_B, df_C]):
                        
            # fill the distributions as they are saved in the dataframes
            plot_labels = [key for key in df_r.keys() if r+key in list(output.keys())]
            for plot in plot_labels: output[r+plot].fill(df_r[plot])  
            
            # fill some new distributions 
            output[r+"2D_SUEP_pt_SUEP_nconst_"+label].fill(df_r['SUEP_pt_'+label], df_r['SUEP_nconst_'+label])
            output[r+"2D_SUEP_nconst_ntracks_"+label].fill(df_r["SUEP_nconst_"+label], df_r["SUEP_ntracks_"+label])
            output[r+"2D_SUEP_spher_ntracks_"+label].fill(df_r["SUEP_spher_"+label], df_r["SUEP_ntracks_"+label])
    
    if options.xrootd: os.system('rm ' + options.dataset+'.hdf5')    
        
# ABCD method to obtain D expected for each selection
for label in labels:
    if sizeA[label]>0.0:
        CoverA =  sizeC[label] / sizeA[label]
        CoverA_var2 =  sizeB[label] / sizeA[label]
    else:
        CoverA = 0.0
        CoverA_var2 = 0.0
        print("A region has no occupancy for selection", label)
    output["D_exp_"+label] = output["D_exp_"+label]*(CoverA)
    output["D_exp_var2_"+label] = output["D_exp_var2_"+label]*(CoverA_var2)
    
# apply normalization
if weight > 0.0 and options.isMC:
    for plot in list(output.keys()): output[plot] = output[plot]*xsection/weight
else:
    print("Weight is 0")
        
#Save to pickle
pickle.dump(output, fpickle)
print("Number of files that failed to be read:", nfailed)

# save to root
with uproot.recreate("outputs/" + options.dataset+ "_" + output_label + '.root') as froot:
    for h, hist in output.items():
        froot[h] = hist