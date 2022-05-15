#Make plots for SUEP analysis. Reads in hdf5 files and outputs to pickle and root files
import os, sys, subprocess
import pandas as pd 
import numpy as np
import argparse
import uproot
import getpass
import pickle
import json
from tqdm import tqdm
from hist import Hist

#Import our own functions
import pileup_weight

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
var1_IRM = 'SUEP_S1_IRM'
var2_IRM = 'ntracks'
var1_IRM_val = 0.50
var2_IRM_val = 100
var1_ML = 'resnet_SUEP_pred_ML'
var2_ML = 'ntracks'
var1_ML_val = 0.50
var2_ML_val = 100
output_label = options.output
redirector = "root://t3serv017.mit.edu/"

# selections
spher_nconst_ABCD = [ 
    ["SUEP_spher_IRM", ">=", 0.25],
    ["SUEP_nconst_IRM", ">=", 10]
]
spher_ntracks_ABCD = [
    ['ntracks','>',0],
    ["SUEP_spher_IRM", ">=", 0.25],
]
S1_ntracks_ABCD = [
    ['ntracks','>', 0],
    ["SUEP_S1_IRM", ">=", 0.35],
]
nPVs_l35_study = spher_nconst_ABCD + [['PV_npvs','<',35]]
nPVs_l35_njets_2_study = spher_nconst_ABCD + [['PV_npvs','<',35], ['ngood_fastjets','==',2]]
inf_ntracksABCD = spher_ntracks_ABCD + [['PV_npvs','<',35]]
raw = [['ntracks','>',0]]
ht_barrel = [['ht_barrel', '>', 1200]]
selections = S1_ntracks_ABCD + ht_barrel
    
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
if options.xrootd:
    dataDir = "/scratch/{}/SUEP/{}/{}/merged/".format(username,options.tag,options.dataset)
    result = subprocess.check_output(["xrdfs",redirector,"ls",dataDir])
    result = result.decode("utf-8")
    files = result.split("\n")
    files = [f for f in files if len(f) > 0]
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
            "A_var2_"+label: Hist.new.Reg(499, 0, 500, name="A_var2_"+label).Weight(),
            "B_var2_"+label: Hist.new.Reg(499, 0, 500, name="B_var2_"+label).Weight(),
            "C_var2_"+label: Hist.new.Reg(499, 0, 500, name="C_var2_"+label).Weight(),
            "D_exp_var2_"+label: Hist.new.Reg(499, 0, 500, name="D_exp_var2_"+label).Weight(),
            "D_obs_var2_"+label: Hist.new.Reg(499, 0, 500, name="D_obs_var2_"+label).Weight(),
    }
    if label == 'IRM':
        output.update({"ABCDvars_2D_"+label : Hist.new.Reg(100, 0, 1, name= var1_IRM+"_IRM").Reg(499, 0, 500, name=var2_IRM+"_IRM").Weight()})
    elif label == 'ML':
        output.update({"ABCDvars_2D_"+label : Hist.new.Reg(100, 0, 1, name= var1_ML+"_ML").Reg(499, 0, 500, name=var2_ML+"_ML").Weight()})

    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in ["", "A_", "B_", "C_"]:
        output.update({
            r+"ht_" + label : Hist.new.Reg(100, 0, 10000, name=r+"ht_"+label, label='HT').Weight(),
            r+"ntracks_" + label : Hist.new.Reg(499, 0, 500, name=r+"ntracks_"+label, label='# Tracks in Event').Weight(),
            r+"ngood_fastjets_" + label : Hist.new.Reg(9,0, 10, name=r+"ngood_fastjets_"+label, label='# FastJets in Event').Weight(),
            r+"nLostTracks_"+label : Hist.new.Reg(49,0, 50, name=r+"nLostTracks_"+label, label="# Lost Tracks in Event ").Weight(),
            r+"PV_npvs_"+label : Hist.new.Reg(199,0, 200, name=r+"PV_npvs_"+label, label="# PVs in Event ").Weight(),
            r+"Pileup_nTrueInt_"+label : Hist.new.Reg(199,0, 200, name=r+"Pileup_nTrueInt_"+label, label="# True Interactions in Event ").Weight(),
            r+"ngood_ak4jets_" + label : Hist.new.Reg(19,0, 20, name=r+"ngood_ak4jets_"+label, label= '# ak4jets in Event').Weight(),
        })
        for i in range(10):
            output.update({
                r+"eta_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100,-5,5, name=r+"eta_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $\eta$").Weight(),
                r+"phi_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"phi_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $\phi$").Weight(),
                r+"pt_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $p_T$").Weight(),
            })
        for i in range(2):
            output.update({
                r+"eta_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100,-5,5, name=r+"eta_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $\eta$").Weight(),
                r+"phi_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"phi_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $\phi$").Weight(),
                r+"pt_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $p_T$").Weight(),
            })
            
    if label == 'IRM':
        output.update({
            # 2D histograms
            "2D_SUEP_S1_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(499, 0, 500, name="ntracks_"+label, label='# Tracks').Weight(),
            "2D_SUEP_S1_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(499, 0, 500, name="nconst_"+label, label='# Constituents').Weight(),     
            "2D_SUEP_S1_SUEP_pt_avg_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label).Reg(500, 0, 5000, name="SUEP_pt_avg_"+label).Weight(),
            "2D_ntracks_SUEP_pt_avg_"+label : Hist.new.Reg(499, 0, 500, name="ntracks_"+label).Reg(500, 0, 5000, name="SUEP_pt_avg_"+label).Weight(),  
        })
        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in ["", "A_", "B_", "C_"]:
            output.update({
                r+"SUEP_nconst_"+label : Hist.new.Reg(199, 0, 200, name=r+"SUEP_nconst_"+label, label="# Tracks in SUEP").Weight(),
                r+"SUEP_ntracks_"+label : Hist.new.Reg(499, 0, 500, name=r+"ntracks_"+label, label="# Tracks in event").Weight(),
                r+"SUEP_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r+"SUEP_pt_"+label, label=r"SUEP $p_T$ [GeV]").Weight(),
                r+"SUEP_pt_avg_"+label : Hist.new.Reg(500, 0, 5000, name=r+"SUEP_pt_avg_"+label, label=r"SUEP Components $p_T$ Avg.").Weight(),
                r+"SUEP_pt_avg_b_"+label : Hist.new.Reg(100, 0, 500, name=r+"SUEP_pt_avg_b_"+label, label=r"SUEP Components $p_T$ avg (boosted frame)").Weight(),
                r+"SUEP_nLostTracks_"+label : Hist.new.Reg(199,0, 200, name=r+"SUEP_nLostTracks_"+label, label="# Lost Tracks in SUEP").Weight(),
                r+"SUEP_eta_"+label : Hist.new.Reg(100,-5,5, name=r+"SUEP_eta_"+label, label=r"SUEP $\eta$").Weight(),
                r+"SUEP_phi_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"SUEP_phi_"+label, label=r"SUEP $\phi$").Weight(),
                r+"SUEP_mass_"+label : Hist.new.Reg(150, 0, 4000, name=r+"SUEP_mass_"+label, label="SUEP Mass [GeV]").Weight(),
                r+"SUEP_S1_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_S1_"+label, label='$Sph_1$').Weight(),
                r+"SUEP_girth": Hist.new.Reg(50, 0, 1.0, name=r+"SUEP_girth_"+label, label=r"SUEP Girth").Weight(),
                r+"SUEP_rho0_"+label : Hist.new.Reg(100, 0, 20, name=r+"SUEP_rho0_"+label, label=r"SUEP $\rho_0$").Weight(),
                r+"SUEP_rho1_"+label : Hist.new.Reg(100, 0, 20, name=r+"SUEP_rho1_"+label, label=r"SUEP $\rho_1$").Weight(),
            })

    if label == 'ML':
        for r in ["", "A_", "B_", "C_"]:
            output.update({
                r+"resnet_SUEP_pred_"+label : Hist.new.Reg(100, 0, 1, name=r+"resnet_SUEP_pred_"+label, label="Resnet Output").Weight(),
                r+"ntracks_"+label : Hist.new.Reg(100, 0, 500, name=r+"ntracks"+label, label="# Tracks in Event").Weight(),
            })
                        
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
for label in ['IRM','ML']: 
    output.update(create_output_file(label))
    sizeA.update({label:0})
    sizeB.update({label:0})
    sizeC.update({label:0})


puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(options.era)   
for ifile in tqdm(files):

    if options.xrootd:
        if os.path.exists(options.dataset+'.hdf5'): os.system('rm ' + options.dataset+'.hdf5')
        xrd_file = redirector + ifile
        os.system("xrdcp -s {} {}.hdf5".format(xrd_file, options.dataset))
        df, metadata = h5load(options.dataset+'.hdf5', 'vars')   
    else:
        df, metadata = h5load(ifile, 'vars')   
 

    # check if file is corrupted
    if type(df) == int: 
        nfailed += 1
        continue
            
    # update the gensumweight
    if options.isMC and metadata != 0: weight += metadata['gensumweight']

    # check if file is empty
    if 'empty' in list(df.keys()): continue
    if df.shape[0] == 0: continue    

    #####################################################################################
    #Additional weights [pileup_weight]
    #####################################################################################
    event_weight = np.ones(df.shape[0])
    if options.isMC == 1:
        Pileup_nTrueInt = np.array(df['Pileup_nTrueInt']).astype(int)
        pu = puweights[Pileup_nTrueInt]
        event_weight *= pu
    #event_weight *= another event weight, etc
    df['event_weight'] = event_weight

    #####################################################################################
    # ---- ML Method Plots
    #####################################################################################
    label = 'ML'

    # keep event wide and ML variables, cut out events that don't pass IRM
    df_ML = df[[c for c in df.keys() if 'IRM' not in c]]
    df_ML = df_ML[~df['resnet_SUEP_pred_ML'].isnull()]
    
    # apply selections
    for sel in selections: 
        if sel[0] in list(df_ML.keys()):
            df_ML = apply_selection(df_ML, sel[0], sel[1], sel[2])
    
  
    # blind
    if options.blind and not options.isMC:
         df_ML = df_ML.loc[((df_ML[var1_ML] < var1_ML_val) & (df_ML[var2_ML] < var2_ML_val)) | ((df_ML[var1_ML] >= var1_ML_val) & (df_ML[var2_ML] < var2_ML_val)) | ((df_ML[var1_ML] < var1_ML_val) & (df_ML[var2_ML] >= var2_ML_val))]

    # divide the dfs by region
    df_A = df_ML.loc[(df_ML[var1_ML] < var1_ML_val) & (df_ML[var2_ML] < var2_ML_val)]
    df_B = df_ML.loc[(df_ML[var1_ML] >= var1_ML_val) & (df_ML[var2_ML] < var2_ML_val)]
    df_C = df_ML.loc[(df_ML[var1_ML] < var1_ML_val) & (df_ML[var2_ML] >= var2_ML_val)]
    df_D_obs = df_ML.loc[(df_ML[var1_ML] >= var1_ML_val) & (df_ML[var2_ML] >= var2_ML_val)]

    # keep track of number of events per region, used to measure D_exp
    sizeC[label] += df_C.shape[0]
    sizeB[label] += df_B.shape[0]
    sizeA[label] += df_A.shape[0]
    
    # fill the ABCD histograms for both variables
    output["A_"+label].fill(df_A[var1_ML], weight=df_A['event_weight'])
    output["B_"+label].fill(df_B[var1_ML], weight=df_B['event_weight'])
    output["D_exp_"+label].fill(df_B[var1_ML], weight=df_B['event_weight'])
    output["C_"+label].fill(df_C[var1_ML], weight=df_C['event_weight'])
    output["D_obs_"+label].fill(df_D_obs[var1_ML], weight=df_D_obs['event_weight'])
    output["A_var2_"+label].fill(df_A[var2_ML], weight=df_A['event_weight'])
    output["B_var2_"+label].fill(df_B[var2_ML], weight=df_B['event_weight'])
    output["D_exp_var2_"+label].fill(df_C[var2_ML], weight=df_C['event_weight'])
    output["C_var2_"+label].fill(df_C[var2_ML], weight=df_C['event_weight'])
    output["D_obs_var2_"+label].fill(df_D_obs[var2_ML], weight=df_D_obs['event_weight'])
    output["ABCDvars_2D_"+label].fill(df_ML[var1_ML], df_ML[var2_ML], weight=df_ML['event_weight'])
        
    # fill the distributions as they are saved in the dataframes
    plot_labels = [key for key in df_ML.keys() if key in list(output.keys())]      # all the _ML things
    for plot in plot_labels: output[plot].fill(df_ML[plot], weight=df_ML['event_weight'])  
    plot_labels = [key for key in df_ML.keys() if key+"_"+label in list(output.keys())]      # event wide variables
    for plot in plot_labels: output[plot+"_"+label].fill(df_ML[plot], weight=df_ML['event_weight']) 
    
    # per region
    for r, df_r in zip(["A_", "B_", "C_"], [df_A, df_B, df_C]):
        plot_labels = [key for key in df_r.keys() if r+key in list(output.keys())]     # all the _ML things
        for plot in plot_labels: output[r+plot].fill(df_r[plot], weight=df_r['event_weight'])  
        plot_labels = [key for key in df_r.keys() if r+key+"_"+label in list(output.keys())]    # event wide variables
        for plot in plot_labels: output[r+plot+"_"+label].fill(df_r[plot], weight=df_r['event_weight'])  

    #####################################################################################
    # ---- ISR Removal Method Plots
    #####################################################################################
    label = 'IRM'
        
    # apply selections
    for sel in selections: 
        df = apply_selection(df, sel[0], sel[1], sel[2])
        
    # keep event wide and IRM variables, cut out events that don't pass IRM
    df_IRM = df[[c for c in df.keys() if 'ML' not in c]]
    df_IRM = df_IRM[~df['SUEP_pt_IRM'].isnull()]
    
    # blind
    if options.blind and not options.isMC:
         df_IRM = df_IRM.loc[((df_IRM[var1_IRM] < var1_IRM_val) & (df_IRM[var2_IRM] < var2_IRM_val)) | ((df_IRM[var1_IRM] >= var1_IRM_val) & (df_IRM[var2_IRM] < var2_IRM_val)) | ((df_IRM[var1_IRM] < var1_IRM_val) & (df_IRM[var2_IRM] >= var2_IRM_val))]

    # divide the dfs by region
    df_A = df_IRM.loc[(df_IRM[var1_IRM] < var1_IRM_val) & (df_IRM[var2_IRM] < var2_IRM_val)]
    df_B = df_IRM.loc[(df_IRM[var1_IRM] >= var1_IRM_val) & (df_IRM[var2_IRM] < var2_IRM_val)]
    df_C = df_IRM.loc[(df_IRM[var1_IRM] < var1_IRM_val) & (df_IRM[var2_IRM] >= var2_IRM_val)]
    df_D_obs = df_IRM.loc[(df_IRM[var1_IRM] >= var1_IRM_val) & (df_IRM[var2_IRM] >= var2_IRM_val)]

    # keep track of number of events per region, used to measure D_exp
    sizeC[label] += df_C.shape[0]
    sizeB[label] += df_B.shape[0]
    sizeA[label] += df_A.shape[0]

    # fill the ABCD histograms for both variables
    output["A_"+label].fill(df_A[var1_IRM], weight=df_A['event_weight'])
    output["B_"+label].fill(df_B[var1_IRM], weight=df_B['event_weight'])
    output["D_exp_"+label].fill(df_B[var1_IRM], weight=df_B['event_weight'])
    output["C_"+label].fill(df_C[var1_IRM], weight=df_C['event_weight'])
    output["D_obs_"+label].fill(df_D_obs[var1_IRM], weight=df_D_obs['event_weight'])
    output["A_var2_"+label].fill(df_A[var2_IRM], weight=df_A['event_weight'])
    output["B_var2_"+label].fill(df_B[var2_IRM], weight=df_B['event_weight'])
    output["D_exp_var2_"+label].fill(df_C[var2_IRM], weight=df_C['event_weight'])
    output["C_var2_"+label].fill(df_C[var2_IRM], weight=df_C['event_weight'])
    output["D_obs_var2_"+label].fill(df_D_obs[var2_IRM], weight=df_D_obs['event_weight'])
    output["ABCDvars_2D_"+label].fill(df_IRM[var1_IRM], df_IRM[var2_IRM], weight=df_IRM['event_weight'])

    # fill the distributions as they are saved in the dataframes
    plot_labels = [key for key in df_IRM.keys() if key in list(output.keys())]     # all the _IRM things
    for plot in plot_labels: output[plot].fill(df_IRM[plot], weight=df_IRM['event_weight'])  
    plot_labels = [key for key in df_IRM.keys() if key+"_"+label in list(output.keys())]     # event wide variables
    for plot in plot_labels: output[plot+"_"+label].fill(df_IRM[plot], weight=df_IRM['event_weight'])  

    # fill some new distributions  
    output["2D_SUEP_S1_ntracks_"+label].fill(df_IRM["SUEP_S1_"+label], df_IRM["SUEP_ntracks_"+label], weight=df_IRM['event_weight'])
    output["2D_SUEP_S1_SUEP_nconst_"+label].fill(df_IRM["SUEP_S1_"+label], df_IRM["SUEP_nconst_"+label], weight=df_IRM['event_weight'])
    output["2D_SUEP_S1_SUEP_pt_avg_"+label].fill(df_IRM["SUEP_S1_"+label], df_IRM["SUEP_pt_avg_"+label], weight=df_IRM['event_weight'])
    output["2D_ntracks_SUEP_pt_avg_"+label].fill(df_IRM["SUEP_ntracks_"+label], df_IRM["SUEP_pt_avg_"+label], weight=df_IRM['event_weight'])
    
    # per region
    for r, df_r in zip(["A_", "B_", "C_"], [df_A, df_B, df_C]):

        # fill the distributions as they are saved in the dataframes
        plot_labels = [key for key in df_r.keys() if r+key in list(output.keys())]    # all the _IRM things
        for plot in plot_labels: output[r+plot].fill(df_r[plot], weight=df_r['event_weight'])  
        plot_labels = [key for key in df_r.keys() if r+key+"_"+label in list(output.keys())]   # event wide variables
        for plot in plot_labels: output[r+plot+"_"+label].fill(df_r[plot], weight=df_r['event_weight']) 
        
        df_r_4jets = df_r[~df_r['pt_ak4jets4'].isnull()]
        output[r+'pt_ak4jets0_4jets_'+label].fill(df_r_4jets['pt_ak4jets0'], weight=df_r_4jets['event_weight'])
        output[r+'phi_ak4jets0_4jets_'+label].fill(df_r_4jets['phi_ak4jets0'], weight=df_r_4jets['event_weight'])
        output[r+'eta_ak4jets0_4jets_'+label].fill(df_r_4jets['eta_ak4jets0'], weight=df_r_4jets['event_weight'])
        output[r+'pt_ak4jets1_4jets_'+label].fill(df_r_4jets['pt_ak4jets1'], weight=df_r_4jets['event_weight'])
        output[r+'phi_ak4jets1_4jets_'+label].fill(df_r_4jets['phi_ak4jets1'], weight=df_r_4jets['event_weight'])
        output[r+'eta_ak4jets1_4jets_'+label].fill(df_r_4jets['eta_ak4jets1'], weight=df_r_4jets['event_weight'])
        
    if options.xrootd: os.system('rm ' + options.dataset+'.hdf5')    

### end plotting loop
        
# ABCD method to obtain D expected for each selection
for label in ['IRM', 'ML']:
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
