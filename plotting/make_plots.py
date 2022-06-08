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
from collections import defaultdict

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
parser.add_argument('--weights', type=int, default=0, help="Apply binned weights (default=False)")
parser.add_argument('--xrootd', type=int, default=0, help="Local data or xrdcp from hadoop (default=False)")
options = parser.parse_args()

# parameters for script
output_label = options.output
redirector = "root://t3serv017.mit.edu/"

# ABCD methods
# include lower and upper bounds for ALL regions
abcd_IRM = {
    'SUEP_S1_IRM' : [0.35, 0.4, 0.5, 1.0],
    'SUEP_nconst_IRM' : [10, 20, 40, 1000],
    'SR' : [['SUEP_S1_IRM', '>=', 0.5], ['SUEP_nconst_IRM', '>=', 40]]
}
abcd_CL = {
    'SUEP_S1_CL' : [0.35, 0.4, 0.5, 1.0],
    'SUEP_nconst_CL' : [20, 40, 80, 1000],
    'SR' : [['SUEP_S1_CL', '>=', 0.5], ['SUEP_nconst_CL', '>=', 80]]
}
abcd_CLi= {
    'ISR_S1_CL' : [0.35, 0.4, 0.5, 1.0],
    'ISR_nconst_CL' : [20, 40, 80, 1000],
    'SR' : [['ISR_S1_CL', '>=', 0.5], ['ISR_nconst_CL', '>=', 80]]
}
abcd_ML = {
    'resnet_SUEP_pred_ML' : [0.0, 0.5, 1.0],
    'ntracks': [0, 100, 1000],
    'SR' : [['resnet_SUEP_pred_ML', '>=', 0.5], ['ntracks', '>=', 100]]
}

# selections
base = [['ht_tracker', '>', 1200], ['ntracks','>',0]]
S1_ntracks_ABCD_IRM = [
    ['ntracks','>', 10],
    ["SUEP_S1_IRM", ">=", 0.35],
]
S1_ntracks_ABCD_CL = [
    ['ntracks','>', 10],
    ["SUEP_S1_CL", ">=", 0.35],
]
S1_ntracks_ABCD_CLi = [
    ['ntracks','>', 10],
    ["ISR_S1_CL", ">=", 0.35],
]
# pick selections to be used for different methods
selections_ML = base
selections_IRM = S1_ntracks_ABCD_IRM + base
selections_CL = S1_ntracks_ABCD_CL + base
selections_CLi = S1_ntracks_ABCD_CLi + base


#############################################################################################################

def make_selection(df, variable, operator, value, apply=True):
    """
    Apply a selection on DataFrame df based on on the df column'variable'
    using the 'operator' and 'value' passed as arguments to the function.
    Returns the resulting DataFrame after the operation is applied.
    
    df: input dataframe.
    variable: df column.
    operator: see code below.
    value: value to cut variable on using operator.
    apply: toggles whether the selection is applied to the dataframe, or
    whether a list of booleans is returned matching the indices that
    passed and failed the selection.
    """
    if operator in ["greater than","gt",">"]:
        if apply: return df.loc[(df[variable] > value)]
        else: return (df[variable] > value)
    if operator in ["greater than or equal to", ">="]:
        if apply: return df.loc[(df[variable] >= value)]
        else: return (df[variable] >= value)
    elif operator in ["less than", "lt", "<"]:
        if apply: return df.loc[(df[variable] < value)]
        else: return (df[variable] < value)
    elif operator in ["less than or equal to", "<="]:
        if apply: return df.loc[(df[variable] <= value)]
        else: return (df[variable] <= value)
    elif operator in ["equal to", "eq", "=="]:
        if apply: return df.loc[(df[variable] == value)]
        else: return (df[variable] == value)
    else:
        sys.exit("Couldn't find operator requested " + operator)
        
def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

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
    
def plot(df_in, size_dict, output, selections, abcd, label='ML', label_out='ML', vars2d = []):
    """
    INPUTS:
        df_in: input DataFrame.
        size_dict: dictionary which holds count of events in each region for each method.
        output: dictionary of histograms.
        
        selections: selections to be applied to df_in for a method.
        abcd: definitions of ABCD regions, signal region.
        
        label: label associated with the method (e.g. "CL") as used in df_in.
        label_out: label associated with the output method (e.g. "Cli") as used in the
                   output dictionary, the selections, and the abcd dict. i.e., multiple
                   label_out's can be defined for the same input label, as different
                   selections and ABCD methods can be applied to the same input method.
        vars2d: list of pairs of variables that you want to plot against each other in 2D.
        
    OUTPUTS: 
        size_dict: now updated with this file's counts.
        output: now with updated histograms.
        
    EXPLANATION:
    The DataFrame generated by ../workflows/SUEP_coffea.py has the form:
    event variables (ht, ...)   IRM vars (SUEP_S1_IRM, ...)  ML vars  Other Methods
          0                                 0                   0          ...
          1                                 NaN                 1          ...
          2                                 NaN                 NaN        ...
          3                                 1                   2          ...
    The event vars are always filled, while the vars for each method are filled only
    if the event passes the method's selections.
    
    N.B.: Histograms are filled only if they already exist in the output dictionary.
    Each input method ('label') is processed using different selections, and
    ABCD regions, and is then outputted to a specific ('label_out').

    e.g. We want to plot CL. 
    Event Selection:
        1. Grab only events that don't have NaN for CL variables.
        2. Grab the event_variables and CL_vars columns.
        3. Blind for data! Use abcd_CL['SR'] to define signal regions and cut it out of df.
        4. Apply selections as defined in selections_CL.

    Fill Histograms:
        1. Plot variables from the DataFrame. 
           1a. Event wide variables
           1b. CL variables
        2. Plot 2D variables from the vars2d list.
        3. Plot variables from the different ABCD regions as defined in the abcd dict.
           3a. Event wide variables
           3b. CL variables
    """

    #####################################################################################
    # ---- Event Selection
    #####################################################################################
    
    # 1. keep only events that passed this method
    df = df_in.copy()
    event_selector = {
        'CL' : 'SUEP_pt_CL',
        'CLi' : 'SUEP_pt_CL',
        'IRM' : 'SUEP_pt_IRM',
        'ML' : 'resnet_SUEP_pred_ML'
    }
    df = df[~df[event_selector[label]].isnull()]

    # 2. keep event wide variables and variables for this method only
    all_labels = ['CL', 'IRM', 'ML']
    exclude_labels = [l for l in all_labels if l != label]
    df = df[[c for c in df.keys() if all([l not in c for l in exclude_labels])]]
        
    # 3. blind
    if options.blind and not options.isMC:       
        SR = abcd['SR']
        if len(SR) != 2: sys.exit(label_out + ": Make sure you have correctly defined your signal region. Exiting.")
        df = df.loc[~(make_selection(df, SR[0][0], SR[0][1], SR[0][2], apply=False) & make_selection(df, SR[1][0], SR[1][1], SR[1][2], apply=False))]
        
    # 4. apply selections
    for sel in selections: 
        df = make_selection(df, sel[0], sel[1], sel[2], apply=True)
        
    #####################################################################################
    # ---- Fill Histograms
    #####################################################################################
    
    # 1. fill the distributions as they are saved in the dataframes
    # 1a. Plot event wide variables
    plot_labels = [key for key in df.keys() if key+"_"+label_out in list(output.keys())]  
    for plot in plot_labels: output[plot+"_"+label_out].fill(df[plot], weight=df['event_weight']) 
    # 1b. Plot method variables
    plot_labels = [key for key in df.keys() if key.replace(label, label_out) in list(output.keys())]
    for plot in plot_labels: output[plot.replace(label, label_out)].fill(df[plot], weight=df['event_weight'])  
    
    # 2. fill some 2D distributions  
    keys = list(output.keys())
    for pair in vars2d:
        x_in = pair[0]
        x_out = pair[0].replace("_"+label, "")
        y_in = pair[1]
        y_out = pair[1].replace("_"+label, "")
        
        plot = "2D_" + x_out + "_" + y_out +"_" +label_out
        plot_inv = "2D_" + y_out + "_" + x_out +"_" +label_out
        if plot in keys: output[plot].fill(df[x_in], df[y_in], weight=df['event_weight'])
        elif plot_inv in keys: output[plot_inv].fill(df[y_in], df[x_in], weight=df['event_weight'])
        else:
            print("Didn't find histogram for", pair)
            continue

    # 3. divide the dfs by region
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    x_var = list(abcd.keys())[0]
    y_var = list(abcd.keys())[1]
    iRegion = 0
    for i in range(len(abcd[x_var])-1):
        x_val_lo = abcd[x_var][i]
        x_val_hi = abcd[x_var][i+1]
        
        for j in range(len(abcd[y_var])-1):
            y_val_lo = abcd[y_var][j]
            y_val_hi = abcd[y_var][j+1]
            
            x_cut = (make_selection(df, x_var, '>=', x_val_lo, False) & make_selection(df, x_var, '<', x_val_hi, False))
            y_cut = (make_selection(df, y_var, '>=', y_val_lo, False) & make_selection(df, y_var, '<', y_val_hi, False))
            df_r = df.loc[(x_cut & y_cut)]
                  
            r = regions[iRegion] + "_"
            size_dict[label_out][regions[iRegion]] += df_r.shape[0]
            iRegion += 1
            
            # double check blinding
            if iRegion == (len(abcd[x_var])-1)*(len(abcd[y_var])-1) and not options.isMC:
                if df_r.shape[0] > 0: 
                    sys.exit(label_out+": You are not blinding correctly! Exiting.")
        
            # 3a. Plot event wide variables
            plot_labels = [key for key in df_r.keys() if r+key+"_"+label_out in list(output.keys())]   # event wide variables
            for plot in plot_labels: output[r+plot+"_"+label_out].fill(df_r[plot], weight=df_r['event_weight']) 
            # 3b. Plot method variables
            plot_labels = [key for key in df_r.keys() if r+key.replace(label, label_out) in list(output.keys())]  # method vars
            for plot in plot_labels: output[r+plot.replace(label, label_out)].fill(df_r[plot], weight=df_r['event_weight'])  
           
    return output, size_dict
        
#############################################################################################################

# get list of files
username = getpass.getuser()
if options.xrootd:
    dataDir = "/scratch/{}/SUEP/{}/{}/merged/".format(username,options.tag,options.dataset)
    result = subprocess.check_output(["xrdfs",redirector,"ls",dataDir])
    result = result.decode("utf-8")
    files = result.split("\n")
    files = [f for f in files if len(f) > 0]
else:
    dataDir = "/data/submit/{}/{}/{}/merged/".format(username, options.tag, options.dataset)
    files = [dataDir + f for f in os.listdir(dataDir)]

# get cross section
xsection = 1.0
if options.isMC:
    with open('../data/xsections_{}.json'.format(options.era)) as file:
        MC_xsecs = json.load(file)
        try:
            xsection *= MC_xsecs[options.dataset]["xsec"]
            xsection *= MC_xsecs[options.dataset]["kr"]
            xsection *= MC_xsecs[options.dataset]["br"]
        except:
            print("WARNING: I did not find the xsection for that MC sample. Check the dataset name and the relevant yaml file")

# get weights
w = np.load('nconst_weights.npy', allow_pickle=True)
weights = defaultdict(lambda: np.zeros(2))
weights.update(w.item())

# output histos
def create_output_file(label, abcd):
    
    x_var = list(abcd.keys())[0]
    y_var = list(abcd.keys())[1]
    output.update({"ABCDvars_2D_"+label : Hist.new.Reg(100, 0, abcd[x_var][-1], name=x_var).Reg(100, 0, abcd[y_var][-1], name=y_var).Weight()})
 
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_regions = (len(abcd[x_var]) - 1) * (len(abcd[y_var]) - 1)
    regions_list =  [""] + [regions[i]+"_" for i in range(n_regions)]
    
    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in regions_list:
        output.update({
            r+"ht_" + label : Hist.new.Reg(100, 0, 10000, name=r+"ht_"+label, label='HT').Weight(),
            r+"ht_tracker_" + label : Hist.new.Reg(100, 0, 10000, name=r+"ht_tracker_"+label, label='HT Tracker').Weight(),
            r+"ntracks_" + label : Hist.new.Reg(101, 0, 500, name=r+"ntracks_"+label, label='# Tracks in Event').Weight(),
            r+"ngood_fastjets_" + label : Hist.new.Reg(9,0, 10, name=r+"ngood_fastjets_"+label, label='# FastJets in Event').Weight(),
            r+"PV_npvs_"+label : Hist.new.Reg(199,0, 200, name=r+"PV_npvs_"+label, label="# PVs in Event ").Weight(),
            r+"Pileup_nTrueInt_"+label : Hist.new.Reg(199,0, 200, name=r+"Pileup_nTrueInt_"+label, label="# True Interactions in Event ").Weight(),
            r+"ngood_ak4jets_" + label : Hist.new.Reg(19,0, 20, name=r+"ngood_ak4jets_"+label, label= '# ak4jets in Event').Weight(),
            r+"ngood_rtacker_ak4jets_" + label : Hist.new.Reg(19,0, 20, name=r+"ngood_tracker_ak4jets_"+label, label= '# ak4jets in Event ($\abs{\eta} < 2.4$)').Weight(),
        })
        # for i in range(10):
        #     output.update({
        #         r+"eta_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100,-5,5, name=r+"eta_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $\eta$").Weight(),
        #         r+"phi_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"phi_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $\phi$").Weight(),
        #         r+"pt_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $p_T$").Weight(),
        #     })
        # for i in range(2):
        #     output.update({
        #         r+"eta_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100,-5,5, name=r+"eta_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $\eta$").Weight(),
        #         r+"phi_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"phi_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $\phi$").Weight(),
        #         r+"pt_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $p_T$").Weight(),
        #     })
            
    if label == 'IRM' or label == 'CL':
        output.update({
            # 2D histograms
            "2D_SUEP_S1_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(100, 0, 500, name="ntracks_"+label, label='# Tracks').Weight(),
            "2D_SUEP_S1_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="nconst_"+label, label='# Constituents').Weight(),     
            "2D_SUEP_S1_SUEP_pt_avg_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="SUEP_pt_avg_"+label, label='$p_T Avg$').Weight(),
            "2D_SUEP_nconst_SUEP_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name="SUEP_nconst_"+label, label='# Const').Reg(200, 0, 500, name="SUEP_pt_avg_"+label, label='$p_T Avg$').Weight(), 
            "2D_ntracks_SUEP_pt_avg_"+label : Hist.new.Reg(100, 0, 500, name="ntracks_"+label, label='# Tracks').Reg(200, 0, 500, name="SUEP_pt_avg_"+label, label='$p_T Avg$').Weight(),  
            "2D_SUEP_S1_SUEP_pt_avg_b_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(50, 0, 50, name="SUEP_pt_avg_b_"+label, label='$p_T Avg (Boosted frame)$').Weight(),
            "2D_ntracks_SUEP_pt_avg_b_"+label : Hist.new.Reg(100, 0, 500, name="ntracks_"+label, label='# Tracks').Reg(50, 0, 50, name="SUEP_pt_avg_b_"+label, label='$p_T Avg (Boosted frame)$').Weight(),  
            "2D_SUEP_nconst_SUEP_pt_avg_b_"+label : Hist.new.Reg(200, 0, 500, name="SUEP_nconst_"+label, label='# Const').Reg(50, 0, 50, name="SUEP_pt_avg_b_"+label, label='$p_T Avg (Boosted frame)$').Weight(), 
            "2D_SUEP_S1_SUEP_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name="SUEP_S1_"+label, label='$Sph_1$').Reg(100, 0, 1, name="SUEP_pt_mean_scaled_"+label, label='$p_T Avg / p_T Max (Boosted frame)$').Weight(),
            "2D_ntracks_SUEP_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 500, name="ntracks_"+label, label='# Tracks').Reg(100, 0, 1, name="SUEP_pt_mean_scaled_"+label, label='$p_T Avg / p_T Max (Boosted frame)$').Weight(),  
            "2D_SUEP_nconst_SUEP_pt_mean_scaled_"+label : Hist.new.Reg(200, 0, 500, name="SUEP_nconst_"+label, label='# Const').Reg(100, 0, 1, name="SUEP_pt_mean_scaled_"+label, label='$p_T Avg / p_T Max (Boosted frame)$').Weight(),  
        })
        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update({
                r+"SUEP_nconst_"+label : Hist.new.Reg(199, 0, 500, name=r+"SUEP_nconst_"+label, label="# Tracks in SUEP").Weight(),
                r+"SUEP_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r+"SUEP_pt_"+label, label=r"SUEP $p_T$ [GeV]").Weight(),
                r+"SUEP_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name=r+"SUEP_pt_avg_"+label, label=r"SUEP Components $p_T$ Avg.").Weight(),
                r+"SUEP_pt_avg_b_"+label : Hist.new.Reg(50, 0, 50, name=r+"SUEP_pt_avg_b_"+label, label=r"SUEP Components $p_T$ avg (Boosted Frame)").Weight(),
                r+"SUEP_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_pt_mean_scaled_"+label, label=r"SUEP Components $p_T$ Mean / Max (Boosted Frame)").Weight(),
                r+"SUEP_eta_"+label : Hist.new.Reg(100,-5,5, name=r+"SUEP_eta_"+label, label=r"SUEP $\eta$").Weight(),
                r+"SUEP_phi_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"SUEP_phi_"+label, label=r"SUEP $\phi$").Weight(),
                r+"SUEP_mass_"+label : Hist.new.Reg(150, 0, 4000, name=r+"SUEP_mass_"+label, label="SUEP Mass [GeV]").Weight(),
                r+"SUEP_S1_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_S1_"+label, label='$Sph_1$').Weight(),
                r+"SUEP_girth": Hist.new.Reg(50, 0, 1.0, name=r+"SUEP_girth_"+label, label=r"SUEP Girth").Weight(),
                r+"SUEP_rho0_"+label : Hist.new.Reg(50, 0, 20, name=r+"SUEP_rho0_"+label, label=r"SUEP $\rho_0$").Weight(),
                r+"SUEP_rho1_"+label : Hist.new.Reg(50, 0, 20, name=r+"SUEP_rho1_"+label, label=r"SUEP $\rho_1$").Weight(),
            })
    
    if label == 'CLi':
        output.update({
            # 2D histograms
            "2D_ISR_S1_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="ISR_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="ntracks_"+label, label='# Tracks').Weight(),
            "2D_ISR_S1_ISR_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="ISR_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="nconst_"+label, label='# Constituents').Weight(),     
            "2D_ISR_S1_ISR_pt_avg_"+label : Hist.new.Reg(100, 0, 1.0, name="ISR_S1_"+label).Reg(500, 0, 500, name="ISR_pt_avg_"+label).Weight(),
            "2D_ISR_nconst_ISR_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name="ISR_nconst_"+label).Reg(500, 0, 500, name="ISR_pt_avg_"+label).Weight(), 
            "2D_ntracks_ISR_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name="ntracks_"+label).Reg(500, 0, 500, name="ISR_pt_avg_"+label).Weight(),  
            "2D_ISR_S1_ISR_pt_avg_b_"+label : Hist.new.Reg(100, 0, 1.0, name="ISR_S1_"+label).Reg(100, 0, 100, name="ISR_pt_avg_"+label).Weight(),
            "2D_ntracks_ISR_pt_avg_b_"+label : Hist.new.Reg(200, 0, 500, name="ntracks_"+label).Reg(100, 0, 100, name="ISR_pt_avg_"+label).Weight(),  
            "2D_ISR_nconst_ISR_pt_avg_b_"+label : Hist.new.Reg(200, 0, 500, name="ISR_nconst_"+label).Reg(100, 0, 100, name="ISR_pt_avg_"+label).Weight(), 
            "2D_ISR_S1_ISR_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name="ISR_S1_"+label).Reg(100, 0, 1, name="ISR_pt_mean_scaled_"+label).Weight(),
            "2D_ntracks_ISR_pt_mean_scaled_"+label : Hist.new.Reg(200, 0, 500, name="ntracks_"+label).Reg(100, 0, 1, name="ISR_pt_mean_scaled_"+label).Weight(),  
            "2D_ISR_nconst_ISR_pt_mean_scaled_"+label : Hist.new.Reg(200, 0, 500, name="ISR_nconst_"+label).Reg(100, 0, 1, name="ISR_pt_mean_scaled_"+label).Weight(),  
            
        })
        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update({
                r+"ISR_nconst_"+label : Hist.new.Reg(199, 0, 500, name=r+"ISR_nconst_"+label, label="# Tracks in ISR").Weight(),
                r+"ISR_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r+"ISR_pt_"+label, label=r"ISR $p_T$ [GeV]").Weight(),
                r+"ISR_pt_avg_"+label : Hist.new.Reg(500, 0, 500, name=r+"ISR_pt_avg_"+label, label=r"ISR Components $p_T$ Avg.").Weight(),
                r+"ISR_pt_avg_b_"+label : Hist.new.Reg(100, 0, 100, name=r+"ISR_pt_avg_b_"+label, label=r"ISR Components $p_T$ avg (Boosted Frame)").Weight(),
                r+"ISR_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name=r+"ISR_pt_mean_scaled_"+label, label=r"ISR Components $p_T$ Mean / Max (Boosted Frame)").Weight(),
                r+"ISR_eta_"+label : Hist.new.Reg(100,-5,5, name=r+"ISR_eta_"+label, label=r"ISR $\eta$").Weight(),
                r+"ISR_phi_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"ISR_phi_"+label, label=r"ISR $\phi$").Weight(),
                r+"ISR_mass_"+label : Hist.new.Reg(150, 0, 4000, name=r+"ISR_mass_"+label, label="ISR Mass [GeV]").Weight(),
                r+"ISR_S1_"+label : Hist.new.Reg(100, 0, 1, name=r+"ISR_S1_"+label, label='$Sph_1$').Weight(),
                r+"ISR_girth": Hist.new.Reg(50, 0, 1.0, name=r+"ISR_girth_"+label, label=r"ISR Girth").Weight(),
                r+"ISR_rho0_"+label : Hist.new.Reg(100, 0, 20, name=r+"ISR_rho0_"+label, label=r"ISR $\rho_0$").Weight(),
                r+"ISR_rho1_"+label : Hist.new.Reg(100, 0, 20, name=r+"ISR_rho1_"+label, label=r"ISR $\rho_1$").Weight(),
            })
    
    if label == 'ML':
        for r in regions_list:
            output.update({
                r+"resnet_SUEP_pred_"+label : Hist.new.Reg(100, 0, 1, name=r+"resnet_SUEP_pred_"+label, label="Resnet Output").Weight(),
                r+"ntracks_"+label : Hist.new.Reg(100, 0, 500, name=r+"ntracks"+label, label="# Tracks in Event").Weight(),
            })
                        
    return output

# fill ABCD hists with dfs from hdf5 files
nfailed = 0
weight = 0
fpickle =  open("outputs/" + options.dataset+ "_" + output_label + '.pkl', "wb")
size_dict = nested_dict(2, float)
output = {}
for label, abcd in zip(['IRM','ML','CL','CLi'], [abcd_IRM, abcd_ML, abcd_CL, abcd_CLi]):  
    output.update(create_output_file(label, abcd))

### Plotting loop #######################################################################
puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(options.era)   
for ifile in tqdm(files):
    
    #####################################################################################
    # ---- Load file
    #####################################################################################

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
    # ---- Additional weights [pileup_weight]
    #####################################################################################
    event_weight = np.ones(df.shape[0])
    df['event_weight'] = event_weight
    #event_weight *= another event weight, etc

    # pileup weights
    if options.isMC == 1:
        Pileup_nTrueInt = np.array(df['Pileup_nTrueInt']).astype(int)
        pu = puweights[Pileup_nTrueInt]
        df['event_weight'] *= pu
    
    # scaling weights
    if options.isMC == 1 and options.weights:
        
        regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        x_var = list(abcd.keys())[0]
        y_var = list(abcd.keys())[1]
        iRegion = 0
        
        # S1 regions
        for i in range(len(abcd[x_var])-1):
            x_val_lo = abcd[x_var][i]
            x_val_hi = abcd[x_var][i+1]

            # nconst regions
            for j in range(len(abcd[y_var])-1):
                y_val_lo = abcd[y_var][j]
                y_val_hi = abcd[y_var][j+1]
                
                r = regions[iRegion]
                
                # from the weights
                bins = weights[r]['bins']
                ratios = weights[r]['ratios']
                
                # nconst bins
                for k in range(len(bins)-1):
                    z_val_lo = bins[k]
                    z_val_hi = bins[k+1]
                    ratio = ratios[k]
                
                    zslice = (df['SUEP_nconst_CL'] >= z_val_lo) & (df['SUEP_nconst_CL'] < z_val_hi)
                    yslice = (df['SUEP_nconst_CL'] >= y_val_lo) & (df['SUEP_nconst_CL'] < y_val_hi)
                    xslice = (df['SUEP_S1_CL'] >= x_val_lo) & (df['SUEP_S1_CL'] < x_val_hi)
                                        
                    df.loc[xslice & yslice & zslice, 'event_weight'] *= ratio
                
                iRegion += 1
    
    #df['event_weight'] = event_weight

    #####################################################################################
    # ---- Make plots
    #####################################################################################
    
    # ISR Removal Method plots
    vars2d = [
        ['SUEP_S1_IRM', 'ntracks'],
        ['SUEP_pt_avg_b_IRM', 'ntracks'],
        ['SUEP_pt_mean_scaled_IRM', 'ntracks'],
        ['SUEP_S1_IRM', 'SUEP_nconst_IRM'],
        ['SUEP_pt_avg_b_IRM', 'SUEP_nconst_IRM'],
        ['SUEP_pt_mean_scaled_IRM', 'SUEP_nconst_IRM'],
        ['SUEP_S1_IRM', 'SUEP_pt_mean_scaled_IRM'],
    ]
    output, size_dict = plot(df, size_dict, output,
                         selections_IRM, abcd_IRM, 
                         label='IRM', label_out='IRM', 
                         vars2d=vars2d)
    
    # Cluster Method plots
    vars2d = [
        ['SUEP_S1_CL', 'ntracks'],
        ['SUEP_pt_avg_b_CL', 'ntracks'],
        ['SUEP_pt_mean_scaled_CL', 'ntracks'],
        ['SUEP_S1_CL', 'SUEP_nconst_CL'],
        ['SUEP_pt_avg_b_CL', 'SUEP_nconst_CL'],
        ['SUEP_pt_mean_scaled_CL', 'SUEP_nconst_CL'],
        ['SUEP_S1_CL', 'SUEP_pt_mean_scaled_CL'],
    ]
    output, size_dict = plot(df.copy(), size_dict, output,
                             selections_CL, abcd_CL, 
                             label='CL', label_out='CL', 
                             vars2d=vars2d)
    
    # Inverted Cluster Method plots
    if options.isMC:
        vars2d = [
            ['ISR_S1_CL', 'ntracks'],
            ['ISR_pt_avg_b_CL', 'ntracks'],
            ['ISR_pt_mean_scaled_CL', 'ntracks'],
            ['ISR_S1_CL', 'ISR_nconst_CL'],
            ['ISR_pt_avg_b_CL', 'ISR_nconst_CL'],
            ['ISR_pt_mean_scaled_CL', 'ISR_nconst_CL'],
            ['ISR_S1_CL', 'ISR_pt_mean_scaled_CL'],
        ]
        output, size_dict = plot(df.copy(), size_dict, output,
                                 selections_CLi, abcd_CLi, 
                                 label='CL', label_out='CLi', 
                                 vars2d=vars2d)
    
    # ML Method plots
    if False:
        vars2d = [
        ]
        output, size_dict = plot(df.copy(), size_dict, output,
                                 selections_ML, abcd_ML, 
                                 label='ML', label_out='ML', 
                                 vars2d=vars2d)
        
    #####################################################################################
    # ---- End
    #####################################################################################
    
    # remove file at the end of loop   
    if options.xrootd: os.system('rm ' + options.dataset+'.hdf5')    

### End plotting loop ###################################################################
    
# apply normalization
if options.isMC:
    if weight > 0.0:
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