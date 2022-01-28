    import pandas as pd 
import numpy as np
from hist import Hist
import argparse
import os
import awkward as ak
import uproot
import getpass
import pickle
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-dataset", "--dataset"  , type=str, default="QCD", help="dataset name", required=True)
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=False)
parser.add_argument("-e"   , "--era"   , type=int, default=2018  , help="era", required=False)
parser.add_argument('--doSyst', type=int, default=0, help="make systematic plots")
parser.add_argument('--isMC', type=int, default=1, help="Is this MC or data")
parser.add_argument('--blind', type=int, default=1, help="Blind the data (default=True)")
parser.add_argument('--local', type=int, default=0, help="Local data or xrdcp from hadoop (default=False)")
options = parser.parse_args()

# other parameters for script
username = getpass.getuser()
if not options.local:
    dataDir = "/mnt/T3_US_MIT/hadoop/scratch/{}/SUEP/{}/{}/".format(username,options.tag,options.dataset)
else:
    dataDir = "/work/submit/{}/SUEP/{}/{}/".format(username, options.tag, options.dataset)
files = [file for file in os.listdir(dataDir)]

# parameters for ABCD method
var1_label = 'spher'
var2_label = 'nconst'
var1_val = 0.50
var2_val = 25
nbins = 100                # applies to var1_label
labels = ['ch']            # which selection to make plots for
output_label = 'noPtCut'

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
def create_output_file(l):
    output = {
            # variables from the dataframe
            "SUEP_nconst_"+label : Hist.new.Reg(499, 0, 500, name="nconst_"+label, label="# Tracks in SUEP").Weight(),
            "SUEP_ntracks_"+label : Hist.new.Reg(499, 0, 500, name="ntracks_"+label, label="# Tracks in event").Weight(),
            "SUEP_pt_"+label : Hist.new.Reg(100, 0, 2000, name="pt_"+label, label=r"$p_T$").Weight(),
            "SUEP_pt_avg_"+label : Hist.new.Reg(100, 0, 100, name="pt_avg_"+label, label=r"Components $p_T$ Avg.").Weight(),
            "SUEP_pt_avg_b_"+label : Hist.new.Reg(100, 0, 100, name="pt_avg_b_"+label, label=r"Components $p_T$ avg (boosted frame)").Weight(),
            "SUEP_eta_"+label : Hist.new.Reg(100, -5, 5, name="eta_"+label, label=r"$\eta$").Weight(),
            "SUEP_phi_"+label : Hist.new.Reg(100, 0, 6.5, name="phi_"+label, label=r"$\phi$").Weight(),
            "SUEP_mass_"+label : Hist.new.Reg(150, 0, 4000, name="mass_"+label, label="Mass").Weight(),
            "SUEP_spher_"+label : Hist.new.Reg(100, 0, 1, name="spher_"+label, label="Sphericity").Weight(),
            "SUEP_aplan_"+label : Hist.new.Reg(100, 0, 1, name="aplan_"+label, label="Aplanarity").Weight(),
            "SUEP_FW2M_"+label : Hist.new.Reg(100, 0, 1, name="FW2M_"+label, label="2nd Fox Wolfram Moment").Weight(),
            "SUEP_D_"+label : Hist.new.Reg(100, 0, 1, name="D_"+label, label="D").Weight(),
            "SUEP_girth": Hist.new.Reg(50, 0, 1.0, name="girth_"+label, label=r"Girth").Weight(),
            "SUEP_rho0_"+label : Hist.new.Reg(100, 0, 20, name="rho0_"+label, label=r"$\rho_0$").Weight(),
            "SUEP_rho1_"+label : Hist.new.Reg(100, 0, 20, name="rho1_"+label, label=r"$\rho_1$").Weight(),

            # new hists
            "A_"+label: Hist.new.Reg(nbins, 0, 1, name="A_"+label).Weight(),
            "B_"+label: Hist.new.Reg(nbins, 0, 1, name="B_"+label).Weight(),
            "C_"+label: Hist.new.Reg(nbins, 0, 1, name="C_"+label).Weight(),
            "D_exp_"+label: Hist.new.Reg(nbins, 0, 1, name="D_exp_"+label).Weight(),
            "D_obs_"+label: Hist.new.Reg(nbins, 0, 1, name="D_obs_"+label).Weight(),
            "ABCDvars_2D_"+label : Hist.new.Reg(100, 0, 1, name= var1_label +label).Reg(99, 0, 200, name=var2_label).Weight(),
            "2D_girth_nconst_"+label : Hist.new.Reg(50, 0, 1.0, name="girth_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "2D_rho0_nconst_"+label : Hist.new.Reg(100, 0, 20, name="rho0_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "2D_rho1_nconst_"+label : Hist.new.Reg(100, 0, 20, name="rho1_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "2D_spher_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="spher_"+label).Reg(200, 0, 500, name="ntracks_"+label).Weight(),
            "2D_spher_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="spher_"+label).Reg(99, 0, 200, name="nconst_"+label).Weight(),
            "ht_" + label : Hist.new.Reg(1000, 0, 30000, name="ht_"+label, label='HT').Weight(),
            "nJets_" + label : Hist.new.Reg(199, 0, 200, name="nJets_"+label, label='# Jets in Event').Weight(),
            "nLostTracks_"+label : Hist.new.Reg(499, 0, 500, name="nLostTracks_"+label, label="# Lost Tracks in Event ").Weight(),
            "2D_nJets_SUEPpT_"+label : Hist.new.Reg(199, 0, 200, name="nJets_"+label).Reg(100, 0, 3000, name="pt_"+label).Weight(),    
    }
    
    # per region
    for r in ["A", "B", "C"]:
        output.update({
            r+"_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r + "pt_"+label, label=r + r" $p_T$").Weight(),
            r+"_nconst_"+label : Hist.new.Reg(499, 0, 500, name=r + " nconst_"+label, label=r + " # Tracks in SUEP").Weight(),
            "2D_"+r+"_pt_nconst_"+label : Hist.new.Reg(100, 0, 2000, name=r+" pt_"+label).Reg(499, 0, 500, name=r+" nconst_"+label).Weight(),
            r+"_eta_"+label : Hist.new.Reg(100, -5, 5, name=r+" eta_"+label, label=r + r"$\eta$").Weight(),
            r+"_phi_"+label : Hist.new.Reg(200, -6.5, 6.5, name=r + " phi_"+label, label=r + r"$\phi$").Weight(),
            r +"_spher_"+label : Hist.new.Reg(100, 0, 1, name=r+"spher_"+label, label=r+"Sphericity").Weight(),
            r + "_ntracks_"+label : Hist.new.Reg(499, 0, 500, name=r+"ntracks_"+label, label=r+"# Tracks in event").Weight()
        })
    if label == 'ch':# Christos only
        output2 = {
            "SUEP_dphi_chcands_ISR_"+label:Hist.new.Reg(100, 0, 4, name="dphi_chcands_ISR").Weight(),
            "SUEP_dphi_SUEPtracks_ISR_"+label: Hist.new.Reg(100, 0, 4, name="dphi_SUEPtracks_ISR").Weight(),
            "SUEP_dphi_ISRtracks_ISR_"+label:Hist.new.Reg(100, 0, 4, name="dphi_ISRtracks_ISR").Weight(),
            "SUEP_dphi_SUEP_ISR_"+label:Hist.new.Reg(100, 0, 4, name="dphi_SUEP_ISR").Weight(),
        }
        output.update(output2)
    if options.isMC and options.doSyst:# Systematic plots
        output3 = {
            #"SUEP_"+label+"_variable_"+sys:Hist.new.Reg(100, 0, 4, name="variable").Weight(),
        }
        output.update(output3)
    return output

# load hdf5 with pandas
def h5load(ifile, label):
    try:
        with pd.HDFStore(ifile) as store:
            try:
                data = store[label] 
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata
            except ValueError: 
                print("Empty file!", ifile)
                return 0, 0
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
output, sizeA, sizeC = {}, {}, {}
for label in labels: 
    output.update(create_output_file(label))
    sizeA.update({label:0})
    sizeC.update({label:0})
    
for ifile in tqdm(files):
    ifile = dataDir+"/"+ifile

    if not options.local:
        if os.path.exists(options.dataset+'.hdf5'): os.system('rm ' + options.dataset+'.hdf5')
        xrd_file = "root://t3serv017.mit.edu:/" + ifile.split('hadoop')[1]
        os.system("xrdcp {} {}.hdf5".format(xrd_file, options.dataset))
        df_vars, metadata = h5load(options.dataset+'.hdf5', 'vars')   
    else:
        df_vars, metadata = h5load(ifile, 'vars')   

    # check if file is corrupted, or empty
    if type(df_vars) == int: 
        nfailed += 1
        continue
    if df_vars.shape[0] == 0: continue
    
    # update the gensumweight
    if options.isMC: weight += metadata['gensumweight']
    
    # store event-wide info to be indexed within each selection
    hts = df_vars['ht']
    nJets = df_vars['ngood_fastjets']
    nLostTracks = df_vars['nLostTracks']
    
    for label in labels:
        if not options.local: df, metadata = h5load(options.dataset+'.hdf5', label) 
        else: df, metadata = h5load(ifile, label)
        if df.shape[0] == 0: continue
        
        # parameters for ABCD plots
        var1 = 'SUEP_' + var1_label + '_' + label
        var2 = 'SUEP_' + var2_label + '_' + label
        
        # selections
        if var2_label == 'nconst': df = df.loc[df['SUEP_nconst_'+label] >= 10]
        if var1_label == 'spher': df = df.loc[df['SUEP_spher_'+label] >= 0.25]
        #df = df.loc[df['SUEP_'+label+'_pt'] >= 300]
        
        # blind
        if options.blind and not options.isMC:
             df = df.loc[((df[var1] < var1_val) & (df[var2] < var2_val)) | ((df[var1] >= var1_val) & (df[var2] < var2_val)) | ((df[var1] < var1_val) & (df[var2] >= var2_val))]

        # divide the dfs by region
        df_A = df.loc[(df[var1] < var1_val) & (df[var2] < var2_val)]
        df_B = df.loc[(df[var1] >= var1_val) & (df[var2] < var2_val)]
        df_C = df.loc[(df[var1] < var1_val) & (df[var2] >= var2_val)]
        df_D_obs = df.loc[(df[var1] >= var1_val) & (df[var2] >= var2_val)]
        
        sizeC[label] += df_C.shape[0]
        sizeA[label] += df_A.shape[0]
        
        # fill the ABCD histograms
        output["A_"+label].fill(df_A[var1])
        output["B_"+label].fill(df_B[var1])
        output["D_exp_"+label].fill(df_B[var1])
        output["C_"+label].fill(df_C[var1])
        output["D_obs_"+label].fill(df_D_obs[var1])
        output["ABCDvars_2D_"+label].fill(df[var1], df[var2])
        
        # fill the distributions as they are saved in the dataframes
        plot_labels = [key for key in df.keys() if key in list(output.keys())]
        for plot in plot_labels: output[plot].fill(df[plot])  

        # fill some new distributions  
        output["2D_girth_nconst_"+label].fill(df["SUEP_girth_"+label], df["SUEP_nconst_"+label])
        output["2D_rho0_nconst_"+label].fill(df["SUEP_rho0_"+label], df["SUEP_nconst_"+label])
        output["2D_rho1_nconst_"+label].fill(df["SUEP_rho1_"+label], df["SUEP_nconst_"+label])
        output["2D_spher_nconst_"+label].fill(df["SUEP_spher_"+label], df["SUEP_nconst_"+label])
        output["2D_spher_ntracks_"+label].fill(df["SUEP_spher_"+label], df["SUEP_ntracks_"+label])
        output["ht_" + label].fill(hts[df['event_index_'+label]])
        output["nJets_" + label].fill(nJets[df['event_index_'+label]])
        output["nLostTracks_" + label].fill(nLostTracks[df['event_index_'+label]])
        output["2D_nJets_SUEPpT_" + label].fill(nJets[df['event_index_'+label]], df['SUEP_pt_'+label])
        
        # per region
        for r, df_r in zip(["A", "B", "C"], [df_A, df_B, df_C]):
        
            output[r + "_pt_"+label].fill(df_r['SUEP_pt_'+label])
            output[r + "_eta_"+label].fill(df_r['SUEP_eta_'+label])
            output[r + "_phi_"+label].fill(df_r['SUEP_phi_'+label])
            output[r + "_spher_"+label].fill(df_r['SUEP_spher_'+label])
            output[r + "_nconst_"+label].fill(df_r['SUEP_nconst_'+label])
            output[r + "_ntracks_"+label].fill(df_r['SUEP_ntracks_'+label])
            output["2D_" + r + "_pt_nconst_"+label].fill(df_r['SUEP_pt_'+label], df_r['SUEP_nconst_'+label])
    
    if not options.local: os.system('rm ' + options.dataset+'.hdf5')    
        
# ABCD method to obtain D expected for each selection
for label in labels:
    if sizeA[label]>0.0:
        CoverA =  sizeC[label] / sizeA[label]
    else:
        CoverA = 0.0
        print("A region has no occupancy for selection", label)
    output["D_exp_"+label] = output["D_exp_"+label]*(CoverA)
    
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