import pandas as pd
import os
import ROOT
import numpy as np

##### VARIABLES #####

channel= "numtrkvars"
var    = "Ntracks"
where  = "G"

lumi  = 137.0 # Luminosity of Run 2
xsecS = 870*0.0336*3  # ZH, times BR to leptons
xsecB = 7181000*0.0336*3 # DY, times BR to lepton

addSys = 0.1 # Systematic uncertainty

fromPV_vals = ["0.0","1.0","2.0"]
pt_vals = ["0.5","1.0","1.5","2.0","2.5","3.0"]

#fromPV_vals = ["0.0"]
#pt_vals = ["0.5"]

# Make this True, and it will generate plots. If False, it will only print out the max significance.
doPlots = True

##### FUNCTION DEFINITION #####

def getSignificance(cut):
    ### This function calculates the significance for a given cut.

    # Normalization factors
    normS = lumi*xsecS/(S.get_storer(channel).attrs.metadata["gensumweight"])
    normB = lumi*xsecB/(B.get_storer(channel).attrs.metadata["gensumweight"])

    # Vectors of weights
    weightS = S[channel]["genweight"]
    weightB = B[channel]["genweight"]

    # Loop over events and fill counters
    nSpass = 0
    nBpass = 0
    nEvent = 0

    ## Signal loop
    for idx, val in enumerate(S[channel][var]):
        if where == "L" and val <= cut:
            nSpass += weightS[idx]*normS
        if where == "G" and val >= cut:
            nSpass += weightS[idx]*normS

    ## Background loop
    for idx, val in enumerate(B[channel][var]):
        if where == "L" and val <= cut:
            nBpass += weightB[idx]*normB
        if where == "G" and val >= cut:
            nBpass += weightB[idx]*normB
            nEvent += 1

    significance = nSpass/(nSpass+nBpass+(addSys*nBpass)**2)**0.5

    return significance, nEvent


##### DICTIONARIES #####

plots = {}
canvases = {}


##### LOOP #####

for i in fromPV_vals:
    for j in pt_vals:
        SFile = "../outputZH_archive/NumTrk["+j+","+i+"]out.hdf5"
        BFile = "../outputDY_archive/NumTrk["+j+","+i+"]out.hdf5"
        
        S = pd.HDFStore(SFile, 'r')
        B = pd.HDFStore(BFile, 'r')
        
        maxNtrk = max(S[channel]["Ntracks"])       

        if doPlots:              
            plots["SigvNCut_plot[{pt},{pv}]".format(pt=j,pv=i)]=ROOT.TH2F("Sig", "Significance Plot"+"(pt="+j+",fromPV="+i+")", 100, 0, maxNtrk, 100, 0, 6)
            canvases["canvas[{pt},{pv}]".format(pt=j,pv=i)]=ROOT.TCanvas("c"+i+j,"c"+i+j,800,600)
        
        max_significance = 0
        max_Ntracks = 0
        
        for idx in range(0,maxNtrk):
            significance, nEvent =  getSignificance(idx)[:2]
            #print(significance, nEvent, "significance, nEven")
            if significance > max_significance:
                max_significance = significance
                max_Ntracks = idx
            if doPlots:
                if nEvent > 20: #Here I am sorting out the bins with small number of events
                    plots["SigvNCut_plot["+j+","+i+"]"].Fill(idx,significance)
       
        if doPlots:
            print("Histogram filling done. Drawing in progress...")
 
            plots["SigvNCut_plot["+j+","+i+"]"].Draw()
            canvases["canvas["+j+","+i+"]"].Draw()
            plots["SigvNCut_plot["+j+","+i+"]"].GetXaxis().SetTitle("Value of Cut")
            plots["SigvNCut_plot["+j+","+i+"]"].GetYaxis().SetTitle("Significance Z = #frac{S}{#sqrt{S+B}}")
            plots["SigvNCut_plot["+j+","+i+"]"].SetMarkerStyle(20)      
 
            canvases["canvas["+j+","+i+"]"].SaveAs("/eos/user/j/jkil/www/SigvNCut_plot["+j+","+i+"].pdf")
            canvases["canvas["+j+","+i+"]"].SaveAs("/eos/user/j/jkil/www/SigvNCut_plot["+j+","+i+"].png")
 
        print("Done with fromPV = "+i+", pt = "+j+" (Max significance=",max_significance," at ", "Value of cut=",max_Ntracks,")")
