import pandas as pd
import os
import ROOT
import numpy as np

#output = "/eos/user/j/jkil/www"

SFile  = "../outputZH_archive/NumTrk[0.5,1.0]out.hdf5" # signal file
BFile  = "../outputDY_archive/NumTrk[0.5,1.0]out.hdf5" # background file
channel= "numtrkvars"
var    = "Ntracks"
where  = "G"

lumi  = 137.0 # Luminosity of Run 2
xsecS = 870*0.0336*3  # ZH, times BR to leptons
xsecB = 7181000*0.0336*3 # DY, times BR to leptons

S = pd.HDFStore(SFile, 'r')
B = pd.HDFStore(BFile, 'r')

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

    significance = nSpass/(nSpass+nBpass)**0.5

    return significance

fromPV_vals = ["0.0","1.0","2.0"]
pt_vals = ["0.5","1.0","1.5","2.0","2.5","3.0"]
cut_vals = np.arange(60)

for i in range(0,10):
    print(getSignificance(i))

SigvNCut_plot = ROOT.TH2F("Sig", "Significance vs Cut Value Plot", 100, 0, 10, 100, 0, 5)

#for i in fromPV_vals:
#    for j in pt_vals:
#        getSignificance(i,j):
#        print("NumTrk["+i+","+j+"]out.hdf5")


