import pandas as pd
import os

######### Configuration parameters ##########
SFile  = "../outputZH/out.hdf5" # signal file
BFile  = "../outputDY/out.hdf5" # background file
channel= "vars" # Channel name inside the analyzer
var    = "nTracks" #Var name inside the analyzer
cut    = 60. # Cut value
where  = "G" # L == less than, G== great than. For which region to compute significance
what   = "StoSqrtSB" # What to report: StoB, StoSqrtSB, StoSqrtB

## Conf but shouldn't change
lumi  = 137.0 # Luminosity of Run 2
xsecS = 870*0.0336*3  # ZH, times BR to leptons
xsecB = 7181000*0.0336*3 # DY, times BR to leptons

######## Code itself

S = pd.HDFStore(SFile, 'r') # Input file for the signal
B = pd.HDFStore(BFile, 'r') # Input file for the background

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

if what == "StoSqrtSB":
  print("S/sqrt(S+B) = %1.3f for a cut of %1.3f in %s"%(nSpass/(nSpass+nBpass)**0.5, cut, var))
elif what == "StoB":
  print("S/B = %1.3f for a cut of %1.3f in %s"%(nSpass/nBpass, cut, var))
elif what == "StoSqrtB":
  print("S/sqrt(B) = %1.3f for a cut of %1.3f in %s"%(nSpass/nBpass**0.5, cut, var))



