import os
import ROOT

def hdf5inpath(path):
  ret = []
  for f in os.listdir(path):
    if "hdf5" in f: 
      ret.append(path + "/" + f)
  return ret

order   = ["ttZ","VV", "WJets","ttto1l", "ttto2l", "DY"]
samples = {
  "DY": {
         "name"     : "DY", #Here plain text
         "label"    : "DY", #Here we can use weird glyphs
         "xsec"     : 2008.*3*1000., # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 7, # Light blue
         "isSig"    : False, 
         "files"    : hdf5inpath("/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_DY/") 
  },

  "ttto2l": {
         "name"     : "TT", #Here plain text
         "label"    : "t#bar{t} (2l)", #Here we can use weird glyphs
         "xsec"     : 831.76*((3*0.108)**2)*1000., # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 2, # Red
         "isSig"    : False,
         "files"    :  hdf5inpath("/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_TT/") 
  },
  "ttto1l": {
         "name"     : "TT_semilep", #Here plain text
         "label"    : "t#bar{t} (1l)", #Here we can use weird glyphs
         "xsec"     : 831.76*(3*0.108)*(1-3*0.108)*1000., # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 5, # Yellow
         "isSig"    : False,
         "files"    : hdf5inpath("/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_TT_semilep/")
  },

  "Wjets": {
         "name"     : "W", #Here plain text
         "label"    : "W", #Here we can use weird glyphs
         "xsec"     : 20508.9*1000, # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 6, # Purple
         "isSig"    : False,
         "files"    : hdf5inpath("/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_W/") 
  },

  "VV": {
         "name"     : "VV", #Here plain text
         "label"    : "VV", #Here we can use weird glyphs
         "xsec"     : 10.481*1000, # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 3, # Green
         "isSig"    : False,
         "files"    : hdf5inpath("/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_WW/"),
  },
  "ttZ": {
         "name"     : "ttZ", #Here plain text
         "label"    : "t#bar{t}X", #Here we can use weird glyphs
         "xsec"     : 0.78*1000*0.4, # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 9, # Dark blue
         "isSig"    : False,
         "files"    : hdf5inpath("/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_TTZ_LL/")
  },
  "SUEP_ZH_generic": {
         "name"     : "suep_zh_125_generic", #Here plain text
         "label"    : "ZS^{gen}, m_{S} = 125 GeV", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": ROOT.kBlack,
         "isSig"    : True,
         "files"    : ["/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_ZHgeneric//out_1_1_1001.hdf5"],
  },

  "SUEP_ZH_leptonic": {
         "name"     : "suep_zh_125_leptonic", #Here plain text
         "label"    : "ZS^{lep}, m_{S} = 125 GeV", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kGreen,
         "fillcolor": ROOT.kGreen,
         "isSig"    : True,
         "files"    : ["/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_ZHleptonic//out_1_1_1101.hdf5"],
  },

  "SUEP_ZH_hadronic": {
         "name"     : "suep_zh_125_hadronic", #Here plain text
         "label"    : "ZS^{had}, m_{S} = 125 GeV", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kBlue,
         "fillcolor": ROOT.kBlue,
         "isSig"    : True,
         "files"    : ["/afs/cern.ch/work/c/cericeci/private/test/SUEPCoffea_dask/output_batch_ZHhadronic//out_1_1_1001.hdf5"],
  },

}
