import ROOT
import copy

def cut(x, name):
  print("--",name)
  return 1 #(x["njets"] >= 0)

def makeFunc(x, y, name):
  return (x, y*cut(x, name))

plotspre = {
  "ntracks": {
             "name"     : "ntracks",
             "bins"     : ["uniform", 100, 0, 500],
             "channel"  : "twoleptons",
             "value"    : lambda x, y : (x["ntracks"], y*cut(x)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 1.,
             "ratiominY": 0.,
             "plotname" : "ntracks",
             "xlabel"   : "N_{tracks}",
             "vars"     : ["ntracks"]
  },
}

plots = {}
ptScan     = [0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.]
fromPVScan = [0]#, 1, 2] # fromPV == 0 not useful
etaScan    = [5.0]#, 2.5, 1.4, 1.0]
for ptcut in ptScan:
  for pvcut in fromPVScan:
    for etacut in etaScan:
      copiedname                     =  copy.deepcopy("ntracks_pv%i_pt%1.2f_eta%1.1f"%(pvcut, ptcut, etacut))
      plots[copiedname]              =  copy.deepcopy(plotspre["ntracks"])
      plots[copiedname]["name"]      =  copiedname
      plots[copiedname]["plotname"]  =  copiedname
      plots[copiedname]["vars"]      =  [copiedname]
      plots[copiedname]["xlabel"]    =  "N_{tracks} p_{T} > %1.1f, |#eta| < %1.1f, PV > %i"%(ptcut, etacut, pvcut)
      plots[copiedname]["value"]     =  "lambda x, y : (x[\"%s\"], y)"%copiedname


#print(plots)

plots = {'ntracks_pv0_pt0.50_eta5.0': {'name': 'ntracks_pv0_pt0.50_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt0.50_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt0.50_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 0.5, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt0.50_eta5.0']}, 'ntracks_pv0_pt0.75_eta5.0': {'name': 'ntracks_pv0_pt0.75_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt0.75_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt0.75_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 0.8, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt0.75_eta5.0']}, 'ntracks_pv0_pt1.00_eta5.0': {'name': 'ntracks_pv0_pt1.00_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt1.00_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt1.00_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 1.0, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt1.00_eta5.0']}, 'ntracks_pv0_pt1.25_eta5.0': {'name': 'ntracks_pv0_pt1.25_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt1.25_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt1.25_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 1.2, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt1.25_eta5.0']}, 'ntracks_pv0_pt1.50_eta5.0': {'name': 'ntracks_pv0_pt1.50_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt1.50_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt1.50_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 1.5, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt1.50_eta5.0']}, 'ntracks_pv0_pt1.75_eta5.0': {'name': 'ntracks_pv0_pt1.75_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt1.75_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt1.75_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 1.8, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt1.75_eta5.0']}, 'ntracks_pv0_pt2.00_eta5.0': {'name': 'ntracks_pv0_pt2.00_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt2.00_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt2.00_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 2.0, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt2.00_eta5.0']}, 'ntracks_pv0_pt2.25_eta5.0': {'name': 'ntracks_pv0_pt2.25_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt2.25_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt2.25_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 2.2, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt2.25_eta5.0']}, 'ntracks_pv0_pt2.50_eta5.0': {'name': 'ntracks_pv0_pt2.50_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt2.50_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt2.50_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 2.5, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt2.50_eta5.0']}, 'ntracks_pv0_pt2.75_eta5.0': {'name': 'ntracks_pv0_pt2.75_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt2.75_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt2.75_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 2.8, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt2.75_eta5.0']}, 'ntracks_pv0_pt3.00_eta5.0': {'name': 'ntracks_pv0_pt3.00_eta5.0', 'bins': ['uniform', 100, 0, 500], 'channel': 'twoleptons', 'value': lambda x, y : (x["ntracks_pv0_pt3.00_eta5.0"], y), 'logY': True, 'normalize': False, 'maxY': 1000000000.0, 'minY': 1.0, 'ratiomaxY': 1.0, 'ratiominY': 0.0, 'plotname': 'ntracks_pv0_pt3.00_eta5.0', 'xlabel': 'N_{tracks} p_{T} > 3.0, |#eta| < 5.0, PV > 0', 'vars': ['ntracks_pv0_pt3.00_eta5.0']}}
