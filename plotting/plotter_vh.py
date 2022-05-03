import ROOT
import imp
import array 
import os
import pandas as pd
import CMS_lumi
import multiprocessing

ROOT.gStyle.SetOptStat(0)

class sample(object):
  def __init__(self, sampledict):
    self.config  = sampledict
    self.name    = sampledict["name"]
    self.hdfiles = []
    for f in sampledict["files"]:
      try:
        self.hdfiles.append(pd.HDFStore(f, "r"))
      except:
        print("File %s broken, will skip loading it"%f)

    self.histos  = {}
    self.plots   = {}
    self.norms   = {}
    self.channels= []
    self.plotsinchannel = {}
    self.varstoload = {}
    self.yields     = {}
  def addPlot(self, plotdict):
    self.plots[plotdict["name"]] = plotdict
    if not(plotdict["channel"] in self.channels): 
      self.channels.append(plotdict["channel"])
      self.plotsinchannel[plotdict["channel"]] = [plotdict["name"]]
      self.varstoload[plotdict["channel"]]     = plotdict["vars"]
    else:
      self.plotsinchannel[plotdict["channel"]].append(plotdict["name"])
      for v in plotdict["vars"]:
        if not(v in self.varstoload[plotdict["channel"]]): 
          self.varstoload[plotdict["channel"]].append(v)

  def getRawHistogramsAndNorms(self, options):
    for plotName in self.plots:
      p = self.plots[plotName]
      if p["bins"][0] == "uniform": #Then it is nbins, minX, maxX
        self.histos[plotName] = ROOT.TH1F(plotName + "_" + self.name, plotName + "_" + self.name, p["bins"][1], p["bins"][2], p["bins"][3])
      elif p["bins"][0] == "limits":
        self.histos[plotName] = ROOT.TH1F(plotName + "_" + self.name, plotName + "_" + self.name, array.array('f', p["bins"][1])) 
      self.norms[plotName] = 0

   
    for iff, f in enumerate(self.hdfiles):
      print("Loading file %i/%i"%(iff, len(self.hdfiles)))
      for c in self.channels:
        weights = f[c]["genweight"]
        for plotName in self.plotsinchannel[c]:
          print("...%s"%plotName)
          p = self.plots[plotName]
          self.norms[plotName] += f.get_storer(p["channel"]).attrs.metadata["gensumweight"]
          values, weightsHere = p["value"](f[c], weights)
          print(p["value"])
          for idx in range(len(values)):
            if (idx+1)%100000 == 0: print("%i/%i"%(idx, len(values)))
            self.histos[p["name"]].Fill(values[idx], weightsHere[idx])
    # After all is said and done, normalize to xsec
    for c in self.channels:
      for plotName in self.plotsinchannel[c]:
        p = self.plots[plotName]
        self.histos[plotName].Sumw2() # To get proper stat. unc.
        self.histos[plotName].Scale(options.luminosity*self.config["xsec"]/self.norms[plotName])
        self.yields[plotName] = self.histos[plotName].Integral()

  def setStyleOptions(self):
    for key in self.histos:
      if "linecolor" in self.config: 
        self.histos[key].SetLineColor(self.config["linecolor"])
      if "fillcolor" in self.config:
        self.histos[key].SetFillColor(self.config["fillcolor"])
      if "style" in self.config:
        self.histos[key].SetLineStyle(self.config["style"])

  def isBackground(self):
    bkg = True
    if "isSig" in self.config: bkg = not(self.config["isSig"])
    return bkg

class plotter(object):
  def __init__(self, plotdicts, sampledicts):
    self.plots      = plotdicts #[plotdicts[d] for d in plotdicts]
    self.samples    = [sample(sampledicts[d]) for d in sampledicts]
    print("...Initializing")
    for p in self.plots:
      for s in self.samples:
        s.addPlot(self.plots[p])

  def doPlots(self, options):
    self.getRawHistogramsAndNorms(options)
    self.doStackPlots(options)

  def getRawHistogramsAndNorms(self, options):
    for s in self.samples:
      print("...Processing histograms for %s"%s.name)
      s.getRawHistogramsAndNorms(options)
      s.setStyleOptions()

  def doStackPlots(self, options):
    for plotName in self.plots:
      print("...Plotting %s"%plotName)
      mode = "stack"
      if "mode" in self.plots[plotName]: mode = self.plots[plotName]["mode"]

      if mode == "stack": self.doStackPlot(plotName, options)
      ## More to be implemented

  def doStackPlot(self, pname, options):
    p = self.plots[pname]
    c = ROOT.TCanvas(pname,pname, 800,1050)
    # Set pads
    p1 = ROOT.TPad("mainpad", "mainpad", 0, 0.30, 1, 1)
    p1.SetBottomMargin(0.025)
    p1.SetTopMargin(0.14)
    p1.SetLeftMargin(0.12)
    if "margins" in p:
      p1.SetBottomMargin(p["margins"][0])
      p1.SetTopMargin(p["margins"][1])
      p1.SetLeftMargin(p["margins"][2])
      p1.SetRightMargin(p["margins"][3])
    p1.Draw()
    p1.SetLogy(True)
    if "logY" in p:
      p1.SetLogy(p["logY"])

    p2 = ROOT.TPad("ratiopad", "ratiopad", 0, 0, 1, 0.30)
    p2.SetTopMargin(0.01)
    p2.SetBottomMargin(0.45)
    p2.SetLeftMargin(0.12)
    p2.SetFillStyle(0)
    if "margins" in p:
      p1.SetLeftMargin(p["margins"][2])
      p1.SetRightMargin(p["margins"][3])
    p2.Draw()
    p1.cd()
    tl = ROOT.TLegend(0.5,0.55,0.9,0.85)
    if "legendPosition" in p:
      tl = ROOT.TLegend(p["legendPosition"][0], p["legendPosition"][1], p["legendPosition"][2], p["legendPosition"][3])  

    # Now get the histograms and build the stack
    theStack = ROOT.THStack(pname+"_stack", pname)
    theIndivs= []
    # Background go into the stack
    stacksize = 0
    back = False
    if options.ordered:
      self.samples.sort(key= lambda x: x.yields[pname], reverse=False)

    for s in self.samples:
      if s.isBackground():
        theStack.Add(s.histos[pname])
        tl.AddEntry(s.histos[pname], s.config["label"], "f")
        if not(back): back = s.histos[pname].Clone("total_background")
        else: back.Add(s.histos[pname])
        #print(pname, s.name)
        #s.histos[pname].Print("all")
        stacksize += s.histos[pname].Integral()
      else:
        s.histos[pname].SetFillStyle(0)
        s.histos[pname].SetLineWidth(3)
        s.histos[pname].SetLineStyle(1)
        theIndivs.append(s.histos[pname])
        tl.AddEntry(s.histos[pname], s.config["label"], "l")

    if p["normalize"]:
      for index in range(len(theIndivs)):
        theIndivs[index].Scale(1./theIndivs[index].Integral())
      theStack = ROOT.THStack(pname+"_stack_norm", pname+ "_norm")
      for s in self.samples:
        if s.isBackground():
          s.histos[pname].Scale(1./stacksize)
          theStack.Add(s.histos[pname])
        
    # Now plotting stuff
    theStack.SetTitle("") 
    theStack.Draw("hist")
    theStack.GetXaxis().SetLabelSize(0)
    theStack.GetYaxis().SetLabelSize(0.04)
    theStack.GetYaxis().SetTitleSize(0.08)
    theStack.GetYaxis().SetTitleOffset(0.72)

    theStack.GetYaxis().SetTitle("Normalized events" if p["normalize"] else "Events")
    theStack.GetXaxis().SetTitle("") # Empty, as it goes into the ratio plot
    if "maxY" in p: 
      theStack.SetMaximum(p["maxY"])
    if "minY" in p:
      theStack.SetMinimum(p["minY"])

    theStack.Draw("hist")
    for ind in theIndivs:
      ind.Draw("hist same")
    tl.Draw("same")

    # Now we go to the ratio
    p2.cd()

    # By default S/B, TODO: add more options
    den  = back.Clone("back_ratio")
    nums = [ind.Clone(ind.GetName()+ "_ratio") for ind in  theIndivs]
    #den.Divide(den)
    for num in nums: 
      num.Divide(den)
    den.Divide(den)
    den.SetLineColor(ROOT.kBlack)
    den.SetTitle("")
    den.GetYaxis().SetTitle("S/B")
    den.GetXaxis().SetTitle(p["xlabel"])
    den.GetYaxis().SetTitleSize(0.12)
    den.GetYaxis().SetTitleOffset(0.32)
    den.GetXaxis().SetTitleSize(0.12)
    den.GetXaxis().SetLabelSize(0.1)
    den.GetYaxis().SetLabelSize(0.06)

    if "ratiomaxY" in p:
      den.SetMaximum(p["ratiomaxY"])
    if "ratiominY" in p:
      den.SetMinimum(p["ratiominY"])
    den.Draw("")
    for num in nums:
      num.Draw("same")
    CMS_lumi.writeExtraText = True
    CMS_lumi.lumi_13TeV = "%.0f fb^{-1}" % options.luminosity
    CMS_lumi.extraText  = "Preliminary"
    CMS_lumi.lumi_sqrtS = "13"
    CMS_lumi.CMS_lumi(c, 4, 0, 0.122)


    c.SaveAs(options.plotdir + "/" + p["plotname"] + ".pdf")
    c.SaveAs(options.plotdir + "/" + p["plotname"] + ".png")
    # Also save as TH1 in root file 
    tf = ROOT.TFile(options.plotdir + "/" + p["plotname"] + ".root", "RECREATE")
    for s in self.samples:
      if s.isBackground():
        s.histos[pname].Write()
      else:
        s.histos[pname].Write()
    theStack.Write()
    tf.Close()


if __name__ == "__main__":
  print("Starting plotting script...")
  from optparse import OptionParser
  parser = OptionParser(usage="%prog [options] samples.py plots.py") 
  parser.add_option("-l","--luminosity", dest="luminosity", type="float", default=137, help="Luminosity")
  parser.add_option("-j","--jobs", dest="jobs", type="int", default=-1, help="Number of jobs (cores to use)")

  parser.add_option("-p","--plotdir", dest="plotdir", type="string", default="./", help="Where to put the plots")
  parser.add_option("--strict-order", dest="ordered", action="store_true", default=False, help="If true, will stack samples in the order of yields")
  (options, args) = parser.parse_args()
  samplesFile = imp.load_source("samples",args[0])
  plotsFile   = imp.load_source("plots",  args[1])
  if not(os.path.isdir(options.plotdir)):
    os.system("mkdir %s"%options.plotdir)
  os.system("cp %s %s %s"%(args[0], args[1], options.plotdir))
  samples = samplesFile.samples
  plots   = plotsFile.plots
  thePlotter = plotter(plots, samples)
  thePlotter.doPlots(options)

