"""
SUEP_coffea_ZH.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer, 2021
"""

import os
import pathlib
import shutil
import awkward as ak
import pandas as pd
import numpy as np
import fastjet
from coffea import hist, processor
import vector
from typing import List, Optional
vector.register_awkward()

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC: int, era: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, flag: bool, output_location: Optional[str]) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}

        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def sphericity(self, particles, r):
        norm = ak.sum(particles.p ** r, axis=1, keepdims=True)
        s = np.array([[
                       ak.sum(particles.px * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(particles.px * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(particles.px * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(particles.py * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(particles.py * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(particles.py * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(particles.pz * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(particles.pz * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(particles.pz * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ]])
        s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
        evals = np.sort(np.linalg.eigvalsh(s))
        return evals

    def rho(self, number, jet, tracks, deltaR, dr=0.05):
        r_start = number*dr
        r_end = (number+1)*dr
        ring = (deltaR > r_start) & (deltaR < r_end)
        rho_values = ak.sum(tracks[ring].pt, axis=1)/(dr*jet.pt)
        return rho_values

    def ak_to_pandas(self, jet_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(jet_collection):
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(jet_collection[field]):
                    output[f"{prefix}_{subfield}"] = ak.to_numpy(
                        jet_collection[field][subfield]
                    )
            else:
                output[field] = ak.to_numpy(jet_collection[field])
        return output

    def h5store(self, store: pd.HDFStore, df: pd.DataFrame, fname: str, gname: str, **kwargs: float) -> None:
        store.put(gname, df)
        store.get_storer(gname).attrs.metadata = kwargs
        
    def save_dfs(self, dfs, df_names):
        fname = "out.hdf5"
        subdirs = []
        store = pd.HDFStore(fname)
        if self.output_location is not None:
            # pandas to hdf5
            for out, gname in zip(dfs, df_names):
                if self.isMC:
                    metadata = dict(gensumweight=self.gensumweight,era=self.era, mc=self.isMC,sample=self.sample)
                    #metadata.update({"gensumweight":self.gensumweight})
                else:
                    metadata = dict(era=self.era, mc=self.isMC,sample=self.sample)    
                    
                store_fin = self.h5store(store, out, fname, gname, **metadata)

            store.close()
            self.dump_table(fname, self.output_location, subdirs)
        else:
            print("self.output_location is None")
            store.close()

    def dump_table(self, fname: str, location: str, subdirs: Optional[List[str]] = None) -> None:
        subdirs = subdirs or []
        xrd_prefix = "root://"
        pfx_len = len(xrd_prefix)
        xrootd = False
        if xrd_prefix in location:
            try:
                import XRootD
                import XRootD.client

                xrootd = True
            except ImportError:
                raise ImportError(
                    "Install XRootD python bindings with: conda install -c conda-forge xroot"
                )
        local_file = (
            os.path.abspath(os.path.join(".", fname))
            if xrootd
            else os.path.join(".", fname)
        )
        merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
        destination = (
            location + merged_subdirs + f"/{fname}"
            if xrootd
            else os.path.join(location, os.path.join(merged_subdirs, fname))
        )
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination)
            copyproc.prepare()
            copyproc.run()
            client = XRootD.client.FileSystem(
                location[: location[pfx_len:].find("/") + pfx_len]
            )
            status = client.locate(
                destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
                XRootD.client.flags.OpenFlags.READ,
            )
            assert status[0].ok
            del client
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            if not os.path.exists(dirname):
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            if not os.path.samefile(local_file, destination):
                shutil.copy2(local_file, destination)
            else:
                fname = "condor_" + fname
                destination = os.path.join(location, os.path.join(merged_subdirs, fname))
                shutil.copy2(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


    def selectByTrigger(self, events, extraColls = []):
        ### Apply trigger selection
        ### TODO:: Save a per-event flag that classifies the event (ee or mumu)
        if self.era == 2018:
           cutAnyHLT = (events.HLT.IsoMu24) | (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8) | (events.HLT.Ele32_WPTight_Gsf) | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL) | (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL)
           return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]
        if self.era == 2017:
           cutAnyHLT = (events.HLT.IsoMu27) | (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8) | (events.HLT.Ele35_WPTight_Gsf) | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL) | (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)
           return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]
        if self.era == 2016:
           cutAnyHLT = (events.HLT.IsoMu24) | (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ) | (events.HLT.Ele27_WPTight_Gsf) | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ)
           return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]
        return events, [coll[cutAnyHLT] for coll in extraColls]

    def selectByLeptons(self, events, extraColls = []):
        ### Very basic lepton selection criteria, prepare lepton 4-momenta collection for plotting and so
        muons = ak.zip({
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
            "charge": events.Muon.pdgId/(-13),
        }, with_name="Momentum4D")
        electrons = ak.zip({
            "pt": events.Electron.pt,
            "eta": events.Electron.eta,
            "phi": events.Electron.phi,
            "mass": events.Electron.mass,
            "charge": events.Electron.pdgId/(-11),
        }, with_name="Momentum4D")
        ###  Some very simple selections on ID ###
        ###  Muons: loose ID + dxy dz cuts mimicking the medium prompt ID https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
        ###  Electrons: loose ID + dxy dz cuts for promptness https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCutBasedIdentification
        cutMuons     = (events.Muon.looseId) & (events.Muon.pt >= 10) & (abs(events.Muon.dxy) <= 0.02) & (abs(events.Muon.dz) <= 0.1) # dxy is impact parameter in transverse plane
        cutElectrons = (events.Electron.cutBased >= 2) & (events.Electron.pt >= 15)
        ### Apply the cuts
        # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
        selMuons     = muons[cutMuons]
        selElectrons = electrons[cutElectrons]
        ### Now global cuts to select events. Notice this means exactly two leptons with pT >= 10, and the leading one pT >= 25
	
        # cutHasTwoMuons imposes three conditions:
        #  First, number of muons (axis=1 means column. Each row is an event.) in an event is 2.
        #  Second, pt of the muons is greater than 25.
        #  Third, Sum of charge of muons should be 0. (because it originates from Z)
        cutHasTwoMuons = (ak.num(selMuons, axis=1)==2) & (ak.max(selMuons.pt, axis=1, mask_identity=False) >= 25) & (ak.sum(selMuons.charge,axis=1) == 0)
        cutHasTwoElecs = (ak.num(selElectrons, axis=1)==2) & (ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25) & (ak.sum(selElectrons.charge,axis=1) == 0)
        cutTwoLeps     = ((ak.num(selElectrons, axis=1)+ak.num(selMuons, axis=1)) < 4)
        cutHasTwoLeps  = ((cutHasTwoMuons) | (cutHasTwoElecs)) & cutTwoLeps
        ### Cut the events, also return the selected leptons for operation down the line
        events = events[ cutHasTwoLeps]
        return events, selElectrons[cutHasTwoLeps], selMuons[cutHasTwoLeps], [coll[cutHasTwoLeps] for coll in extraColls]

    def selectByJets(self, events, extraColls = []):
        # These are just standard jets, as available in the nanoAOD
        Jets = ak.zip({
            "pt": events.Jet.pt,
            "eta": events.Jet.eta,
            "phi": events.Jet.phi,
            "mass": events.Jet.mass,
            "jetId": events.Jet.jetId
        }, with_name="Momentum4D")
        jetCut = (Jets.pt > 30) & (abs(Jets.eta)<4.7)
        ak4jets = Jets[jetCut]

        # This is a cut that selects one/two/three jet(s) (It's just an array of Booleans)
        cutHasOneJet = (ak.num(ak4jets, axis=1)==1)
        cutHasTwoJets = (ak.num(ak4jets, axis=1)==2)
        cutHasThreeJets = (ak.num(ak4jets, axis=1)==3)
        
        # The following is the collection of jets with appropriate cuts applied.
        onejet = ak4jets[cutHasOneJet] #format: [[13.8],[28.8],...[32.4]]
        twojets = ak4jets[cutHasTwoJets] #format: [[13.8,19.0],[28.8,17.4],...[32.4,58.1]]
        threejets = ak4jets[cutHasThreeJets] #format: [[13.8,19.0,16.4],[28.8,17.4,28.6],...[32.4,58.1,28.8]]      
 
        # The following is the collection of events with respective number of jets
        event_onejet = events[cutHasOneJet]
        event_twojets = events[cutHasTwoJets]
        event_threejets = events[cutHasThreeJets]

        return event_onejet, event_twojets, event_threejets, onejet, twojets, threejets,[coll for coll in extraColls]

    def selectByTracks(self, events, leptons, extraColls = []):

        # Prepare the clean PFCand matched to tracks collection
        # PFCands (Particle-Flow Candidates) : every particle in the particle flow is saved here.
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")
        #print(len(ak.num(Cands, axis=1)),"Printing len of ak.num(Cands, axis=0)(hopefully matches the number of events)")
        # Track selection requirements
        cut = (events.PFCands.fromPV > 1) & \
            (events.PFCands.trkPt >= 0.7) & \
            (abs(events.PFCands.trkEta) <= 2.5) & \
            (abs(events.PFCands.dz) < 10) & \
            (events.PFCands.dzErr < 0.05)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        # Prepare the Lost Track collection
	# LostTracks : Tracks are reconstructed but not identified as one of the particles.
	# SUEP particles usually fall into this category.
        LostTracks = ak.zip({
            "pt": events.lostTracks.pt,
            "eta": events.lostTracks.eta,
            "phi": events.lostTracks.phi,
            "mass": 0.0
        }, with_name="Momentum4D")
        # More track selection requirement
        cut = (events.lostTracks.fromPV > 1) & \
            (events.lostTracks.pt >= 0.7) & \
            (abs(events.lostTracks.eta) <= 1.0) \
            & (abs(events.lostTracks.dz) < 10) & \
            (events.lostTracks.dzErr < 0.05)
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)

        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
	# Here we are concatenating the pf tracks and lost tracks.
        Total_Tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)
        tracks = Total_Tracks
        print(tracks,"Printing the first 'tracks'",len(tracks))
        ## Tracks that overlap with the leptons are taken out
        tracks = tracks[(tracks.deltaR(leptons[:,0])>= 0.4) & (tracks.deltaR(leptons[:,1])>= 0.4)]
        print(tracks,"Printing the second 'tracks'",len(tracks))
        return events, leptons, tracks, [coll for coll in extraColls]

    def selectByGEN(self, events):
        GenParts = ak.zip({
            "pt": events.GenPart.pt,
            "eta": events.GenPart.eta,
            "phi": events.GenPart.phi,
            "mass": events.GenPart.mass
        }, with_name="Momentum4D")
        cutgenZ    = (events.GenPart.pdgId == 23) & (events.GenPart.status == 62)
        cutgenH    = (events.GenPart.pdgId == 25) & (events.GenPart.status == 62)
        cutgenSUEP = (events.GenPart.pdgId == 999999) & (events.GenPart.status == 2)
        return events, GenParts[cutgenZ], GenParts[cutgenH], GenParts[cutgenSUEP]

    def shouldContinueAfterCut(self, events, out = pd.DataFrame(['empty'], columns=['empty'])):
        if len(events) == 0:
            self.save_dfs([out,out],["lepvars","jetvars"])
            return False
        else:
            return True

    def process(self, events):
        debug    = True  # If we want some prints in the middle
        doTracks = True # Just to speed things up
        doGen    = False # In case we want info on the gen level 
        # Main processor code
        # Define outputs
        output  = self.accumulator.identity()
        outlep  = {}
        out1jet  = {}
        out2jets  = {}
        out3jets  = {}
        outnumtrk = {}
        outgen  = {}

        # Data dependant stuff
        dataset = events.metadata['dataset']
        if self.isMC:      self.gensumweight = ak.sum(events.genWeight)
        if not(self.isMC): doGen = False

        # ------------------------------------------------------------------------------------
        # ------------------------------- OBJECT LOADING -------------------------------------
        # ------------------------------------------------------------------------------------
        # First, the trigger selection, as it takes out significant load for other steps
        if debug: print("Applying lepton requirements.... %i events in"%len(events))
        events, electrons, muons = self.selectByLeptons(events)[:3]

        if not(self.shouldContinueAfterCut(events)): return output # If we have no events, we simply stop

        # Then, the lepton selection
        if debug: print("%i events pass lepton cuts. Applying trigger requirements...."%len(events))
        events, [electrons, muons] = self.selectByTrigger(events,[electrons, muons])
        # Once passed the trigger, we can just merge the collections
        leptons = ak.concatenate([electrons, muons], axis=1)
        # Sort leptons by pt
        highpt_leptons = ak.argsort(leptons.pt, axis=1, ascending=False, stable=True)
        leptons = leptons[highpt_leptons]

        if not(self.shouldContinueAfterCut(events)): return output

        if debug: print("%i events pass trigger cuts. Selecting jets..."%len(events))
        event_onejet, event_twojets, event_threejets, onejet, twojets, threejets, [electrons, muons] = self.selectByJets(events, [electrons, muons])
	# Sorting jets by pt.
        highpt_1jet = ak.argsort(onejet.pt, axis=1, ascending=False, stable=True)
        highpt_2jet = ak.argsort(twojets.pt, axis=1, ascending=False, stable=True)
        highpt_3jet = ak.argsort(threejets.pt, axis=1, ascending=False, stable=True)
        onejet = onejet[highpt_1jet]
        twojets = twojets[highpt_2jet]
        threejets = threejets[highpt_3jet]

        if not(self.shouldContinueAfterCut(events)): return output
        if debug: print("%i events pass jet cuts. Selecting tracks..."%len(events))
        
        if doTracks:
          # Right now no track cuts, only selecting tracks
          events, leptons, tracks = self.selectByTracks(events, leptons) [:3]
          if not(self.shouldContinueAfterCut(events)): return output
          if debug: print("%i events pass track cuts. Doing more stuff..."%len(events))

        if doGen:
          events, genZ, genH, genSUEP = self.selectByGEN(events)[:4]
          if not(self.shouldContinueAfterCut(events)): return output
          if debug: print("%i events pass gen cuts. Doing more stuff..."%len(events))

        # ------------------------------------------------------------------------------
        # ------------------------------- PLOTTING -------------------------------------
        # ------------------------------------------------------------------------------

        # Define outputs for plotting
        if debug: print("Saving reco variables")
	# The variables that I can get are listed above in "SelectByLeptons" function
        #outlep["leadlep_pt"]    = leptons.pt[:,0]
        #outlep["subleadlep_pt"] = leptons.pt[:,1]
        #outlep["leadlep_eta"]   = leptons.eta[:,0]
        #outlep["subleadlep_eta"]= leptons.eta[:,1]
        #outlep["leadlep_phi"] = leptons.phi[:,0]
        #outlep["subleadlep_phi"] = leptons.phi[:,1]

	# From here I am working with Z boson reconstruction from the daugther leptons
        #outlep["Z_pt"] = np.sqrt((leptons.pt[:,0])**2 + (leptons.pt[:,1])**2 + 2*leptons.pt[:,0]*leptons.pt[:,1]*np.cos(leptons.phi[:,0]-leptons.phi[:,1]))
        #outlep["Z_eta"] = np.arcsinh((leptons.pt[:,0]*np.sinh(leptons.eta[:,0])+leptons.pt[:,1]*np.sinh(leptons.eta[:,1]))/np.sqrt((leptons.pt[:,0])**2 + (leptons.pt[:,1])**2 + 2*leptons.pt[:,0]*leptons.pt[:,1]*np.cos(leptons.phi[:,0]-leptons.phi[:,1])))
        #outlep["Z_phi"] = np.arcsin((leptons.pt[:,0]*np.sin(leptons.phi[:,0]) + leptons.pt[:,1]*np.sin(leptons.phi[:,1]))/(np.sqrt((leptons.pt[:,0])**2 + (leptons.pt[:,1])**2 + 2*leptons.pt[:,0]*leptons.pt[:,1]*np.cos(leptons.phi[:,0]-leptons.phi[:,1]))))
        #outlep["Z_m"] = np.sqrt(2*leptons.pt[:,0]*leptons.pt[:,1]*(np.cosh(leptons.eta[:,1]-leptons.eta[:,0])-np.cos(leptons.phi[:,1]-leptons.phi[:,0])))

        # From here I am working with jets
	# ak4jets is an array of arrays. 
        # Each element in the big array is an event, and each element (which is an array) has n entries, where n = # of jets in an event.
        #out1jet["onejet_pt"] = onejet.pt[:,0]
        #out1jet["onejet_eta"] = onejet.eta[:,0]
        #out1jet["onejet_phi"] = onejet.phi[:,0]

        #out2jets["twojets1_pt"] = twojets.pt[:,0]
        #out2jets["twojets1_eta"] = twojets.eta[:,0]
        #out2jets["twojets1_phi"] = twojets.phi[:,0]

        #out2jets["twojets2_pt"] = twojets.pt[:,1]
        #out2jets["twojets2_eta"] = twojets.eta[:,1]
        #out2jets["twojets2_phi"] = twojets.phi[:,1]

        #out3jets["threejets1_pt"] = threejets.pt[:,0]
        #out3jets["threejets1_eta"] = threejets.eta[:,0]
        #out3jets["threejets1_phi"] = threejets.phi[:,0]

        #out3jets["threejets2_pt"] = threejets.pt[:,1]
        #out3jets["threejets2_eta"] = threejets.eta[:,1]
        #out3jets["threejets2_phi"] = threejets.phi[:,1]

        #out3jets["threejets3_pt"] = threejets.pt[:,2]
        #out3jets["threejets3_eta"] = threejets.eta[:,2]
        #out3jets["threejets3_phi"] = threejets.phi[:,2]

        #From here I am working with track multiplicity
        #outnumtrk["outnumtrk"]

        if doGen:
          if debug: print("Saving gen variables")
          outlep["genZpt"]  = genZ.pt[:,0]
          outlep["genZeta"] = genZ.eta[:,0]
          outlep["genZphi"] = genZ.phi[:,0]

          outlep["genHpt"]  = genH.pt[:,0]
          outlep["genHeta"] = genH.eta[:,0]
          outlep["genHphi"] = genH.phi[:,0]


        if self.isMC:
          # We need this to be able to normalize the samples 
          outlep["genweight"]= events.genWeight[:]
          out1jet["genweight"]= event_onejet.genWeight[:]
          out2jets["genweight"]= event_twojets.genWeight[:]
          out3jets["genweight"]= event_threejets.genWeight[:]
          outnumtrk["genweight"]= events.genWeight[:]

        # This goes last, convert from awkward array to pandas and save the hdf5
        if debug: print("Conversion to pandas...")
        if not isinstance(outlep, pd.DataFrame): outlep = self.ak_to_pandas(outlep)
        if not isinstance(out1jet, pd.DataFrame): out1jet = self.ak_to_pandas(out1jet)
        if not isinstance(out2jets, pd.DataFrame): out2jets = self.ak_to_pandas(out2jets)
        if not isinstance(out3jets, pd.DataFrame): out3jets = self.ak_to_pandas(out3jets)
        if not isinstance(outnumtrk, pd.DataFrame): outnumtrk = self.ak_to_pandas(outnumtrk)
        if debug: print("DFS saving....")
        self.save_dfs([outlep, out1jet, out2jets, out3jets, outnumtrk],["lepvars","jetvars1","jetvars2","jetvars3","numtrkvars"])

        return output

    def postprocess(self, accumulator):
        return accumulator
