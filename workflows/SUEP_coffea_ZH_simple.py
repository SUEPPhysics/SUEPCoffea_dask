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
            print(field)
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
        
    def save_dfs(self, dfs, df_names, fname=None):
        if not(fname): fname = "out.hdf5"
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
            if os.path.isfile(destination):
              if not os.path.samefile(local_file, destination):
                shutil.copy2(local_file, destination)
              else:
                fname = "condor_" + fname
                destination = os.path.join(location, os.path.join(merged_subdirs, fname))
                shutil.copy2(local_file, destination)
            else:
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
        cutMuons     = (events.Muon.looseId) & (events.Muon.pt >= 10) & (abs(events.Muon.dxy) <= 0.02) & (abs(events.Muon.dz) <= 0.1)
        cutElectrons = (events.Electron.cutBased >= 2) & (events.Electron.pt >= 15)
        ### Apply the cuts
        selMuons     = muons[cutMuons]
        selElectrons = electrons[cutElectrons]
        ### Now global cuts to select events. Notice this means exactly two leptons with pT >= 10, and the leading one pT >= 25
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
        # No cut applied, really, but we could do it
        return events, ak4jets, [coll for coll in extraColls]

    def selectByTracks(self, events, leptons, extraColls = []):

        # Prepare the clean PFCand matched to tracks collection
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")
        # Track selection requirements
        cut = (events.PFCands.fromPV > 1) & \
            (events.PFCands.trkPt >= 0.7) & \
            (abs(events.PFCands.trkEta) <= 2.5) & \
            (abs(events.PFCands.dz) < 10) & \
            (events.PFCands.dzErr < 0.05)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        # Prepare the Lost Track collection
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
        Total_Tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)
        tracks = Total_Tracks
        ## Tracks that overlap with the leptons are taken out
        tracks = tracks[(tracks.deltaR(leptons[:,0])>= 0.4) & (tracks.deltaR(leptons[:,1])>= 0.4)]
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
            self.save_dfs([out],["vars"], chunkTag)
            return False
        else:
            return True

    def process(self, events):
        debug    = True  # If we want some prints in the middle
        doTracks = True # Just to speed things up
        doGen    = True # In case we want info on the gen level 
        chunkTag = "out_%i_%i_%i.hdf5"%(events.event[0], events.luminosityBlock[0], events.run[0]) #Unique tag to get different outputs per tag
        # Main processor code
        # Define outputs
        output  = self.accumulator.identity()
        out     = {}
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
        # Right now no jet cuts, only selecting jets
        events, ak4jets, [electrons, muons] = self.selectByJets(events, [electrons, muons])
        highpt_jets = ak.argsort(ak4jets.pt, axis=1, ascending=False, stable=True)
        ak4jets = ak4jets[highpt_jets]

        if not(self.shouldContinueAfterCut(events)): return output
        if debug: print("%i events pass jet cuts. Selecting tracks..."%len(events))
        
        if doTracks:
          # Right now no track cuts, only selecting tracks
          events, leptons, tracks     = self.selectByTracks(events, leptons) [:3]
          if not(self.shouldContinueAfterCut(events)): return output
          if debug: print("%i events pass track cuts. Doing more stuff..."%len(events))

        if doGen:
          events, genZ, genH, genSUEP = self.selectByGEN(events)[:4]
          if not(self.shouldContinueAfterCut(events)): return output
          if debug: print("%i events pass gen cuts. Doing more stuff..."%len(events))


        # Now deal with the Z candidate

        Zcands = leptons[:,0] + leptons[:,1]
        
        # ------------------------------------------------------------------------------
        # ------------------------------- PLOTTING -------------------------------------
        # ------------------------------------------------------------------------------

        # Define outputs for plotting
        if debug: print("Saving reco variables")
	# The variables that I can get are listed above in "SelectByLeptons" function
        out["leadlep_pt"]    = leptons.pt[:,0]
        out["subleadlep_pt"] = leptons.pt[:,1]
        out["leadlep_eta"]   = leptons.eta[:,0]
        out["subleadlep_eta"]= leptons.eta[:,1]
        out["leadlep_phi"] = leptons.phi[:,0]
        out["subleadlep_phi"] = leptons.phi[:,1]

        out["Zpt"]  = Zcands.pt[:]
        out["Zeta"] = Zcands.eta[:]
        out["Zphi"] = Zcands.phi[:]
 
        if doTracks:
          out["nTracks"]     = ak.num(tracks, axis=1)
          spher =  self.sphericity(tracks, 2)
          out["spher_lab"] = 1.5*(spher[:,1] + spher[:,0])
          print(spher[0:4,0], spher[0:4,1], spher[0:4,2])

          boost_Zinv = ak.zip({
            "px": Zcands.px,
            "py": Zcands.py,
            "pz": Zcands.pz,
            "mass": Zcands.mass
          }, with_name="Momentum4D")

          boost_tracks = ak.zip({
            "px": ak.sum(tracks.px, axis=1)*-1,
            "py": ak.sum(tracks.py, axis=1)*-1,
            "pz": ak.sum(tracks.pz, axis=1)*-1,
            "mass": 125 # Assuming it is a Higgs?
          }, with_name="Momentum4D")

          tracks_boostedagainstZ      = tracks.boost_p4(boost_Zinv)
          tracks_boostedagainsttracks = tracks.boost_p4(boost_tracks)
          
          print("_")
          spherZ =  self.sphericity(tracks_boostedagainstZ, 2)
          out["spher_Z"] = 1.5*(spherZ[:,1] + spherZ[:,0])
          print(spherZ[0:4,0], spherZ[0:4,1], spherZ[0:4,2])
          print("_")
          sphertracks =  self.sphericity(tracks_boostedagainsttracks, 2)
          out["spher_tracks"] = 1.5*(sphertracks[:,1] + sphertracks[:,0])
          print(sphertracks[0:4,0], sphertracks[0:4,1], sphertracks[0:4,2])



          """maxTracks = ak.max(out["nTracks"], axis=0)
          print("Max Tracks: %i"%maxTracks)
          print(ak.pad_none(tracks.pt, maxTracks, clip=True).tolist())
          print(ak.pad_none(tracks.pt, maxTracks, clip=True).type)
          out["Tracks_pt"]   = ak.pad_none(tracks.pt , maxTracks,clip=True).tolist()
          out["Tracks_eta"]  = ak.pad_none(tracks.eta, maxTracks,clip=True).tolist()
          out["Tracks_phi"]  = ak.pad_none(tracks.phi, maxTracks,clip=True).tolist()
          out["dZTracksphi_mean"] = ak.mean(tracks.deltaphi(Zcands), axis=1)
          out["dZTrackseta_mean"] = ak.mean(tracks.deltaeta(Zcands), axis=1)
          out["dZTrackR_mean"]    = ak.mean(tracks.deltaR(Zcands),axis=1)"""

 
  
        """if doGen:
          if debug: print("Saving gen variables")
          out["genZpt"]  = genZ.pt[:,0]
          out["genZeta"] = genZ.eta[:,0]
          out["genZphi"] = genZ.phi[:,0]

          out["genHpt"]  = genH.pt[:,0]
          out["genHeta"] = genH.eta[:,0]
          out["genHphi"] = genH.phi[:,0]
        """ 

        if self.isMC:
          # We need this to be able to normalize the samples 
          out["genweight"]= events.genWeight[:]

        # This goes last, convert from awkward array to pandas and save the hdf5
        if debug: print("Conversion to pandas...")
        if not isinstance(out, pd.DataFrame): out = self.ak_to_pandas(out)
        if debug: print("DFS saving....")
        self.save_dfs([out],["vars"], chunkTag)

        return output

    def postprocess(self, accumulator):
        return accumulator
