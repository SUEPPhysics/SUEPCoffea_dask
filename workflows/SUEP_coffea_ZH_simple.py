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

    def sphericity(self, events, particles, r):

        cleaningCut = (ak.num(particles) != 0) & ak.all(9999998 > (ak.nan_to_num(particles.x,nan = 9999999)),axis = 1) & ak.all(9999998 > (ak.nan_to_num(particles.p,nan = 9999999)),axis = 1)
        events = events[cleaningCut]
        particles = particles[cleaningCut] 

        norm = np.squeeze(ak.sum(particles.p ** r, axis=1, keepdims=True))

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
        evals = np.sort(np.linalg.eigvals(s))

        # eval1 < eval2 < eval3
        eval1 = np.moveaxis(evals,0,1)[0]
        eval2 = np.moveaxis(evals,0,1)[1]
        eval3 = np.moveaxis(evals,0,1)[2]

        scalarSpher = 1.5*(eval1[:] + eval2[:])
        selectionCut = scalarSpher > 0.4

        events = events[selectionCut]
        particles = particles[selectionCut] 
        evals = evals[selectionCut]
        eval1 = eval1[selectionCut]
        eval2 = eval2[selectionCut]
        eval3 = eval3[selectionCut]
        
        return events, particles, evals, eval1, eval2, eval3

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
    ###lepton selection criteria--4momenta collection for plotting

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
        ### PARTICLE FLOW CANDIDATES ###
        # Every particle in particle flow (clean PFCand matched to tracks collection)
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")

        cutPF = (events.PFCands.fromPV > 1) & \
            (events.PFCands.trkPt >= 1) & \
            (abs(events.PFCands.trkEta) <= 2.5) & \
            (abs(events.PFCands.dz) < 10) & \
            (events.PFCands.dzErr < 0.05)
        Cleaned_cands = ak.packed(Cands[cutPF])

	    ### LOST TRACKS ###
        # Unidentified tracks, usually SUEP Particles
        LostTracks = ak.zip({
            "pt": events.lostTracks.pt,
            "eta": events.lostTracks.eta,
            "phi": events.lostTracks.phi,
            "mass": 0.0
        }, with_name="Momentum4D")

        cutLost = (events.lostTracks.fromPV > 1) & \
            (events.lostTracks.pt >= 1) & \
            (abs(events.lostTracks.eta) <= 1.0) \
            & (abs(events.lostTracks.dz) < 10) & \
            (events.lostTracks.dzErr < 0.05)
        Lost_Tracks_cands = ak.packed(LostTracks[cutLost])

        # dimensions of tracks = events x tracks in event x 4 momenta
        totalTracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)

        # Sorting out the tracks that overlap with leptons
        totalTracks = totalTracks[(totalTracks.deltaR(leptons[:,0])>= 0.4) & (totalTracks.deltaR(leptons[:,1])>= 0.4)]
        nTracks = ak.num(totalTracks,axis=1)
        return events, leptons, totalTracks, nTracks, [coll for coll in extraColls]

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
        #if not(events.event[0]==208940120 and events.luminosityBlock[0]==77328 and events.run[0]==1): return self.accumulator.identity()
        debug    = True  # If we want some prints in the middle
        chunkTag = "out_%i_%i_%i.hdf5"%(events.event[0], events.luminosityBlock[0], events.run[0]) #Unique tag to get different outputs per tag
        doTracks = True # Make it false, and it will speed things up
        doGen    = False #i In case we want info on the gen level 
        # Main processor code
        # Define outputs
        output  = self.accumulator.identity()
        outlep  = {}
        out1jet  = {}
        out2jets  = {}
        out3jets  = {}
        outnumtrk = {}
        outSpherL = {}
        outSpherZ = {}
        outSpherT = {}
        outnTrkL = {}
        outnTrkZ = {}
        outnTrkT = {}
        outgen  = {}

        # Data dependant stuff
        dataset = events.metadata['dataset']
        if self.isMC:      self.gensumweight = ak.sum(events.genWeight)
        if not(self.isMC): doGen = False

        # ------------------------------------------------------------------------------------
        # ------------------------------- OBJECT LOADING -------------------------------------
        # ------------------------------------------------------------------------------------
        # Trigger selection
        if debug: print("Applying lepton requirements.... %i events in"%len(events))
        events, electrons, muons = self.selectByLeptons(events)[:3]

        if not(self.shouldContinueAfterCut(events)): return output # If we have no events, we simply stop

        # Lepton selection
        if debug: print("%i events pass lepton cuts. Applying trigger requirements...."%len(events))
        events, [electrons, muons] = self.selectByTrigger(events,[electrons, muons])
        leptons = ak.concatenate([electrons, muons], axis=1)
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
          events, leptons, tracks, nTracks = self.selectByTracks(events, leptons) [:4]
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

        #region: LEPTONS
        outlep["leadlep_pt"]    = leptons.pt[:,0]
        outlep["subleadlep_pt"] = leptons.pt[:,1]
        outlep["leadlep_eta"]   = leptons.eta[:,0]
        outlep["subleadlep_eta"]= leptons.eta[:,1]
        outlep["leadlep_phi"] = leptons.phi[:,0]
        outlep["subleadlep_phi"] = leptons.phi[:,1]


        # From here I am working with Z boson reconstruction from the daugther leptons
        outlep["Z_pt"] = Zcands.pt[:] 
        outlep["Z_eta"] = Zcands.eta[:] 
        outlep["Z_phi"] = Zcands.phi[:] 
        outlep["Z_m"] =  Zcands.mass[:]
        #endregion

        #region: JETS
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
        #endregion

        #region: TRACK MULTIPLICITY & SPHERICITY
        if doTracks:
            outnumtrk["Ntracks"] = nTracks

            # Reconstructing by setting pS = -pZ 
            boost_Zinv = ak.zip({
                "px": Zcands.px,
                "py": Zcands.py,
                "pz": Zcands.pz,
                "mass": Zcands.mass
            }, with_name="Momentum4D") 

            # Reconstructing by summing all tracks
            boost_tracks = ak.zip({
                "px": ak.sum(tracks.px, axis=1)*-1,
                "py": ak.sum(tracks.py, axis=1)*-1,
                "pz": ak.sum(tracks.pz, axis=1)*-1,
                "mass": 125 # Assuming it is a Higgs?
            }, with_name="Momentum4D")

            tracks_boostedagainstZ      = tracks.boost_p4(boost_Zinv)
            tracks_boostedagainsttracks = tracks.boost_p4(boost_tracks)

            clean_eventsL, particlesL, evalsL, evalL1, evalL2, evalL3 = self.sphericity(events, tracks, 2) # Gives the sphericity in Lab frame
            clean_eventsZ, particlesZ, evalsZ, evalZ1, evalZ2, evalZ3 = self.sphericity(events, tracks_boostedagainstZ, 2) #Gives the sphericity in -Z frame (-pZ = pS)
            clean_eventsT, particlesT, evalsT, evalT1, evalT2, evalT3 = self.sphericity(events, tracks_boostedagainsttracks, 2) #Gives the sphericity in -Z frame (tracks)

            #region: OUTPUT FOR SPHERICITY

            ### Evals themselves ###
            outSpherL["eval_L1"] = evalL1[:]
            outSpherL["eval_L2"] = evalL2[:]
            outSpherL["eval_L3"] = evalL3[:]
            outSpherZ["eval_Z1"] = evalZ1[:]
            outSpherZ["eval_Z2"] = evalZ2[:]
            outSpherZ["eval_Z3"] = evalZ3[:]
            outSpherT["eval_T1"] = evalT1[:]
            outSpherT["eval_T2"] = evalT2[:]
            outSpherT["eval_T3"] = evalT3[:]

            ### Scalar Sphericity ###
            outSpherL["scalarSpher_L"] = 1.5*(evalL1[:] + evalL2[:])
            outSpherZ["scalarSpher_Z"] = 1.5*(evalZ1[:] + evalZ2[:])
            outSpherT["scalarSpher_T"] = 1.5*(evalT1[:] + evalT2[:])

            ### Mean Difference ###
            meandiffL = np.empty(len(evalsL))
            meandiffZ = np.empty(len(evalsZ))
            meandiffT = np.empty(len(evalsT))
            
            for i in range(len(evalsL)):
                meandiffL[i] = np.mean([abs(evalsL[i][0]-evalsL[i][1]),abs(evalsL[i][1]-evalsL[i][2]),abs(evalsL[i][2]-evalsL[i][0])])

            for i in range(len(evalsZ)):
                meandiffZ[i] = np.mean([abs(evalsZ[i][0]-evalsZ[i][1]),abs(evalsZ[i][1]-evalsZ[i][2]),abs(evalsZ[i][2]-evalsZ[i][0])])

            for i in range(len(evalsT)):
                meandiffT[i] = np.mean([abs(evalsT[i][0]-evalsT[i][1]),abs(evalsT[i][1]-evalsT[i][2]),abs(evalsT[i][2]-evalsT[i][0])])

            outSpherL["meanDiff_L"] = meandiffL
            outSpherZ["meanDiff_Z"] = meandiffZ
            outSpherT["meanDiff_T"] = meandiffT

            ### Tracks with Sphericity Selection ###
            """
            outnTrkL["spherSel_tracksL"] = particlesL
            outnTrkZ["spherSel_tracksZ"] = particlesZ
            outnTrkT["spherSel_tracksT"] = particlesT 
            """

            #endregion
        #endregion

        if doGen:
          if debug: print("Saving gen variables")
          
          outlep["genZpt"]  = genZ.pt[:,0]
          outlep["genZeta"] = genZ.eta[:,0]
          outlep["genZphi"] = genZ.phi[:,0]

          outlep["genHpt"]  = genH.pt[:,0]
          outlep["genHeta"] = genH.eta[:,0]
          outlep["genHphi"] = genH.phi[:,0]

        outputs = {
            "lepvars":[outlep,events],
            "jetvars1":[out1jet,event_onejet],
            "jetvars2":[out2jets,event_twojets],
            "jetvars3":[out3jets,event_threejets],
            "numtrkvars":[outnumtrk,events],
            "sphervarsL":[outSpherL,clean_eventsL],
            "sphervarsZ":[outSpherZ,clean_eventsZ],
            "sphervarsT":[outSpherT,clean_eventsT],
            "spherTrkL":[outnTrkL,clean_eventsL],
            "spherTrkZ":[outnTrkZ,clean_eventsZ],
            "spherTrkT":[outnTrkT,clean_eventsT]
            }

        for output in outputs:
            if self.isMC:
                outputs[output][0]["genweight"] = outputs[output][1].genWeight[:]
            if debug: print("Conversion to pandas...")
            if not isinstance(outputs[output][0], pd.DataFrame): 
                outputs[output][0] = self.ak_to_pandas(outputs[output][0])

        """
        if self.isMC:
            # We need this to be able to normalize the samples 
            outlep["genweight"]= events.genWeight[:]
            out1jet["genweight"]= event_onejet.genWeight[:]
            out2jets["genweight"]= event_twojets.genWeight[:]
            out3jets["genweight"]= event_threejets.genWeight[:]
            outnumtrk["genweight"]= events.genWeight[:]
            outSpherL["genweight"]= clean_eventsL.genWeight[:]
            outSpherZ["genweight"]= clean_eventsZ.genWeight[:]
            outSpherT["genweight"]= clean_eventsT.genWeight[:]
          

        # This goes last, convert from awkward array to pandas and save the hdf5
        if debug: print("Conversion to pandas...")
    
        if not isinstance(outlep, pd.DataFrame): outlep = self.ak_to_pandas(outlep)
        if not isinstance(out1jet, pd.DataFrame): out1jet = self.ak_to_pandas(out1jet)
        if not isinstance(out2jets, pd.DataFrame): out2jets = self.ak_to_pandas(out2jets)
        if not isinstance(out3jets, pd.DataFrame): out3jets = self.ak_to_pandas(out3jets)
        if not isinstance(outnumtrk, pd.DataFrame): outnumtrk = self.ak_to_pandas(outnumtrk)
        if not isinstance(outSpherL, pd.DataFrame): outSpherL = self.ak_to_pandas(outSpherL)
        if not isinstance(outSpherZ, pd.DataFrame): outSpherZ = self.ak_to_pandas(outSpherZ)
        if not isinstance(outSpherT, pd.DataFrame): outSpherT = self.ak_to_pandas(outSpherT)
        """

        if debug: print("DFS saving....")

        self.save_dfs(
            [ outlep,    out1jet,  out2jets,  out3jets,  outnumtrk,   outSpherL,   outSpherZ,   outSpherT,   outnTrkL,   outnTrkZ,   outnTrkT],
            ["lepvars","jetvars1","jetvars2","jetvars3","numtrkvars","sphervarsL","sphervarsZ","sphervarsT","spherTrkL","spherTrkZ","spherTrkT"],
            chunkTag
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
