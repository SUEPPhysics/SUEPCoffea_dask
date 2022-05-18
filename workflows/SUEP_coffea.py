"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
from coffea import hist, processor
from typing import List, Optional
import awkward as ak
import pandas as pd
import numpy as np
import fastjet
import vector
vector.register_awkward()

#Importing SUEP specific functions
from workflows.inference_utils import *
from workflows.pandas_utils import *
from workflows.math_utils import *

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC: int, era: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, flag: bool, do_inf: bool, output_location: Optional[str]) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.do_inf = do_inf
        self.prefixes = {"SUEP": "SUEP"}
        
        #Set up the image size and pixels
        self.eta_pix = 280
        self.phi_pix = 360
        self.eta_span = (-2.5, 2.5)
        self.phi_span = (-np.pi, np.pi)
        self.eta_scale = self.eta_pix/(self.eta_span[1]-self.eta_span[0])
        self.phi_scale = self.phi_pix/(self.phi_span[1]-self.phi_span[0])

        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        if self.isMC: self.gensumweight = ak.sum(events.genWeight)
        
        #####################################################################################
        # ---- Trigger
        #####################################################################################
        
        if self.era == 2016:
            trigger = (events.HLT.PFHT900 == 1)
        else:
            trigger = (events.HLT.PFHT1050 == 1)
        
        # cut based on ak4 jets to replicate the trigger
        Jets = ak.zip({
            "pt": events.Jet.pt,
            "eta": events.Jet.eta,
            "phi": events.Jet.phi,
            "mass": events.Jet.mass,
            "jetId": events.Jet.jetId
        })
        jetCut = (Jets.pt > 30) & (abs(Jets.eta)<4.7)
        ak4jets = Jets[jetCut]
        ht = ak.sum(ak4jets.pt,axis=-1)
        
        # apply trigger selection
        events = events[(trigger & (ht > 1200))]
        ak4jets = ak4jets[(trigger & (ht > 1200))]
        
        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            print("No events passed trigger. Saving empty outputs.")
            out_vars = pd.DataFrame(['empty'], columns=['empty'])
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        #####################################################################################
        # ---- Track selection
        #####################################################################################

        #Prepare the clean PFCand matched to tracks collection
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")        
        cut = (events.PFCands.fromPV > 1) & \
                (events.PFCands.trkPt >= 0.7) & \
                (abs(events.PFCands.trkEta) <= 2.5) & \
                (abs(events.PFCands.dz) < 10) & \
                (events.PFCands.dzErr < 0.05)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        #Prepare the Lost Track collection
        LostTracks = ak.zip({
            "pt": events.lostTracks.pt,
            "eta": events.lostTracks.eta,
            "phi": events.lostTracks.phi,
            "mass": 0.0
        }, with_name="Momentum4D")
        cut = (events.lostTracks.fromPV > 1) & \
            (events.lostTracks.pt >= 0.7) & \
            (abs(events.lostTracks.eta) <= 1.0) & \
            (abs(events.lostTracks.dz) < 10) & \
            (events.lostTracks.dzErr < 0.05)
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)
                                                
        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
        tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)
        
        #####################################################################################
        # ---- FastJet reclustering
        #####################################################################################
        
        # minimum pT for reclustered jets
        minPt = 150
                
        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)        
        cluster = fastjet.ClusterSequence(tracks, jetdef)
        
        # have to set min_pt = 0 and cut later to avoid some memory issues
        # FIXME: should try to understand this failure
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(),"Momentum4D") 
        ak_inclusive_cluster = ak.with_name(cluster.constituents(),"Momentum4D")
        
        # apply minimum pT cut
        minPtCut = ak_inclusive_jets.pt > minPt
        ak_inclusive_jets = ak_inclusive_jets[minPtCut]
        ak_inclusive_cluster = ak_inclusive_cluster[minPtCut]
        
        #####################################################################################
        # ---- Event level information
        #####################################################################################
                
        # from https://twiki.cern.ch/twiki/bin/view/CMS/JetID:
        # jetId==2 means: pass tight ID, fail tightLepVeto
        # jetId==6 means: pass tight and tightLepVeto ID. 
        tightJetId = (ak4jets.jetId > 2)
        tight_ak4jets = ak4jets[tightJetId]
        looseJetId = (ak4jets.jetId >= 2)
        loose_ak4jets = ak4jets[looseJetId]
        
        # tracker jets
        trackerCut = abs(ak4jets.eta) < 2.4
        tracker_ak4jets = ak4jets[trackerCut]
            
        # save per event variables to a dataframe
        out_vars = pd.DataFrame()
        out_vars["ntracks"] = ak.num(tracks).to_list()
        out_vars["ngood_fastjets"] = ak.num(ak_inclusive_jets).to_list()
        out_vars["ht"] = ak.sum(ak4jets.pt,axis=-1).to_list()
        if self.era == 2016:
            out_vars["HLT_PFHT900"] = events.HLT.PFHT900
        else:
            out_vars["HLT_PFHT1050"] = events.HLT.PFHT1050
        # store first n jets infos per event
        for i in range(10):
            iAk4jet = (ak.num(ak4jets) > i)  
            out_vars["eta_ak4jets"+str(i)] = [x[i] if j else np.nan for j, x in zip(iAk4jet, ak4jets.eta)]
            out_vars["phi_ak4jets"+str(i)] = [x[i] if j else np.nan for j, x in zip(iAk4jet, ak4jets.phi)]
            out_vars["pt_ak4jets"+str(i)] = [x[i] if j else np.nan for j, x in zip(iAk4jet, ak4jets.pt)]
        out_vars["ngood_ak4jets"] = ak.num(ak4jets).to_list()
        out_vars["n_loose_ak4jets"] = ak.num(loose_ak4jets).to_list()
        out_vars["n_tight_ak4jets"] = ak.num(tight_ak4jets).to_list()
        out_vars["ht_loose"] = ak.sum(loose_ak4jets.pt,axis=-1).to_list()
        out_vars["ht_tight"] = ak.sum(tight_ak4jets.pt,axis=-1).to_list()
        out_vars["ht_tracker"] = ak.sum(tracker_ak4jets.pt,axis=-1).to_list()
        if self.isMC: out_vars["Pileup_nTrueInt"] = events.Pileup.nTrueInt
        out_vars["PV_npvs"] = events.PV.npvs
        out_vars["PV_npvsGood"] = events.PV.npvsGood

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0,len(tracks))
 
        #####################################################################################
        # ---- ML METHOD
        #####################################################################################
        
        #These lines control the inference from ML models. Conversion is done elsewhere
        #The inference skips the lost tracks for now. 
        inf_cands = Cleaned_cands
        SUEP_pred = np.ones(len(inf_cands))*np.nan
        if self.do_inf:    
            ort_sess = ort.InferenceSession('data/resnet.onnx')
            options = ort.SessionOptions() 
            options.inter_op_num_threads = 1 # number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.
            
            # convert events to images and run inference in batches
            # in order to avoid memory issues
            # also exploits the numba-compiled convert_to_images function
            batch_size = 100
            for i in range(0, len(inf_cands), batch_size):

                if i + batch_size > len(inf_cands): batch_size = len(inf_cands) - i
                batch = inf_cands[i:i+batch_size]
                imgs = convert_to_images(self, batch)
                batch_resnet_jets = run_inference(self, imgs, ort_sess)
                if i == 0: resnet_jets = batch_resnet_jets
                else: resnet_jets = np.concatenate((resnet_jets, batch_resnet_jets))    

            # highest SUEP prediction per event
            SUEP_pred = resnet_jets[:,1]
                
        out_vars["resnet_SUEP_pred_ML"] = SUEP_pred
                
        #####################################################################################
        # ---- ISR Removal Method (IRM)
        #####################################################################################
        
        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        columns_IRM = [
                "SUEP_nconst_IRM", "SUEP_ntracks_IRM", "SUEP_pt_avg_IRM", "SUEP_pt_avg_b_IRM", 
                "SUEP_S1_IRM", "SUEP_rho0_IRM", "SUEP_rho1_IRM", 
                "SUEP_pt_IRM", "SUEP_eta_IRM", "SUEP_phi_IRM", "SUEP_mass_IRM"
        ]
        columns_IRM += [c.replace("SUEP", "ISR") for c in columns_IRM]
        columns_IRM += ["dphi_SUEP_ISR_IRM"]
                
        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        
        # output file if no events pass selections for ISR, avoids errors later on
        if len(tracks) == 0:
            print("No events in ISR Removal Method, clusterCut.")
            for c in columns_IRM: out_vars[c] = np.nan
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output

        # order the reclustered jets by pT (will take top 2 for ISR removal method)
        highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
        jets_pTsorted = ak_inclusive_jets[highpt_jet]
        clusters_pTsorted = ak_inclusive_cluster[highpt_jet]     
        
        # at least 2 tracks in highest pt jet
        highpt_cands = clusters_pTsorted[:,0]                    # tracks for highest pt jet         
        singletrackCut = (ak.num(highpt_cands)>1)             
        jets_pTsorted = jets_pTsorted[singletrackCut]          
        clusters_pTsorted = clusters_pTsorted[singletrackCut]
        tracks = tracks[singletrackCut]
        indices = indices[singletrackCut]
        
        # number of constituents per jet, sorted by pT
        nconst_pTsorted = ak.num(clusters_pTsorted, axis=-1)

        # Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap
        SUEP_cand = ak.where(nconst_pTsorted[:,1]<=nconst_pTsorted[:,0],jets_pTsorted[:,0],jets_pTsorted[:,1])
        ISR_cand = ak.where(nconst_pTsorted[:,1]>nconst_pTsorted[:,0],jets_pTsorted[:,0],jets_pTsorted[:,1])
        
        # ISR REMOVAL: boost into frame of SUEP and use dphi cut to select tracks away from ISR
        boost_SUEP = ak.zip({
            "px": SUEP_cand.px*-1,
            "py": SUEP_cand.py*-1,
            "pz": SUEP_cand.pz*-1,
            "mass": SUEP_cand.mass
        }, with_name="Momentum4D")        
        ISR_cand_b = ISR_cand.boost_p4(boost_SUEP)
        tracks_b = tracks.boost_p4(boost_SUEP)
        SUEP_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) > 1.6]
        ISR_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) <= 1.6]
        oneIRMtrackCut = (ak.num(SUEP_tracks_b)>1)
        
        # output file if no events pass selections for ISR, avoids errors later on
        if not any(oneIRMtrackCut):
            print("No events in ISR Removal Method, oneIRMtrackCut.")
            for c in columns_IRM: out_vars[c] = np.nan
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        #remove the events left with one track
        SUEP_tracks_b = SUEP_tracks_b[oneIRMtrackCut]
        ISR_tracks_b = ISR_tracks_b[oneIRMtrackCut]
        SUEP_cand = SUEP_cand[oneIRMtrackCut]
        ISR_cand = ISR_cand[oneIRMtrackCut]        
        tracks = tracks[oneIRMtrackCut]
        indices = indices[oneIRMtrackCut]

        out_vars.loc[indices, "SUEP_dphi_SUEP_ISR_IRM"] = ak.mean(abs(SUEP_cand.deltaphi(ISR_cand)), axis=-1)
        
        # SUEP jet variables
        eigs = sphericity(self, SUEP_tracks_b,1.0) #Set r=1.0 for IRC safe
        out_vars.loc[indices, "SUEP_nconst_IRM"] = ak.num(SUEP_tracks_b)
        out_vars.loc[indices, "SUEP_ntracks_IRM"] = ak.num(tracks)
        out_vars.loc[indices, "SUEP_pt_avg_b_IRM"] = ak.mean(SUEP_tracks_b.pt, axis=-1)
        out_vars.loc[indices, "SUEP_S1_IRM"] = 1.5 * (eigs[:,1]+eigs[:,0])
       
        # ISR jet variables
        out_vars.loc[indices, "ISR_nconst_IRM"] = ak.num(ISR_tracks_b)
        out_vars.loc[indices, "ISR_ntracks_IRM"] = ak.num(tracks)
        # unboost to lab frame, boost to ISR frame
        ISR_tracks = ISR_tracks_b.boost_p4(SUEP_cand)
        boost_ISR = ak.zip({
            "px": ISR_cand.px*-1,
            "py": ISR_cand.py*-1,
            "pz": ISR_cand.pz*-1,
            "mass": ISR_cand.mass
        }, with_name="Momentum4D") 
        ISR_tracks_bISR = ISR_tracks.boost_p4(boost_ISR)
        eigs = sphericity(self, ISR_tracks_bISR, 1.0) #Set r=1.0 for IRC safe
        out_vars.loc[indices, "ISR_pt_avg_b_IRM"] = ak.mean(ISR_tracks_bISR.pt, axis=-1)
        out_vars.loc[indices, "ISR_S1_IRM"] = 1.5 * (eigs[:,1]+eigs[:,0])

        # unboost for these
        SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
        out_vars.loc[indices, "SUEP_pt_avg_IRM"] = ak.mean(SUEP_tracks.pt, axis=-1)
        out_vars.loc[indices, "ISR_pt_avg_IRM"] = ak.mean(ISR_tracks.pt, axis=-1)
        
        deltaR = SUEP_tracks.deltaR(SUEP_cand)
        out_vars.loc[indices, "SUEP_rho0_IRM"] = rho(self, 0, SUEP_cand, SUEP_tracks, deltaR)
        out_vars.loc[indices, "SUEP_rho1_IRM"] = rho(self, 1, SUEP_cand, SUEP_tracks, deltaR)
        deltaR = ISR_tracks.deltaR(ISR_cand)
        out_vars.loc[indices, "ISR_rho0_IRM"] = rho(self, 0, ISR_cand, ISR_tracks, deltaR)
        out_vars.loc[indices, "ISR_rho1_IRM"] = rho(self, 1, ISR_cand, ISR_tracks, deltaR)
        
        # redefine the jets using the tracks as selected by IRM
        SUEP = ak.zip({
            "px": ak.sum(SUEP_tracks.px, axis=-1),
            "py": ak.sum(SUEP_tracks.py, axis=-1),
            "pz": ak.sum(SUEP_tracks.pz, axis=-1),
            "energy": ak.sum(SUEP_tracks.energy, axis=-1),
        }, with_name="Momentum4D")
        out_vars.loc[indices, "SUEP_pt_IRM"] = SUEP.pt
        out_vars.loc[indices, "SUEP_eta_IRM"] = SUEP.eta
        out_vars.loc[indices, "SUEP_phi_IRM"] = SUEP.phi
        out_vars.loc[indices, "SUEP_mass_IRM"] = SUEP.mass
        ISR = ak.zip({
            "px": ak.sum(ISR_tracks.px, axis=-1),
            "py": ak.sum(ISR_tracks.py, axis=-1),
            "pz": ak.sum(ISR_tracks.pz, axis=-1),
            "energy": ak.sum(ISR_tracks.energy, axis=-1),
        }, with_name="Momentum4D")
        out_vars.loc[indices, "ISR_pt_IRM"] = ISR.pt
        out_vars.loc[indices, "ISR_eta_IRM"] = ISR.eta
        out_vars.loc[indices, "ISR_phi_IRM"] = ISR.phi
        out_vars.loc[indices, "ISR_mass_IRM"] = ISR.mass
        
        
        #####################################################################################
        # ---- Save outputs
        #####################################################################################
        
        save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
        
        
        return output

    def postprocess(self, accumulator):
        return accumulator
