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
        # ---- Fastejet reclustering
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
        minPt = 150
        
        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)        
        cluster = fastjet.ClusterSequence(tracks, jetdef)
        
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt= 0),"Momentum4D") 
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt= 0),"Momentum4D")
        
        minPtCut = ak_inclusive_jets.pt > minPt
        ak_inclusive_jets = ak_inclusive_jets[minPtCut]
        ak_inclusive_cluster = ak_inclusive_cluster[minPtCut]
        
        atLeastOneJet = ak.num(ak_inclusive_jets) > 0
        ak_inclusive_jets = ak_inclusive_jets[atLeastOneJet]
        ak_inclusive_cluster = ak_inclusive_cluster[atLeastOneJet]
        tracks = tracks[atLeastOneJet]
        
        #####################################################################################
        # ---- Event level information and selections
        #####################################################################################

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
        ak4jets = ak4jets[atLeastOneJet]
        
        # from https://twiki.cern.ch/twiki/bin/view/CMS/JetID:
        # jetId==2 means: pass tight ID, fail tightLepVeto
        # jetId==6 means: pass tight and tightLepVeto ID. 
        tightJetId = (ak4jets.jetId > 2)
        tight_ak4jets = ak4jets[tightJetId]
        looseJetId = (ak4jets.jetId >= 2)
        loose_ak4jets = ak4jets[looseJetId]
        
        # barrel jets
        barrelCut = abs(ak4jets.eta) < 2.4
        barrel_ak4jets = ak4jets[barrelCut]
            
        # save per event variables to a dataframe
        out_vars = pd.DataFrame()
        out_vars["uncleaned_tracks"] = ak.num(Cands[atLeastOneJet]).to_list()
        out_vars["ntracks"] = ak.num(tracks).to_list()
        out_vars["ngood_fastjets"] = ak.num(ak_inclusive_jets).to_list()
        out_vars["ht"] = ak.sum(ak4jets.pt,axis=-1).to_list()
        out_vars["nLostTracks"] = ak.num(Lost_Tracks_cands[atLeastOneJet]).to_list()
        if self.era == 2016:
            out_vars["HLT_PFHT900"] = events.HLT.PFHT900[atLeastOneJet]
        else:
            out_vars["HLT_PFHT1050"] = events.HLT.PFHT1050[atLeastOneJet]
        
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
        out_vars["ht_barrel"] = ak.sum(barrel_ak4jets.pt,axis=-1).to_list()
        if self.isMC: out_vars["Pileup_nTrueInt"] = events.Pileup.nTrueInt[atLeastOneJet]
        out_vars["PV_npvs"] = events.PV.npvs[atLeastOneJet]
        out_vars["PV_npvsGood"] = events.PV.npvsGood[atLeastOneJet]
            
        # define triggers by era
        if self.era == 2016:
            trigger = ((out_vars['HLT_PFHT900'] == 1))
        else:
            trigger = ((out_vars['HLT_PFHT1050'] == 1))
            
        # remove events that fail the trigger and HT cuts
        htCut = ((out_vars["ht"]>1200) & (trigger))    
        ak_inclusive_cluster = ak_inclusive_cluster[htCut]
        ak_inclusive_jets = ak_inclusive_jets[htCut]
        tracks = tracks[htCut]
        Lost_Tracks_cands = Lost_Tracks_cands[htCut]
        
        # indices of events in tracks, used to keep track which events pass IRM
        indices = np.arange(0,len(tracks))
        
        # only care about events that pass either of IRM or ML methods
        out_vars = out_vars.loc[htCut, :]
        out_vars = out_vars.reset_index(drop=True)
        
        # output empty dataframe if no events pass trigger
        if len(indices) == 0:
            print("No events passed trigger. Saving empty outputs.")
            out_vars = pd.DataFrame(['empty'], columns=['empty'])
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        #####################################################################################
        # ---- ML METHOD
        #####################################################################################
        
        #These lines control the inference from JetSSD. Conversion is done elsewhere
        #The inference skips the lost tracks for now. 
        SUEP_pred = np.ones(len(Cleaned_cands[htCut]))*np.nan
        if self.do_inf:    
            ort_sess = ort.InferenceSession('data/resnet.onnx')
            options = ort.SessionOptions() 
            options.inter_op_num_threads = 1 # number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.
            
            if ak.any(htCut): 
                inf_cands = Cleaned_cands[atLeastOneJet][htCut]
                
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
        # ---- ISR Removal Method
        #####################################################################################
        
        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        columns_ISR = [
                "SUEP_nLostTracks_IRM", "SUEP_pt_IRM", "SUEP_eta_IRM", "SUEP_phi_IRM",
                "SUEP_mass_IRM", "SUEP_dphi_SUEP_ISR_IRM", "SUEP_nconst_IRM", "SUEP_ntracks_IRM",
                "SUEP_pt_avg_b_IRM", "SUEP_spher_IRM", "SUEP_S1_IRM", "SUEP_pt_avg_IRM",         
                "SUEP_girth_IRM", "SUEP_rho0_IRM", "SUEP_rho1_IRM", "SUEP_pt_IRM", 
                "SUEP_eta_IRM", "SUEP_phi_IRM", "SUEP_mass_IRM"
        ]
        
        # remove events without a cluster
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        Lost_Tracks_cands = Lost_Tracks_cands[clusterCut]
        
        # output file if no events pass selections for ISR, avoids errors later on
        if len(tracks) == 0:
            print("No events in ISR Removal Method, clusterCut.")
            for c in columns_ISR: out_vars[c] = np.nan
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        ### SUEP_mult
        chonkocity = ak.num(ak_inclusive_cluster, axis=2)
        #chonkiest_jet = ak.argsort(chonkocity, axis=1, ascending=True, stable=True)[:, ::-1]
        #thicc_jets = ak_inclusive_jets[chonkiest_jet]
        #chonkiest_cands = ak_inclusive_cluster[chonkiest_jet][:,0]

        ### Order the reclustered jets by pT (will take top 2 for ISR removal method)
        highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
        SUEP_pt = ak_inclusive_jets[highpt_jet]
        SUEP_pt_nconst = chonkocity[highpt_jet]
        SUEP_pt_tracks = ak_inclusive_cluster[highpt_jet]        
        highpt_cands = SUEP_pt_tracks[:,0]                  #tracks for highest pt
        singletrackCut = (ak.num(highpt_cands)>1)
        SUEP_pt = SUEP_pt[singletrackCut]           #We dont want to look at single track jets
        SUEP_pt_nconst = SUEP_pt_nconst[singletrackCut]
        SUEP_pt_tracks = SUEP_pt_tracks[singletrackCut]
        highpt_cands = highpt_cands[singletrackCut] 
        tracks_IRM = tracks[singletrackCut]
        indices_IRM = indices[singletrackCut]
        Lost_Tracks_cands = Lost_Tracks_cands[singletrackCut]

        # ISR removal method (IRM): Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap. Boost and remove ISR 
        SUEP_cand = ak.where(SUEP_pt_nconst[:,1]<=SUEP_pt_nconst[:,0],SUEP_pt[:,0],SUEP_pt[:,1])
        SUEP_cand_tracks = ak.where(SUEP_pt_nconst[:,1]<=SUEP_pt_nconst[:,0],SUEP_pt_tracks[:,0],SUEP_pt_tracks[:,1])
        ISR_cand = ak.where(SUEP_pt_nconst[:,1]>SUEP_pt_nconst[:,0],SUEP_pt[:,0],SUEP_pt[:,1])
        ISR_cand_tracks = ak.where(SUEP_pt_nconst[:,1]>SUEP_pt_nconst[:,0],SUEP_pt_tracks[:,0],SUEP_pt_tracks[:,1])
        dphi_SUEP_ISR = abs(SUEP_cand.deltaphi(ISR_cand))
        boost_IRM = ak.zip({
            "px": SUEP_cand.px*-1,
            "py": SUEP_cand.py*-1,
            "pz": SUEP_cand.pz*-1,
            "mass": SUEP_cand.mass
        }, with_name="Momentum4D")
        ISR_cand_b = ISR_cand.boost_p4(boost_IRM)
        tracks_IRM = tracks_IRM.boost_p4(boost_IRM)
        Lost_Tracks_IRM = Lost_Tracks_cands.boost_p4(boost_IRM)
        IRM_cands = tracks_IRM[abs(tracks_IRM.deltaphi(ISR_cand_b)) > 1.6]
        Lost_IRM_cands = Lost_Tracks_IRM[abs(Lost_Tracks_IRM.deltaphi(ISR_cand_b)) > 1.6]
        oneIRMtrackCut = (ak.num(IRM_cands)>1)
        
        # output file if no events pass selections for ISR, avoids errors later on
        if not any(oneIRMtrackCut):
            print("No events in ISR Removal Method, oneIRMtrackCut.")
            for c in columns_ISR: out_vars[c] = np.nan
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        IRM_cands = IRM_cands[oneIRMtrackCut]#remove the events left with one track
        tracks_IRM = tracks_IRM[oneIRMtrackCut]
        SUEP_cand = SUEP_cand[oneIRMtrackCut]
        ISR_cand = ISR_cand[oneIRMtrackCut]
        ISR_cand_b = ISR_cand_b[oneIRMtrackCut]
        SUEP_cand_tracks = SUEP_cand_tracks[oneIRMtrackCut]
        ISR_cand_tracks = ISR_cand_tracks[oneIRMtrackCut]
        boost_IRM = boost_IRM[oneIRMtrackCut]
        indices_IRM = indices_IRM[oneIRMtrackCut]
        Lost_IRM_cands = Lost_IRM_cands[oneIRMtrackCut]

        out_vars.loc[indices_IRM, "SUEP_nLostTracks_IRM"] = ak.num(Lost_IRM_cands)        
        out_vars.loc[indices_IRM, "SUEP_dphi_SUEP_ISR_IRM"] = ak.mean(abs(SUEP_cand.deltaphi(ISR_cand)), axis=-1)
        eigs = sphericity(self, IRM_cands,1.0) #Set r=1.0 for IRC safe
        out_vars.loc[indices_IRM, "SUEP_nconst_IRM"] = ak.num(IRM_cands)
        out_vars.loc[indices_IRM, "SUEP_ntracks_IRM"] = ak.num(tracks_IRM)
        out_vars.loc[indices_IRM, "SUEP_pt_avg_b_IRM"] = ak.mean(IRM_cands.pt, axis=-1)
        out_vars.loc[indices_IRM, "SUEP_S1_IRM"] = 1.5 * (eigs[:,1]+eigs[:,0])
        #out_vars.loc[indices_IRM, "SUEP_aplan_IRM"] = 1.5 * eigs_2[:,0]
        #out_vars.loc[indices_IRM, "SUEP_FW2M_IRM"] = 1.0 - 3.0 * (eigs_2[:,2]*eigs_2[:,1] + eigs_2[:,2]*eigs_2[:,0] + eigs_2[:,1]*eigs_2[:,0])
        #out_vars.loc[indices_IRM, "SUEP_D_IRM"] = 27.0 * eigs_2[:,2]*eigs_2[:,1]*eigs_2[:,0]

        # unboost for these
        IRM_cands_ub = IRM_cands.boost_p4(SUEP_cand)
        deltaR = IRM_cands_ub.deltaR(SUEP_cand)
        out_vars.loc[indices_IRM, "SUEP_pt_avg_IRM"] = ak.mean(IRM_cands_ub.pt, axis=-1)
        out_vars.loc[indices_IRM, "SUEP_girth_IRM"] = ak.sum((deltaR/1.5)*IRM_cands_ub.pt, axis=-1)/SUEP_cand.pt
        out_vars.loc[indices_IRM, "SUEP_rho0_IRM"] = rho(self, 0, SUEP_cand, IRM_cands_ub, deltaR)
        out_vars.loc[indices_IRM, "SUEP_rho1_IRM"] = rho(self, 1, SUEP_cand, IRM_cands_ub, deltaR)

        SUEPs = ak.zip({
            "px": ak.sum(IRM_cands.px, axis=-1),
            "py": ak.sum(IRM_cands.py, axis=-1),
            "pz": ak.sum(IRM_cands.pz, axis=-1),
            "energy": ak.sum(IRM_cands.energy, axis=-1),
        }, with_name="Momentum4D")

        out_vars.loc[indices_IRM, "SUEP_pt_IRM"] = SUEPs.pt
        out_vars.loc[indices_IRM, "SUEP_eta_IRM"] = SUEPs.eta
        out_vars.loc[indices_IRM, "SUEP_phi_IRM"] = SUEPs.phi
        out_vars.loc[indices_IRM, "SUEP_mass_IRM"] = SUEPs.mass
        
        ### save outputs
        save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
        
        return output

    def postprocess(self, accumulator):
        return accumulator
