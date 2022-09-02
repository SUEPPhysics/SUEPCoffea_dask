"""
ML_coffea.py
Coffea producer for SUEP ML analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2022
"""
from coffea import hist, processor
from typing import List, Optional
import awkward as ak
import pandas as pd
import numpy as np
import h5py
import fastjet
import vector
vector.register_awkward()

#Importing SUEP specific functions
from workflows.inference_utils import *
from workflows.pandas_utils import *
from workflows.math_utils import *

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC: int, era: int, scouting: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, flag: bool, do_inf: bool, output_location: Optional[str]) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.scouting = scouting
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.do_inf = do_inf
        self.prefixes = {"SUEP": "SUEP"}

        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        if self.isMC and self.scouting==1: self.gensumweight = ak.count(events.PFcand.pt)
        elif self.isMC: self.gensumweight = ak.sum(events.genWeight)
        
        # cut based on ak4 jets to replicate the trigger
        if self.scouting == 1:
            Jets = ak.zip({
                "pt": events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
            })
        else:
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
        if self.scouting == 1:
            events = events[(ht > 600)]
            ak4jets = ak4jets[(ht > 600)]
        else:
            if self.era == 2016:
                trigger = (events.HLT.PFHT900 == 1)
            else:
                trigger = (events.HLT.PFHT1050 == 1)
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
        if self.scouting == 1:
            Cands = ak.zip({
                "pt": events.PFcand.pt,
                "eta": events.PFcand.eta,
                "phi": events.PFcand.phi,
                "mass": events.PFcand.mass
            }, with_name="Momentum4D")
            cut = (events.PFcand.pt >= 0.75) & \
                    (abs(events.PFcand.eta) <= 2.5) & \
                    (events.PFcand.vertex == 0) & \
                    (events.PFcand.q != 0)
            Cleaned_cands = Cands[cut]
            tracks =  ak.packed(Cleaned_cands)
            
        else:
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
        if self.scouting == 1:
            tight_ak4jets = ak4jets
            loose_ak4jets = ak4jets
        else:
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
        out_vars["ngood_tracker_ak4jets"] = ak.num(tracker_ak4jets).to_list()

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0,len(tracks))
                
        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################
        
        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        columns_IRM = [
                "SUEP_nconst_IRM", "SUEP_ntracks_IRM", 
                "SUEP_pt_avg_IRM", "SUEP_pt_avg_b_IRM", "SUEP_pt_mean_scaled",
                "SUEP_S1_IRM", "SUEP_rho0_IRM", "SUEP_rho1_IRM", 
                "SUEP_pt_IRM", "SUEP_eta_IRM", "SUEP_phi_IRM", "SUEP_mass_IRM",
                "dphi_SUEP_ISR_IRM"
        ]
        columns_CL = [c.replace("IRM", "CL") for c in columns_IRM]
        columns_CO = [c.replace("IRM", "CO") for c in columns_IRM]
        columns = columns_IRM + columns_CL + columns_CO
        if self.isMC: 
            columns_CL_ISR = [c.replace("IRM", "CL".replace("SUEP", "ISR")) for c in columns_IRM]
            columns_CO_ISR = [c.replace("IRM", "CO".replace("SUEP", "ISR")) for c in columns_IRM]
            columns += columns_CL_ISR + columns_CO_ISR
                
        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        
        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            for c in columns: out_vars[c] = np.nan
            save_dfs(self, [out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        # drop events that don't pass the cut
        out_vars = out_vars[out_vars['ngood_fastjets']>1]
        assert len(out_vars) == len(tracks)
        
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
        
        # drop events that don't pass the cut
        out_vars = out_vars[ak.to_numpy(singletrackCut)]
        assert len(out_vars) == len(tracks)
        
        # number of constituents per jet, sorted by pT
        nconst_pTsorted = ak.num(clusters_pTsorted, axis=-1)

        # Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap
        SUEP_cand = ak.where(nconst_pTsorted[:,1]<=nconst_pTsorted[:,0],jets_pTsorted[:,0],jets_pTsorted[:,1])
        ISR_cand = ak.where(nconst_pTsorted[:,1]>nconst_pTsorted[:,0],jets_pTsorted[:,0],jets_pTsorted[:,1])
        SUEP_cluster_tracks = ak.where(nconst_pTsorted[:,1]<=nconst_pTsorted[:,0],clusters_pTsorted[:,0],clusters_pTsorted[:,1])
        ISR_cluster_tracks = ak.where(nconst_pTsorted[:,1]>nconst_pTsorted[:,0],clusters_pTsorted[:,0],clusters_pTsorted[:,1])
        
        #####################################################################################
        # ---- Cluster Method (CL)
        # In this method, we use the tracks that were already clustered into the SUEP jet
        # to be the SUEP jet. Variables such as sphericity are calculated using these.
        #####################################################################################
        
        # no cut needed, but still define new variables for this method
        # so we don't mix things up
        SUEP_cand_CL = SUEP_cand
        ISR_cand_CL = ISR_cand
        tracks_CL = tracks
        
        # boost into frame of SUEP
        boost_SUEP = ak.zip({
            "px": SUEP_cand_CL.px*-1,
            "py": SUEP_cand_CL.py*-1,
            "pz": SUEP_cand_CL.pz*-1,
            "mass": SUEP_cand_CL.mass
        }, with_name="Momentum4D")        
        
        # SUEP tracks for this method are defined to be the ones from the cluster
        # that was picked to be the SUEP jet
        SUEP_tracks_b_CL = SUEP_cluster_tracks.boost_p4(boost_SUEP)        
        
        # SUEP jet variables
        eigs = sphericity(self, SUEP_tracks_b_CL,1.0) #Set r=1.0 for IRC safe
        out_vars["SUEP_nconst_CL"] = ak.num(SUEP_tracks_b_CL)
        out_vars["SUEP_S1_CL"] = 1.5 * (eigs[:,1]+eigs[:,0])
        
        tracks_b_CL = tracks_CL.boost_p4(boost_SUEP)      
        
        # event variables
        eigs = sphericity(self, tracks_b_CL,1.0) #Set r=1.0 for IRC safe
        out_vars["ntracks_b_CL"] = ak.num(SUEP_tracks_b_CL)
        out_vars["event_S1_CL"] = 1.5 * (eigs[:,1]+eigs[:,0])
        
        #####################################################################################
        # ---- Save outputs
        #####################################################################################
        
        # max number of tracks to save per event
        npfcands=1000
        
        # convert to this format
        l1event_feat = self.store_event_features(out_vars)
        l1pfcand_cyl, l1pfcand_cart = self.store_objects_coordinates(tracks_CL, npfcands, obj='PFcand')
        l1bpfcand_cyl, l1bpfcand_cart = self.store_objects_coordinates(tracks_b_CL, npfcands, obj='bPFcand')
        
        # save to file
        with h5py.File('out.hdf5', 'w') as outFile:
            outFile.create_dataset('Pfcand_cyl', data=l1pfcand_cyl, compression='gzip')
            outFile.create_dataset('Pfcand_cart', data=l1pfcand_cart, compression='gzip')
            outFile.create_dataset('bPfcand_cyl', data=l1bpfcand_cyl, compression='gzip')
            outFile.create_dataset('bPfcand_cart', data=l1bpfcand_cart, compression='gzip')
            outFile.create_dataset('event_feat', data=l1event_feat, compression='gzip')
            
        return output

    def postprocess(self, accumulator):
        return accumulator
    
    def store_event_features(self, df):
        '''store objects in zero-padded numpy arrays'''
        nentries = df.shape[0]
        l1Oevent_features = np.zeros((nentries,3))
        n_pfcand = df['ntracks'].to_numpy()
        eventBoosted_sphericity = df['event_S1_CL'].to_numpy()
        suepJetBoosted_sphericity = df['SUEP_S1_CL'].to_numpy()

        l1Oevent_features[:,0] = n_pfcand
        l1Oevent_features[:,1] = eventBoosted_sphericity
        l1Oevent_features[:,2] = suepJetBoosted_sphericity

        return l1Oevent_features

    def store_objects_coordinates(self, tracks, nobj=10, obj='FatJet_'):
        '''store objects in zero-padded numpy arrays'''
        
        # bad naming, tracks dimensions is (events x tracks)
        nentries = len(tracks)
        
        l1Obj_cyl = np.zeros((nentries,nobj,4))
        l1Obj_cart = np.zeros((nentries,nobj,4))

        pt = self.to_np_array(tracks.pt,maxN=nobj)
        eta = self.to_np_array(tracks.eta,maxN=nobj)
        phi = self.to_np_array(tracks.phi,maxN=nobj)
        m = self.to_np_array(tracks.mass,maxN=nobj)

        l1Obj_cyl[:,:,0] = pt
        l1Obj_cyl[:,:,1] = eta
        l1Obj_cyl[:,:,2] = phi
        l1Obj_cyl[:,:,3] = m
        l1Obj_cart[:,:,0] = pt*np.cos(phi)
        l1Obj_cart[:,:,1] = pt*np.sin(phi)
        l1Obj_cart[:,:,2] = pt*np.sinh(eta)
        l1Obj_cart[:,:,3] = m

        return l1Obj_cyl, l1Obj_cart
    
    def to_np_array(self, ak_array, maxN=100, pad=0):
        '''convert awkward array to regular numpy array'''
        return ak.to_numpy(ak.fill_none(ak.pad_none(ak_array,maxN,clip=True,axis=-1),pad))