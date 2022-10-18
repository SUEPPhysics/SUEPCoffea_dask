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
import workflows.SUEP_utils 

class ML_cluster(processor.ProcessorABC):
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
        
        # apply trigger
        events = SUEP_utils.eventSelection(self, events)
        
        # cut based on ak4 jets to replicate the trigger
        ak4jets = SUEP_utils.jet_awkward(self, events.Jet)    
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
            outFile = events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5"
            with h5py.File(outFile, 'w') as outFile:
                outFile.create_dataset('empty', data=['empty'], compression='gzip')
            return output
        
        #####################################################################################
        # ---- Track selection
        #####################################################################################

        #Prepare the clean PFCand matched to tracks collection      
        if self.scouting == 1: tracks, _ = SUEP_utils.getScoutingTracks(self, events)
        else: tracks, _ = SUEP_utils.getTracks(self, events)
        
        #####################################################################################
        # ---- FastJet reclustering
        #####################################################################################
        
        ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(self, tracks, r=1.5, minPt=150)
        
        #####################################################################################
        # ---- Event level information
        #####################################################################################
            
        # save per event variables to a dataframe
        out_vars = pd.DataFrame()
        out_vars["ntracks"] = ak.num(tracks).to_list()
        out_vars["ngood_fastjets"] = ak.num(ak_inclusive_jets).to_list()
        out_vars["ht"] = ak.sum(ak4jets.pt,axis=-1).to_list()
        out_vars["ngood_ak4jets"] = ak.num(ak4jets).to_list()

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0,len(tracks))
                
        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################
        
        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        columns = ["SUEP_nconst_CL", "SUEP_S1_CL"]
                
        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        
        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            outFile = events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5"
            with h5py.File(outFile, 'w') as outFile:
                outFile.create_dataset('empty', data=['empty'], compression='gzip')
            return output
        
        # drop events that don't pass the cut
        out_vars = out_vars[out_vars['ngood_fastjets']>1]
        assert len(out_vars) == len(tracks)
        indices = np.arange(0, len(tracks))
        
        tracks, indices, topTwoJets = SUEP_utils.getTopTwoJets(self, tracks, indices, ak_inclusive_jets, ak_inclusive_cluster)
        SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets
        
        # drop events that don't pass the cut
        out_vars = out_vars.iloc[indices, :]
        indices = np.arange(0, len(tracks))
        
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
        eigs = sphericity(SUEP_tracks_b_CL,1.0) #Set r=1.0 for IRC safe
        out_vars["SUEP_nconst_CL"] = ak.num(SUEP_tracks_b_CL)
        out_vars["SUEP_S1_CL"] = 1.5 * (eigs[:,1]+eigs[:,0])
        
        tracks_b_CL = tracks_CL.boost_p4(boost_SUEP)      
        
        # event variables
        eigs = sphericity(tracks_b_CL,1.0) #Set r=1.0 for IRC safe
        out_vars["event_S1_CL"] = 1.5 * (eigs[:,1]+eigs[:,0])
        
        #####################################################################################
        # ---- Save outputs
        #####################################################################################
        
        # max number of tracks to save per event
        npfcands=1000
        
        # convert to this format
        l1event_feat = self.store_event_features(out_vars)
        
        l1pfcand_cyl = convert_cyl(tracks_CL, npfcands)
        l1pfcand_cart = convert_cart(tracks_CL, npfcands)
        l1pfcand_p4 = convert_p4(tracks_CL, npfcands)
        
        l1bpfcand_cyl = convert_cyl(tracks_b_CL, npfcands)
        l1bpfcand_cart = convert_cart(tracks_b_CL, npfcands)
        l1bpfcand_p4 = convert_p4(tracks_b_CL, npfcands)

        # save to file
        outFile = events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5"
        with h5py.File(outFile, 'w') as outFile:
            outFile.create_dataset('Pfcand_cyl', data=l1pfcand_cyl, compression='gzip')
            outFile.create_dataset('Pfcand_cart', data=l1pfcand_cart, compression='gzip')
            outFile.create_dataset('Pfcand_p4', data=l1pfcand_p4, compression='gzip')
            outFile.create_dataset('bPfcand_cyl', data=l1bpfcand_cyl, compression='gzip')
            outFile.create_dataset('bPfcand_cart', data=l1bpfcand_cart, compression='gzip')
            outFile.create_dataset('bPfcand_p4', data=l1bpfcand_p4, compression='gzip')
            outFile.create_dataset('event_feat', data=l1event_feat, compression='gzip')
            
        return output

    def postprocess(self, accumulator):
        return accumulator
    
    def store_event_features(self, df):
        '''store objects in zero-padded numpy arrays'''
        nentries = df.shape[0]
        l1Oevent_features = np.zeros((nentries, 4))
        n_pfcand = df['ntracks'].to_numpy()
        eventBoosted_sphericity = df['event_S1_CL'].to_numpy()
        suepJetBoosted_sphericity = df['SUEP_S1_CL'].to_numpy()
        suepJetBoosted_nconst = df['SUEP_nconst_CL'].to_numpy()

        l1Oevent_features[:,0] = n_pfcand
        l1Oevent_features[:,1] = eventBoosted_sphericity
        l1Oevent_features[:,2] = suepJetBoosted_sphericity
        l1Oevent_features[:,3] = suepJetBoosted_nconst

        return l1Oevent_features