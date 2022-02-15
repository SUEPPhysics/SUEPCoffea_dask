"""
SUEP_coffea.py
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
                    #metadata.update({gensumweight:self.gensumweight})
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

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        if self.isMC: self.gensumweight = ak.sum(events.genWeight)

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
            (abs(events.lostTracks.eta) <= 1.0) \
            & (abs(events.lostTracks.dz) < 10) & \
            (events.lostTracks.dzErr < 0.05)
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)
     
        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
        Total_Tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)
        tracks = Total_Tracks
        minPt = 150
        
        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)        
        cluster = fastjet.ClusterSequence(tracks, jetdef)
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt= minPt),"Momentum4D")  
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt= minPt),"Momentum4D")
                        
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
        
        # save variables to a dataframe
        col1 = pd.Series(ak.num(Cands).to_list(), name = "uncleaned_tracks")
        col2 = pd.Series(ak.num(tracks).to_list(), name = "ntracks")
        col3 = pd.Series(ak.num(ak_inclusive_jets).to_list(), name = "ngood_fastjets")
        col4 = pd.Series(ak.sum(ak4jets.pt,axis=-1).to_list(), name = "ht")
        col5 = pd.Series(ak.num(Lost_Tracks_cands).to_list(), name = "nLostTracks")
        col6 = pd.Series(events.HLT.PFJet500.to_list(), name="HLT_PFJet500")
        col7 = pd.Series(events.HLT.PFHT1050.to_list(), name="HLT_PFHT1050")
        col8 = pd.Series(ak.num(ak4jets).to_list(), name = "ngood_ak4jets")
        out_vars = pd.concat([col1, col2, col3, col4, col5, col6, col7, col8], axis=1)
        
        # indices of events in tracks, used to keep track which events pass the selections
        indices = np.arange(0,len(tracks))
         
        # remove events that fail the HT cut
        #trigger = ((out_vars['HLT_PFJet500'] == 1) | (out_vars['HLT_PFHT1050'] == 1))
        trigger = ((out_vars['HLT_PFHT1050'] == 1))
        htCut = ((out_vars['ht'] > 1200) & (trigger))    
        ak_inclusive_cluster = ak_inclusive_cluster[htCut]
        ak_inclusive_jets = ak_inclusive_jets[htCut]
        tracks = tracks[htCut]
        indices = indices[htCut]
        Lost_Tracks_cands = Lost_Tracks_cands[htCut]
           
        # remove events without a cluster
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        Lost_Tracks_cands = Lost_Tracks_cands[clusterCut]
 
        # output an empty file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass the selections. Saving empty outputs.")
            out_ch = pd.DataFrame(['empty'], columns=['empty'])
            self.save_dfs([out_ch, out_vars],["ch","vars"])
            return output
        
        ### SUEP_mult
        chonkocity = ak.num(ak_inclusive_cluster, axis=2)
        #chonkiest_jet = ak.argsort(chonkocity, axis=1, ascending=True, stable=True)[:, ::-1]
        #thicc_jets = ak_inclusive_jets[chonkiest_jet]
        #chonkiest_cands = ak_inclusive_cluster[chonkiest_jet][:,0]
        #singletrackCut = (ak.num(chonkiest_cands)>1)
        
        # account for no events passing our selections
#         if not any(singletrackCut): 
#             print("No events in Multiplicity Method.")
#             out_mult = pd.DataFrame()
#         else:            
#             #cut events with single track highest mult jets
#             thicc_jets = thicc_jets[singletrackCut]
#             chonkiest_cands = chonkiest_cands[singletrackCut]
#             tracks_mult = tracks[singletrackCut]
#             indices_mult = indices[singletrackCut]
        
#             out_mult = thicc_jets[:,0]
#             out_mult["event_index_mult"] = indices_mult
#             out_mult["SUEP_ntracks_mult"] = ak.num(tracks_mult, axis=1)
#             out_mult["SUEP_nconst_mult"] = ak.num(chonkiest_cands, axis=1)
#             out_mult["SUEP_pt_mult"] = thicc_jets[:,0].pt
#             out_mult["SUEP_pt_avg_mult"] = ak.mean(chonkiest_cands.pt, axis=-1)
#             out_mult["SUEP_eta_mult"] = thicc_jets[:,0].eta
#             out_mult["SUEP_phi_mult"] = thicc_jets[:,0].phi
#             out_mult["SUEP_mass_mult"] = thicc_jets[:,0].mass
#             deltaR = chonkiest_cands.deltaR(thicc_jets[:,0])
#             out_mult["SUEP_girth_mult"] = ak.sum((deltaR/(1.5))*chonkiest_cands.pt/thicc_jets[:,0].pt, axis=-1)
#             out_mult["SUEP_rho0_mult"] = self.rho(0, thicc_jets[:,0], chonkiest_cands, deltaR)
#             out_mult["SUEP_rho1_mult"] = self.rho(1, thicc_jets[:,0], chonkiest_cands, deltaR)

#             #SUEP_mult boosting, sphericity and rho
#             boost_mult = ak.zip({
#                 "px": thicc_jets[:,0].px*-1,
#                 "py": thicc_jets[:,0].py*-1,
#                 "pz": thicc_jets[:,0].pz*-1,
#                 "mass": thicc_jets[:,0].mass
#             }, with_name="Momentum4D")
#             chonkiest_cands = chonkiest_cands.boost_p4(boost_mult)
#             mult_eigs = self.sphericity(chonkiest_cands,2.0)  
#             out_mult["SUEP_pt_avg_b_mult"] = ak.mean(chonkiest_cands.pt, axis=-1)
#             out_mult["SUEP_spher_mult"] = 1.5 * (mult_eigs[:,1]+mult_eigs[:,0])
#             out_mult["SUEP_aplan_mult"] =  1.5 * mult_eigs[:,0]
#             out_mult["SUEP_FW2M_mult"] = 1.0 - 3.0 * (mult_eigs[:,2]*mult_eigs[:,1] + mult_eigs[:,0]*mult_eigs[:,2] + mult_eigs[:,1]*mult_eigs[:,0])
#             out_mult["SUEP_D_mult"] = 27.0 * mult_eigs[:,2]*mult_eigs[:,1]*mult_eigs[:,0]


        ### SUEP_pt
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
        tracks_ch = tracks[singletrackCut]
        indices_ch = indices[singletrackCut]
        Lost_Tracks_cands = Lost_Tracks_cands[singletrackCut]

        # ISR removal method
        SUEP_cand = ak.where(SUEP_pt_nconst[:,1]<=SUEP_pt_nconst[:,0],SUEP_pt[:,0],SUEP_pt[:,1])
        SUEP_cand_tracks = ak.where(SUEP_pt_nconst[:,1]<=SUEP_pt_nconst[:,0],SUEP_pt_tracks[:,0],SUEP_pt_tracks[:,1])
        ISR_cand = ak.where(SUEP_pt_nconst[:,1]>SUEP_pt_nconst[:,0],SUEP_pt[:,0],SUEP_pt[:,1])
        ISR_cand_tracks = ak.where(SUEP_pt_nconst[:,1]>SUEP_pt_nconst[:,0],SUEP_pt_tracks[:,0],SUEP_pt_tracks[:,1])
        dphi_SUEP_ISR = abs(SUEP_cand.deltaphi(ISR_cand))
        boost_ch = ak.zip({
            "px": SUEP_cand.px*-1,
            "py": SUEP_cand.py*-1,
            "pz": SUEP_cand.pz*-1,
            "mass": SUEP_cand.mass
        }, with_name="Momentum4D")
        ISR_cand_b = ISR_cand.boost_p4(boost_ch)
        tracks_ch = tracks_ch.boost_p4(boost_ch)
        Lost_Tracks_ch = Lost_Tracks_cands.boost_p4(boost_ch)
        Christos_cands = tracks_ch[abs(tracks_ch.deltaphi(ISR_cand_b)) > 1.6]
        Lost_Christos_cands = Lost_Tracks_ch[abs(Lost_Tracks_ch.deltaphi(ISR_cand_b)) > 1.6]
        onechtrackCut = (ak.num(Christos_cands)>1)
        
        # account for no events passing our selections
        if not any(onechtrackCut):
            print("No events in ISR Removal Method.")
            out_ch = pd.DataFrame()
        else:
            Christos_cands = Christos_cands[onechtrackCut]#remove the events left with one track
            tracks_ch = tracks_ch[onechtrackCut]
            SUEP_cand = SUEP_cand[onechtrackCut]
            ISR_cand = ISR_cand[onechtrackCut]
            ISR_cand_b = ISR_cand_b[onechtrackCut]
            SUEP_cand_tracks = SUEP_cand_tracks[onechtrackCut]
            ISR_cand_tracks = ISR_cand_tracks[onechtrackCut]
            boost_ch = boost_ch[onechtrackCut]
            indices_ch = indices_ch[onechtrackCut]
            Lost_Christos_cands = Lost_Christos_cands[onechtrackCut]
            
            out_ch = SUEP_cand
            out_ch["event_index_ch"] = indices_ch
            out_ch["SUEP_nLostTracks_ch"] = ak.num(Lost_Christos_cands)
            out_ch["SUEP_pt_ch"] = SUEP_cand.pt
            out_ch["SUEP_eta_ch"] = SUEP_cand.eta
            out_ch["SUEP_phi_ch"] = SUEP_cand.phi
            out_ch["SUEP_mass_ch"] = SUEP_cand.mass
            out_ch["SUEP_dphi_SUEP_ISR_ch"] = ak.mean(abs(SUEP_cand.deltaphi(ISR_cand)), axis=-1)
            ch_eigs = self.sphericity(Christos_cands,2.0)
            out_ch["SUEP_nconst_ch"] = ak.num(Christos_cands)
            out_ch["SUEP_ntracks_ch"] = ak.num(tracks_ch)
            out_ch["SUEP_pt_avg_b_ch"] = ak.mean(Christos_cands.pt, axis=-1)
            out_ch["SUEP_spher_ch"] = 1.5 * (ch_eigs[:,1]+ch_eigs[:,0])
            out_ch["SUEP_aplan_ch"] = 1.5 * ch_eigs[:,0]
            out_ch["SUEP_FW2M_ch"] = 1.0 - 3.0 * (ch_eigs[:,2]*ch_eigs[:,1] + ch_eigs[:,2]*ch_eigs[:,0] + ch_eigs[:,1]*ch_eigs[:,0])
            out_ch["SUEP_D_ch"] = 27.0 * ch_eigs[:,2]*ch_eigs[:,1]*ch_eigs[:,0]
            out_ch["SUEP_dphi_chcands_ISR_ch"] = ak.mean(abs(Christos_cands.deltaphi(ISR_cand_b)), axis=-1)
            out_ch["SUEP_dphi_ISRtracks_ISR_ch"] = ak.mean(abs(ISR_cand_tracks.boost_p4(boost_ch).deltaphi(ISR_cand_b)), axis=-1)
            out_ch["SUEP_dphi_SUEPtracks_ISR_ch"] = ak.mean(abs(SUEP_cand_tracks.boost_p4(boost_ch).deltaphi(ISR_cand_b)), axis=-1)    

            # unboost for these
            Christos_cands_ub = Christos_cands.boost_p4(SUEP_cand)
            deltaR = Christos_cands_ub.deltaR(SUEP_cand)
            out_ch["SUEP_pt_avg_ch"] = ak.mean(Christos_cands_ub.pt, axis=-1)
            out_ch["SUEP_girth_ch"] = ak.sum((deltaR/1.5)*Christos_cands_ub.pt, axis=-1)/SUEP_cand.pt
            out_ch["SUEP_rho0_ch"] = self.rho(0, SUEP_cand, Christos_cands_ub, deltaR)
            out_ch["SUEP_rho1_ch"] = self.rho(1, SUEP_cand, Christos_cands_ub, deltaR)

            
        ### save outputs
        # ak to pandas, if needed
        #if not isinstance(out_mult, pd.DataFrame): out_mult = self.ak_to_pandas(out_mult)
        if not isinstance(out_ch, pd.DataFrame): out_ch = self.ak_to_pandas(out_ch)
        
        # pandas to hdf5 file
        #self.save_dfs([out_ch, out_mult, out_vars],["ch","mult","vars"])
        self.save_dfs([out_ch, out_vars],["ch","vars"])

        return output

    def postprocess(self, accumulator):
        return accumulator
