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
import onnxruntime as ort

# temporary
import os, psutil
vector.register_awkward()

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

        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator
    
    def softmax(self, data):
        # some numpy magic
        return np.exp(data)/(np.exp(data).sum(axis=-1)[:,:,None])

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

    def process_images(self, events, ort_sess):

        #Set up the image size and pixels
        eta_pix = 280
        phi_pix = 360
        eta_span = (-2.5, 2.5)
        phi_span = (-np.pi, np.pi)
        eta_scale = eta_pix/(eta_span[1]-eta_span[0])
        phi_scale = phi_pix/(phi_span[1]-phi_span[0])

        #Turn the PFcand info into indexes on the image map
        idx_eta = ak.values_astype(np.floor((events.eta-eta_span[0])*eta_scale),"int64")
        idx_phi = ak.values_astype(np.floor((events.phi-phi_span[0])*phi_scale),"int64")
        idx_eta = ak.where(idx_eta == eta_pix, eta_pix-1, idx_eta)
        idx_phi = ak.where(idx_phi == phi_pix, phi_pix-1, idx_phi)
        pt = events.pt

        #Running the inference in batch mode
        input_name = ort_sess.get_inputs()[0].name
        cl_outputs = np.array([])
        
        for event_i in range(len(events)):
            
            # form image
            to_infer = np.zeros((1, eta_pix, phi_pix))
            to_infer[0,idx_eta[event_i],idx_phi[event_i]] = pt[event_i]  
            
            # normalize pt
            m = np.mean(to_infer[0,:,:])
            s = np.std(to_infer[0,:,:])
            if s != 0: 
                to_infer[0,:,:] = (to_infer[0,:,:]-m)/s
           
            # SSD: grab classification outputs (0 - loc, 1 - classifation, 2 - regression)
            # resnet: only classification as output
            cl_output =  ort_sess.run(None, {input_name: np.array([to_infer.astype(np.float32)])})
            cl_output_softmax = self.softmax(cl_output)[0]
            if event_i == 0: 
                cl_outputs = cl_output_softmax
            else: 
                cl_outputs = np.concatenate((cl_outputs, cl_output_softmax))
                                
        return cl_outputs

    def rho(self, number, jet, tracks, deltaR, dr=0.05):
        r_start = number*dr
        r_end = (number+1)*dr
        ring = (deltaR > r_start) & (deltaR < r_end)
        rho_values = ak.sum(tracks[ring].pt, axis=1)/(dr*jet.pt)
        return rho_values

    def ak_to_pandas(self, jet_collection: ak.Array) -> pd.DataFrame:
        out_df = pd.DataFrame()
        for field in ak.fields(jet_collection):
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(jet_collection[field]):
                    out_df[f"{prefix}_{subfield}"] = ak.to_numpy(
                        jet_collection[field][subfield]
                    )
            else:
                out_df[field] = ak.to_numpy(jet_collection[field])
        return out_df

    def h5store(self, store: pd.HDFStore, df: pd.DataFrame, fname: str, gname: str, **kwargs: float) -> None:
        store.put(gname, df)
        store.get_storer(gname).attrs.metadata = kwargs
        
    def save_dfs(self, dfs, df_names, fname):
        #fname = "out.hdf5"
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
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt= minPt),"Momentum4D") 
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt= minPt),"Momentum4D")
                        
        # cut based on ak4 jets to replicate the trigCger
        Jets = ak.zip({
            "pt": events.Jet.pt,
            "eta": events.Jet.eta,
            "phi": events.Jet.phi,
            "mass": events.Jet.mass,
            "jetId": events.Jet.jetId
        })
        jetCut = (Jets.pt > 30) & (abs(Jets.eta)<4.7)
        ak4jets = Jets[jetCut]
        
        # from https://twiki.cern.ch/twiki/bin/view/CMS/JetID:
        # jetId==2 means: pass tight ID, fail tightLepVeto
        # jetId==6 means: pass tight and tightLepVeto ID. 
        tightJetId = (ak4jets.jetId > 2)
        tight_ak4jets = ak4jets[tightJetId]
        looseJetId = (ak4jets.jetId >= 2)
        loose_ak4jets = ak4jets[looseJetId]
        
        # save per event variables to a dataframe
        out_vars = pd.DataFrame()
        out_vars["uncleaned_tracks"] = ak.num(Cands).to_list()
        out_vars["ntracks"] = ak.num(tracks).to_list()
        out_vars["ngood_fastjets"] = ak.num(ak_inclusive_jets).to_list()
        out_vars["ht"] = ak.sum(ak4jets.pt,axis=-1).to_list()
        out_vars["nLostTracks"] = ak.num(Lost_Tracks_cands).to_list()
        if self.era == 2016:
            out_vars["HLT_PFHT900"] = events.HLT.PFHT900.to_list()
        else:
            out_vars["HLT_PFHT1050"] = events.HLT.PFHT1050.to_list()
        oneAk4jet = (ak.num(ak4jets) >= 1)
        out_vars["eta_ak4jets1"] = [x[0] if i else -100 for i, x in zip(oneAk4jet, ak4jets.eta)]
        out_vars["phi_ak4jets1"] = [x[0] if i else -100 for i, x in zip(oneAk4jet, ak4jets.phi)]
        twoAk4jets = (ak.num(ak4jets) >= 2)
        out_vars["eta_ak4jets2"] = [x[1] if i else -100 for i, x in zip(twoAk4jets, ak4jets.eta)]
        out_vars["phi_ak4jets2"] = [x[1] if i else -100 for i, x in zip(twoAk4jets, ak4jets.phi)]
        out_vars["ngood_ak4jets"] = ak.num(ak4jets).to_list()
        out_vars["n_loose_ak4jets"] = ak.num(loose_ak4jets).to_list()
        out_vars["n_tight_ak4jets"] = ak.num(tight_ak4jets).to_list()
        out_vars["ht_loose"] = ak.sum(loose_ak4jets.pt,axis=-1).to_list()
        out_vars["ht_tight"] = ak.sum(tight_ak4jets.pt,axis=-1).to_list()
        out_vars["PV_npvs"] = events.PV.npvs
        out_vars["PV_npvsGood"] = events.PV.npvsGood
         
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
            self.save_dfs([out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
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
            options.intra_op_num_threads = 1 # number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose.
            options.inter_op_num_threads = 1 # number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.
            
            if ak.any(htCut): 
                inf_cands = Cleaned_cands[htCut]
                resnet_jets = self.process_images(inf_cands, ort_sess)
                                
                # highest SUEP prediction per event
                SUEP_pred = resnet_jets[:,1]
                
        out_vars["resnet_SUEP_pred"] = SUEP_pred
                
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
            self.save_dfs([out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
            return output
        
        ### SUEP_mult
        chonkocity = ak.num(ak_inclusive_cluster, axis=2)
        #chonkiest_jet = ak.argsort(chonkocity, axis=1, ascending=True, stable=True)[:, ::-1]
        #thicc_jets = ak_inclusive_jets[chonkiest_jet]
        #chonkiest_cands = ak_inclusive_cluster[chonkiest_jet][:,0]
        #singletrackCut = (ak.num(chonkiest_cands)>1)

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
        tracks_IRM = tracks[singletrackCut]
        indices_IRM = indices[singletrackCut]
        Lost_Tracks_cands = Lost_Tracks_cands[singletrackCut]

        # ISR removal method (IRM)
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
            self.save_dfs([out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
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
        out_vars.loc[indices_IRM, "SUEP_pt_IRM"] = SUEP_cand.pt     
        out_vars.loc[indices_IRM, "SUEP_eta_IRM"] = SUEP_cand.eta
        out_vars.loc[indices_IRM, "SUEP_phi_IRM"] = SUEP_cand.phi
        out_vars.loc[indices_IRM, "SUEP_mass_IRM"] = SUEP_cand.mass
        out_vars.loc[indices_IRM, "SUEP_dphi_SUEP_ISR_IRM"] = ak.mean(abs(SUEP_cand.deltaphi(ISR_cand)), axis=-1)
        eigs_2 = self.sphericity(IRM_cands,2.0)
        eigs = self.sphericity(IRM_cands,1.0)
        out_vars.loc[indices_IRM, "SUEP_nconst_IRM"] = ak.num(IRM_cands)
        out_vars.loc[indices_IRM, "SUEP_ntracks_IRM"] = ak.num(tracks_IRM)
        out_vars.loc[indices_IRM, "SUEP_pt_avg_b_IRM"] = ak.mean(IRM_cands.pt, axis=-1)
        out_vars.loc[indices_IRM, "SUEP_spher_IRM"] = 1.5 * (eigs_2[:,1]+eigs_2[:,0])
        out_vars.loc[indices_IRM, "SUEP_S1_IRM"] = 1.5 * (eigs[:,1]+eigs[:,0])
        #out_vars.loc[indices_IRM, "SUEP_aplan_IRM"] = 1.5 * eigs_2[:,0]
        #out_vars.loc[indices_IRM, "SUEP_FW2M_IRM"] = 1.0 - 3.0 * (eigs_2[:,2]*eigs_2[:,1] + eigs_2[:,2]*eigs_2[:,0] + eigs_2[:,1]*eigs_2[:,0])
        #out_vars.loc[indices_IRM, "SUEP_D_IRM"] = 27.0 * eigs_2[:,2]*eigs_2[:,1]*eigs_2[:,0]
        #out_vars.loc[indices_IRM, "SUEP_dphi_IRMcands_ISR_IRM"] = ak.mean(abs(IRM_cands.deltaphi(ISR_cand_b)), axis=-1)
        #out_vars.loc[indices_IRM, "SUEP_dphi_ISRtracks_ISR_IRM"] = ak.mean(abs(ISR_cand_tracks.boost_p4(boost_IRM).deltaphi(ISR_cand_b)), axis=-1)
        #out_vars.loc[indices_IRM, "SUEP_dphi_SUEPtracks_ISR_IRM"] = ak.mean(abs(SUEP_cand_tracks.boost_p4(boost_IRM).deltaphi(ISR_cand_b)), axis=-1)    

        # unboost for these
        IRM_cands_ub = IRM_cands.boost_p4(SUEP_cand)
        deltaR = IRM_cands_ub.deltaR(SUEP_cand)
        out_vars.loc[indices_IRM, "SUEP_pt_avg_IRM"] = ak.mean(IRM_cands_ub.pt, axis=-1)
        out_vars.loc[indices_IRM, "SUEP_girth_IRM"] = ak.sum((deltaR/1.5)*IRM_cands_ub.pt, axis=-1)/SUEP_cand.pt
        out_vars.loc[indices_IRM, "SUEP_rho0_IRM"] = self.rho(0, SUEP_cand, IRM_cands_ub, deltaR)
        out_vars.loc[indices_IRM, "SUEP_rho1_IRM"] = self.rho(1, SUEP_cand, IRM_cands_ub, deltaR)

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
        self.save_dfs([out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
        
        return output

    def postprocess(self, accumulator):
        return accumulator
