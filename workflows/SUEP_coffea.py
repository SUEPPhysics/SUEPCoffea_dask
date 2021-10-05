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
    def __init__(self, isMC: int, era: int, xsec: float, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, flag: bool, output_location: Optional[str]) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.xsec = xsec
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

        #Prepare the clean track collection
        Cands = ak.zip({
            "pt": events.PFCands_trkPt,
            "eta": events.PFCands_trkEta,
            "phi": events.PFCands_trkPhi,
            "mass": events.PFCands_mass
        }, with_name="Momentum4D")
        cut = (events.PFCands_fromPV > 1) & (events.PFCands_trkPt >= 1) & (events.PFCands_trkEta <= 2.5)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
        cluster = fastjet.ClusterSequence(Cleaned_cands, jetdef) 
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt=150),"Momentum4D")
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt=150),"Momentum4D")
        col1 = pd.Series(ak.num(Cands).to_list(), name = "uncleaned_tracks")
        col2 = pd.Series(ak.num(Cleaned_cands).to_list(), name = "nCleaned_Cands")
        col3 = pd.Series(ak.num(ak_inclusive_jets).to_list(), name = "ngood_fastjets")

        #remove events without a cluster
        ak_inclusive_cluster = ak_inclusive_cluster[ak.num(ak_inclusive_jets, axis=1)>1]
        ak_inclusive_jets = ak_inclusive_jets[ak.num(ak_inclusive_jets, axis=1)>1]
        
        #SUEP_mult
        chonkocity = ak.num(ak_inclusive_cluster, axis=2)
        chonkiest_jet = ak.argsort(chonkocity, axis=1, ascending=True, stable=True)[:, ::-1] 
        thicc_jets = ak_inclusive_jets[chonkiest_jet]
        chonkiest_cands = ak_inclusive_cluster[chonkiest_jet][:,0]
        thicc_jets = thicc_jets[ak.num(chonkiest_cands)>1]#We dont want to look at single track jets
        chonkiest_cands = chonkiest_cands[ak.num(chonkiest_cands)>1]#We dont want to look at single track jets
        out_mult = thicc_jets[:,0]
        out_mult["SUEP_mult_nconst"] = ak.max(ak.num(ak_inclusive_cluster, axis=2),axis=1)
        out_mult["SUEP_mult_pt"] = thicc_jets[:,0].pt
        out_mult["SUEP_mult_eta"] = thicc_jets[:,0].eta
        out_mult["SUEP_mult_phi"] = thicc_jets[:,0].phi
        out_mult["SUEP_mult_mass"] = thicc_jets[:,0].mass

        #SUEP_mult boosting and sphericity
        boost_mult = ak.zip({
            "px": thicc_jets[:,0].px*-1,
            "py": thicc_jets[:,0].py*-1,
            "pz": thicc_jets[:,0].pz*-1,
            "mass": thicc_jets[:,0].mass
        }, with_name="Momentum4D")
        chonkiest_cands = chonkiest_cands.boost_p4(boost_mult)
        mult_eigs = self.sphericity(chonkiest_cands,2.0)  
        out_mult["SUEP_mult_spher"] = 1.5 * (mult_eigs[:,1]+mult_eigs[:,0])
        out_mult["SUEP_mult_aplan"] =  1.5 * mult_eigs[:,0]
        out_mult["SUEP_mult_FW2M"] = 1.0 - 3.0 * (mult_eigs[:,2]*mult_eigs[:,1] + mult_eigs[:,0]*mult_eigs[:,2] + mult_eigs[:,1]*mult_eigs[:,0])
        out_mult["SUEP_mult_D"] = 27.0 * mult_eigs[:,2]*mult_eigs[:,1]*mult_eigs[:,0]    

        #SUEP_pt
        highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
        SUEP_pt = ak_inclusive_jets[highpt_jet]
        SUEP_pt_nconst = chonkocity[highpt_jet]
        SUEP_pt_tracks = ak_inclusive_cluster[highpt_jet]
        highpt_cands = SUEP_pt_tracks[:,0]                  #tracks for highest pt
        SUEP_pt = SUEP_pt[ak.num(highpt_cands)>1]           #We dont want to look at single track jets
        SUEP_pt_nconst = SUEP_pt_nconst[ak.num(highpt_cands)>1]
        SUEP_pt_tracks = SUEP_pt_tracks[ak.num(highpt_cands)>1]
        highpt_cands = highpt_cands[ak.num(highpt_cands)>1] #We dont want to look at single track jets
      
        #Christos Method for ISR removal
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
        SUEP_cand = SUEP_cand.boost_p4(boost_ch)
        ISR_cand = ISR_cand.boost_p4(boost_ch)
        Christos_cands = Cleaned_cands[ak.num(ak_inclusive_jets)>1]
        Christos_cands = Christos_cands[ak.num(highpt_cands)>1]#remove the jets with one track
        Christos_cands = Christos_cands.boost_p4(boost_ch)
        Christos_cands = Christos_cands[abs(Christos_cands.deltaphi(ISR_cand)) > 1.6]
        Christos_cands = Christos_cands[ak.num(Christos_cands)>1]#remove the events left with one track
        ch_eigs = self.sphericity(Christos_cands,2.0)
        out_ch = ak.zip({"SUEP_ch_nconst":ak.num(Christos_cands)})
        out_ch["SUEP_ch_spher"] = 1.5 * (ch_eigs[:,1]+ch_eigs[:,0])
        out_ch["SUEP_ch_aplan"] = 1.5 * ch_eigs[:,0]
        out_ch["SUEP_ch_FW2M"] = 1.0 - 3.0 * (ch_eigs[:,2]*ch_eigs[:,1] + ch_eigs[:,2]*ch_eigs[:,0] + ch_eigs[:,1]*ch_eigs[:,0])
        out_ch["SUEP_ch_D"] = 27.0 * ch_eigs[:,2]*ch_eigs[:,1]*ch_eigs[:,0]

        # calculate some variables to plot
        SUEP_cand_tracks = SUEP_cand_tracks.boost_p4(boost_ch)
        ISR_cand_tracks = ISR_cand_tracks.boost_p4(boost_ch)
        dphi_ISRtracks_ISR = ak.flatten(abs(ISR_cand_tracks.deltaphi(ISR_cand)), axis=-1)
        dphi_SUEPtracks_ISR = ak.flatten(abs(SUEP_cand_tracks.deltaphi(ISR_cand)), axis=-1)

        # save plots to a dataframe
        col4 = pd.Series(dphi_SUEP_ISR.to_list(), name = "dphi_SUEP_ISR")
        col5 = pd.Series(dphi_ISRtracks_ISR.to_list(), name = "dphi_ISRtracks_ISR")
        col6 = pd.Series(dphi_SUEPtracks_ISR.to_list(), name = "dphi_SUEPtracks_ISR")
        out_vars = pd.concat([col1, col2, col3, col4, col5, col6], axis=1)

        #Prepare for writing to HDF5 file (xsec stored in metadata)
        fname = (events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".hdf5")
        subdirs = []
        store = pd.HDFStore(fname)
        if self.output_location is not None:

            # ak to pandas to hdf5
            for out, gname in [[out_mult, "mult"], [out_ch, "ch"]]:
                df = self.ak_to_pandas(out)
                metadata = dict(xsec=self.xsec,era=self.era,
                                mc=self.isMC,sample=self.sample)
                store_fin = self.h5store(store, df, fname, gname, **metadata)

            for df, gname in [[out_vars, "out_vars"]]:

                metadata = dict(xsec=self.xsec,era=self.era,
                                mc=self.isMC,sample=self.sample)
                store_fin = self.h5store(store, df, fname, gname, **metadata)

            store.close()
            self.dump_table(fname, self.output_location, subdirs)
        else:
            store.close()

        
        return output

    def postprocess(self, accumulator):
        return accumulator