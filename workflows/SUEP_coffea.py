"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer, 2021
"""

import os
import pathlib
import shutil
import warnings
import awkward as ak
import pandas
import numpy as np
import fastjet
import math
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
        self._accumulator = processor.dict_accumulator({
            "uncleaned_tracks": hist.Hist("Events",
                hist.Bin("Uncleaned_Ntracks", "Uncleaned NTracks", 10000, 0, 10000)),
            "nCleaned_Cands": hist.Hist("Events",
                hist.Bin("nCleaned_Cands", "NTracks", 200, 0, 200)),
            "ngood_fastjets" : hist.Hist("Events",
                hist.Bin("ngood_fastjets", "# Fastjets", 15, 0, 15)),
            "SUEP_mult_nconst" : hist.Hist("Events",
                hist.Bin("SUEP_mult_nconst", "# Tracks", 250, 0, 250)),
            "SUEP_mult_pt" : hist.Hist("Events",
                hist.Bin("SUEP_mult_pt", "pT", 100, 0, 2000)),
            "SUEP_mult_eta" : hist.Hist("Events",
                hist.Bin("SUEP_mult_eta", "eta", 100, -5, 5)),
            "SUEP_mult_phi" : hist.Hist("Events",
                hist.Bin("SUEP_mult_phi", "phi", 100, 0, 6.5)),
            "SUEP_mult_mass" : hist.Hist("Events",
                hist.Bin("SUEP_mult_mass", "mass", 150, 0, 4000)),
            "SUEP_mult_spher" : hist.Hist("Events",
                hist.Bin("SUEP_mult_spher", "Sphericity", 100, 0, 1)),
            "SUEP_mult_aplan" : hist.Hist("Events",
                hist.Bin("SUEP_mult_aplan", "Aplanarity", 100, 0, 1)),
            "SUEP_mult_FW2M" : hist.Hist("Events",
                hist.Bin("SUEP_mult_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)),
            "SUEP_mult_D" : hist.Hist("Events",
                hist.Bin("SUEP_mult_D", "D", 100, 0, 1)),
            "SUEP_pt_nconst" : hist.Hist("Events",
                hist.Bin("SUEP_pt_nconst", "# Tracks", 250, 0, 250)),
            "SUEP_pt_pt" : hist.Hist("Events",
                hist.Bin("SUEP_pt_pt", "pT", 100, 0, 2000)),
            "SUEP_pt_eta" : hist.Hist("Events",
                hist.Bin("SUEP_pt_eta", "eta", 100, -5, 5)),
            "SUEP_pt_phi" : hist.Hist("Events",
                hist.Bin("SUEP_pt_phi", "phi", 100, 0, 6.5)),
            "SUEP_pt_mass" : hist.Hist("Events",
                hist.Bin("SUEP_pt_mass", "mass", 150, 0, 4000)),
            "SUEP_pt_spher" : hist.Hist("Events",
                hist.Bin("SUEP_pt_spher", "Sphericity", 100, 0, 1)),  
            "SUEP_pt_aplan" : hist.Hist("Events",
                hist.Bin("SUEP_pt_aplan", "Aplanarity", 100, 0, 1)),  
            "SUEP_pt_FW2M" : hist.Hist("Events", 
                hist.Bin("SUEP_pt_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)),  
            "SUEP_pt_D" : hist.Hist("Events", 
                hist.Bin("SUEP_pt_D", "D", 100, 0, 1)),
            "SUEP_ch_nconst" : hist.Hist("Events",
                hist.Bin("SUEP_ch_nconst", "# Tracks", 250, 0, 250)),
            "SUEP_ch_spher" : hist.Hist("Events",
                hist.Bin("SUEP_ch_spher", "Sphericity", 100, 0, 1)),
            "SUEP_ch_aplan" : hist.Hist("Events",
                hist.Bin("SUEP_ch_aplan", "Aplanarity", 100, 0, 1)),
            "SUEP_ch_FW2M" : hist.Hist("Events",
                hist.Bin("SUEP_ch_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)),
            "SUEP_ch_D" : hist.Hist("Events",
                hist.Bin("SUEP_ch_D", "D", 100, 0, 1)),
            "A": hist.Hist(
                "Events", hist.Bin("A", "A", 100, 0, 1)),
            "B": hist.Hist(
                "Events", hist.Bin("B", "B", 100, 0, 1)),
            "C": hist.Hist(
                "Events", hist.Bin("C", "C", 100, 0, 1)),
            "D_exp": hist.Hist(
                "Events", hist.Bin("D_exp", "D_exp", 100, 0, 1)),
            "D_obs": hist.Hist(
                "Events", hist.Bin("D_obs", "D_obs", 100, 0, 1)),
        })

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

    def ak_to_pandas(self, jet_collection: ak.Array) -> pandas.DataFrame:
        output = pandas.DataFrame()
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

    def dump_pandas(self, pddf: pandas.DataFrame, fname: str, location: str, subdirs: Optional[List[str]] = None,) -> None:
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
        pddf.to_parquet(local_file)
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
                shutil.copy(local_file, destination)
            else:
                fname = "condor_" + fname
                destination = os.path.join(location, os.path.join(merged_subdirs, fname))
                shutil.copy(local_file, destination)
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
        output["uncleaned_tracks"].fill(Uncleaned_Ntracks = ak.num(Cands))
        output["nCleaned_Cands"].fill(nCleaned_Cands = ak.num(Cleaned_cands))

        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
        cluster = fastjet.ClusterSequence(Cleaned_cands, jetdef) 
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt=150),"Momentum4D")
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt=150),"Momentum4D")
        output["ngood_fastjets"].fill(ngood_fastjets = ak.num(ak_inclusive_jets))

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
        out_mult["xsec"] = [self.xsec] * len(thicc_jets[:,0])
        out_mult["SUEP_mult_nconst"] = ak.max(ak.num(ak_inclusive_cluster, axis=2),axis=1)
        out_mult["SUEP_mult_pt"] = thicc_jets[:,0].pt
        out_mult["SUEP_mult_eta"] = thicc_jets[:,0].eta
        out_mult["SUEP_mult_phi"] = thicc_jets[:,0].phi
        out_mult["SUEP_mult_mass"] = thicc_jets[:,0].mass
        output["SUEP_mult_nconst"].fill(SUEP_mult_nconst = out_mult["SUEP_mult_nconst"])
        output["SUEP_mult_pt"].fill(SUEP_mult_pt = out_mult["SUEP_mult_pt"])
        output["SUEP_mult_eta"].fill(SUEP_mult_eta = out_mult["SUEP_mult_eta"])
        output["SUEP_mult_phi"].fill(SUEP_mult_phi = out_mult["SUEP_mult_phi"])
        output["SUEP_mult_mass"].fill(SUEP_mult_mass = out_mult["SUEP_mult_mass"])

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
        output["SUEP_mult_spher"].fill(SUEP_mult_spher = out_mult["SUEP_mult_spher"])
        output["SUEP_mult_aplan"].fill(SUEP_mult_aplan = out_mult["SUEP_mult_aplan"])
        output["SUEP_mult_FW2M"].fill(SUEP_mult_FW2M = out_mult["SUEP_mult_FW2M"])
        output["SUEP_mult_D"].fill(SUEP_mult_D = out_mult["SUEP_mult_D"])       

        #SUEP_pt
        highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
        SUEP_pt = ak_inclusive_jets[highpt_jet]
        SUEP_pt_constituent = chonkocity[highpt_jet]
        highpt_cands = ak_inclusive_cluster[highpt_jet][:,0]
        SUEP_pt = SUEP_pt[ak.num(highpt_cands)>1]#We dont want to look at single track jets
        highpt_cands = highpt_cands[ak.num(highpt_cands)>1]#We dont want to look at single track jets
        output["SUEP_pt_nconst"].fill(SUEP_pt_nconst = SUEP_pt_constituent[:,0])
        output["SUEP_pt_pt"].fill(SUEP_pt_pt = SUEP_pt[:,0].pt)
        output["SUEP_pt_eta"].fill(SUEP_pt_eta = SUEP_pt[:,0].eta)
        output["SUEP_pt_phi"].fill(SUEP_pt_phi = SUEP_pt[:,0].phi)
        output["SUEP_pt_mass"].fill(SUEP_pt_mass = SUEP_pt[:,0].mass)

        #SUEP_pt boosting and sphericity
        boost_pt = ak.zip({
            "px": SUEP_pt[:,0].px*-1,
            "py": SUEP_pt[:,0].py*-1,
            "pz": SUEP_pt[:,0].pz*-1,
            "mass": SUEP_pt[:,0].mass
        }, with_name="Momentum4D")
        highpt_cands = highpt_cands.boost_p4(boost_pt)
        pt_eigs = self.sphericity(highpt_cands,2.0)
        output["SUEP_pt_spher"].fill(SUEP_pt_spher = 1.5 * (pt_eigs[:,1]+pt_eigs[:,0]))
        output["SUEP_pt_aplan"].fill(SUEP_pt_aplan = 1.5 * pt_eigs[:,0])
        output["SUEP_pt_FW2M"].fill(SUEP_pt_FW2M = 1.0 - 3.0 * (pt_eigs[:,2]*pt_eigs[:,1] + pt_eigs[:,2]*pt_eigs[:,0] + pt_eigs[:,1]*pt_eigs[:,0]))
        output["SUEP_pt_D"].fill(SUEP_pt_D = 27.0 * pt_eigs[:,2]*pt_eigs[:,1]*pt_eigs[:,0])

        #Christos Method for ISR removal
        SUEP_pt_constituent = SUEP_pt_constituent[ak.num(highpt_cands)>1]
        SUEP_cand = ak.where(SUEP_pt_constituent[:,1]<SUEP_pt_constituent[:,0],SUEP_pt[:,0],SUEP_pt[:,1])
        ISR_cand = ak.where(SUEP_pt_constituent[:,1]>SUEP_pt_constituent[:,0],SUEP_pt[:,0],SUEP_pt[:,1])
        boost_ch = ak.zip({
            "px": SUEP_cand.px*-1,
            "py": SUEP_cand.py*-1,
            "pz": SUEP_cand.pz*-1,
            "mass": SUEP_cand.mass
        }, with_name="Momentum4D")
        ISR_cand = ISR_cand.boost_p4(boost_ch)
        Christos_cands = Cleaned_cands[ak.num(ak_inclusive_jets)>1]
        Christos_cands = Christos_cands[ak.num(highpt_cands)>1]#remove the jets with one track
        Christos_cands = Christos_cands.boost_p4(boost_ch)
        Christos_cands = Christos_cands[abs(Christos_cands.deltaphi(ISR_cand)) > 1.6]
        Christos_cands = Christos_cands[ak.num(Christos_cands)>1]#remove the events left with one track
        ch_eigs = self.sphericity(Christos_cands,2.0)
        out_ch = Christos_cands[:,0]
        out_ch["xsec"] = [self.xsec] * len(Christos_cands[:,0])
        out_ch["SUEP_ch_nconst"] = ak.num(Christos_cands)
        out_ch["SUEP_ch_spher"] = 1.5 * (ch_eigs[:,1]+ch_eigs[:,0])
        out_ch["SUEP_ch_aplan"] = 1.5 * ch_eigs[:,0]
        out_ch["SUEP_ch_FW2M"] = 1.0 - 3.0 * (ch_eigs[:,2]*ch_eigs[:,1] + ch_eigs[:,2]*ch_eigs[:,0] + ch_eigs[:,1]*ch_eigs[:,0])
        out_ch["SUEP_ch_D"] = 27.0 * ch_eigs[:,2]*ch_eigs[:,1]*ch_eigs[:,0]
        output["SUEP_ch_nconst"].fill(SUEP_ch_nconst = out_ch["SUEP_ch_nconst"])
        output["SUEP_ch_spher"].fill(SUEP_ch_spher = out_ch["SUEP_ch_spher"])
        output["SUEP_ch_aplan"].fill(SUEP_ch_aplan = out_ch["SUEP_ch_aplan"])
        output["SUEP_ch_FW2M"].fill(SUEP_ch_FW2M = out_ch["SUEP_ch_FW2M"])
        output["SUEP_ch_D"].fill(SUEP_ch_D = out_ch["SUEP_ch_D"])

        #ABCD method plots
        SUEP_ch_spher = 1.5 * (ch_eigs[:,1]+ch_eigs[:,0])
        SUEP_ch_nconst = ak.num(Christos_cands)
        A_reg = (SUEP_ch_spher < 0.60) & (SUEP_ch_nconst < 150)
        B_reg = (SUEP_ch_spher >= 0.60) & (SUEP_ch_nconst < 150)
        C_reg = (SUEP_ch_spher < 0.60) & (SUEP_ch_nconst >= 150)
        D_reg = (SUEP_ch_spher >= 0.60) & (SUEP_ch_nconst >= 150)

        A_hist = SUEP_ch_spher[A_reg]
        B_hist = SUEP_ch_spher[B_reg]
        C_hist = SUEP_ch_spher[C_reg]
        D_observed = SUEP_ch_spher[D_reg]

        output["A"].fill(A = A_hist)
        output["B"].fill(B = B_hist)
        output["C"].fill(C = C_hist)
        output["D_obs"].fill(D_obs = D_observed)

        if ak.size(A_hist)>0.0:
            CoverA =  ak.size(C_hist) /  ak.size(A_hist)
        else:
            CoverA = 0.0
            print("A region has no occupancy")
        D_expected = B_hist
        output["D_exp"].fill(D_exp = D_expected)
        output["D_exp"].scale(CoverA)

        if self.output_location is not None:

            for out, label in [[out_mult, "mult"], [out_ch, "ch"]]:
                df = self.ak_to_pandas(out)
                fname = (
                    events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                    + "_" + label + ".parquet"
                )
                subdirs = []
                #if "dataset" in events.metadata:
                #    subdirs.append(events.metadata["dataset"])
                self.dump_pandas(df, fname, self.output_location, subdirs)


        return output

    def postprocess(self, accumulator):
        return accumulator