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

class generator(processor.ProcessorABC):
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

    def ak_to_pandas_lists(self, collection: ak.Array, fields = None) -> pd.DataFrame:
        output = pd.DataFrame()
        if fields is None: fields = ak.fields(collection)
        for field in fields:
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(collection[field]):
                    output[f"{prefix}_{subfield}"] = collection[field][subfield].to_list()
                    
            else:
                output[field] = collection[field].to_list()
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
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")
        cut = (events.PFCands.fromPV > 1) & (events.PFCands.trkPt >= 1) & (events.PFCands.trkEta <= 2.5)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
        cluster = fastjet.ClusterSequence(Cleaned_cands, jetdef)

        Jets = ak.zip({
                "pt": events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
                "jetId": events.Jet.jetId
            })
        jetCut = (Jets.jetId) & (Jets.pt > 30) & (abs(Jets.eta)<4.7)
        ak4jets = Jets[jetCut]

        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt=150),"Momentum4D")
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt=150),"Momentum4D")
        
        # remove events that fail the HT cut
        ht = ak.sum(ak4jets.pt,axis=-1)
        htCut = (ht > 1200)
        ak_inclusive_cluster = ak_inclusive_cluster[htCut]
        ak_inclusive_jets = ak_inclusive_jets[htCut]
        Cleaned_cands = Cleaned_cands[htCut]

        # remove events without a cluster
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        Cleaned_cands = Cleaned_cands[clusterCut]

        # prepare outputs
        out_cands = Cleaned_cands
        out_labels = ak.zip({
                "pt": ak_inclusive_jets.pt,
                "eta": ak_inclusive_jets.eta,
                "phi": ak_inclusive_jets.phi
        })

        #Prepare for writing to HDF5 file (xsec stored in metadata)
        #fname = (events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".hdf5")
        fname = "out.hdf5"
        subdirs = []
        store = pd.HDFStore(fname)
        if self.output_location is not None:

            # ak to pandas to hdf5
            for out, gname in [[out_cands, "cands"], [out_labels, "labels"]]:
                df = self.ak_to_pandas_lists(out, ['eta', 'phi', 'pt'])
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
