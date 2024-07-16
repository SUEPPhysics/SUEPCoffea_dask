import os
import pathlib
import shutil
from typing import List, Optional

import awkward as ak
import coffea
import pandas as pd


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


def h5store(
    self, store: pd.HDFStore, df: pd.DataFrame, fname: str, gname: str, **kwargs: float
) -> None:
    store.put(gname, df)
    store.get_storer(gname).attrs.metadata = kwargs


def save_dfs(self, dfs, df_names, fname="out.hdf5", metadata=None, mode='w'):
    store = pd.HDFStore(fname, mode=mode)
    
    # pandas to hdf5
    for out, gname in zip(dfs, df_names):
        if metadata is None:
            if self.isMC:
                metadata = dict(
                    gensumweight=self.gensumweight,
                    era=self.era,
                    mc=self.isMC,
                    sample=self.sample,
                )
            else:
                metadata = dict(era=self.era, mc=self.isMC, sample=self.sample)

        store_fin = h5store(self, store, out, fname, gname, **metadata)

    store.close()
        

def format_dataframe(dataframe: pd.DataFrame):
    """
    Applies some formatting to efficiently store the data
    """
    for key, value in dataframe.items():
        # hdf5 doesn't store well coffea accumulators, and we don't need them anymore, so convert them to their values
        if type(value) == coffea.processor.accumulator.value_accumulator:
            dataframe[key] = value.value
    return dataframe


def format_metadata(metadata):
    """
    Applies some formatting to efficiently store the metadata
    """
    for key in metadata.keys():
        if type(metadata[key]) == coffea.processor.accumulator.value_accumulator:
            metadata[key] = metadata[key].value
    return metadata
