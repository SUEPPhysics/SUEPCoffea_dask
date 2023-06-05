import pandas as pd
from rich import pretty


def h5load(ifile, label):
    with pd.HDFStore(ifile, "r") as store:
        data = store[label]
        metadata = store.get_storer(label).attrs.metadata
        return data, metadata


f = "condor_dataset.hdf5"
f_old = "../old_stuff/SUEPCoffea_dask/condor_dataset.hdf5"
event_vars, m = h5load(f, "vars")
event_vars_old, m_old = h5load(f_old, "vars")

pretty.pprint("Printing new file columns")
pretty.pprint(event_vars.columns)

pretty.pprint("Printing old file columns")
pretty.pprint(event_vars_old.columns)
