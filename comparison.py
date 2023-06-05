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
pretty.pprint(event_vars)
pretty.pprint(m)

pretty.pprint("Printing old file columns")
pretty.pprint(event_vars_old)
pretty.pprint(m_old)

# Check if the two datasets are the same
print("Checking if the two datasets are the same")
print(event_vars.equals(event_vars_old))
